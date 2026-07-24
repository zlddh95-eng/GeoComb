from __future__ import annotations

import contextlib
import io
import os
import runpy
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .database import ProjectDatabase
from .excel_export import export_excel
from .overlay_export import export_all_overlays
from .resources import models_dir, scripts_dir
from .utils import copy_images, write_json


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[str, int, int], None]

_RUN_SCRIPT_LOCK = threading.Lock()


class CallbackWriter(io.TextIOBase):
    """Forward redirected output without recursively re-entering stdout."""

    def __init__(
        self,
        callback: LogCallback,
        fallback_stdout: io.TextIOBase,
        fallback_stderr: io.TextIOBase,
    ):
        self.callback = callback
        self.buffer = ""
        self.fallback_stdout = fallback_stdout
        self.fallback_stderr = fallback_stderr
        self._emitting = False

    def _emit(self, line: str) -> None:
        if not line:
            return

        if self._emitting:
            self.fallback_stdout.write(line + "\n")
            self.fallback_stdout.flush()
            return

        self._emitting = True
        current_stdout = sys.stdout
        current_stderr = sys.stderr

        try:
            sys.stdout = self.fallback_stdout
            sys.stderr = self.fallback_stderr
            self.callback(line)
        finally:
            sys.stdout = current_stdout
            sys.stderr = current_stderr
            self._emitting = False

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)

        self.buffer += text

        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._emit(line.rstrip("\r"))

        return len(text)

    def flush(self) -> None:
        if self.buffer:
            self._emit(self.buffer.rstrip("\r"))
            self.buffer = ""


@dataclass(frozen=True)
class PipelinePaths:
    project: Path
    input_images: Path
    pipeline_root: Path
    crop_root: Path
    hough_root: Path
    step1_root: Path
    step2_root: Path
    final_detection_root: Path
    prediction_root: Path
    initial_overlay_root: Path

    @classmethod
    def create(cls, project: Path) -> "PipelinePaths":
        pipeline_root = project / "pipeline"
        return cls(
            project=project,
            input_images=project / "input_images",
            pipeline_root=pipeline_root,
            crop_root=pipeline_root / "00_frame_crop",
            hough_root=pipeline_root / "01_hough" / "test",
            step1_root=pipeline_root / "02_pitch_filter",
            step2_root=pipeline_root / "03_localhex",
            final_detection_root=pipeline_root / "04_magenta_filter",
            prediction_root=pipeline_root / "05_predictions",
            initial_overlay_root=pipeline_root / "06_initial_overlays",
        )


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()

    if requested not in {"auto", "cpu", "cuda", "gpu", "0"}:
        raise ValueError(
            "Device must be one of: auto, cpu, cuda"
        )

    if requested == "cpu":
        return "cpu"

    try:
        import torch
        available = bool(torch.cuda.is_available())
    except Exception:
        available = False

    if requested in {"cuda", "gpu", "0"} and not available:
        raise RuntimeError(
            "NVIDIA GPU mode was requested, but CUDA is not available."
        )

    return "0" if available else "cpu"


def run_python_script(
    script: Path,
    arguments: list[str],
    log: LogCallback,
) -> None:
    if not script.is_file():
        raise FileNotFoundError(script)

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    writer = CallbackWriter(
        log,
        fallback_stdout=original_stdout,
        fallback_stderr=original_stderr,
    )

    with _RUN_SCRIPT_LOCK:
        try:
            sys.argv = [str(script), *map(str, arguments)]
            os.chdir(script.parent)

            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                try:
                    runpy.run_path(
                        str(script),
                        run_name="__main__",
                    )
                except SystemExit as exc:
                    code = 0 if exc.code is None else int(exc.code)
                    if code != 0:
                        raise RuntimeError(
                            f"{script.name} exited with code {code}"
                        ) from exc
        finally:
            writer.flush()
            sys.argv = old_argv
            os.chdir(old_cwd)


class PipelineRunner:
    def __init__(
        self,
        log_callback: LogCallback | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        self.log = log_callback or print
        self.progress = progress_callback or (
            lambda stage, current, total: None
        )

    def _stage(
        self,
        name: str,
        current: int,
        total: int,
    ) -> None:
        self.log("")
        self.log(f"===== {current}/{total}: {name} =====")
        self.progress(name, current, total)

    def analyze(
        self,
        input_folder: Path,
        project_folder: Path,
        device: str = "auto",
        overwrite: bool = False,
    ) -> ProjectDatabase:
        input_folder = Path(input_folder).resolve()
        project_folder = Path(project_folder).resolve()

        if not input_folder.is_dir():
            raise NotADirectoryError(input_folder)

        if (
            project_folder == input_folder
            or input_folder.is_relative_to(project_folder)
        ):
            raise ValueError(
                "The project folder cannot be the input folder or a parent "
                "of the input folder, because overwrite mode would remove "
                "the source images."
            )

        if project_folder.exists() and any(project_folder.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Project folder is not empty: {project_folder}"
                )
            shutil.rmtree(project_folder)

        project_folder.mkdir(parents=True, exist_ok=True)
        paths = PipelinePaths.create(project_folder)
        paths.input_images.mkdir(parents=True, exist_ok=True)

        resolved_device = resolve_device(device)
        self.log(f"Resolved device: {resolved_device}")
        roi_device = "cpu"
        self.log(
            "Platform-stable preprocessing: ROI device=cpu, bbox quantum=32, "
            "Hough selection=plateau_median"
        )

        manifest = copy_images(
            input_folder,
            paths.input_images,
        )
        write_json(
            project_folder / "input_manifest.json",
            manifest,
        )

        scripts = scripts_dir()
        models = models_dir()

        frame_model = models / "frame_roi_yolo26m_1024_best.pt"
        stage1_model = models / "stage1_best.pt"
        stage2_model = models / "stage2_best.pt"
        stage3_model = models / "stage3_best.pt"

        required = [
            frame_model,
            stage1_model,
            stage2_model,
            stage3_model,
            scripts / "detect_frame_crop_resize_puremagenta_from_raw_v1.py",
            scripts / "detect_hough_circles_lattice_auto_radius_on_warped_v1.py",
            scripts / "filter_existing_hough_pitch_fixedR046_overlaparea30_step1_v1.py",
            scripts / "alternating_regularize_localhex_from_gray_step1_v1.py",
            scripts / "filter_pure_magenta_cells_after_step2_v1.py",
            scripts / "apply_3stage_current_detection_5000coords_v1.py",
            scripts / "draw_cell_overlay_outline_legend_corner_v2.py",
        ]

        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)

        total_stages = 9

        self._stage("Frame ROI segmentation and magenta crops", 1, total_stages)
        run_python_script(
            scripts / "detect_frame_crop_resize_puremagenta_from_raw_v1.py",
            [
                "--image_dir", str(paths.input_images),
                "--model", str(frame_model),
                "--out_root", str(paths.crop_root),
                "--split", "test",
                "--target_longs", "3000", "5000",
                "--imgsz", "1024",
                "--conf", "0.25",
                "--margin_ratio", "0.02",
                "--preview_w", "2200",
                "--device", roi_device,
                "--bbox_quantum", "32",
                "--opencv_threads", "1",
                "--clean",
            ],
            self.log,
        )

        image_root_3000 = (
            paths.crop_root
            / "00_frame_crop_masked_magenta_3000"
            / "images"
            / "test"
        )
        image_root_5000 = (
            paths.crop_root
            / "01_frame_crop_masked_magenta_5000"
            / "images"
        )

        self._stage("Adaptive Hough seed detection", 2, total_stages)
        run_python_script(
            scripts / "detect_hough_circles_lattice_auto_radius_on_warped_v1.py",
            [
                "--image_dir", str(image_root_3000),
                "--out_root", str(paths.hough_root),
                "--preprocess", "gray_clahe_blur",
                "--radius_min", "12",
                "--radius_max", "24",
                "--score_param2_list", "12",
                "--final_param2_list", "10", "12", "14",
                "--radius_selection", "plateau_median",
                "--plateau_score_fraction", "0.60",
                "--plateau_min_candidates", "3",
                "--opencv_threads", "1",
                "--preview_w", "2200",
                "--clean",
            ],
            self.log,
        )

        self._stage("Pitch-based overlap filtering", 3, total_stages)
        run_python_script(
            scripts / "filter_existing_hough_pitch_fixedR046_overlaparea30_step1_v1.py",
            [
                "--image_dir", str(image_root_3000),
                "--old_hough_root", str(paths.hough_root),
                "--out_root", str(paths.step1_root),
                "--split", "test",
                "--fixed_radius_ratio", "0.48",
                "--overlap_frac_thr", "0.33",
                "--preview_w", "2200",
                "--clean",
            ],
            self.log,
        )

        self._stage("Local regularization and hexagonal gap filling", 4, total_stages)
        run_python_script(
            scripts / "alternating_regularize_localhex_from_gray_step1_v1.py",
            [
                "--image_dir", str(image_root_3000),
                "--input_root", str(paths.step1_root),
                "--out_root", str(paths.step2_root),
                "--split", "test",
                "--fixed_radius_ratio", "0.48",
                "--phase0_max_move_ratio", "0.12",
                "--phase0_max_iters", "30",
                "--phaseA_strict_iters", "2",
                "--phaseA_recall_iters", "3",
                "--phaseB_max_move_ratio", "0.06",
                "--phaseB_max_iters", "20",
                "--phaseC_strict_iters", "0",
                "--phaseC_recall_iters", "5",
                "--max_step_ratio", "0.025",
                "--force_gain", "0.25",
                "--neighbor_low_ratio", "0.70",
                "--neighbor_high_ratio", "1.35",
                "--close_ratio", "0.92",
                "--close_weight", "3.0",
                "--candidate_distance_ratio", "1.00",
                "--existing_exclude_ratio", "0.52",
                "--candidate_nms_ratio", "0.52",
                "--support_low_ratio", "0.70",
                "--support_high_ratio", "1.35",
                "--strict_min_support", "3",
                "--strict_min_orig_support", "2",
                "--strict_min_support_dirs", "2",
                "--recall_min_support", "2",
                "--recall_min_orig_support", "0",
                "--recall_min_support_dirs", "2",
                "--max_overlap_sum_frac", "0.33",
                "--max_overlap_pair_frac", "0.33",
                "--final_overlap_frac_thr", "0.33",
                "--preview_w", "2200",
            ],
            self.log,
        )

        self._stage("Magenta-background filtering", 5, total_stages)
        run_python_script(
            scripts / "filter_pure_magenta_cells_after_step2_v1.py",
            [
                "--image_dir", str(image_root_3000),
                "--input_root", str(paths.step2_root),
                "--out_root", str(paths.final_detection_root),
                "--split", "test",
                "--fixed_radius_ratio", "0.48",
                "--max_magenta_frac", "0.30",
                "--magenta_tolerance", "3",
                "--preview_w", "2200",
                "--clean",
            ],
            self.log,
        )

        center_root = (
            paths.final_detection_root
            / "08_pure_magenta_cut_seed"
        )

        self._stage("Three-stage cell classification", 6, total_stages)
        run_python_script(
            scripts / "apply_3stage_current_detection_5000coords_v1.py",
            [
                "--image_root", str(image_root_5000),
                "--center_root", str(center_root),
                "--out_root", str(paths.prediction_root),
                "--stage1_model", str(stage1_model),
                "--cap_model", str(stage2_model),
                "--egg_model", str(stage3_model),
                "--splits", "test",
                "--coord_scale", "1.6666667",
                "--stage1_half_step", "0.80",
                "--stage1_imgsz", "320",
                "--cap_half_step", "1.80",
                "--cap_imgsz", "320",
                "--egg_half_step", "0.45",
                "--egg_imgsz", "384",
                "--batch_size", "64",
                "--device", resolved_device,
                "--preview_width", "2200",
                "--label_every", "999999999",
            ],
            self.log,
        )

        self._stage("Initial frozen-workflow overlays", 7, total_stages)
        run_python_script(
            scripts / "draw_cell_overlay_outline_legend_corner_v2.py",
            [
                "--image_root", str(image_root_5000),
                "--pred_root", str(paths.prediction_root),
                "--out_root", str(paths.initial_overlay_root),
                "--split", "test",
                "--radius_ratio", "0.46",
                "--thickness", "2",
                "--legend_position", "top_right",
                "--legend_scale", "0.75",
                "--legend_alpha", "0.72",
                "--preview_width", "0",
            ],
            self.log,
        )

        self._stage("Editable project database", 8, total_stages)
        database = ProjectDatabase(project_folder)
        database.import_pipeline_results(
            input_manifest=manifest,
            crop_summary_path=(
                paths.crop_root
                / "summaries"
                / "frame_detection_crop_magenta_summary.csv"
            ),
            prediction_root=paths.prediction_root,
            crop_image_root=image_root_5000,
            replace_existing=True,
        )
        database.set_meta("device", resolved_device)
        database.set_meta("roi_device", roi_device)
        database.set_meta("bbox_quantum", "32")
        database.set_meta("hough_radius_selection", "plateau_median")
        database.set_meta("hough_plateau_score_fraction", "0.60")
        database.set_meta("input_folder", str(input_folder))
        database.set_meta(
            "pipeline_version",
            "GeoComb platform-stable candidate v2",
        )

        self._stage("Excel, CSV, and editable overlays", 9, total_stages)
        database.export_cells_csv(
            project_folder / "results" / "cells"
        )
        export_all_overlays(database)
        excel_path, excel_ok, excel_message = export_excel(database)
        self.log(f"{excel_message}: {excel_path}")

        self.progress("Complete", total_stages, total_stages)
        self.log(f"Project complete: {project_folder}")
        return database
