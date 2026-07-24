from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .constants import CLASS_COLORS_BGR, CLASS_ORDER
from .database import ProjectDatabase


def draw_legend(
    image: np.ndarray,
    counts: dict[str, int],
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, width / 7000.0)
    thickness = max(1, int(round(font_scale * 2)))
    pad = int(round(16 * font_scale))
    row_height = int(round(34 * font_scale))
    swatch = int(round(20 * font_scale))

    labels = CLASS_ORDER
    text_width = max(
        cv2.getTextSize(
            f"{label}: {counts.get(label, 0)}",
            font,
            font_scale,
            thickness,
        )[0][0]
        for label in labels
    )
    box_width = pad * 3 + swatch + text_width
    box_height = pad * 2 + row_height * (len(labels) + 1)
    margin = max(12, pad)

    x0 = max(0, width - box_width - margin)
    y0 = margin
    x1 = min(width - 1, x0 + box_width)
    y1 = min(height - 1, y0 + box_height)

    layer = output.copy()
    cv2.rectangle(
        layer,
        (x0, y0),
        (x1, y1),
        (18, 18, 18),
        -1,
    )
    cv2.addWeighted(
        layer,
        0.72,
        output,
        0.28,
        0,
        dst=output,
    )
    cv2.rectangle(
        output,
        (x0, y0),
        (x1, y1),
        (240, 240, 240),
        thickness,
    )

    y = y0 + pad + row_height // 2
    cv2.putText(
        output,
        f"Total: {sum(counts.values())}",
        (x0 + pad, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    y += row_height

    for label in labels:
        color = CLASS_COLORS_BGR[label]
        y_mid = y
        cv2.rectangle(
            output,
            (
                x0 + pad,
                y_mid - swatch // 2,
            ),
            (
                x0 + pad + swatch,
                y_mid + swatch // 2,
            ),
            color,
            -1,
        )
        cv2.putText(
            output,
            f"{label}: {counts.get(label, 0)}",
            (
                x0 + pad * 2 + swatch,
                y_mid + int(round(7 * font_scale)),
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += row_height

    return output


def _draw_cells(
    image: np.ndarray,
    rows: Iterable,
    coordinate_mode: str,
    scale_5000: float,
    thickness: int = 2,
) -> np.ndarray:
    output = image.copy()

    for row in rows:
        class_name = str(row["class_current"])
        color = CLASS_COLORS_BGR[class_name]

        if coordinate_mode == "original":
            x = float(row["x_original"])
            y = float(row["y_original"])
            step = (
                float(row["estimated_step_canvas"])
                / float(scale_5000)
            )
        else:
            x = float(row["x_canvas"])
            y = float(row["y_canvas"])
            step = float(row["estimated_step_canvas"])

        radius = max(3, min(100, int(round(step * 0.46))))
        center = (int(round(x)), int(round(y)))

        cv2.circle(
            output,
            center,
            radius,
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        if row["edited"]:
            cv2.circle(
                output,
                center,
                max(2, thickness + 1),
                (0, 0, 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    return output


def export_image_overlays(
    database: ProjectDatabase,
    image_id: int,
) -> tuple[Path, Path]:
    image = database.get_image(image_id)
    rows = database.cells_for_image(image_id)
    counts = database.class_counts(image_id)

    crop = cv2.imread(
        str(image["crop_image_path"]),
        cv2.IMREAD_COLOR,
    )
    original = cv2.imread(
        str(image["copied_path"]),
        cv2.IMREAD_COLOR,
    )

    if crop is None:
        raise FileNotFoundError(image["crop_image_path"])
    if original is None:
        raise FileNotFoundError(image["copied_path"])

    crop_overlay = _draw_cells(
        crop,
        rows,
        "canvas",
        float(image["scale_5000"]),
    )
    crop_overlay = draw_legend(crop_overlay, counts)

    original_overlay = _draw_cells(
        original,
        rows,
        "original",
        float(image["scale_5000"]),
    )
    original_overlay = draw_legend(
        original_overlay,
        counts,
    )

    overlay_root = (
        database.project_folder
        / "results"
        / "overlays"
    )
    crop_folder = overlay_root / "frame_crop_5000"
    original_folder = overlay_root / "original_image"
    crop_folder.mkdir(parents=True, exist_ok=True)
    original_folder.mkdir(parents=True, exist_ok=True)

    crop_path = (
        crop_folder
        / f"{image['safe_stem']}_overlay.png"
    )
    original_path = (
        original_folder
        / f"{image['safe_stem']}_overlay.png"
    )

    cv2.imwrite(str(crop_path), crop_overlay)
    cv2.imwrite(str(original_path), original_overlay)

    return crop_path, original_path


def export_all_overlays(
    database: ProjectDatabase,
) -> list[tuple[Path, Path]]:
    return [
        export_image_overlays(
            database,
            int(image["id"]),
        )
        for image in database.list_images()
    ]
