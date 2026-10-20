"""Dependency-free packaging contract checks, not a PyInstaller/Windows build."""

import ast
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "build" / "GeoComb.spec"


class BuildSpecTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="GeoComb build test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ("build", "config", "pipeline_scripts"):
            shutil.copytree(ROOT / folder, self.root / folder)
        for entry in ("geocomb_entry.py", "geocomb_gpu_entry.py"):
            shutil.copyfile(ROOT / entry, self.root / entry)
        # Deliberately fake data, used only with mocked PyInstaller targets.
        # No real weights are read, altered, loaded, or used for inference.
        self.models = self.root / "models"
        self.models.mkdir()
        for name in ("frame_roi_yolo26m_1024_best.pt", "stage1_best.pt",
                     "stage2_best.pt", "stage3_best.pt"):
            (self.models / name).write_bytes(b"packaging-test-only")

    def evaluate(self, variant="CPU"):
        hooks = types.ModuleType("PyInstaller.utils.hooks")
        hooks.collect_all = lambda package: (
            [(f"/{package}/cfg/default.yaml", f"{package}/cfg")],
            [(f"/{package}/native.dll", package)],
            [f"{package}.nn.tasks"],
        )
        modules = {name: types.ModuleType(name) for name in (
            "PyInstaller", "PyInstaller.utils")}
        modules[hooks.__name__] = hooks
        calls = {}

        def target(name):
            def capture(*args, **kwargs):
                calls[name] = (args, kwargs)
                return types.SimpleNamespace(
                    pure=[], scripts=[], binaries=kwargs.get("binaries", []),
                    datas=kwargs.get("datas", []))
            return capture

        environment = {key: value for key, value in os.environ.items()
                       if key != "GEOCOMB_VARIANT"}
        if variant is not None:
            environment["GEOCOMB_VARIANT"] = variant
        globals_ = {name: target(name) for name in ("Analysis", "PYZ", "EXE", "COLLECT")}
        globals_["SPECPATH"] = str(self.root / "build")
        with patch.dict(sys.modules, modules), patch.dict(os.environ, environment, clear=True):
            exec(compile(SPEC.read_text(encoding="utf-8"), str(SPEC), "exec"), globals_)
        return calls

    def test_variants_match_batch_contract(self):
        for variant, entry in (("CPU", "geocomb_entry.py"), ("GPU", "geocomb_gpu_entry.py")):
            with self.subTest(variant=variant):
                calls = self.evaluate(variant)
                args, options = calls["Analysis"]
                self.assertEqual(args[0], [str(self.root / entry)])
                self.assertEqual(options["pathex"], [str(self.root)])
                name = f"GeoComb_{variant}"
                self.assertEqual(calls["EXE"][1]["name"], name)
                self.assertEqual(calls["COLLECT"][1]["name"], name)
                self.assertTrue(calls["EXE"][1]["exclude_binaries"])
                self.assertTrue(calls["EXE"][1]["console"])
                batch = (ROOT / f"build_{variant.lower()}.bat").read_text()
                self.assertIn('set "SPEC=build\\GeoComb.spec"', batch)
                self.assertIn(f'set "GEOCOMB_VARIANT={variant}"', batch)
                self.assertIn(f'%DIST%\\{name}\\{name}.exe', batch)

    def test_default_normalized_and_invalid_variants(self):
        self.assertEqual(self.evaluate(None)["EXE"][1]["name"], "GeoComb_CPU")
        self.assertEqual(self.evaluate(" gpu ")["EXE"][1]["name"], "GeoComb_GPU")
        with self.assertRaisesRegex(ValueError, "CPU or GPU"):
            self.evaluate("invalid")

    def test_runtime_assets_and_dynamic_script_analysis(self):
        options = self.evaluate()["Analysis"][1]
        assets = {(Path(src).name, dest) for src, dest in options["datas"]}
        tree = ast.parse((ROOT / "geocomb/pipeline.py").read_text())
        filenames = {node.value for node in ast.walk(tree)
                     if isinstance(node, ast.Constant) and isinstance(node.value, str)
                     and node.value.endswith((".py", ".pt"))}
        self.assertEqual(len(filenames), 11)
        for name in filenames:
            destination = "models" if name.endswith(".pt") else "pipeline_scripts"
            self.assertIn((name, destination), assets)
            if name.endswith(".py"):
                self.assertIn(f"pipeline_scripts.{Path(name).stem}", options["hiddenimports"])
        self.assertIn(("defaults.json", "config"), assets)
        self.assertIn("ultralytics.nn.tasks", options["hiddenimports"])
        self.assertIn("torchvision", options["hiddenimports"])
        self.assertTrue(options["binaries"])

    def test_paths_are_independent_of_working_directory(self):
        cwd = Path.cwd()
        try:
            os.chdir(self.root.parent)
            calls = self.evaluate()
        finally:
            os.chdir(cwd)
        self.assertTrue(all(Path(src).is_absolute() for src, _ in calls["Analysis"][1]["datas"]))

    def test_missing_assets_fail_before_analysis(self):
        for relative in ("geocomb_entry.py", "models/stage1_best.pt", "config/defaults.json",
                         "pipeline_scripts/apply_3stage_current_detection_5000coords_v1.py"):
            with self.subTest(asset=relative):
                path = self.root / relative
                original = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaisesRegex(FileNotFoundError, "Required build asset"):
                        self.evaluate()
                finally:
                    path.write_bytes(original)

    def test_lfs_pointers_and_empty_weights_are_rejected(self):
        for model in self.models.iterdir():
            original = model.read_bytes()
            for data, message in ((b"version https://git-lfs.github.com/spec/v1\n", "Git LFS pointer"),
                                  (b"", "empty")):
                with self.subTest(model=model.name, data=data):
                    model.write_bytes(data)
                    try:
                        with self.assertRaisesRegex(ValueError, message):
                            self.evaluate()
                    finally:
                        model.write_bytes(original)

    def test_private_regression_assets_are_not_collected(self):
        (self.root / "regression_test").mkdir()
        (self.root / "regression_test/private.jpg").write_bytes(b"not for distribution")
        datas = self.evaluate()["Analysis"][1]["datas"]
        self.assertFalse(any("regression_test" in src or "regression_test" in dest for src, dest in datas))

    def test_frozen_resource_layout_matches_spec(self):
        internal = self.root / self.evaluate()["EXE"][1]["contents_directory"]
        with patch.object(sys, "frozen", True, create=True), patch.object(sys, "_MEIPASS", str(internal), create=True):
            resources = runpy.run_path(str(ROOT / "geocomb/resources.py"))
            self.assertEqual(resources["models_dir"](), internal / "models")
            self.assertEqual(resources["scripts_dir"](), internal / "pipeline_scripts")

    def test_gitignore_preserves_spec_only(self):
        # Isolated Git repository: works even when these tests came from a ZIP.
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        shutil.copyfile(ROOT / ".gitignore", self.root / ".gitignore")
        for path, ignored in (("build/GeoComb.spec", False), ("build/temp.obj", True),
                              ("build/cache/file", True), ("build_output/cpu/work/file", True),
                              ("dist/output", True), ("release/package.zip", True)):
            result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=self.root)
            self.assertEqual(result.returncode, 0 if ignored else 1, path)


if __name__ == "__main__":
    unittest.main()
