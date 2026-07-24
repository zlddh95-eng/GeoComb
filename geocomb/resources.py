from __future__ import annotations

import sys
from pathlib import Path


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    return application_root().joinpath(*parts)


def models_dir() -> Path:
    return resource_path("models")


def scripts_dir() -> Path:
    return resource_path("pipeline_scripts")


def regression_dir() -> Path:
    return resource_path("regression_test")
