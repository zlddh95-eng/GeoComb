from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

from .constants import SUPPORTED_IMAGE_EXTENSIONS


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.casefold(),
    )


def sanitize_stem(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    text = text.strip("._")
    return text or "image"


def make_safe_image_names(paths: Iterable[Path]) -> list[tuple[Path, str]]:
    used: set[str] = set()
    rows: list[tuple[Path, str]] = []

    for index, source in enumerate(paths, 1):
        base = sanitize_stem(source.stem)
        extension = source.suffix.lower()
        candidate = f"{base}{extension}"

        if candidate.casefold() in used:
            candidate = f"{base}_{index:04d}{extension}"

        while candidate.casefold() in used:
            candidate = (
                f"{base}_{index:04d}_"
                f"{hashlib.sha1(str(source).encode()).hexdigest()[:6]}"
                f"{extension}"
            )

        used.add(candidate.casefold())
        rows.append((source, candidate))

    return rows


def copy_images(
    input_folder: Path,
    destination: Path,
) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    sources = image_files(input_folder)

    if not sources:
        raise FileNotFoundError(
            f"No supported images were found in {input_folder}"
        )

    manifest: list[dict] = []

    for source, safe_name in make_safe_image_names(sources):
        target = destination / safe_name
        shutil.copy2(source, target)
        manifest.append(
            {
                "original_name": source.name,
                "source_path": str(source.resolve()),
                "safe_name": safe_name,
                "safe_stem": Path(safe_name).stem,
                "copied_path": str(target.resolve()),
            }
        )

    return manifest


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
