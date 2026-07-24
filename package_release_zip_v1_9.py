from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()

    if not source.is_dir():
        raise NotADirectoryError(source)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        destination.unlink()

    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
    )

    total_bytes = sum(
        path.stat().st_size
        for path in files
    )
    processed = 0

    print(
        f"Creating ZIP64 archive with {len(files):,} files "
        f"({total_bytes / (1024 ** 3):.2f} GiB before compression)"
    )

    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for index, path in enumerate(files, start=1):
            relative = path.relative_to(source)
            archive.write(
                path,
                arcname=relative,
            )
            processed += path.stat().st_size

            if index % 250 == 0 or index == len(files):
                percent = (
                    100.0 * processed / total_bytes
                    if total_bytes
                    else 100.0
                )
                print(
                    f"  {index:,}/{len(files):,} files "
                    f"({percent:.1f}%)"
                )

    if not destination.is_file():
        raise RuntimeError(
            f"ZIP was not created: {destination}"
        )

    print(
        f"ZIP created: {destination} "
        f"({destination.stat().st_size / (1024 ** 3):.2f} GiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
