from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .database import ProjectDatabase
from .excel_export import export_excel
from .overlay_export import export_all_overlays
from .pipeline import PipelineRunner
from .validation import validate_installation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="GeoComb",
        description=(
            "Geometry-aware honey bee comb-cell detection, "
            "classification, editing, and export."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze every image in a folder.",
    )
    analyze.add_argument(
        "--input-folder",
        type=Path,
        required=True,
    )
    analyze.add_argument(
        "--project-folder",
        type=Path,
        required=True,
    )
    analyze.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    analyze.add_argument(
        "--overwrite",
        action="store_true",
    )

    export = subparsers.add_parser(
        "export",
        help="Regenerate Excel, CSV, and overlays from an edited project.",
    )
    export.add_argument(
        "--project-folder",
        type=Path,
        required=True,
    )

    validate = subparsers.add_parser(
        "validate",
        help=("Run the optional local regression test; "
              "authorized regression_test assets are required."),
    )
    validate.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
    )
    validate.add_argument(
        "--output-folder",
        type=Path,
        default=None,
    )

    return parser


def main(arguments: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(arguments)

    if args.command == "analyze":
        runner = PipelineRunner()
        runner.analyze(
            input_folder=args.input_folder,
            project_folder=args.project_folder,
            device=args.device,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "export":
        database = ProjectDatabase(
            args.project_folder
        )
        if not database.database_path.is_file():
            raise FileNotFoundError(
                database.database_path
            )

        database.export_cells_csv(
            database.project_folder
            / "results"
            / "cells"
        )
        export_all_overlays(database)
        path, ok, message = export_excel(database)
        print(message)
        print(path)
        return 0 if ok else 2

    if args.command == "validate":
        result = validate_installation(
            device=args.device,
            output_folder=args.output_folder,
        )
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result["passed"] else 3

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
