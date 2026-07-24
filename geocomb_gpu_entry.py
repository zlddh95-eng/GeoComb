from __future__ import annotations

import multiprocessing
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from geocomb.cli import main as cli_main
from geocomb.gui import GeoCombWindow


CLI_COMMANDS = {
    "analyze",
    "export",
    "validate",
}


def run_gpu_gui() -> int:
    application = (
        QApplication.instance()
        or QApplication([])
    )
    application.setApplicationName(
        "GeoComb NVIDIA GPU"
    )

    try:
        import torch

        if not torch.cuda.is_available():
            QMessageBox.critical(
                None,
                "NVIDIA GPU unavailable",
                (
                    "GeoComb GPU could not access an NVIDIA CUDA GPU.\n\n"
                    "Install a compatible NVIDIA driver, or use the "
                    "GeoComb CPU edition."
                ),
            )
            return 2

        gpu_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        QMessageBox.critical(
            None,
            "GPU initialization failed",
            str(exc),
        )
        return 2

    window = GeoCombWindow()
    window.device_combo.setCurrentText("cuda")
    window.setWindowTitle(
        f"GeoComb — NVIDIA GPU ({gpu_name})"
    )
    window.statusBar().showMessage(
        f"CUDA ready: {gpu_name}"
    )
    window.show()

    return application.exec()


def main() -> int:
    multiprocessing.freeze_support()

    if (
        len(sys.argv) > 1
        and sys.argv[1] in CLI_COMMANDS
    ):
        return cli_main(sys.argv[1:])

    return run_gpu_gui()


if __name__ == "__main__":
    raise SystemExit(main())
