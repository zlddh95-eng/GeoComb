from __future__ import annotations

import sys

from geocomb.cli import main as cli_main
from geocomb.gui import run_gui


CLI_COMMANDS = {
    "analyze",
    "export",
    "validate",
}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in CLI_COMMANDS:
        return cli_main(sys.argv[1:])
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
