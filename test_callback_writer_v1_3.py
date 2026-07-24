from __future__ import annotations

import tempfile
from pathlib import Path

from geocomb.pipeline import run_python_script


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="geocomb_writer_test_") as temp:
        script = Path(temp) / "print_test.py"
        script.write_text(
            'print("stdout line 1")\n'
            'print("stdout line 2")\n',
            encoding="utf-8",
        )

        captured: list[str] = []

        def callback(line: str) -> None:
            print(f"[callback] {line}")
            captured.append(line)

        run_python_script(script, [], callback)
        assert captured == ["stdout line 1", "stdout line 2"], captured

    print("CallbackWriter recursion test PASSED.")


if __name__ == "__main__":
    main()
