from __future__ import annotations

"""Backward-compatible V3-only entry point."""

import subprocess
import sys


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "jobs.run_pipeline", "--refresh-current", "--skip-v2"],
        check=True,
    )


if __name__ == "__main__":
    main()
