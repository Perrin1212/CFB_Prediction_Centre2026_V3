from __future__ import annotations

"""Backward-compatible alias for the complete V3 production update."""

import subprocess
import sys


def main() -> None:
    print("run_daily_update now delegates to the complete V3 production pipeline.")
    subprocess.run(
        [sys.executable, "-m", "jobs.run_pipeline", "--refresh-current"],
        check=True,
    )


if __name__ == "__main__":
    main()
