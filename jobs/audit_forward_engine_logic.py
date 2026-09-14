from __future__ import annotations

from pathlib import Path
import inspect
import sys


# ============================================================
# FORCE UTF-8 OUTPUT
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace",
    )


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# IMPORTS
# ============================================================

from engine.ratings import TeamRatingsEngine
from engine.opponent_adjustment import OpponentAdjustmentEngine


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_class_source(
    title: str,
    cls,
) -> None:

    section(title)

    try:

        source = inspect.getsource(
            cls
        )

        print(source)

    except Exception as error:

        print(
            f"Could not inspect {cls.__name__}: "
            f"{error}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— REMAINING FORWARD ENGINE LOGIC AUDIT"
    )

    print_class_source(
        "RAW TEAM RATINGS ENGINE SOURCE",
        TeamRatingsEngine,
    )

    print_class_source(
        "OPPONENT ADJUSTMENT ENGINE SOURCE",
        OpponentAdjustmentEngine,
    )

    section(
        "REMAINING FORWARD ENGINE LOGIC AUDIT COMPLETE"
    )

    print(
        "No model state was changed."
    )

    print(
        "No data or model artifact was changed."
    )


if __name__ == "__main__":
    main()