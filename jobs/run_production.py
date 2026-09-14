from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

V1_PROJECT_ROOT = (
    PROJECT_ROOT.parent
    / "CFB_Prediction_Centre2026"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "predictions"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

RUN_MANIFEST_PATH = (
    PREDICTIONS_DIR
    / "production_run_manifest.json"
)


# ============================================================
# PRODUCTION PIPELINE
# ============================================================

V2_STEPS = [
    (
        "Audit refreshed 2026 source",
        "jobs.audit_2026_source",
    ),
    (
        "Fetch completed 2026 box scores",
        "jobs.fetch_2026_box_scores",
    ),
    (
        "Build 2026 forward data",
        "jobs.build_2026_forward_data",
    ),
    (
        "Run frozen winner model",
        "jobs.run_2026_forward",
    ),
    (
        "Run frozen score / simulation model",
        "jobs.run_2026_scoring",
    ),
    (
        "Update immutable prediction tracker",
        "jobs.update_prediction_tracker",
    ),
    (
        "Update 24-hour official locks",
        "jobs.update_official_locks",
    ),
]


# ============================================================
# DISPLAY
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# TIME
# ============================================================

def utc_now() -> datetime:

    return datetime.now(
        timezone.utc
    )


def iso_utc(
    value: datetime,
) -> str:

    return (
        value
        .isoformat(
            timespec="seconds"
        )
        .replace(
            "+00:00",
            "Z",
        )
    )


# ============================================================
# PYTHON EXECUTABLES
# ============================================================

def v1_python_executable() -> Path:

    windows_venv = (
        V1_PROJECT_ROOT
        / ".venv"
        / "Scripts"
        / "python.exe"
    )

    if windows_venv.exists():
        return windows_venv

    unix_venv = (
        V1_PROJECT_ROOT
        / ".venv"
        / "bin"
        / "python"
    )

    if unix_venv.exists():
        return unix_venv

    return Path(
        sys.executable
    )


def v2_python_executable() -> Path:

    return Path(
        sys.executable
    )


# ============================================================
# FILE SAFETY
# ============================================================

def require_file(
    path: Path,
    label: str,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def preflight() -> None:

    section(
        "1. PRODUCTION PREFLIGHT"
    )

    if not V1_PROJECT_ROOT.exists():

        raise FileNotFoundError(
            "V1 project folder not found:\n"
            f"{V1_PROJECT_ROOT}"
        )

    print(
        f"V1 project: "
        f"{V1_PROJECT_ROOT}"
    )

    print(
        f"V2 project: "
        f"{PROJECT_ROOT}"
    )

    print(
        f"V1 Python:  "
        f"{v1_python_executable()}"
    )

    print(
        f"V2 Python:  "
        f"{v2_python_executable()}"
    )

    require_file(
        PROCESSED_DIR
        / "historical_team_games.csv",
        "Historical team-game data",
    )

    require_file(
        PROCESSED_DIR
        / "matchup_history.csv",
        "Historical matchup data",
    )

    require_file(
        PROCESSED_DIR
        / "v2_probability_model.joblib",
        "Frozen winner model",
    )

    require_file(
        PROCESSED_DIR
        / "v2_score_model.joblib",
        "Frozen score model",
    )

    require_file(
        PROCESSED_DIR
        / "v2_corrected_score_uncertainty.json",
        "Corrected Monte Carlo uncertainty artifact",
    )

    print()

    print(
        "PASS - frozen model artifacts and "
        "historical state inputs are present."
    )


# ============================================================
# STEP RUNNER
# ============================================================

def run_module(
    *,
    step_number: int,
    total_steps: int,
    label: str,
    module: str,
    cwd: Path,
    python_executable: Path,
) -> dict[str, Any]:

    section(
        f"{step_number}. {label.upper()}"
    )

    command = [
        str(
            python_executable
        ),
        "-m",
        module,
    ]

    print(
        f"Working directory: {cwd}"
    )

    print(
        "Command: "
        + " ".join(
            command
        )
    )

    print()

    started = utc_now()
    timer = time.perf_counter()

    environment = os.environ.copy()

    environment[
        "PYTHONIOENCODING"
    ] = "utf-8"

    environment[
        "PYTHONUTF8"
    ] = "1"

    result = subprocess.run(
        command,
        cwd=str(
            cwd
        ),
        env=environment,
        check=False,
    )

    duration_seconds = (
        time.perf_counter()
        -
        timer
    )

    finished = utc_now()

    status = (
        "passed"
        if result.returncode == 0
        else
        "failed"
    )

    print()

    print(
        f"Step result: {status.upper()}"
    )

    print(
        f"Return code: {result.returncode}"
    )

    print(
        f"Duration:    "
        f"{duration_seconds:.2f}s"
    )

    step_result = {
        "step_number": (
            step_number
        ),
        "total_steps": (
            total_steps
        ),
        "label": (
            label
        ),
        "module": (
            module
        ),
        "working_directory": (
            str(
                cwd
            )
        ),
        "python_executable": (
            str(
                python_executable
            )
        ),
        "started_at_utc": (
            iso_utc(
                started
            )
        ),
        "finished_at_utc": (
            iso_utc(
                finished
            )
        ),
        "duration_seconds": round(
            duration_seconds,
            3,
        ),
        "return_code": (
            int(
                result.returncode
            )
        ),
        "status": (
            status
        ),
    }

    if result.returncode != 0:

        raise RuntimeError(
            f"Production pipeline stopped because "
            f"step '{label}' failed with return code "
            f"{result.returncode}."
        )

    return step_result


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_final_outputs() -> dict[str, Any]:

    section(
        "FINAL OUTPUT VALIDATION"
    )

    required_outputs = [
        PREDICTIONS_DIR
        / "2026_forward_predictions.csv",

        PREDICTIONS_DIR
        / "2026_scoring_predictions.csv",

        PREDICTIONS_DIR
        / "prediction_tracker_history.csv",

        PREDICTIONS_DIR
        / "prediction_tracker_current.csv",

        PREDICTIONS_DIR
        / "official_locked_predictions.csv",

        PREDICTIONS_DIR
        / "official_lock_status.csv",
    ]

    output_state: dict[
        str,
        Any,
    ] = {}

    for path in required_outputs:

        if not path.exists():

            raise FileNotFoundError(
                "Production output validation failed. "
                f"Missing:\n{path}"
            )

        stat = path.stat()

        output_state[
            path.name
        ] = {
            "path": (
                str(
                    path
                )
            ),
            "size_bytes": (
                int(
                    stat.st_size
                )
            ),
            "modified_epoch": (
                float(
                    stat.st_mtime
                )
            ),
        }

        print(
            f"FOUND  {path.name:<38} "
            f"{stat.st_size:>12,} bytes"
        )

    print()

    print(
        "PASS - all required production outputs exist."
    )

    return output_state


# ============================================================
# MANIFEST
# ============================================================

def save_run_manifest(
    *,
    started_at: datetime,
    finished_at: datetime,
    steps: list[dict[str, Any]],
    outputs: dict[str, Any],
    skipped_v1_refresh: bool,
) -> None:

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "pipeline_name": (
            "CFB Prediction Centre 2026 V2 "
            "Production Pipeline"
        ),
        "pipeline_version": (
            "v2_production_runner_1"
        ),
        "started_at_utc": (
            iso_utc(
                started_at
            )
        ),
        "finished_at_utc": (
            iso_utc(
                finished_at
            )
        ),
        "duration_seconds": round(
            (
                finished_at
                -
                started_at
            )
            .total_seconds(),
            3,
        ),
        "status": (
            "passed"
        ),
        "v1_schedule_refresh_skipped": (
            bool(
                skipped_v1_refresh
            )
        ),
        "steps": (
            steps
        ),
        "outputs": (
            outputs
        ),
        "production_rules": {
            "winner_model": (
                "frozen"
            ),
            "score_model": (
                "frozen compact_full"
            ),
            "monte_carlo_uncertainty": (
                "corrected OOS residual state"
            ),
            "prediction_history": (
                "append-only logical history"
            ),
            "official_lock_hours_before_kickoff": (
                24
            ),
            "historical_reconstructions_can_lock": (
                False
            ),
            "official_locks_immutable": (
                True
            ),
        },
    }

    RUN_MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved production manifest -> "
        f"{RUN_MANIFEST_PATH}"
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete CFB Prediction Centre "
            "2026 V2 production refresh."
        )
    )

    parser.add_argument(
        "--skip-v1-refresh",
        action="store_true",
        help=(
            "Skip the V1 CFBD schedule/results refresh "
            "and use the current V1 database state."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "- PRODUCTION RUNNER"
    )

    print(
        "Pipeline:"
    )

    print()

    print(
        "CFBD schedule/results refresh"
    )

    print(
        "  -> 2026 source audit"
    )

    print(
        "  -> completed box-score fetch"
    )

    print(
        "  -> 2026 forward adapter"
    )

    print(
        "  -> frozen winner model"
    )

    print(
        "  -> frozen score + Monte Carlo model"
    )

    print(
        "  -> immutable prediction tracker"
    )

    print(
        "  -> 24-hour official prediction locks"
    )

    print()

    print(
        "Any failed step stops the pipeline immediately."
    )

    started_at = utc_now()

    step_results: list[
        dict[str, Any]
    ] = []

    try:

        preflight()

        total_steps = (
            len(
                V2_STEPS
            )
            +
            (
                0
                if args.skip_v1_refresh
                else
                1
            )
        )

        step_number = 1

        if args.skip_v1_refresh:

            section(
                "2. V1 REFRESH SKIPPED"
            )

            print(
                "Using the current V1 SQLite schedule/"
                "results state."
            )

        else:

            step_number += 1

            result = run_module(
                step_number=(
                    step_number
                ),
                total_steps=(
                    total_steps
                ),
                label=(
                    "Refresh V1 schedule and results from CFBD"
                ),
                module=(
                    "jobs.update_data"
                ),
                cwd=(
                    V1_PROJECT_ROOT
                ),
                python_executable=(
                    v1_python_executable()
                ),
            )

            step_results.append(
                result
            )

        for (
            label,
            module,
        ) in V2_STEPS:

            step_number += 1

            result = run_module(
                step_number=(
                    step_number
                ),
                total_steps=(
                    total_steps
                ),
                label=(
                    label
                ),
                module=(
                    module
                ),
                cwd=(
                    PROJECT_ROOT
                ),
                python_executable=(
                    v2_python_executable()
                ),
            )

            step_results.append(
                result
            )

        outputs = (
            validate_final_outputs()
        )

        finished_at = utc_now()

        section(
            "PRODUCTION RUN COMPLETE"
        )

        print(
            "✓ Schedule/results source refreshed."
            if not args.skip_v1_refresh
            else
            "✓ Existing schedule/results source used."
        )

        print(
            "✓ 2026 source universe audited."
        )

        print(
            "✓ Completed box scores refreshed."
        )

        print(
            "✓ Forward model data rebuilt."
        )

        print(
            "✓ Frozen winner predictions regenerated."
        )

        print(
            "✓ Frozen score/simulation predictions regenerated."
        )

        print(
            "✓ Immutable tracker updated."
        )

        print(
            "✓ 24-hour official locks updated."
        )

        print(
            "✓ Existing official locks preserved."
        )

        print()

        print(
            f"Total duration: "
            f"{(finished_at - started_at).total_seconds():.2f}s"
        )

        save_run_manifest(
            started_at=(
                started_at
            ),
            finished_at=(
                finished_at
            ),
            steps=(
                step_results
            ),
            outputs=(
                outputs
            ),
            skipped_v1_refresh=(
                args.skip_v1_refresh
            ),
        )

        print()

        print(
            "Normal production command:"
        )

        print(
            "python -m jobs.run_production"
        )

    except Exception as exc:

        finished_at = utc_now()

        PREDICTIONS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        failure_manifest = {
            "pipeline_name": (
                "CFB Prediction Centre 2026 V2 "
                "Production Pipeline"
            ),
            "pipeline_version": (
                "v2_production_runner_1"
            ),
            "started_at_utc": (
                iso_utc(
                    started_at
                )
            ),
            "finished_at_utc": (
                iso_utc(
                    finished_at
                )
            ),
            "status": (
                "failed"
            ),
            "completed_steps": (
                step_results
            ),
            "error_type": (
                type(
                    exc
                ).__name__
            ),
            "error": (
                str(
                    exc
                )
            ),
        }

        RUN_MANIFEST_PATH.write_text(
            json.dumps(
                failure_manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

        section(
            "PRODUCTION RUN FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()

        print(
            "The pipeline stopped immediately. "
            "Downstream steps were not allowed to "
            "continue after the failure."
        )

        print()

        print(
            f"Failure manifest -> "
            f"{RUN_MANIFEST_PATH}"
        )

        raise


if __name__ == "__main__":
    main()
