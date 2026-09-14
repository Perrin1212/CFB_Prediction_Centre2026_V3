from __future__ import annotations

"""One-command V2 control + V3 production data, model, forecast and app update."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = ROOT / "data" / "processed" / "v3" / "frozen_2021_2025" / "manifest.json"
RUN_MANIFEST = ROOT / "data" / "audits" / "v3_pipeline_run.json"
V2_REQUIRED = (
    ROOT / "data" / "processed" / "matchup_history.csv",
    ROOT / "data" / "processed" / "v2_probability_model.joblib",
    ROOT / "data" / "processed" / "v2_score_model.joblib",
    ROOT / "data" / "processed" / "v2_corrected_score_uncertainty.json",
)


def run(module: str, arguments: list[str] | None = None, required: bool = True) -> bool:
    command = [sys.executable, "-m", module, *(arguments or [])]
    print("\n" + "=" * 78)
    print("RUNNING:", " ".join(command))
    print("=" * 78, flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode and required:
        raise RuntimeError(f"{module} failed with exit code {result.returncode}")
    if result.returncode:
        print(f"WARNING: {module} failed with exit code {result.returncode}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-current", action="store_true")
    parser.add_argument("--rebuild-history", action="store_true")
    parser.add_argument("--simulations", type=int, default=50_000)
    parser.add_argument("--quick", action="store_true", help="Use 10,000 simulations per game.")
    parser.add_argument("--skip-v2", action="store_true")
    parser.add_argument("--skip-market", action="store_true")
    args = parser.parse_args()
    simulations = 10_000 if args.quick else args.simulations
    if simulations < 100:
        raise ValueError("Simulation count must be at least 100")

    started = datetime.now(timezone.utc)
    completed_steps: list[str] = []
    if not args.skip_market:
        run("jobs.capture_market_lines", required=False)
        completed_steps.append("market_capture_attempted")
    missing_v2 = [path for path in V2_REQUIRED if not path.exists()]
    run_v2 = not args.skip_v2 and not missing_v2
    if missing_v2 and not args.skip_v2:
        print("\nV2 control refresh skipped because its frozen artifacts are not present:")
        for path in missing_v2:
            print(f"  - {path.name}")
        print("V3 will continue normally; canonical V3 results will refresh the app status.")
        completed_steps.append("v2_control_unavailable")
    if run_v2:
        run("jobs.run_production")
        run("jobs.build_app_data")
        run("jobs.build_betting_performance")
        completed_steps.extend(["v2_control", "app_base", "betting_performance"])

    run("jobs.v3_acquire_history", ["--refresh-current"] if args.refresh_current else [])
    completed_steps.append("v3_acquisition")
    rebuild = args.rebuild_history or not FROZEN_MANIFEST.exists()
    if rebuild:
        run("jobs.v3_build_canonical")
        run("jobs.v3_freeze_history", ["--replace"])
        run("jobs.v3_train_and_validate")
        run("jobs.evaluate_v3_production")
        completed_steps.extend(["canonical_full", "history_frozen", "model_fit", "holdout_validation"])
    else:
        run("jobs.v3_refresh_current")
        completed_steps.append("canonical_current_refresh")

    run("jobs.v3_predict_2026", ["--simulations", str(simulations)])
    run("jobs.build_v3_app_overlay")
    run("jobs.build_v3_tracker")
    run("jobs.build_v3_betting_performance")
    run("jobs.build_v3_elo_app")
    run("jobs.build_v3_team_profiles")
    completed_steps.extend(["v3_predictions", "v3_app_overlay", "v3_tracker", "v3_betting"])
    manifest = {
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "refresh_current": bool(args.refresh_current),
        "rebuild_history": bool(rebuild),
        "simulations_per_game": int(simulations),
        "steps": completed_steps,
    }
    RUN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    RUN_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print("CFB V3 PRODUCTION UPDATE COMPLETE")
    print("=" * 78)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
