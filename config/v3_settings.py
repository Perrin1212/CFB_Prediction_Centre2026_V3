from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class V3Settings:
    seasons: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025, 2026)
    train_seasons: tuple[int, ...] = (2021, 2022, 2023, 2024)
    holdout_season: int = 2025
    live_season: int = 2026
    weeks: tuple[int, ...] = tuple(range(0, 17))
    classification: str = "fbs"
    raw_root: Path = ROOT / "data" / "raw" / "cfbd"
    processed_root: Path = ROOT / "data" / "processed" / "v3"
    model_root: Path = ROOT / "data" / "models" / "v3"
    prediction_root: Path = ROOT / "data" / "predictions"
    audit_root: Path = ROOT / "data" / "audits"
    # Fast local diagnostics can explicitly request fewer simulations, but the
    # production forecast is generated from 50,000 reproducible trials.
    simulations: int = 50000
    validation_simulations: int = 5000
    seed: int = 20260908
    preseason_regression: float = 0.42
    opponent_iterations: int = 8
    min_games_for_full_weight: int = 5
    current_season_weights: tuple[float, ...] = (
        0.00,
        0.35,
        0.55,
        0.70,
        0.80,
        0.90,
        0.92,
    )
    home_field_points: float = 2.25
    same_kickoff_tolerance_minutes: int = 1
    max_reasonable_games_per_season: int = 1200
    min_reasonable_games_per_full_season: int = 500
    market_forbidden_tokens: tuple[str, ...] = (
        "spread", "moneyline", "odds", "market", "book", "consensus", "closing", "open_line", "vegas"
    )

SETTINGS = V3Settings()
for p in (SETTINGS.raw_root, SETTINGS.processed_root, SETTINGS.model_root,
          SETTINGS.prediction_root, SETTINGS.audit_root):
    p.mkdir(parents=True, exist_ok=True)
