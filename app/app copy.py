from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:  # pragma: no cover - graceful fallback if Altair is unavailable
    alt = None


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DATA_DIR = PROJECT_ROOT / "data" / "app"

GAMES_PATH = APP_DATA_DIR / "games.csv"
GAMES_V3_PATH = APP_DATA_DIR / "games_v3.csv"
ELO_WEEKLY_PATH = APP_DATA_DIR / "elo_weekly.csv"
ELO_RANKINGS_PATH = APP_DATA_DIR / "elo_rankings_current.csv"
PERFORMANCE_PATH = APP_DATA_DIR / "performance.json"
BETTING_PATH = APP_DATA_DIR / "betting_performance.csv"
TEAM_PROFILES_PATH = APP_DATA_DIR / "team_profiles.csv"


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="CFB Prediction Centre",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
<style>
:root {
    --bg: #eaf0f4;
    --bg-soft: #f2f6f8;
    --panel: #f8fafb;
    --panel-2: #ffffff;
    --panel-3: #eef4f7;
    --line: #d5e0e7;
    --line-soft: rgba(82, 125, 163, .14);
    --text: #24313d;
    --muted: #687988;
    --muted-2: #84939f;
    --accent: #527da3;
    --accent-2: #6e95b6;
    --green: #5f8c73;
    --amber: #a9844e;
    --red: #b96c63;
    --shadow: 0 12px 34px rgba(52, 73, 94, .08);
}

html, body, .stApp {
    background: var(--bg);
}

.stApp {
    background:
        radial-gradient(circle at 50% -14%, rgba(82,125,163,.08), transparent 30%),
        linear-gradient(180deg, #edf3f6 0%, #e8eff3 52%, #edf3f6 100%);
    color: var(--text);
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.block-container {
    max-width: 1080px;
    padding-top: max(.55rem, env(safe-area-inset-top));
    padding-left: .9rem;
    padding-right: .9rem;
    padding-bottom: calc(2.4rem + env(safe-area-inset-bottom));
}

#MainMenu, footer, header, [data-testid="stToolbar"] {
    visibility: hidden;
}

/* ---------- brand header ---------- */

.cfb-header {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    margin:2px 0 10px;
}

.cfb-brand-wrap {
    display:flex;
    align-items:center;
    gap:10px;
    min-width:0;
}

.cfb-mark {
    width:38px;
    height:38px;
    border-radius:12px;
    display:grid;
    place-items:center;
    background:linear-gradient(145deg,#18324a,#244e70);
    border:1px solid #315c7c;
    box-shadow:0 8px 24px rgba(0,0,0,.18);
    font-size:18px;
}

.cfb-brand-title {
    font-size:.95rem;
    line-height:1.05;
    font-weight:900;
    letter-spacing:.02em;
    color:var(--text);
}

.cfb-brand-sub {
    color:var(--muted);
    font-size:.63rem;
    margin-top:4px;
}

.cfb-season-pill {
    padding:6px 9px;
    border:1px solid #b8cad7;
    border-radius:999px;
    color:#bfd5e8;
    background:rgba(20,36,55,.72);
    font-size:.62rem;
    font-weight:800;
    white-space:nowrap;
}

/* ---------- top navigation ---------- */

div[data-testid="stRadio"] {
    position:sticky;
    top:max(0px, env(safe-area-inset-top));
    z-index:999;
    margin:0 -.15rem 14px;
    padding:5px;
    border:1px solid var(--line);
    border-radius:14px;
    background:rgba(248,250,251,.96);
    backdrop-filter:blur(16px);
    -webkit-backdrop-filter:blur(16px);
    box-shadow:0 8px 24px rgba(52,73,94,.10);
}

div[data-testid="stRadio"] > label {
    display:none !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    display:flex;
    gap:3px;
    width:100%;
    overflow-x:auto;
    scrollbar-width:none;
}

div[data-testid="stRadio"] div[role="radiogroup"]::-webkit-scrollbar {
    display:none;
}

div[data-testid="stRadio"] div[role="radiogroup"] label {
    flex:1 0 auto;
    min-width:74px;
    min-height:40px;
    border-radius:10px;
    display:flex;
    align-items:center;
    justify-content:center;
    padding:4px 8px;
    color:var(--muted);
    border:1px solid transparent;
    transition:.15s ease;
}

div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    color:#233947;
    background:#e8f0f5;
    border-color:#c8d8e3;
}

div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    color:#ffffff !important;
    background:#527da3;
    border-color:#527da3;
}

div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
    color:#ffffff !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] label p {
    margin:0 !important;
    font-size:.69rem !important;
    font-weight:800 !important;
    white-space:nowrap;
}

div[data-testid="stRadio"] input { display:none; }

/* ---------- hero ---------- */

.hero {
    border:1px solid var(--line);
    border-radius:20px;
    background:
        linear-gradient(135deg, #f8fafb, #f1f6f8);
    padding:20px;
    margin-bottom:12px;
    box-shadow:var(--shadow);
    overflow:hidden;
    position:relative;
}

.hero:after {
    content:"";
    position:absolute;
    width:190px;
    height:190px;
    border-radius:50%;
    right:-85px;
    top:-95px;
    background:radial-gradient(circle, rgba(82,125,163,.10), transparent 67%);
    pointer-events:none;
}

.hero-kicker {
    color:var(--accent-2);
    font-size:.62rem;
    font-weight:850;
    letter-spacing:.12em;
    text-transform:uppercase;
}

.hero-title {
    color:var(--text);
    font-size:clamp(1.55rem, 6vw, 2.45rem);
    line-height:1.02;
    font-weight:950;
    letter-spacing:-.035em;
    margin:6px 0 7px;
}

.hero-copy {
    max-width:660px;
    color:var(--muted);
    font-size:.79rem;
    line-height:1.5;
}

/* ---------- section headers ---------- */

.section-head {
    display:flex;
    align-items:flex-end;
    justify-content:space-between;
    gap:12px;
    margin:22px 0 9px;
}

.section-title {
    color:var(--text);
    font-size:1.03rem;
    font-weight:900;
    letter-spacing:-.015em;
}

.section-sub {
    color:var(--muted);
    font-size:.66rem;
    margin-top:3px;
}

/* ---------- cards ---------- */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color:var(--line) !important;
    border-radius:17px !important;
    background:linear-gradient(180deg, #ffffff, #f8fafb) !important;
    box-shadow:0 10px 28px rgba(0,0,0,.11);
}

.metric-card {
    border:1px solid var(--line);
    border-radius:15px;
    padding:13px;
    background:linear-gradient(180deg,#ffffff,#f4f8fa);
    min-height:98px;
    box-shadow:0 5px 16px rgba(52,73,94,.05);
}

.metric-label {
    color:#52718b;
    font-size:.59rem;
    text-transform:uppercase;
    letter-spacing:.08em;
    font-weight:850;
}

.metric-value {
    color:#1f2f3c !important;
    font-size:1.48rem;
    font-weight:950;
    line-height:1;
    margin-top:7px;
    letter-spacing:-.025em;
}

.metric-note {
    color:#70818f !important;
    font-size:.62rem;
    margin-top:6px;
    line-height:1.3;
}

.status-chip {
    display:inline-flex;
    align-items:center;
    gap:5px;
    padding:4px 7px;
    border-radius:999px;
    border:1px solid #c5d5e0;
    background:#edf4f8;
    color:#4b687f;
    font-size:.57rem;
    font-weight:850;
    letter-spacing:.04em;
    text-transform:uppercase;
}

.status-chip.good {
    color:#bce9d2;
    border-color:rgba(105,201,154,.35);
    background:rgba(105,201,154,.09);
}

.status-chip.accent {
    color:#cae9fb;
    border-color:rgba(107,183,236,.35);
    background:rgba(107,183,236,.09);
}

.status-chip.warn {
    color:#ead49f;
    border-color:rgba(217,184,107,.35);
    background:rgba(217,184,107,.08);
}

/* ---------- matchup card ---------- */

.game-meta {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:8px;
    color:var(--muted);
    font-size:.62rem;
    margin-bottom:6px;
}

.team-name {
    color:var(--text);
    font-size:.84rem;
    font-weight:900;
    line-height:1.12;
}

.team-meta {
    color:var(--muted);
    font-size:.60rem;
    margin-top:3px;
}

.score-number {
    color:var(--text);
    font-size:1.72rem;
    font-weight:950;
    line-height:1;
    text-align:center;
    letter-spacing:-.035em;
}

.score-caption {
    color:var(--muted);
    font-size:.56rem;
    text-transform:uppercase;
    letter-spacing:.08em;
    text-align:center;
    margin-bottom:4px;
}

.prob {
    color:var(--accent-2);
    font-weight:900;
    font-size:.75rem;
}

.favourite-line {
    display:flex;
    align-items:center;
    justify-content:center;
    gap:6px;
    margin-top:8px;
    color:#4f6c83;
    font-size:.64rem;
    font-weight:800;
}

.prob-track {
    display:flex;
    width:100%;
    height:8px;
    overflow:hidden;
    border-radius:999px;
    background:#d8e3ea;
    margin-top:10px;
}

.prob-away {
    height:100%;
    background:#9aafbf;
}

.prob-home {
    height:100%;
    background:#648eb1;
}


.market-strip {
    margin-top:9px;
    padding:8px 10px;
    border:1px solid #d6e2ea;
    border-radius:10px;
    background:#f3f7fa;
    color:#526a7d;
    text-align:center;
    font-size:.62rem;
    line-height:1.25;
}

.market-strip strong {
    color:#263a49;
    font-weight:900;
}

.upset-banner {
    margin:0 0 7px;
    padding:10px 12px;
    border:1px solid #d5e0e7;
    border-radius:12px;
    background:linear-gradient(90deg,#f8fbfc,#eef5f8);
    color:#344b5d;
    font-size:.67rem;
    line-height:1.35;
}

.upset-banner strong {
    color:#294e6b;
    font-weight:950;
}

.upset-badge {
    display:inline-block;
    margin-right:6px;
    padding:3px 7px;
    border-radius:999px;
    background:#dbe9f3;
    color:#315d7e;
    font-size:.57rem;
    font-weight:900;
    text-transform:uppercase;
    letter-spacing:.035em;
}

.team-hero {
    border:1px solid var(--line);
    border-radius:18px;
    background:linear-gradient(135deg,#ffffff,#f3f8fa);
    padding:16px;
    margin-bottom:8px;
}

.team-hero-name {
    color:#1f2f3c;
    font-size:1.35rem;
    font-weight:950;
    line-height:1.05;
    letter-spacing:-.025em;
}

.team-hero-meta {
    color:#6d7f8d;
    font-size:.68rem;
    margin-top:5px;
}

/* ---------- detail ---------- */

.detail-team {
    color:var(--text);
    font-size:.86rem;
    font-weight:900;
    text-align:center;
    line-height:1.15;
}

.detail-prob {
    color:var(--text);
    font-size:2rem;
    font-weight:950;
    text-align:center;
    line-height:1;
    margin-top:6px;
}

.factor-row {
    display:grid;
    grid-template-columns:minmax(0,1fr) auto;
    gap:10px;
    align-items:center;
    padding:10px 0;
    border-bottom:1px solid var(--line-soft);
}

.factor-label {
    color:var(--muted);
    font-size:.66rem;
}

.factor-value {
    color:var(--text);
    font-size:.76rem;
    font-weight:900;
    text-align:right;
}

/* ---------- ELO ---------- */

.elo-row {
    display:grid;
    grid-template-columns:34px 38px minmax(0,1fr) auto;
    gap:9px;
    align-items:center;
    padding:9px 1px;
    border-bottom:1px solid var(--line-soft);
}

.elo-rank {
    color:#8ea3b8;
    font-size:.70rem;
    font-weight:850;
    text-align:center;
}

.elo-team {
    color:var(--text);
    font-size:.80rem;
    font-weight:900;
}

.elo-conf {
    color:var(--muted);
    font-size:.58rem;
    margin-top:2px;
}

.elo-value {
    color:var(--text);
    font-size:.84rem;
    font-weight:950;
    text-align:right;
}

.elo-delta {
    font-size:.58rem;
    font-weight:850;
    text-align:right;
    margin-top:2px;
}

.up { color:var(--green); }
.down { color:var(--red); }
.flat { color:var(--muted); }

/* ---------- native controls ---------- */

.stButton > button {
    width:100%;
    min-height:42px;
    border-radius:11px;
    border:1px solid #b8cad7;
    background:#e4edf3;
    color:#29465d;
    font-size:.73rem;
    font-weight:850;
    box-shadow:none;
}

.stButton > button:hover {
    border-color:#527da3;
    color:#fff;
    background:#527da3;
}

.stButton > button:active,
.stButton > button:focus,
.stButton > button:focus-visible {
    border-color:#3f6688 !important;
    background:#3f6688 !important;
    color:#ffffff !important;
    box-shadow:0 0 0 3px rgba(82,125,163,.16) !important;
}

.stButton > button:disabled {
    background:#edf2f5 !important;
    color:#8a98a4 !important;
    border-color:#d5e0e7 !important;
}

div[data-baseweb="select"] > div,
input {
    background:#ffffff !important;
    border-color:#c9d7e1 !important;
    color:var(--text) !important;
}

.stSelectbox label,
.stTextInput label,
.stSegmentedControl label {
    color:var(--muted) !important;
    font-size:.68rem !important;
}


/* ---------- segmented controls ---------- */

[data-testid="stSegmentedControl"] button,
button[data-testid="stBaseButton-segmented_control"] {
    background:#ffffff !important;
    color:#304657 !important;
    border-color:#c8d6e0 !important;
    min-height:38px !important;
    font-weight:800 !important;
}

[data-testid="stSegmentedControl"] button *,
button[data-testid="stBaseButton-segmented_control"] * {
    color:inherit !important;
}

[data-testid="stSegmentedControl"] button:hover,
button[data-testid="stBaseButton-segmented_control"]:hover {
    background:#e7f0f6 !important;
    color:#203847 !important;
    border-color:#7fa1bb !important;
}

[data-testid="stSegmentedControl"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] button[data-selected="true"],
button[data-testid="stBaseButton-segmented_controlActive"] {
    background:#527da3 !important;
    color:#ffffff !important;
    border-color:#527da3 !important;
}

[data-testid="stSegmentedControl"] button[aria-pressed="true"] *,
[data-testid="stSegmentedControl"] button[data-selected="true"] *,
button[data-testid="stBaseButton-segmented_controlActive"] * {
    color:#ffffff !important;
}

[data-testid="stSegmentedControl"] button[aria-pressed="true"]:hover,
[data-testid="stSegmentedControl"] button[data-selected="true"]:hover,
button[data-testid="stBaseButton-segmented_controlActive"]:hover {
    background:#426c90 !important;
    color:#ffffff !important;
    border-color:#426c90 !important;
}

[data-testid="stSegmentedControl"] button:focus-visible {
    outline:3px solid rgba(82,125,163,.22) !important;
    outline-offset:2px !important;
}

[data-testid="stDataFrame"] {
    border:1px solid var(--line);
    border-radius:14px;
    overflow:hidden;
}

/* ---------- responsive ---------- */

@media (max-width: 640px) {
    .block-container {
        padding-left:.68rem;
        padding-right:.68rem;
    }

    .cfb-header {
        margin-bottom:8px;
    }

    .cfb-brand-title { font-size:.86rem; }
    .cfb-brand-sub { font-size:.58rem; }
    .cfb-season-pill { font-size:.56rem; padding:5px 7px; }

    div[data-testid="stRadio"] {
        border-radius:12px;
        margin-left:-.12rem;
        margin-right:-.12rem;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label {
        min-width:66px;
        padding:4px 7px;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label p {
        font-size:.64rem !important;
    }

    .hero { padding:17px; border-radius:17px; }
    .metric-card { min-height:92px; padding:12px; }
    .metric-value { font-size:1.30rem; }
    .score-number { font-size:1.48rem; }
    .detail-prob { font-size:1.68rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


@st.cache_data(ttl=120)
def load_data():
    games_source = GAMES_V3_PATH if GAMES_V3_PATH.exists() and GAMES_V3_PATH.stat().st_size > 0 else GAMES_PATH
    games = safe_read_csv(games_source)
    weekly = safe_read_csv(ELO_WEEKLY_PATH)
    rankings = safe_read_csv(ELO_RANKINGS_PATH)
    betting = safe_read_csv(BETTING_PATH)
    profiles = safe_read_csv(TEAM_PROFILES_PATH)

    performance: dict[str, Any] = {}
    if PERFORMANCE_PATH.exists() and PERFORMANCE_PATH.stat().st_size > 0:
        try:
            performance = json.loads(PERFORMANCE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            performance = {}

    if not games.empty:
        source = "start_date_utc" if "start_date_utc" in games.columns else "start_date"
        if source in games.columns:
            games["start_date_utc"] = pd.to_datetime(games[source], errors="coerce", utc=True)

    if not betting.empty and "start_date" in betting.columns:
        betting["start_date"] = pd.to_datetime(betting["start_date"], errors="coerce", utc=True)

    return games, weekly, rankings, performance, betting, profiles


# ============================================================
# HELPERS
# ============================================================


def clean(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def num(value: Any, digits: int = 1, default: str = "—") -> str:
    try:
        x = float(value)
        if np.isfinite(x):
            return f"{x:.{digits}f}"
    except Exception:
        pass
    return default


def pct(value: Any, digits: int = 1, default: str = "—") -> str:
    try:
        x = float(value)
        if np.isfinite(x):
            return f"{x * 100:.{digits}f}%"
    except Exception:
        pass
    return default


def signed(value: Any, digits: int = 0, default: str = "—") -> str:
    try:
        if value is None or pd.isna(value):
            return default
        number = float(value)
    except Exception:
        return default

    return f"{number:+.{digits}f}"


def pounds(value: Any) -> str:
    try:
        x = float(value)
        if np.isfinite(x):
            sign = "+" if x > 0 else ""
            return f"{sign}£{x:,.2f}"
    except Exception:
        pass
    return "—"


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def display_value(row: pd.Series, column: str) -> Any:
    display_column = f"display_{column}"
    if display_column in row.index and pd.notna(row[display_column]):
        return row[display_column]
    return row[column] if column in row.index else np.nan


def kickoff_text(value: Any) -> str:
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return "Kickoff TBD"
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("Europe/London").strftime("%a %d %b · %H:%M UK")
    except Exception:
        return "Kickoff TBD"


def lock_text(row: pd.Series) -> str:
    if clean(row.get("game_status", "")).lower() == "completed":
        return "FINAL"

    if boolish(row.get("is_locked", False)):
        quality = clean(row.get("official_lock_quality", ""))
        return "LOCKED · LATE" if quality == "late_initial_capture" else "OFFICIAL LOCK"

    try:
        hours = float(row.get("hours_until_lock", np.nan))
        if not np.isfinite(hours):
            return "DYNAMIC"
        if hours <= 0:
            return "LOCK DUE"
        if hours >= 336:
            return f"LOCKS IN {hours / 168:.0f} WK"
        if hours >= 72:
            return f"LOCKS IN {hours / 24:.0f} D"
        if hours >= 48:
            return f"LOCKS IN {hours / 24:.1f} D"
        if hours >= 1:
            return f"LOCKS IN {hours:.0f} H"
        return f"LOCKS IN {hours * 60:.0f} MIN"
    except Exception:
        return "DYNAMIC"


def chip_class(row: pd.Series) -> str:
    text = lock_text(row)
    if text == "FINAL":
        return "good"
    if "LOCK" in text and "IN" not in text and text != "LOCK DUE":
        return "accent"
    if text == "LOCK DUE":
        return "warn"
    return ""


def next_upcoming_week(games: pd.DataFrame) -> int | None:
    if games.empty or "start_date_utc" not in games.columns:
        return None
    now = pd.Timestamp.now(tz="UTC")
    view = games[
        games["start_date_utc"].notna() & games["start_date_utc"].gt(now)
    ].sort_values("start_date_utc", kind="stable")
    if view.empty:
        return None
    weeks = pd.to_numeric(view.get("week"), errors="coerce").dropna()
    return int(weeks.iloc[0]) if not weeks.empty else None


def current_top_25(rankings: pd.DataFrame) -> set[str]:
    if rankings.empty or "rank" not in rankings.columns or "team" not in rankings.columns:
        return set()
    ranks = pd.to_numeric(rankings["rank"], errors="coerce")
    return set(rankings.loc[ranks.le(25), "team"].dropna().astype(str).tolist())


def conferences_for_games(games: pd.DataFrame) -> list[str]:
    values: set[str] = set()
    for column in ["home_conference", "away_conference"]:
        if column in games.columns:
            values.update(games[column].dropna().astype(str).str.strip().tolist())
    values.discard("")
    return sorted(values)


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        '<div class="metric-card">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str = "") -> None:
    st.markdown(
        '<div class="section-head"><div>'
        f'<div class="section-title">{html.escape(title)}</div>'
        f'<div class="section-sub">{html.escape(subtitle)}</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def request_game(game_id: int) -> None:
    st.session_state["selected_game_id"] = int(game_id)
    st.session_state["pending_nav"] = "◎ Game"
    st.rerun()


def request_team(team: str) -> None:
    st.session_state["selected_team"] = str(team)
    st.session_state["pending_nav"] = "◉ Teams"
    st.rerun()


def build_elo_lookup(rankings: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if rankings.empty:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in rankings.iterrows():
        team = clean(row.get("team", "")).strip()
        if team:
            lookup[team] = {"rank": row.get("rank", np.nan), "elo": row.get("elo", np.nan)}
    return lookup


def elo_text(team: str, elo_lookup: dict[str, dict[str, Any]]) -> str:
    item = elo_lookup.get(team)
    if not item:
        return "ELO —"
    rank = item.get("rank", np.nan)
    elo = item.get("elo", np.nan)
    rank_text = "—"
    try:
        if pd.notna(rank):
            rank_text = f"#{int(float(rank))}"
    except Exception:
        pass
    return f"{rank_text} · ELO {num(elo, 0)}"


def predicted_winner(row: pd.Series) -> str:
    return clean(display_value(row, "predicted_winner"))


def model_probability(row: pd.Series) -> float:
    try:
        x = float(row.get("model_favourite_probability", np.nan))
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def get_actual_score(row: pd.Series, side: str) -> Any:
    for col in [f"actual_{side}_points", f"actual_{side}_score", f"{side}_points", f"final_{side}_score"]:
        if col in row.index and pd.notna(row.get(col)):
            return row.get(col)
    return np.nan


def v3_available(row: pd.Series) -> bool:
    try:
        value = float(row.get("v3_home_win_probability", np.nan))
        return np.isfinite(value)
    except Exception:
        return False


def v3_winner(row: pd.Series) -> str:
    if not v3_available(row):
        return ""
    winner = clean(row.get("v3_predicted_winner", "")).strip()
    if winner:
        return winner
    try:
        hp = float(row.get("v3_home_win_probability", np.nan))
        ap = float(row.get("v3_away_win_probability", np.nan))
        if np.isfinite(hp) and np.isfinite(ap):
            return clean(row.get("home_team" if hp >= ap else "away_team", ""))
    except Exception:
        pass
    return ""

def v3_margin_text(row: pd.Series) -> str:
    try:
        margin = float(row.get("v3_projected_margin", np.nan))
    except Exception:
        return "—"
    if not np.isfinite(margin):
        return "—"
    team = clean(row.get("home_team" if margin >= 0 else "away_team", ""))
    return f"{team} {abs(margin):.1f}" if team else f"{margin:+.1f}"



def market_spread_text(row: pd.Series) -> str:
    """Human-readable current captured ATS spread for display only."""
    formatted = clean(row.get("market_formatted_spread", "")).strip()
    if formatted:
        return formatted

    favourite = clean(row.get("market_spread_favourite", "")).strip()
    raw = pd.to_numeric(
        pd.Series([row.get("market_spread", np.nan)]),
        errors="coerce",
    ).iloc[0]

    if favourite and pd.notna(raw):
        line = float(raw)
        # CFBD formatted data normally stores the favourite and a negative line.
        signed_line = line if line <= 0 else -abs(line)
        return f"{favourite} {signed_line:+.1f}"

    return "Market spread unavailable"


def market_moneyline_text(row: pd.Series) -> str:
    home_team = clean(row.get("home_team", ""))
    away_team = clean(row.get("away_team", ""))
    home_ml = pd.to_numeric(pd.Series([row.get("market_home_moneyline", np.nan)]), errors="coerce").iloc[0]
    away_ml = pd.to_numeric(pd.Series([row.get("market_away_moneyline", np.nan)]), errors="coerce").iloc[0]

    parts: list[str] = []
    if pd.notna(away_ml):
        parts.append(f"{away_team} {float(away_ml):+.0f}")
    if pd.notna(home_ml):
        parts.append(f"{home_team} {float(home_ml):+.0f}")
    return " · ".join(parts) if parts else "Moneyline unavailable"


def upset_details(row: pd.Series) -> dict[str, Any] | None:
    """Return lower-ELO upset details when the lower-ELO side has >=30% win probability."""
    home_elo = pd.to_numeric(pd.Series([row.get("home_pregame_elo", np.nan)]), errors="coerce").iloc[0]
    away_elo = pd.to_numeric(pd.Series([row.get("away_pregame_elo", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(home_elo) or pd.isna(away_elo) or float(home_elo) == float(away_elo):
        return None

    if float(home_elo) < float(away_elo):
        underdog = clean(row.get("home_team", ""))
        favourite = clean(row.get("away_team", ""))
        underdog_elo = float(home_elo)
        favourite_elo = float(away_elo)
        probability = pd.to_numeric(
            pd.Series([display_value(row, "home_win_probability")]),
            errors="coerce",
        ).iloc[0]
        location = "Home"
    else:
        underdog = clean(row.get("away_team", ""))
        favourite = clean(row.get("home_team", ""))
        underdog_elo = float(away_elo)
        favourite_elo = float(home_elo)
        probability = pd.to_numeric(
            pd.Series([display_value(row, "away_win_probability")]),
            errors="coerce",
        ).iloc[0]
        location = "Away"

    if pd.isna(probability) or float(probability) < 0.30:
        return None

    p = float(probability)
    if p > 0.50:
        label = "Model upset pick"
    elif p >= 0.45:
        label = "Upset alert"
    elif p >= 0.38:
        label = "Strong upset watch"
    else:
        label = "Upset watch"

    return {
        "underdog": underdog,
        "favourite": favourite,
        "underdog_elo": underdog_elo,
        "favourite_elo": favourite_elo,
        "elo_gap": favourite_elo - underdog_elo,
        "probability": p,
        "location": location,
        "label": label,
    }


def chart_theme(chart):
    if alt is None:
        return chart
    return (
        chart.configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor="#687988",
            titleColor="#687988",
            gridColor="#dce5eb",
            gridOpacity=.45,
            domainColor="#cbd8e1",
            tickColor="#cbd8e1",
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_legend(
            labelColor="#687988",
            titleColor="#687988",
            labelFontSize=11,
            titleFontSize=11,
        )
    )


def branded_line_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    tooltip: list[str] | None = None,
    height: int = 300,
    y_title: str | None = None,
) -> None:
    data = frame[[c for c in [x, y] if c in frame.columns]].copy()
    if data.empty:
        st.info("No chart data available.")
        return

    if alt is None:
        st.line_chart(data.set_index(x), height=height)
        return

    tooltips = tooltip or [x, y]
    base = alt.Chart(frame).encode(
        x=alt.X(f"{x}:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{y}:Q", title=y_title or y, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip(t) for t in tooltips if t in frame.columns],
    )
    line = base.mark_line(color="#527da3", strokeWidth=3)
    points = base.mark_circle(color="#6e95b6", size=52, stroke="#ffffff", strokeWidth=1)
    st.altair_chart(chart_theme((line + points).properties(height=height)), use_container_width=True)


def branded_pl_chart(frame: pd.DataFrame, height: int = 320) -> None:
    if frame.empty:
        return
    if alt is None:
        st.line_chart(frame.set_index("bet_number")[["cumulative_profit_loss"]], height=height)
        return

    base = alt.Chart(frame).encode(
        x=alt.X("bet_number:Q", title="Bet number", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("cumulative_profit_loss:Q", title="Cumulative P/L (£)", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("bet_number:Q", title="Bet", format=".0f"),
            alt.Tooltip("cumulative_profit_loss:Q", title="P/L", format="£,.2f"),
        ],
    )
    area = base.mark_area(color="#527da3", opacity=.09)
    line = base.mark_line(color="#527da3", strokeWidth=3)
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#84939f", opacity=.55).encode(y="y:Q")
    st.altair_chart(chart_theme((area + line + zero).properties(height=height)), use_container_width=True)


# ============================================================
# HEADER + NAV
# ============================================================


def render_header() -> None:
    st.markdown(
        """
<div class="cfb-header">
    <div class="cfb-brand-wrap">
        <div class="cfb-mark">🏈</div>
        <div>
            <div class="cfb-brand-title">CFB PREDICTION CENTRE</div>
            <div class="cfb-brand-sub">Model-first college football analytics</div>
        </div>
    </div>
    <div class="cfb-season-pill">2026 · V2 PROD · V3 CHALLENGER</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# MATCHUP CARD
# ============================================================


def compact_matchup_card(
    row: pd.Series,
    key_prefix: str,
    elo_lookup: dict[str, dict[str, Any]],
) -> None:
    game_id = int(row["cfbd_game_id"])
    away_team = clean(row.get("away_team", ""))
    home_team = clean(row.get("home_team", ""))
    away_conf = clean(row.get("away_conference", ""))
    home_conf = clean(row.get("home_conference", ""))

    away_prob = display_value(row, "away_win_probability")
    home_prob = display_value(row, "home_win_probability")
    away_proj = display_value(row, "projected_away_score")
    home_proj = display_value(row, "projected_home_score")
    status = clean(row.get("game_status", "")).lower()

    if status == "completed":
        away_score = get_actual_score(row, "away")
        home_score = get_actual_score(row, "home")
        score_label = "Final"
        if pd.isna(away_score) or pd.isna(home_score):
            away_score, home_score = away_proj, home_proj
            score_label = "Projected"
    else:
        away_score, home_score = away_proj, home_proj
        score_label = "Projected"

    favourite = predicted_winner(row)
    confidence = clean(row.get("confidence_bucket", ""))

    try:
        ap = float(away_prob)
        hp = float(home_prob)
        ap_width = max(0.0, min(100.0, ap * 100)) if np.isfinite(ap) else 50.0
        hp_width = max(0.0, min(100.0, hp * 100)) if np.isfinite(hp) else 50.0
    except Exception:
        ap_width, hp_width = 50.0, 50.0

    with st.container(border=True):
        st.markdown(
            '<div class="game-meta">'
            f'<span>{html.escape(kickoff_text(row.get("start_date_utc")))}</span>'
            f'<span class="status-chip {chip_class(row)}">{html.escape(lock_text(row))}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        away_col, score_col, home_col = st.columns([1, .68, 1], vertical_alignment="center")

        with away_col:
            logo = clean(row.get("away_logo_url", ""))
            if logo.startswith("http"):
                st.image(logo, width=56)
            st.markdown(
                f'<div class="team-name">{html.escape(away_team)}</div>'
                f'<div class="team-meta">{html.escape(away_conf)} · {html.escape(elo_text(away_team, elo_lookup))}</div>'
                f'<div class="prob">{pct(away_prob)}</div>',
                unsafe_allow_html=True,
            )
            if st.button("Team page", key=f"{key_prefix}_away_team_{game_id}", use_container_width=True):
                request_team(away_team)

        with score_col:
            st.markdown(
                f'<div class="score-caption">{html.escape(score_label)}</div>'
                f'<div class="score-number">{num(away_score,0)}–{num(home_score,0)}</div>',
                unsafe_allow_html=True,
            )

        with home_col:
            logo = clean(row.get("home_logo_url", ""))
            if logo.startswith("http"):
                st.image(logo, width=56)
            st.markdown(
                f'<div class="team-name">{html.escape(home_team)}</div>'
                f'<div class="team-meta">{html.escape(home_conf)} · {html.escape(elo_text(home_team, elo_lookup))}</div>'
                f'<div class="prob">{pct(home_prob)}</div>',
                unsafe_allow_html=True,
            )
            if st.button("Team page", key=f"{key_prefix}_home_team_{game_id}", use_container_width=True):
                request_team(home_team)

        st.markdown(
            '<div class="market-strip"><strong>Vegas spread</strong> · '
            f'{html.escape(market_spread_text(row))}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="prob-track"><div class="prob-away" style="width:{ap_width:.1f}%"></div>'
            f'<div class="prob-home" style="width:{hp_width:.1f}%"></div></div>'
            f'<div class="favourite-line">Model: {html.escape(favourite or "—")}'
            f'{" · " + html.escape(confidence) if confidence else ""}</div>',
            unsafe_allow_html=True,
        )

        if v3_available(row) and status != "completed":
            v3_fav = v3_winner(row)
            v3_prob = row.get("v3_home_win_probability" if v3_fav == home_team else "v3_away_win_probability", np.nan)
            st.markdown(
                '<div class="market-strip"><strong>V3 challenger</strong> · '
                f'{html.escape(v3_fav or "—")} {pct(v3_prob)} · '
                f'Projected {num(row.get("v3_projected_away_points"),0)}–{num(row.get("v3_projected_home_points"),0)}</div>',
                unsafe_allow_html=True,
            )

        if st.button("Open matchup →", key=f"{key_prefix}_{game_id}", use_container_width=True):
            request_game(game_id)


# ============================================================
# HOME
# ============================================================


def home_games(
    games: pd.DataFrame,
    rankings: pd.DataFrame,
    focus: str,
    conference: str,
) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    next_week = next_upcoming_week(games)
    view = games[games["start_date_utc"].notna() & games["start_date_utc"].gt(now)].copy()

    if next_week is not None:
        view = view[view["week"].eq(next_week)].copy()

    if conference != "All":
        view = view[
            view.get("home_conference", "").astype(str).eq(conference)
            | view.get("away_conference", "").astype(str).eq(conference)
        ].copy()

    if focus == "Top 25 ELO":
        top_25 = current_top_25(rankings)
        view = view[
            view["home_team"].astype(str).isin(top_25)
            | view["away_team"].astype(str).isin(top_25)
        ].copy()
    elif focus == "Official Locks" and "is_locked" in view.columns:
        view = view[view["is_locked"].apply(boolish)].copy()

    if focus == "Featured" and "model_favourite_probability" in view.columns:
        view = view.sort_values(
            ["model_favourite_probability", "start_date_utc"],
            ascending=[False, True],
            kind="stable",
        )
    else:
        view = view.sort_values("start_date_utc", kind="stable")

    return view.head(5)


def render_home(games: pd.DataFrame, rankings: pd.DataFrame, performance: dict[str, Any]) -> None:
    next_week = next_upcoming_week(games)
    kicker = f"WEEK {next_week}" if next_week is not None else "2026 SEASON"

    st.markdown(
        '<div class="hero">'
        f'<div class="hero-kicker">{html.escape(kicker)} · LIVE MODEL VIEW</div>'
        '<div class="hero-title">College football, modelled clearly.</div>'
        '<div class="hero-copy">Winner probabilities, projected scores, weekly ELO movement and immutable official locks — designed to be quick to scan on mobile without hiding the detail underneath.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    official = performance.get("official_locked", {}) if performance else {}
    upcoming = games[games["start_date_utc"].gt(pd.Timestamp.now(tz="UTC"))] if "start_date_utc" in games.columns else pd.DataFrame()
    upcoming_week_games = upcoming[upcoming["week"].eq(next_week)] if next_week is not None and not upcoming.empty else upcoming

    c1, c2 = st.columns(2)
    with c1:
        metric_card("Official record", official.get("record") or "—", f'{official.get("games",0)} completed locks')
    with c2:
        accuracy = official.get("accuracy")
        metric_card("Official accuracy", pct(accuracy) if accuracy is not None else "—", "Genuine locked predictions")

    c3, c4 = st.columns(2)
    with c3:
        locks = int(games["is_locked"].apply(boolish).sum()) if "is_locked" in games.columns else 0
        metric_card("Official locks", str(locks), "Immutable once captured")
    with c4:
        metric_card("Next slate", f"{len(upcoming_week_games):,}", f"Week {next_week}" if next_week is not None else "Upcoming games")

    section_title("This week", "Featured games first, with fast filters")
    f1, f2 = st.columns(2)
    with f1:
        focus = st.selectbox("View", ["Featured", "Top 25 ELO", "Official Locks"], key="home_focus")
    with f2:
        conference = st.selectbox("Conference", ["All"] + conferences_for_games(games), key="home_conference")

    featured = home_games(games, rankings, focus, conference)
    if featured.empty:
        st.info("No games match those filters for the next upcoming week.")
        return

    elo_lookup = build_elo_lookup(rankings)
    for index, (_, row) in enumerate(featured.iterrows()):
        compact_matchup_card(row, f"home_{index}", elo_lookup)


# ============================================================
# PICKS
# ============================================================


def render_picks(games: pd.DataFrame, rankings: pd.DataFrame) -> None:
    section_title("Predictions", "Browse every game without losing the mobile-first layout")

    weeks = sorted(int(x) for x in pd.to_numeric(games["week"], errors="coerce").dropna().unique().tolist())
    next_week = next_upcoming_week(games)
    week_options: list[Any] = ["All"] + weeks
    default_index = week_options.index(next_week) if next_week in week_options else 0

    c1, c2 = st.columns(2)
    with c1:
        week_choice = st.selectbox("Week", week_options, index=default_index, key="picks_week")
    with c2:
        conference = st.selectbox("Conference", ["All"] + conferences_for_games(games), key="picks_conf")

    search = st.text_input("Search teams", placeholder="Georgia, Ohio State, USC…")

    view = games.copy()
    if week_choice != "All":
        view = view[view["week"].eq(int(week_choice))]
    if conference != "All":
        view = view[
            view["home_conference"].astype(str).eq(conference)
            | view["away_conference"].astype(str).eq(conference)
        ]
    if search.strip():
        q = search.strip().lower()
        view = view[
            view["home_team"].astype(str).str.lower().str.contains(q, regex=False)
            | view["away_team"].astype(str).str.lower().str.contains(q, regex=False)
        ]

    view = view.sort_values("start_date_utc", kind="stable")
    st.caption(f"{len(view):,} games")

    elo_lookup = build_elo_lookup(rankings)
    for index, (_, row) in enumerate(view.head(100).iterrows()):
        compact_matchup_card(row, f"picks_{index}", elo_lookup)



# ============================================================
# UPSET WATCH
# ============================================================


def render_upsets(games: pd.DataFrame, rankings: pd.DataFrame) -> None:
    section_title(
        "Upset Watch",
        "Lower-ELO teams with at least a 30% model chance to beat a higher-ELO opponent",
    )

    now = pd.Timestamp.now(tz="UTC")
    candidates = games.copy()

    if "game_status" in candidates.columns:
        candidates = candidates[
            ~candidates["game_status"].astype(str).str.lower().eq("completed")
        ].copy()

    if "start_date_utc" in candidates.columns:
        candidates = candidates[
            candidates["start_date_utc"].isna()
            | candidates["start_date_utc"].ge(now - pd.Timedelta(hours=6))
        ].copy()

    # Upset Watch compares teams inside the current FBS ELO universe only.
    # FCS/non-ranked opponents may carry a pregame rating in production for
    # matchup modelling, but they do not have a comparable FBS ELO rank and
    # therefore must not create an Upset Watch candidate.
    ranked_fbs_teams = set(
        rankings.get("team", pd.Series(dtype="string"))
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    detail_rows: list[dict[str, Any]] = []
    for idx, row in candidates.iterrows():
        home_team = clean(row.get("home_team", ""))
        away_team = clean(row.get("away_team", ""))

        if (
            not home_team
            or not away_team
            or home_team not in ranked_fbs_teams
            or away_team not in ranked_fbs_teams
        ):
            continue

        details = upset_details(row)
        if details is not None:
            details["_index"] = idx
            detail_rows.append(details)

    if not detail_rows:
        st.info("No current matchups meet the 30% Upset Watch threshold.")
        return

    details_frame = pd.DataFrame(detail_rows)

    weeks = sorted(
        int(x)
        for x in pd.to_numeric(candidates.get("week", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .unique()
        .tolist()
    )
    week_options: list[Any] = ["All"] + weeks

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        week_choice = st.selectbox("Week", week_options, key="upset_week")
    with c2:
        min_prob = st.selectbox(
            "Minimum upset chance",
            [30, 35, 40, 45, 50],
            index=0,
            format_func=lambda x: f"{x}%+",
            key="upset_threshold",
        )
    with c3:
        conference = st.selectbox(
            "Conference",
            ["All"] + conferences_for_games(games),
            key="upset_conf",
        )

    selected: list[tuple[pd.Series, dict[str, Any]]] = []
    for detail in detail_rows:
        row = candidates.loc[detail["_index"]]
        if week_choice != "All":
            try:
                if int(row.get("week")) != int(week_choice):
                    continue
            except Exception:
                continue

        if detail["probability"] < (float(min_prob) / 100.0):
            continue

        if conference != "All":
            if (
                clean(row.get("home_conference", "")) != conference
                and clean(row.get("away_conference", "")) != conference
            ):
                continue

        selected.append((row, detail))

    selected.sort(
        key=lambda item: (
            -float(item[1]["probability"]),
            -float(item[1]["elo_gap"]),
            pd.Timestamp(item[0].get("start_date_utc"))
            if pd.notna(item[0].get("start_date_utc"))
            else pd.Timestamp.max.tz_localize("UTC"),
        )
    )

    st.caption(f"{len(selected):,} matchup{'s' if len(selected) != 1 else ''} on watch")

    if not selected:
        st.info("No games match the selected Upset Watch filters.")
        return

    elo_lookup = build_elo_lookup(rankings)

    for position, (row, detail) in enumerate(selected):
        badge = detail["label"]
        st.markdown(
            '<div class="upset-banner">'
            f'<span class="upset-badge">{html.escape(badge)}</span>'
            f'<strong>{html.escape(detail["underdog"])}</strong> has a '
            f'<strong>{detail["probability"] * 100:.1f}%</strong> model win chance despite a '
            f'<strong>{detail["elo_gap"]:.0f}-point</strong> pregame ELO disadvantage.'
            '</div>',
            unsafe_allow_html=True,
        )
        compact_matchup_card(
            row,
            f"upset_{position}",
            elo_lookup,
        )



# ============================================================
# GAME DETAIL
# ============================================================


def choose_game(games: pd.DataFrame) -> pd.Series:
    game_ids = pd.to_numeric(games["cfbd_game_id"], errors="coerce").dropna().astype(int).tolist()
    selected = st.session_state.get("selected_game_id")

    if selected not in game_ids:
        upcoming = games[games["start_date_utc"].gt(pd.Timestamp.now(tz="UTC"))].sort_values("start_date_utc", kind="stable")
        selected = int(upcoming.iloc[0]["cfbd_game_id"]) if not upcoming.empty else game_ids[0]
        st.session_state["selected_game_id"] = selected

    options = {
        int(row["cfbd_game_id"]): f'{row["away_team"]} @ {row["home_team"]}'
        for _, row in games.iterrows()
        if pd.notna(row["cfbd_game_id"])
    }

    selected = st.selectbox(
        "Matchup",
        list(options.keys()),
        index=list(options.keys()).index(int(selected)),
        format_func=lambda game_id: options[game_id],
        key="game_selector",
    )
    st.session_state["selected_game_id"] = int(selected)
    return games.loc[pd.to_numeric(games["cfbd_game_id"], errors="coerce").eq(int(selected))].iloc[0]


def factor_row(label: str, value: str) -> None:
    st.markdown(
        f'<div class="factor-row"><div class="factor-label">{html.escape(label)}</div>'
        f'<div class="factor-value">{html.escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


def render_game(games: pd.DataFrame) -> None:
    section_title("Game detail", "One clean view of the model, not a wall of numbers")
    row = choose_game(games)

    away_team = clean(row.get("away_team", ""))
    home_team = clean(row.get("home_team", ""))
    away_prob = display_value(row, "away_win_probability")
    home_prob = display_value(row, "home_win_probability")
    away_score = display_value(row, "projected_away_score")
    home_score = display_value(row, "projected_home_score")

    with st.container(border=True):
        st.markdown(
            '<div class="game-meta">'
            f'<span>{html.escape(kickoff_text(row.get("start_date_utc")))}</span>'
            f'<span class="status-chip {chip_class(row)}">{html.escape(lock_text(row))}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        away_col, mid_col, home_col = st.columns([1, .68, 1], vertical_alignment="center")
        with away_col:
            logo = clean(row.get("away_logo_url", ""))
            if logo.startswith("http"):
                st.image(logo, width=76)
            st.markdown(
                f'<div class="detail-team">{html.escape(away_team)}</div>'
                f'<div class="detail-prob">{pct(away_prob)}</div>',
                unsafe_allow_html=True,
            )
            if st.button("View team", key="detail_away_team", use_container_width=True):
                request_team(away_team)
        with mid_col:
            st.markdown(
                '<div class="score-caption">Projected score</div>'
                f'<div class="score-number">{num(away_score,0)}–{num(home_score,0)}</div>'
                '<div class="market-strip"><strong>Vegas spread</strong><br>'
                f'{html.escape(market_spread_text(row))}</div>',
                unsafe_allow_html=True,
            )
        with home_col:
            logo = clean(row.get("home_logo_url", ""))
            if logo.startswith("http"):
                st.image(logo, width=76)
            st.markdown(
                f'<div class="detail-team">{html.escape(home_team)}</div>'
                f'<div class="detail-prob">{pct(home_prob)}</div>',
                unsafe_allow_html=True,
            )
            if st.button("View team", key="detail_home_team", use_container_width=True):
                request_team(home_team)

        try:
            ap = max(0.0, min(100.0, float(away_prob) * 100))
            hp = max(0.0, min(100.0, float(home_prob) * 100))
        except Exception:
            ap, hp = 50.0, 50.0
        st.markdown(
            f'<div class="prob-track"><div class="prob-away" style="width:{ap:.1f}%"></div>'
            f'<div class="prob-home" style="width:{hp:.1f}%"></div></div>',
            unsafe_allow_html=True,
        )

    favourite = predicted_winner(row)
    margin = display_value(row, "final_expected_home_margin")
    try:
        margin_float = float(margin)
    except Exception:
        margin_float = np.nan
    margin_text = "—"
    if np.isfinite(margin_float):
        margin_team = home_team if margin_float >= 0 else away_team
        margin_text = f"{margin_team} {abs(margin_float):.1f}"

    section_title("Projection", "The four numbers that matter first")
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Model favourite", favourite or "—", clean(row.get("confidence_bucket", "")))
    with c2:
        metric_card("Model margin", margin_text, "Projected, not a betting line")
    c3, c4 = st.columns(2)
    with c3:
        metric_card("Projected total", num(display_value(row, "final_expected_total_points"), 1), "Expected combined score")
    with c4:
        metric_card("Simulation home win", pct(display_value(row, "simulation_home_win_probability")), "Monte Carlo diagnostic")

    if v3_available(row):
        section_title("V3 challenger", "Football-first opponent adjustment + possession simulation")
        v3_fav = v3_winner(row)
        v3_fav_prob = row.get("v3_home_win_probability" if v3_fav == home_team else "v3_away_win_probability", np.nan)
        c1, c2 = st.columns(2)
        with c1:
            metric_card("V3 favourite", v3_fav or "—", f'{pct(v3_fav_prob)} · {clean(row.get("v3_confidence_bucket", ""))}')
        with c2:
            metric_card("V3 margin", v3_margin_text(row), "Possession + residual blend")
        c3, c4 = st.columns(2)
        with c3:
            metric_card("V3 projected score", f'{num(row.get("v3_projected_away_points"),1)}–{num(row.get("v3_projected_home_points"),1)}', f'Total {num(row.get("v3_projected_total"),1)}')
        with c4:
            metric_card("V3 simulation home win", pct(row.get("sim_home_win_probability")), f'{num(row.get("sim_simulations"),0)} simulations')

        section_title("V2 vs V3", "Where the challenger disagrees with production")
        c1, c2 = st.columns(2)
        with c1:
            delta = row.get("v3_vs_v2_home_probability_delta", np.nan)
            metric_card("Home win probability Δ", f'{signed(float(delta) * 100,1)} pp' if pd.notna(delta) else "—", "V3 minus V2")
        with c2:
            metric_card("Projected margin Δ", signed(row.get("v3_vs_v2_margin_delta"),1), "Home-margin points · V3 minus V2")
        c3, c4 = st.columns(2)
        with c3:
            metric_card("Projected total Δ", signed(row.get("v3_vs_v2_total_delta"),1), "Points · V3 minus V2")
        with c4:
            metric_card("State maturity", pct(row.get("state_maturity"),0), "2026 in-season state maturity")

        section_title("V3 simulation range", "10th–90th percentile predictive distribution")
        c1, c2 = st.columns(2)
        with c1:
            metric_card(f"{away_team} score", f'{num(row.get("sim_away_p10"),0)}–{num(row.get("sim_away_p90"),0)}', f'≤10 {pct(row.get("sim_away_10_or_less_probability"))} · 40+ {pct(row.get("sim_away_40_plus_probability"))}')
        with c2:
            metric_card(f"{home_team} score", f'{num(row.get("sim_home_p10"),0)}–{num(row.get("sim_home_p90"),0)}', f'≤10 {pct(row.get("sim_home_10_or_less_probability"))} · 40+ {pct(row.get("sim_home_40_plus_probability"))}')
        c3, c4 = st.columns(2)
        with c3:
            metric_card("Margin range", f'{num(row.get("sim_margin_p10"),0)} to {num(row.get("sim_margin_p90"),0)}', "Home minus away")
        with c4:
            metric_card("Total range", f'{num(row.get("sim_total_p10"),0)}–{num(row.get("sim_total_p90"),0)}', "10th–90th percentile")

        section_title("V3 matchup engine", "Opponent-adjusted state entering this specific game")
        with st.container(border=True):
            factor_row("Expected possessions", f'{num(row.get("sim_expected_away_drives"),1)} {away_team} · {num(row.get("sim_expected_home_drives"),1)} {home_team}')
            factor_row("Opponent-adjusted PPD", f'{num(row.get("matchup_away_ppd"),2)} {away_team} · {num(row.get("matchup_home_ppd"),2)} {home_team}')
            factor_row("Matchup ratio", f'{num(row.get("matchup_away_matchup_ratio"),3)} {away_team} · {num(row.get("matchup_home_matchup_ratio"),3)} {home_team}')
            factor_row("Success rate", f'{pct(row.get("matchup_away_success_rate"))} {away_team} · {pct(row.get("matchup_home_success_rate"))} {home_team}')
            factor_row("Explosive rate", f'{pct(row.get("matchup_away_explosive_rate"))} {away_team} · {pct(row.get("matchup_home_explosive_rate"))} {home_team}')
            factor_row("PPA", f'{num(row.get("matchup_away_ppa"),3)} {away_team} · {num(row.get("matchup_home_ppa"),3)} {home_team}')
            factor_row("V3 ELO difference", f'{num(row.get("matchup_elo_difference"),1)} · home minus away')
            factor_row("V3 rating difference", f'{num(row.get("matchup_rating_difference"),1)} · home minus away')
    elif clean(row.get("game_status", "")).lower() != "completed":
        st.info("No V3 challenger forecast is stored for this matchup.")

    section_title("Market", "Captured market information — display only, never a model input")
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Vegas spread", market_spread_text(row), clean(row.get("market_provider", "")) or "Current captured line")
    with c2:
        metric_card("Moneyline", market_moneyline_text(row), "Current captured prices")

    section_title("Why the model leans this way", "Core matchup factors, kept readable")
    with st.container(border=True):
        factor_row(f"{away_team} pregame ELO", num(row.get("away_pregame_elo"), 0))
        factor_row(f"{home_team} pregame ELO", num(row.get("home_pregame_elo"), 0))
        factor_row("ELO difference", f'{num(row.get("elo_difference"), 1)} · home minus away')
        factor_row("Offensive matchup Δ", f'{num(row.get("offensive_matchup_difference"), 2)} · home minus away')
        factor_row("Expected total plays", num(row.get("expected_total_plays"), 1))
        factor_row("Expected drives", num(row.get("expected_total_drives"), 1))

    section_title("Simulation range", "10th–90th percentile score bands")
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Away range", f'{num(row.get("simulation_away_score_p10"),0)}–{num(row.get("simulation_away_score_p90"),0)}', away_team)
    with c2:
        metric_card("Home range", f'{num(row.get("simulation_home_score_p10"),0)}–{num(row.get("simulation_home_score_p90"),0)}', home_team)


# ============================================================
# ELO
# ============================================================


def render_elo_rows(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No ELO rows available.")
        return

    for _, row in frame.iterrows():
        logo = clean(row.get("logo_url", ""))
        delta = pd.to_numeric(pd.Series([row.get("elo_change")]), errors="coerce").iloc[0]
        if pd.isna(delta):
            delta_text, delta_class = "—", "flat"
        elif delta > 0:
            delta_text, delta_class = f"+{delta:.0f}", "up"
        elif delta < 0:
            delta_text, delta_class = f"{delta:.0f}", "down"
        else:
            delta_text, delta_class = "0", "flat"

        logo_html = (
            f'<img src="{html.escape(logo)}" width="32" height="32" style="object-fit:contain;" />'
            if logo.startswith("http") else ""
        )
        rank = clean(row.get("rank", "—"))
        st.markdown(
            '<div class="elo-row">'
            f'<div class="elo-rank">#{html.escape(rank)}</div>'
            f'<div>{logo_html}</div>'
            '<div>'
            f'<div class="elo-team">{html.escape(clean(row.get("team","")))}</div>'
            f'<div class="elo-conf">{html.escape(clean(row.get("conference","")))}</div>'
            '</div>'
            '<div>'
            f'<div class="elo-value">{num(row.get("elo"),0)}</div>'
            f'<div class="elo-delta {delta_class}">{html.escape(delta_text)}</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )


def render_elo(weekly: pd.DataFrame) -> None:
    section_title("ELO rankings", "Weekly team-strength snapshots with movement")
    if weekly.empty:
        st.info("No ELO data available.")
        return

    weeks = sorted(int(x) for x in pd.to_numeric(weekly["week"], errors="coerce").dropna().unique().tolist())
    c1, c2 = st.columns(2)
    with c1:
        selected_week = st.selectbox(
            "Snapshot",
            weeks,
            index=len(weeks) - 1,
            format_func=lambda week: "Preseason" if week == 0 else f"Week {week}",
        )
    conferences = ["All"] + sorted(weekly["conference"].dropna().astype(str).unique().tolist())
    with c2:
        conference = st.selectbox("Conference", conferences, key="elo_conf")

    view = weekly[weekly["week"].eq(selected_week)].copy()
    if conference != "All":
        view = view[view["conference"].astype(str).eq(conference)]

    with st.container(border=True):
        render_elo_rows(view.sort_values("rank", kind="stable"))

    section_title("Team ELO trend", "Branded weekly rating chart")
    teams = sorted(weekly["team"].dropna().astype(str).unique().tolist())
    selected_team = st.selectbox("Team", teams, key="elo_team")
    trend = weekly[weekly["team"].astype(str).eq(selected_team)].sort_values("week", kind="stable")

    if not trend.empty:
        branded_line_chart(trend, "week_label", "elo", ["week_label", "elo"], 300, "ELO")
        latest = trend.iloc[-1]
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Current ELO", num(latest.get("elo"), 0), clean(latest.get("week_label", "")))
        with c2:
            rank = latest.get("rank")
            rank_text = f"#{int(rank)}" if pd.notna(rank) else "—"
            metric_card("Current rank", rank_text, f'ELO Δ {num(latest.get("elo_change"),0)}')

        if st.button("Open team home →", key="elo_open_team", use_container_width=True):
            request_team(selected_team)


# ============================================================
# TEAMS
# ============================================================


def render_teams(games: pd.DataFrame, weekly: pd.DataFrame, profiles: pd.DataFrame) -> None:
    section_title("Team home", "Schedule, results, strength, form and model context in one place")
    if weekly.empty:
        st.info("No team data available.")
        return

    teams = sorted(weekly["team"].dropna().astype(str).unique().tolist())
    requested = st.session_state.get("selected_team")
    default_index = teams.index(requested) if requested in teams else 0
    selected = st.selectbox("Team", teams, index=default_index, key="teams_team_selector")
    st.session_state["selected_team"] = selected

    team_weekly = weekly[weekly["team"].astype(str).eq(selected)].sort_values("week", kind="stable")
    latest = team_weekly.iloc[-1]
    profile = pd.Series(dtype=object)
    if not profiles.empty and "team" in profiles.columns:
        match = profiles[profiles["team"].astype(str).eq(selected)]
        if not match.empty:
            profile = match.iloc[0]

    logo = clean(profile.get("logo_url", latest.get("logo_url", "")))
    conference = clean(profile.get("conference", latest.get("conference", "")))
    record = clean(profile.get("record", "—")) or "—"
    rank_val = profile.get("model_rank", latest.get("rank", np.nan))
    rank_text = f"#{int(float(rank_val))}" if pd.notna(rank_val) else "—"
    elo_val = profile.get("current_elo", latest.get("elo", np.nan))

    with st.container(border=True):
        logo_col, identity_col = st.columns([.24, .76], vertical_alignment="center")
        with logo_col:
            if logo.startswith("http"):
                st.image(logo, width=96)
        with identity_col:
            st.markdown(f"## {selected}")
            st.caption(f"{conference} · {record} · Model rank {rank_text}")
            st.markdown(f"**ELO {num(elo_val, 0)}**")

    section_title("Team outlook", "Current model state and early-season context")
    c1, c2 = st.columns(2)
    with c1:
        metric_card("ELO", num(elo_val, 0), f"Model rank {rank_text}")
    with c2:
        metric_card("Model trend", clean(profile.get("momentum_label", "Building sample")) or "Building sample",
                    f"Recent ELO {signed(profile.get('recent_elo_change'),0)}")
    c3, c4 = st.columns(2)
    with c3:
        off = profile.get("offensive_rating", np.nan)
        metric_card("Offensive rating", num(off, 2), "Production model metric" if pd.notna(off) else "Awaiting exported production rating")
    with c4:
        deff = profile.get("defensive_rating", np.nan)
        metric_card("Defensive rating", num(deff, 2), "Production model metric" if pd.notna(deff) else "Awaiting exported production rating")
    c5, c6 = st.columns(2)
    with c5:
        metric_card("Schedule", clean(profile.get("schedule_strength_label", "Building sample")) or "Building sample",
                    f"Opp. avg ELO {num(profile.get('schedule_strength_elo'),0)}")
    with c6:
        metric_card("Vs expectation", signed(profile.get("recent_margin_vs_expectation"),1), "Recent margin vs model")

    trend_label = clean(profile.get("momentum_label", "Building sample")) or "Building sample"
    expectation = pd.to_numeric(pd.Series([profile.get("recent_margin_vs_expectation", np.nan)]), errors="coerce").iloc[0]
    elo_move = pd.to_numeric(pd.Series([profile.get("recent_elo_change", np.nan)]), errors="coerce").iloc[0]
    if trend_label == "Building sample":
        outlook = "The season sample is still too small for a meaningful trend label."
    else:
        parts = [f"The model currently classifies {selected} as **{trend_label.lower()}**."]
        if pd.notna(elo_move): parts.append(f"Recent ELO movement is {elo_move:+.0f}.")
        if pd.notna(expectation): parts.append(f"Recent performance is {expectation:+.1f} points per game versus model expectation.")
        outlook = " ".join(parts)
    st.info(outlook)

    section_title("ELO trend", "Weekly model strength — no future snapshots")
    branded_line_chart(team_weekly, "week_label", "elo", ["week_label", "elo", "rank"], 300, "ELO")

    team_games = games[
        games["home_team"].astype(str).eq(selected) | games["away_team"].astype(str).eq(selected)
    ].copy().sort_values("start_date_utc", kind="stable")
    now = pd.Timestamp.now(tz="UTC")
    completed_rows = []
    upcoming_rows = []
    for _, game in team_games.iterrows():
        is_home = clean(game.get("home_team")) == selected
        opponent = clean(game.get("away_team" if is_home else "home_team"))
        venue = "vs" if is_home else "@"
        hs = get_actual_score(game, "home"); as_ = get_actual_score(game, "away")
        completed = pd.notna(hs) and pd.notna(as_)
        if completed:
            tp = float(hs if is_home else as_); op = float(as_ if is_home else hs)
            completed_rows.append({"Week": game.get("week"), "Opponent": f"{venue} {opponent}", "Result": f"{'W' if tp > op else 'L'} {tp:.0f}–{op:.0f}"})
        else:
            team_prob = display_value(game, "home_win_probability" if is_home else "away_win_probability")
            upcoming_rows.append({
                "Week": game.get("week"),
                "Opponent": f"{venue} {opponent}",
                "Kickoff": kickoff_text(game.get("start_date_utc")),
                "Win %": pct(team_prob),
                "Vegas spread": market_spread_text(game),
            })

    section_title("Results", "Completed 2026 games")
    if completed_rows:
        st.dataframe(pd.DataFrame(completed_rows).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.caption("No completed results yet.")

    section_title("Schedule", "Upcoming games and current model win probability")
    if upcoming_rows:
        st.dataframe(pd.DataFrame(upcoming_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No upcoming games in the current app dataset.")

    section_title("Season snapshot", "Simple context while the richer efficiency layer develops")
    a, b = st.columns(2)
    with a:
        metric_card("Points for / game", num(profile.get("points_for_per_game"), 1), record)
    with b:
        metric_card("Points against / game", num(profile.get("points_against_per_game"), 1), "Completed games")


# ============================================================
# BETTING PERFORMANCE
# ============================================================


def render_betting_performance(betting: pd.DataFrame) -> None:
    section_title("Betting performance", "Moneyline first; market tracking stays separate from the football model")
    if betting.empty:
        st.info("Betting performance data has not been built yet.")
        st.code("$env:PYTHONIOENCODING='utf-8'\npython -m jobs.build_betting_performance")
        return

    types = betting.get("bet_type", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
    ordered = [x for x in ["Moneyline", "Spread"] if x in types] + [x for x in types if x not in {"Moneyline", "Spread"}]
    default = "Moneyline" if "Moneyline" in ordered else (ordered[0] if ordered else None)
    if default is None:
        st.info("No settled bets are available yet.")
        return

    bet_type = st.segmented_control("Bet type", options=ordered, default=default, selection_mode="single", key="betting_type") or default
    view = betting[betting["bet_type"].astype(str).eq(bet_type)].copy().sort_values("bet_number", kind="stable")
    if view.empty:
        st.info(f"No settled {bet_type.lower()} bets are available yet.")
        return

    stake = pd.to_numeric(view["stake"], errors="coerce").fillna(0.0)
    profit_loss = pd.to_numeric(view["profit_loss"], errors="coerce").fillna(0.0)
    total_staked = float(stake.sum())
    total_pl = float(profit_loss.sum())
    roi = total_pl / total_staked if total_staked > 0 else np.nan

    normalized_results = (
        view.get("result", pd.Series(dtype=str))
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({
            "WIN": "W",
            "WON": "W",
            "LOSS": "L",
            "LOST": "L",
            "PUSH": "P",
            "TIE": "P",
        })
    )
    result_counts = normalized_results.value_counts()
    record = f'{int(result_counts.get("W",0))}–{int(result_counts.get("L",0))}'
    pushes = int(result_counts.get("P", 0))
    if pushes:
        record += f"–{pushes}"

    c1, c2 = st.columns(2)
    with c1:
        metric_card("Record", record, f"{len(view):,} settled bets")
    with c2:
        metric_card("Staked", f"£{total_staked:,.0f}", "£100 baseline stake")
    c3, c4 = st.columns(2)
    with c3:
        metric_card("P / L", pounds(total_pl), "Chronological cumulative")
    with c4:
        metric_card("ROI", f"{roi * 100:+.1f}%" if np.isfinite(roi) else "—", "Profit / total staked")

    section_title("Cumulative P / L", "A calmer, branded equity curve")
    chart = view[["bet_number", "cumulative_profit_loss"]].copy()
    chart["bet_number"] = pd.to_numeric(chart["bet_number"], errors="coerce")
    chart["cumulative_profit_loss"] = pd.to_numeric(chart["cumulative_profit_loss"], errors="coerce")
    chart = pd.concat([pd.DataFrame({"bet_number": [0], "cumulative_profit_loss": [0.0]}), chart], ignore_index=True).dropna()
    branded_pl_chart(chart, 330)

    reconstructed = int(view.get("source_type", pd.Series(dtype=str)).astype(str).eq("historical_reconstruction").sum())
    if reconstructed:
        st.caption(f"Includes {reconstructed} historical-reconstruction {bet_type.lower()} bets. Official live results remain separately identifiable in the history.")

    with st.expander("Bet history"):
        history_columns = [c for c in [
            "bet_number", "week", "away_team", "home_team", "bet_team", "market_line",
            "market_odds", "result", "profit_loss", "cumulative_profit_loss", "source_type"
        ] if c in view.columns]
        history = view[history_columns].rename(columns={
            "bet_number": "Bet", "week": "Week", "away_team": "Away", "home_team": "Home",
            "bet_team": "Model Bet", "market_line": "Line / Odds", "market_odds": "Price",
            "result": "Result", "profit_loss": "P/L £", "cumulative_profit_loss": "Running P/L £",
            "source_type": "Source",
        })
        st.dataframe(history, use_container_width=True, hide_index=True)


# ============================================================
# MORE
# ============================================================


def render_more(performance: dict[str, Any], betting: pd.DataFrame) -> None:
    choice = st.segmented_control(
        "Section",
        options=["Betting", "Performance", "Model"],
        default="Betting",
        selection_mode="single",
        key="more_section",
    ) or "Betting"

    if choice == "Betting":
        render_betting_performance(betting)
        return

    if choice == "Performance":
        section_title("Model performance", "Official forward performance is kept separate from reconstruction")
        official = performance.get("official_locked", {}) if performance else {}
        reconstructed = performance.get("historical_reconstruction", {}) if performance else {}

        c1, c2 = st.columns(2)
        with c1:
            metric_card("Official record", official.get("record") or "—", f'{official.get("games",0)} completed')
        with c2:
            metric_card("Official accuracy", pct(official.get("accuracy")) if official.get("accuracy") is not None else "—", "Genuine locked predictions")

        section_title("2026 diagnostic", "Historical reconstruction only — never mixed into official performance")
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Record", reconstructed.get("record") or "—", f'{reconstructed.get("games",0)} games')
        with c2:
            metric_card("Accuracy", pct(reconstructed.get("accuracy")) if reconstructed.get("accuracy") is not None else "—", f'Brier {num(reconstructed.get("brier"),3)}')
        return

    section_title("Model architecture", "V2 remains production while V3 is evaluated as the challenger")
    with st.container(border=True):
        st.markdown(
            """
**Winner probability**  
ELO → opponent-adjusted matchup → frozen `elo_matchup` winner model.

**Score projection**  
Game environment → possessions / drives → scoring → frozen `compact_full` correction.

**Monte Carlo**  
Simulation around corrected expected scores using uncertainty learned from walk-forward 2024–2025 score residuals.

**ELO**  
Every completed result updates team strength chronologically. Weekly snapshots are carried through bye weeks.

**Official lifecycle**  
Predictions stay dynamic until the lock window. Once locked, the official prediction is immutable. Historical reconstruction can never become an official prediction.

**Betting layer**  
Market prices are used for tracking / settlement only. They do not feed the football prediction model.
"""
        )


# ============================================================
# APP
# ============================================================


render_header()

games, weekly, rankings, performance, betting, profiles = load_data()

if games.empty:
    st.error("App data has not been built.")
    st.code("$env:PYTHONIOENCODING='utf-8'\npython -m jobs.build_app_data")
    st.stop()

pending_nav = st.session_state.pop("pending_nav", None)
if pending_nav is not None:
    st.session_state["nav"] = pending_nav

legacy_nav = {
    "Home": "⌂ Home",
    "Picks": "◈ Picks",
    "Upsets": "⚡ Upsets",
    "Game": "◎ Game",
    "ELO": "↕ ELO",
    "Teams": "◉ Teams",
    "More": "•• More",
}
if st.session_state.get("nav") in legacy_nav:
    st.session_state["nav"] = legacy_nav[st.session_state["nav"]]

if "nav" not in st.session_state:
    st.session_state["nav"] = "⌂ Home"

page = st.radio(
    "Navigation",
    ["⌂ Home", "◈ Picks", "⚡ Upsets", "◎ Game", "↕ ELO", "◉ Teams", "•• More"],
    horizontal=True,
    label_visibility="collapsed",
    key="nav",
)

if page == "⌂ Home":
    render_home(games, rankings, performance)
elif page == "◈ Picks":
    render_picks(games, rankings)
elif page == "⚡ Upsets":
    render_upsets(games, rankings)
elif page == "◎ Game":
    render_game(games)
elif page == "↕ ELO":
    render_elo(weekly)
elif page == "◉ Teams":
    render_teams(games, weekly, profiles)
else:
    render_more(performance, betting)
