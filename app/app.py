from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

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
V3_TRACKER_PATH = APP_DATA_DIR / "v3_tracker.csv"
ELO_WEEKLY_PATH = APP_DATA_DIR / "elo_weekly.csv"
V3_ELO_WEEKLY_PATH = APP_DATA_DIR / "v3_elo_weekly.csv"
ELO_RANKINGS_PATH = APP_DATA_DIR / "elo_rankings_current.csv"
PERFORMANCE_PATH = APP_DATA_DIR / "performance.json"
BETTING_PATH = APP_DATA_DIR / "betting_performance.csv"
V3_BETTING_PATH = APP_DATA_DIR / "v3_betting_performance.csv"
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

# V3 production visual layer.  This intentionally sits after the original
# stylesheet so the proven component markup can keep working while the app
# adopts the dark, broadcast-style design used in the product mock-ups.
st.markdown(
    """
<style>
:root {
    --bg:#050a12;
    --bg-soft:#08111d;
    --panel:#0c1725;
    --panel-2:#101e2e;
    --panel-3:#14263a;
    --line:#1d3348;
    --line-soft:rgba(117,151,181,.16);
    --text:#f4f8fb;
    --muted:#8fa4b8;
    --muted-2:#657c91;
    --accent:#39a7ff;
    --accent-2:#70c7ff;
    --green:#48e09b;
    --amber:#ffbd57;
    --orange:#ff7a45;
    --purple:#a880ff;
    --red:#ff6475;
    --shadow:0 18px 50px rgba(0,0,0,.28);
}

html, body, .stApp { background:#050a12 !important; color:var(--text); }
.stApp {
    background:
      radial-gradient(circle at 74% -10%,rgba(35,126,200,.18),transparent 30%),
      radial-gradient(circle at 8% 28%,rgba(92,60,179,.10),transparent 25%),
      linear-gradient(180deg,#050a12 0%,#07101a 55%,#050a12 100%) !important;
}
.block-container { max-width:1260px; padding-top:.75rem; padding-bottom:5rem; }
p, li, label, .stMarkdown { color:var(--text); }
[data-testid="stCaptionContainer"], .stCaption { color:var(--muted) !important; }

.cfb-header { margin:0 0 14px; padding:0 2px; }
.cfb-mark {
    width:44px;height:44px;border-radius:14px;
    background:linear-gradient(145deg,#0f6eb4,#164a75);
    border:1px solid rgba(112,199,255,.50);
    box-shadow:0 0 30px rgba(57,167,255,.22);
}
.cfb-brand-title { color:#fff; font-size:1rem; letter-spacing:.055em; }
.cfb-brand-sub { color:var(--muted); font-size:.66rem; }
.cfb-season-pill {
    color:#b9e2ff;background:rgba(21,72,111,.32);border-color:#245b82;
    box-shadow:inset 0 0 18px rgba(57,167,255,.07);
}

div[data-testid="stRadio"] {
    background:rgba(8,17,29,.94) !important;border-color:#1b3146 !important;
    border-radius:14px !important;box-shadow:0 14px 38px rgba(0,0,0,.24) !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label {
    color:var(--muted) !important;background:transparent;border-color:transparent;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    color:#fff !important;background:#112237;border-color:#203b54;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    color:#fff !important;background:linear-gradient(135deg,#136da9,#1b8ed1) !important;
    border-color:#32a8ed !important;box-shadow:0 7px 20px rgba(21,136,205,.24);
}

.hero {
    border-color:#213a51 !important;
    background:
      linear-gradient(100deg,rgba(7,20,34,.97),rgba(13,35,55,.93)),
      radial-gradient(circle at 86% 20%,rgba(72,224,155,.18),transparent 35%) !important;
    box-shadow:0 22px 60px rgba(0,0,0,.34),inset 0 1px 0 rgba(255,255,255,.03) !important;
    padding:26px !important;border-radius:22px !important;
}
.hero:before {
    content:"V3";position:absolute;right:24px;bottom:-27px;
    font-size:8.5rem;font-weight:1000;letter-spacing:-.08em;
    color:rgba(96,190,255,.045);pointer-events:none;
}
.hero:after { background:radial-gradient(circle,rgba(72,224,155,.14),transparent 67%); }
.hero-kicker { color:var(--green);font-size:.65rem; }
.hero-title { color:#fff !important;font-size:clamp(1.7rem,5vw,3rem);max-width:800px; }
.hero-copy { color:#9eb2c5;font-size:.83rem;max-width:760px; }

.section-title { color:#f6f9fc;font-size:1.08rem; }
.section-sub { color:var(--muted); }
[data-testid="stVerticalBlockBorderWrapper"] {
    background:linear-gradient(180deg,rgba(15,29,44,.98),rgba(10,21,34,.98)) !important;
    border-color:#1d344a !important;border-radius:18px !important;
    box-shadow:0 12px 32px rgba(0,0,0,.18) !important;
}
.metric-card {
    background:linear-gradient(155deg,#101f30,#0b1725) !important;
    border-color:#1d354b !important;border-radius:16px;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.025),0 12px 30px rgba(0,0,0,.15);
    min-height:104px;position:relative;overflow:hidden;
}
.metric-card:after {
    content:"";position:absolute;left:0;top:0;bottom:0;width:3px;
    background:linear-gradient(180deg,var(--accent),rgba(57,167,255,.10));
}
.metric-label { color:#7f9ab1;font-size:.61rem; }
.metric-value { color:#fff !important;font-size:1.55rem; }
.metric-note { color:#7790a6 !important; }

.game-meta { color:#8198ad; }
.team-name,.detail-team,.score-number,.factor-value,.elo-team,.elo-value { color:#f7fbff !important; }
.team-meta,.score-caption,.factor-label,.elo-conf { color:var(--muted) !important; }
.prob,.detail-prob { color:var(--accent-2) !important; }
.prob-track { height:9px;background:#172a3d; }
.prob-away { background:linear-gradient(90deg,#704dde,#9d7cff); }
.prob-home { background:linear-gradient(90deg,#168ccc,#48c8ff); }
.favourite-line { color:#9db3c6; }
.market-strip {
    background:#0a1623 !important;border-color:#1a344c !important;color:#8fa9be !important;
}
.market-strip strong { color:#dcebf6 !important; }
.upset-banner {
    background:linear-gradient(90deg,rgba(255,122,69,.13),rgba(255,189,87,.05)) !important;
    border-color:rgba(255,145,79,.28) !important;color:#d6b29e !important;
}
.upset-banner strong { color:#ffd7b9 !important; }
.upset-badge { background:rgba(255,122,69,.17);color:#ffad7d; }

.status-chip { color:#91a8bc;background:#102236;border-color:#25415a; }
.status-chip.good { color:#79edb5;border-color:rgba(72,224,155,.34);background:rgba(72,224,155,.08); }
.status-chip.accent { color:#78cbff;border-color:rgba(57,167,255,.34);background:rgba(57,167,255,.08); }
.status-chip.warn { color:#ffd078;border-color:rgba(255,189,87,.34);background:rgba(255,189,87,.08); }

.elo-rank { color:#6e8599; }.up{color:var(--green)}.down{color:var(--red)}.flat{color:var(--muted)}
.factor-row,.elo-row { border-color:var(--line-soft); }
.team-hero { background:linear-gradient(145deg,#112238,#0b1725);border-color:#1f3a52; }
.team-hero-name { color:#fff; }.team-hero-meta { color:var(--muted); }

.stButton>button {
    background:#102236 !important;color:#b9d0e2 !important;border-color:#24425d !important;
    border-radius:11px !important;font-weight:850;
}
.stButton>button:hover {
    background:#173854 !important;color:#fff !important;border-color:#38a5e9 !important;
}
div[data-baseweb="select"]>div,input,[data-baseweb="input"] {
    background:#0c1927 !important;border-color:#223b52 !important;color:#edf6fc !important;
}
div[data-baseweb="popover"],ul[role="listbox"] { background:#0d1b2a !important;color:#fff !important; }
li[role="option"]:hover { background:#17324a !important; }
.stSelectbox label,.stTextInput label,.stSegmentedControl label,.stMultiSelect label,.stSlider label { color:#8ba2b7 !important; }
[data-testid="stSegmentedControl"] button,button[data-testid="stBaseButton-segmented_control"] {
    background:#0c1927 !important;color:#91a9bd !important;border-color:#21394f !important;
}
[data-testid="stSegmentedControl"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] button[data-selected="true"],
button[data-testid="stBaseButton-segmented_controlActive"] {
    color:#fff !important;background:linear-gradient(135deg,#146ea9,#178bc9) !important;border-color:#2aa7ed !important;
}
[data-testid="stDataFrame"] { border-color:#1d344a; }
[data-testid="stExpander"] { background:#0b1724;border-color:#1d344a !important; }
[data-testid="stAlert"] { background:#0d1c2b;color:#c7d6e2;border-color:#25425b; }

.kpi-grid { display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:8px 0 18px; }
.kpi {
    padding:14px;border:1px solid #1c344a;border-radius:15px;
    background:linear-gradient(150deg,#101e2f,#0a1623);min-height:96px;
}
.kpi-label { color:#7690a6;font-size:.59rem;font-weight:850;text-transform:uppercase;letter-spacing:.08em; }
.kpi-value { color:#fff;font-size:1.52rem;font-weight:950;margin-top:8px;letter-spacing:-.025em; }
.kpi-delta { color:#6f8aa0;font-size:.61rem;margin-top:5px; }
.kpi.good .kpi-value{color:var(--green)}.kpi.warn .kpi-value{color:var(--amber)}

.confidence-grid { display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:7px;margin:8px 0; }
.confidence-tile { padding:11px 8px;border-radius:13px;background:#0b1826;border:1px solid #1c3349;text-align:center; }
.confidence-count { color:#fff;font-size:1.28rem;font-weight:950; }
.confidence-name { color:#8098ac;font-size:.55rem;text-transform:uppercase;font-weight:850;margin-top:4px; }
.confidence-tile:nth-child(1){border-top:3px solid #48e09b}.confidence-tile:nth-child(2){border-top:3px solid #39a7ff}
.confidence-tile:nth-child(3){border-top:3px solid #a880ff}.confidence-tile:nth-child(4){border-top:3px solid #ffbd57}
.confidence-tile:nth-child(5){border-top:3px solid #ff7a45}.confidence-tile:nth-child(6){border-top:3px solid #64798c}

.insight-card { padding:15px;border:1px solid #1c344a;border-radius:15px;background:#0b1826;margin:8px 0; }
.insight-kicker { color:var(--green);font-size:.57rem;font-weight:900;letter-spacing:.08em;text-transform:uppercase; }
.insight-title { color:#f4f8fb;font-size:.88rem;font-weight:900;margin-top:6px; }
.insight-copy { color:#8da4b8;font-size:.68rem;line-height:1.45;margin-top:5px; }
.model-pill { display:inline-block;padding:4px 8px;border-radius:999px;font-size:.56rem;font-weight:900;margin-right:5px; }
.model-pill.v2 { color:#78cbff;background:rgba(57,167,255,.12);border:1px solid rgba(57,167,255,.28); }
.model-pill.v3 { color:#75edb5;background:rgba(72,224,155,.11);border:1px solid rgba(72,224,155,.27); }
.model-pill.split { color:#ffd078;background:rgba(255,189,87,.10);border:1px solid rgba(255,189,87,.26); }

@media(max-width:800px){
  .kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.confidence-grid{grid-template-columns:repeat(3,minmax(0,1fr))}
  .hero{padding:19px !important}.hero:before{font-size:6rem;right:14px}.cfb-season-pill{display:none}
}
@media(max-width:520px){
  .block-container{padding-left:.62rem;padding-right:.62rem;padding-bottom:5.5rem}
  .cfb-brand-sub{display:none}.cfb-mark{width:38px;height:38px}.cfb-brand-title{font-size:.82rem}
  div[data-testid="stRadio"]{position:fixed !important;top:auto !important;left:7px;right:7px;bottom:max(7px,env(safe-area-inset-bottom));z-index:9999;margin:0 !important;padding:5px !important}
  div[data-testid="stRadio"] div[role="radiogroup"] label{min-width:58px !important;min-height:44px !important;padding:3px 6px !important}
  div[data-testid="stRadio"] div[role="radiogroup"] label p{font-size:.58rem !important}
  .kpi-value{font-size:1.32rem}.confidence-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
}
</style>
""",
    unsafe_allow_html=True,
)


# Final approved application shell and page-specific components.
st.markdown(
    """
<style>
/* ---------- production shell ---------- */
.block-container{max-width:1480px!important;padding-top:1rem!important;padding-left:1.2rem!important;padding-right:1.2rem!important}
.cfb-header{height:52px;border-bottom:1px solid rgba(67,111,148,.22);margin:0 0 18px!important}
.cfb-brand-title{letter-spacing:.14em!important}.cfb-brand-sub{letter-spacing:.04em}

@media(min-width:901px){
  .block-container{margin-left:205px!important;width:calc(100% - 205px)!important}
  div[data-testid="stRadio"]{
    position:fixed!important;left:0!important;top:0!important;bottom:0!important;width:190px!important;
    height:100vh!important;margin:0!important;padding:102px 12px 28px!important;border-width:0 1px 0 0!important;
    border-radius:0!important;background:
      linear-gradient(180deg,rgba(6,18,33,.99),rgba(5,13,24,.99)),
      radial-gradient(circle at 50% 12%,rgba(24,160,224,.16),transparent 28%)!important;
    box-shadow:12px 0 40px rgba(0,0,0,.20)!important;overflow:hidden!important;
  }
  div[data-testid="stRadio"]:before{
    content:"CFB\\A PREDICTION CENTRE";white-space:pre;position:absolute;left:22px;top:24px;
    color:#f7fbff;font-size:.78rem;font-weight:950;line-height:1.25;letter-spacing:.13em;
    padding-left:29px;background:radial-gradient(circle at 9px 11px,#25c8ff 0 5px,transparent 6px);
  }
  div[data-testid="stRadio"]:after{
    content:"DATA · MODELS · INSIGHT";position:absolute;left:22px;bottom:24px;
    color:#456b88;font-size:.48rem;font-weight:850;letter-spacing:.14em;
  }
  div[data-testid="stRadio"] div[role="radiogroup"]{display:flex!important;flex-direction:column!important;gap:7px!important;overflow:visible!important}
  div[data-testid="stRadio"] div[role="radiogroup"] label{
    width:100%!important;min-height:46px!important;justify-content:flex-start!important;padding:0 14px!important;border-radius:11px!important;
  }
  div[data-testid="stRadio"] div[role="radiogroup"] label p{font-size:.72rem!important;text-align:left!important}
}

/* ---------- shared editorial pieces ---------- */
.page-eyebrow{color:#31c9ff;font-size:.58rem;font-weight:900;letter-spacing:.18em;text-transform:uppercase;margin-bottom:6px}
.page-title{color:#fff;font-size:clamp(1.65rem,4vw,2.75rem);line-height:.98;font-weight:1000;letter-spacing:-.035em;margin:0}
.page-subtitle{color:#8ca5ba;font-size:.73rem;line-height:1.5;margin-top:8px;max-width:760px}
.page-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin:5px 0 20px}
.live-dot{display:inline-flex;align-items:center;gap:7px;color:#8ea8bc;font-size:.60rem;font-weight:800}
.live-dot:before{content:"";width:7px;height:7px;border-radius:50%;background:#4be49d;box-shadow:0 0 14px rgba(75,228,157,.8)}

.command-hero{
  position:relative;overflow:hidden;border:1px solid #1d4b6b;border-radius:23px;padding:25px;
  background:
   linear-gradient(100deg,rgba(5,17,30,.97) 0%,rgba(9,30,49,.88) 52%,rgba(15,31,48,.93) 100%),
   repeating-linear-gradient(90deg,transparent 0 84px,rgba(255,255,255,.018) 85px 86px);
  box-shadow:0 24px 70px rgba(0,0,0,.34),inset 0 1px rgba(255,255,255,.04);margin-bottom:14px
}
.command-hero:after{content:"";position:absolute;right:-80px;top:-130px;width:430px;height:430px;border-radius:50%;background:radial-gradient(circle,rgba(36,172,235,.19),transparent 68%);pointer-events:none}
.command-title{font-size:clamp(1.7rem,5vw,3.2rem);font-weight:1000;letter-spacing:-.045em;line-height:.97;color:#fff;max-width:750px;position:relative;z-index:1}
.command-copy{font-size:.74rem;color:#8ea8bd;max-width:690px;line-height:1.55;margin-top:9px;position:relative;z-index:1}
.micro-row{display:flex;gap:7px;flex-wrap:wrap;margin-top:15px}.micro-pill{padding:5px 9px;border-radius:999px;border:1px solid #23415b;background:rgba(12,29,45,.76);color:#91abc0;font-size:.55rem;font-weight:850}.micro-pill.good{border-color:rgba(72,224,155,.35);color:#70e8ad}.micro-pill.warn{border-color:rgba(255,189,87,.35);color:#ffd078}

.featured-stage{border:1px solid #24516f;border-radius:20px;background:linear-gradient(110deg,rgba(50,13,24,.92),rgba(8,24,40,.98) 48%,rgba(54,10,25,.90));padding:18px;margin:11px 0 16px;box-shadow:0 18px 50px rgba(0,0,0,.25)}
.featured-label{font-size:.56rem;color:#45cfff;font-weight:900;letter-spacing:.13em;text-transform:uppercase;margin-bottom:12px}
.team-lockup{text-align:center}.team-lockup img{width:86px;height:86px;object-fit:contain;filter:drop-shadow(0 10px 20px rgba(0,0,0,.34))}.team-lockup-name{font-size:1.05rem;font-weight:950;color:#fff;line-height:1.05;margin-top:7px}.team-lockup-meta{font-size:.58rem;color:#91a8bb;margin-top:4px}
.stage-score{text-align:center}.stage-score-label{color:#6f8aa0;font-size:.52rem;font-weight:900;letter-spacing:.12em;text-transform:uppercase}.stage-score-value{font-size:2.65rem;color:#fff;font-weight:1000;letter-spacing:-.06em;line-height:1;margin-top:7px}.stage-prob{font-size:.72rem;font-weight:900;color:#70e8ad;margin-top:7px}

.section-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:8px 0 16px}
.signal-card{background:linear-gradient(155deg,#0f2032,#091624);border:1px solid #1c3b54;border-radius:15px;padding:13px;min-height:96px}.signal-card .label{font-size:.55rem;color:#7390a6;font-weight:900;letter-spacing:.08em;text-transform:uppercase}.signal-card .value{font-size:1.5rem;color:#fff;font-weight:1000;margin-top:7px;letter-spacing:-.03em}.signal-card .note{font-size:.56rem;color:#6f889d;margin-top:5px}.signal-card.cyan{border-top:2px solid #35c8ff}.signal-card.green{border-top:2px solid #48e09b}.signal-card.amber{border-top:2px solid #ffbd57}.signal-card.purple{border-top:2px solid #a880ff}

.broadcast-card{border:1px solid #1d3c55;border-radius:18px;background:linear-gradient(160deg,rgba(15,32,49,.98),rgba(7,18,30,.98));padding:14px;box-shadow:0 14px 34px rgba(0,0,0,.18);margin-bottom:10px;position:relative;overflow:hidden}.broadcast-card:before{content:"";position:absolute;left:0;right:0;top:0;height:2px;background:linear-gradient(90deg,transparent,#2cc8ff,transparent);opacity:.55}.broadcast-meta{display:flex;justify-content:space-between;align-items:center;color:#7792a7;font-size:.54rem;font-weight:850;letter-spacing:.04em}.broadcast-teams{display:grid;grid-template-columns:minmax(0,1fr) 100px minmax(0,1fr);gap:10px;align-items:center;margin:13px 0}.broadcast-team{display:flex;align-items:center;gap:9px;min-width:0}.broadcast-team.right{flex-direction:row-reverse;text-align:right}.broadcast-logo{width:52px;height:52px;object-fit:contain;filter:drop-shadow(0 7px 13px rgba(0,0,0,.3))}.broadcast-name{font-size:.80rem;font-weight:950;color:#fff;line-height:1.06}.broadcast-record{font-size:.54rem;color:#7893a9;margin-top:4px}.broadcast-score{text-align:center}.broadcast-score strong{font-size:1.45rem;color:#fff;letter-spacing:-.04em}.broadcast-score span{display:block;font-size:.48rem;color:#6f8a9f;text-transform:uppercase;letter-spacing:.08em}.probability-line{height:7px;border-radius:999px;overflow:hidden;background:#172c40;display:flex;margin:8px 0 5px}.probability-line .away{background:linear-gradient(90deg,#8a65ee,#a986ff)}.probability-line .home{background:linear-gradient(90deg,#24aeea,#47d1ff)}.broadcast-footer{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;margin-top:9px}.badge{display:inline-flex;padding:4px 7px;border-radius:999px;font-size:.50rem;font-weight:900;border:1px solid #24435b;color:#8ea9be;background:#0d1d2d}.badge.strong{color:#74ecb2;border-color:rgba(72,224,155,.34);background:rgba(72,224,155,.07)}.badge.alert{color:#ffd078;border-color:rgba(255,189,87,.34);background:rgba(255,189,87,.07)}.badge.split{color:#d0baff;border-color:rgba(168,128,255,.34);background:rgba(168,128,255,.07)}

.team-banner{position:relative;overflow:hidden;border-radius:22px;border:1px solid var(--team-border,#274a66);padding:22px;background:linear-gradient(105deg,var(--team-glow,rgba(24,90,132,.35)),rgba(7,20,33,.97) 55%,rgba(7,18,30,.98));margin-bottom:14px}.team-banner:after{content:"";position:absolute;width:330px;height:330px;border-radius:50%;right:-80px;top:-130px;background:radial-gradient(circle,var(--team-glow,rgba(43,177,238,.20)),transparent 67%)}.team-banner-inner{display:grid;grid-template-columns:125px minmax(0,1fr) minmax(340px,.8fr);gap:18px;align-items:center;position:relative;z-index:1}.team-banner-logo{width:115px;height:115px;object-fit:contain;filter:drop-shadow(0 12px 20px rgba(0,0,0,.38))}.team-banner-name{font-size:clamp(1.7rem,4vw,3.1rem);color:#fff;font-weight:1000;line-height:.92;letter-spacing:-.04em}.team-banner-meta{font-size:.66rem;color:#a4b8c9;margin-top:8px}.team-kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.team-kpi{background:rgba(7,18,29,.62);border:1px solid rgba(104,145,177,.25);border-radius:13px;padding:12px}.team-kpi span{display:block;color:#718da3;font-size:.49rem;font-weight:900;text-transform:uppercase;letter-spacing:.08em}.team-kpi strong{display:block;color:#fff;font-size:1.25rem;margin-top:6px}.team-kpi strong.up{color:#48e09b}

.matchup-stage{border:1px solid #254962;border-radius:22px;overflow:hidden;background:radial-gradient(circle at 50% 40%,rgba(27,130,185,.17),transparent 26%),linear-gradient(110deg,rgba(67,10,28,.90),rgba(5,17,29,.98) 42% 58%,rgba(74,9,27,.9));padding:20px;margin-bottom:12px}.matchup-grid{display:grid;grid-template-columns:minmax(0,1fr) 230px minmax(0,1fr);gap:16px;align-items:center}.matchup-team{text-align:center}.matchup-team img{width:125px;height:125px;object-fit:contain;filter:drop-shadow(0 15px 24px rgba(0,0,0,.4))}.matchup-name{font-size:1.35rem;color:#fff;font-weight:1000;line-height:1;margin-top:5px}.matchup-meta{font-size:.59rem;color:#a0b3c4;margin-top:6px}.matchup-centre{text-align:center}.matchup-score{font-size:3.15rem;color:#fff;font-weight:1000;letter-spacing:-.07em;line-height:1}.matchup-probs{display:flex;justify-content:center;gap:18px;margin-top:7px;font-size:.82rem;font-weight:950}.matchup-probs .left{color:#ff6077}.matchup-probs .right{color:#69d5ff}.matchup-confidence{display:inline-block;margin-top:9px;border:1px solid rgba(72,224,155,.4);background:rgba(72,224,155,.10);color:#6fe7ac;border-radius:999px;padding:5px 13px;font-size:.55rem;font-weight:950;letter-spacing:.06em}

.model-card{height:100%;border:1px solid #21415a;border-radius:15px;background:#0b1a29;padding:14px}.model-card.v2{border-top:2px solid #35c8ff}.model-card.v3{border-top:2px solid #a880ff}.model-card-title{font-size:.61rem;color:#8ba6bb;font-weight:900;letter-spacing:.08em}.model-card-score{font-size:1.7rem;color:#fff;font-weight:1000;margin-top:9px}.model-card-note{font-size:.57rem;color:#728da2;margin-top:5px}.agreement{display:inline-flex;margin-top:10px;padding:4px 8px;border-radius:999px;background:rgba(72,224,155,.09);border:1px solid rgba(72,224,155,.3);color:#70e8ad;font-size:.50rem;font-weight:950}

.rank-row{display:grid;grid-template-columns:42px 40px minmax(0,1fr) 92px 78px 75px;gap:8px;align-items:center;padding:9px 7px;border-bottom:1px solid rgba(101,139,168,.13);transition:.15s}.rank-row:hover{background:#10253a;transform:translateX(2px)}.rank-num{font-size:.72rem;color:#7892a7;font-weight:900;text-align:center}.rank-logo{width:30px;height:30px;object-fit:contain}.rank-team{font-size:.73rem;color:#fff;font-weight:900}.rank-conf{font-size:.51rem;color:#6f899f;margin-top:2px}.rank-elo{font-size:.76rem;color:#fff;font-weight:950;text-align:right}.rank-delta{text-align:right;font-size:.62rem;font-weight:900}.rank-delta.rise{color:#48e09b}.rank-delta.fall{color:#ff6577}.rank-record{text-align:right;color:#8ba2b5;font-size:.59rem}

.empty-state{padding:32px 20px;text-align:center;border:1px dashed #29455c;border-radius:16px;background:rgba(10,24,37,.6);color:#7893a7;font-size:.68rem}.subnav-note{font-size:.55rem;color:#6d879b;margin:-3px 0 8px}.integrity-card{padding:14px;border:1px solid rgba(72,224,155,.25);border-radius:15px;background:linear-gradient(145deg,rgba(72,224,155,.065),rgba(8,22,34,.92))}.integrity-title{color:#6ee6aa;font-size:.64rem;font-weight:950;letter-spacing:.08em}.integrity-copy{color:#829bad;font-size:.60rem;line-height:1.5;margin-top:6px}

@media(max-width:1100px){.section-grid{grid-template-columns:repeat(2,1fr)}.team-banner-inner{grid-template-columns:100px 1fr}.team-kpis{grid-column:1/-1}.matchup-grid{grid-template-columns:1fr 180px 1fr}.matchup-team img{width:95px;height:95px}}
@media(max-width:900px){.block-container{margin-left:0!important;width:100%!important}.cfb-header{margin-bottom:10px!important}.team-banner-inner{grid-template-columns:80px 1fr}.team-banner-logo{width:76px;height:76px}.team-kpis{grid-template-columns:repeat(3,1fr)}.page-heading{align-items:flex-start;flex-direction:column}.matchup-grid{grid-template-columns:1fr 135px 1fr}.matchup-score{font-size:2.2rem}}
@media(max-width:640px){.section-grid{grid-template-columns:repeat(2,1fr);gap:7px}.signal-card{min-height:84px;padding:11px}.signal-card .value{font-size:1.2rem}.command-hero{padding:18px}.featured-stage{padding:12px}.broadcast-teams{grid-template-columns:1fr 72px 1fr}.broadcast-logo{width:42px;height:42px}.broadcast-name{font-size:.68rem}.matchup-stage{padding:13px}.matchup-grid{grid-template-columns:1fr 92px 1fr;gap:5px}.matchup-team img{width:66px;height:66px}.matchup-name{font-size:.77rem}.matchup-score{font-size:1.65rem}.matchup-probs{gap:7px;font-size:.64rem}.team-banner{padding:15px}.team-banner-inner{grid-template-columns:62px 1fr;gap:10px}.team-banner-logo{width:58px;height:58px}.team-banner-name{font-size:1.55rem}.team-kpis{grid-template-columns:repeat(3,1fr);gap:5px}.team-kpi{padding:8px}.team-kpi strong{font-size:.92rem}.rank-row{grid-template-columns:28px 30px minmax(0,1fr) 58px 50px}.rank-record{display:none}}
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
    games_source = V3_TRACKER_PATH if V3_TRACKER_PATH.exists() and V3_TRACKER_PATH.stat().st_size > 0 else (GAMES_V3_PATH if GAMES_V3_PATH.exists() and GAMES_V3_PATH.stat().st_size > 0 else GAMES_PATH)
    games = safe_read_csv(games_source)
    weekly = safe_read_csv(V3_ELO_WEEKLY_PATH if V3_ELO_WEEKLY_PATH.exists() and V3_ELO_WEEKLY_PATH.stat().st_size > 0 else ELO_WEEKLY_PATH)
    rankings = safe_read_csv(ELO_RANKINGS_PATH)
    betting = safe_read_csv(V3_BETTING_PATH if V3_BETTING_PATH.exists() and V3_BETTING_PATH.stat().st_size > 0 else BETTING_PATH)
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
    """Return the app's authoritative display value.

    V3 owns the primary forecast UI.  The overlay still carries the legacy
    generic/V2 columns for comparison, so generic score/probability fields
    must never silently win when a V3 forecast is present.
    """
    v3_column = {
        "projected_away_score": "v3_projected_away_points",
        "projected_home_score": "v3_projected_home_points",
        "away_win_probability": "v3_away_win_probability",
        "home_win_probability": "v3_home_win_probability",
        "predicted_winner": "v3_predicted_winner",
    }.get(column)
    if v3_column and v3_column in row.index and pd.notna(row.get(v3_column)):
        return row.get(v3_column)
    display_column = f"display_{column}"
    if display_column in row.index and pd.notna(row[display_column]):
        return row[display_column]
    return row[column] if column in row.index else np.nan


def legacy_display_value(row: pd.Series, column: str) -> Any:
    """Explicit V2/legacy value used only in comparison panels."""
    display_column = f"display_{column}"
    if display_column in row.index and pd.notna(row.get(display_column)):
        return row.get(display_column)
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
    st.session_state["pending_nav"] = "◈ Matchup"
    try:
        st.query_params["page"] = "matchup"
        st.query_params["game"] = str(int(game_id))
        if "team" in st.query_params:
            del st.query_params["team"]
    except Exception:
        pass
    st.rerun()


def request_team(team: str) -> None:
    st.session_state["selected_team"] = str(team)
    st.session_state["pending_nav"] = "◉ Team"
    try:
        st.query_params["page"] = "team"
        st.query_params["team"] = str(team)
        if "game" in st.query_params:
            del st.query_params["game"]
    except Exception:
        pass
    st.rerun()


def request_page(page: str) -> None:
    """Navigate while retaining a shareable page route where supported."""
    st.session_state["pending_nav"] = page
    slug_lookup = {
        "⌂ Home": "home",
        "▦ Games": "games",
        "✓ Tracker": "tracker",
        "◈ Matchup": "matchup",
        "◉ Team": "team",
        "↕ ELO": "elo",
        "◫ Analytics": "analytics",
        "•• More": "more",
    }
    try:
        st.query_params["page"] = slug_lookup.get(page, "home")
    except Exception:
        pass
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


def legacy_predicted_winner(row: pd.Series) -> str:
    return clean(legacy_display_value(row, "predicted_winner"))


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
            <div class="cfb-brand-sub">Production forecasts · live tracking · model intelligence</div>
        </div>
    </div>
    <div class="cfb-season-pill">2026 · V3 PRODUCTION CENTRE</div>
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
        f'<div class="hero-kicker">{html.escape(kicker)} · V3 PRODUCTION APP</div>'
        '<div class="hero-title">Your college football command centre.</div>'
        '<div class="hero-copy">Every forecast, official lock, projected score and model signal in one broadcast-style view — fast on mobile, deep when you need the analytics.</div>'
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

    section_title("Confidence distribution", "The full 2026 slate, grouped by model conviction")
    confidence_tiles(games)

    section_title("Top predictions", "This week's strongest and most relevant matchups")
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
    section_title("2026 games", "Search, filter and inspect every production forecast")

    weeks = sorted(int(x) for x in pd.to_numeric(games["week"], errors="coerce").dropna().unique().tolist())
    next_week = next_upcoming_week(games)
    week_options: list[Any] = ["All"] + weeks
    default_index = week_options.index(next_week) if next_week in week_options else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        week_choice = st.selectbox("Week", week_options, index=default_index, key="picks_week")
    with c2:
        conference = st.selectbox("Conference", ["All"] + conferences_for_games(games), key="picks_conf")
    with c3:
        confidence = st.selectbox("Confidence", ["All"] + CONFIDENCE_ORDER, key="picks_confidence")
    with c4:
        sort_by = st.selectbox("Sort", ["Kickoff", "Confidence", "Closest games"], key="picks_sort")

    search = st.text_input("Search teams", placeholder="Georgia, Ohio State, USC…")

    view = games.copy()
    if week_choice != "All":
        view = view[view["week"].eq(int(week_choice))]
    if conference != "All":
        view = view[
            view["home_conference"].astype(str).eq(conference)
            | view["away_conference"].astype(str).eq(conference)
        ]
    if confidence != "All" and "confidence_bucket" in view.columns:
        view = view[view["confidence_bucket"].astype(str).eq(confidence)]
    if search.strip():
        q = search.strip().lower()
        view = view[
            view["home_team"].astype(str).str.lower().str.contains(q, regex=False)
            | view["away_team"].astype(str).str.lower().str.contains(q, regex=False)
        ]

    if sort_by == "Confidence" and "model_favourite_probability" in view.columns:
        view = view.sort_values(["model_favourite_probability", "start_date_utc"], ascending=[False, True], kind="stable")
    elif sort_by == "Closest games" and "model_favourite_probability" in view.columns:
        view = view.assign(_closeness=(pd.to_numeric(view["model_favourite_probability"], errors="coerce") - .5).abs()).sort_values(["_closeness", "start_date_utc"], kind="stable")
    else:
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

    # Older V3 exports may predate the chart column. Recover it from the
    # individual P/L rows so the UI remains backwards-compatible.
    if "cumulative_profit_loss" not in view.columns:
        view["cumulative_profit_loss"] = pd.to_numeric(
            view.get("profit_loss", 0), errors="coerce"
        ).fillna(0.0).cumsum()

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
            "bet_number", "week", "away_team", "home_team", "selection", "bet_team", "market_line",
            "market_odds", "result", "profit_loss", "cumulative_profit_loss", "source_type"
        ] if c in view.columns]
        history = view[history_columns].rename(columns={
            "bet_number": "Bet", "week": "Week", "away_team": "Away", "home_team": "Home",
            "selection": "Selection", "bet_team": "Model Bet", "market_line": "Line / Odds", "market_odds": "Price",
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
# V3 PRODUCTION ANALYTICS + TRACKER
# ============================================================


CONFIDENCE_ORDER = ["Elite", "Very Strong", "Strong", "Moderate", "Lean", "Coin Flip"]


def first_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return the first available numeric column, aligned to frame.index."""
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def first_text(frame: pd.DataFrame, columns: list[str], default: str = "") -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype=str)


def completed_analysis(games: pd.DataFrame) -> pd.DataFrame:
    """Create one leakage-safe evaluation frame from completed exported games."""
    if games.empty:
        return pd.DataFrame()
    view = games.copy()
    view["_home_score"] = first_numeric(view, ["actual_home_points", "actual_home_score", "home_points", "final_home_score"])
    view["_away_score"] = first_numeric(view, ["actual_away_points", "actual_away_score", "away_points", "final_away_score"])
    view = view[view["_home_score"].notna() & view["_away_score"].notna()].copy()
    if view.empty:
        return view

    view["_home_won"] = (view["_home_score"] > view["_away_score"]).astype(float)
    view["_v2_home_prob"] = first_numeric(view, ["display_home_win_probability", "home_win_probability"])
    view["_v3_home_prob"] = first_numeric(view, ["v3_home_win_probability"])
    view["_v2_pick"] = np.where(view["_v2_home_prob"].ge(.5), view.get("home_team", ""), view.get("away_team", ""))
    view["_v3_pick"] = np.where(view["_v3_home_prob"].ge(.5), view.get("home_team", ""), view.get("away_team", ""))
    view["_actual_winner"] = np.where(view["_home_won"].eq(1), view.get("home_team", ""), view.get("away_team", ""))
    view["_v2_correct"] = view["_v2_pick"].eq(view["_actual_winner"])
    view["_v3_correct"] = view["_v3_pick"].eq(view["_actual_winner"])
    view["_v2_conf"] = np.maximum(view["_v2_home_prob"], 1 - view["_v2_home_prob"])
    view["_v3_conf"] = np.maximum(view["_v3_home_prob"], 1 - view["_v3_home_prob"])
    return view


def kpi_html(items: list[tuple[str, str, str, str]]) -> None:
    cards = []
    for label, value, note, tone in items:
        cards.append(
            f'<div class="kpi {html.escape(tone)}"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(value)}</div><div class="kpi-delta">{html.escape(note)}</div></div>'
        )
    st.markdown('<div class="kpi-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def confidence_tiles(games: pd.DataFrame) -> None:
    buckets = first_text(games, ["confidence_bucket"]).str.strip()
    counts = buckets.value_counts()
    tiles = "".join(
        f'<div class="confidence-tile"><div class="confidence-count">{int(counts.get(name, 0)):,}</div>'
        f'<div class="confidence-name">{html.escape(name)}</div></div>'
        for name in CONFIDENCE_ORDER
    )
    st.markdown(f'<div class="confidence-grid">{tiles}</div>', unsafe_allow_html=True)


def production_summary(games: pd.DataFrame, performance: dict[str, Any]) -> None:
    official = performance.get("official_locked", {}) if performance else {}
    reconstructed = performance.get("historical_reconstruction", {}) if performance else {}
    locks = int(games["is_locked"].apply(boolish).sum()) if "is_locked" in games.columns else 0
    next_week = next_upcoming_week(games)
    upcoming = games[games.get("start_date_utc", pd.Series(pd.NaT, index=games.index)).gt(pd.Timestamp.now(tz="UTC"))]
    slate = int(upcoming["week"].eq(next_week).sum()) if next_week is not None and "week" in upcoming.columns else len(upcoming)
    kpi_html([
        ("Official accuracy", pct(official.get("accuracy")), official.get("record") or "No settled locks", "good"),
        ("Official locks", f"{locks:,}", "Immutable forecasts", ""),
        ("Next slate", f"{slate:,}", f"Week {next_week}" if next_week is not None else "Upcoming", ""),
        ("Diagnostic Brier", num(reconstructed.get("brier"), 3), f'{reconstructed.get("games", 0)} reconstructed', ""),
    ])


def render_tracker(games: pd.DataFrame, rankings: pd.DataFrame) -> None:
    section_title("Prediction tracker", "Official locks, live results and auditable model history")
    if games.empty:
        st.info("No prediction rows are available.")
        return

    c1, c2, c3 = st.columns(3)
    weeks = sorted(int(x) for x in pd.to_numeric(games.get("week"), errors="coerce").dropna().unique())
    with c1:
        week = st.selectbox("Week", ["All"] + weeks, key="tracker_week")
    with c2:
        lifecycle = st.selectbox("Status", ["All", "Official locks", "Completed", "Upcoming"], key="tracker_status")
    with c3:
        model_view = st.selectbox("Model", ["V2 production", "V3 challenger", "Disagreements"], key="tracker_model")

    view = games.copy()
    if week != "All":
        view = view[pd.to_numeric(view["week"], errors="coerce").eq(int(week))]
    status = first_text(view, ["game_status"]).str.lower()
    if lifecycle == "Official locks" and "is_locked" in view.columns:
        view = view[view["is_locked"].apply(boolish)]
    elif lifecycle == "Completed":
        view = view[status.eq("completed")]
    elif lifecycle == "Upcoming":
        view = view[~status.eq("completed")]

    if model_view == "Disagreements":
        mask = view.apply(lambda row: v3_available(row) and v3_winner(row) != predicted_winner(row), axis=1)
        view = view[mask]

    view = view.sort_values([c for c in ["start_date_utc", "cfbd_game_id"] if c in view.columns], kind="stable")
    locked = int(view["is_locked"].apply(boolish).sum()) if "is_locked" in view.columns else 0
    complete = int(first_text(view, ["game_status"]).str.lower().eq("completed").sum())
    kpi_html([
        ("Visible", f"{len(view):,}", "Filtered forecasts", ""),
        ("Locked", f"{locked:,}", "Official predictions", "good"),
        ("Completed", f"{complete:,}", "Results attached", ""),
        ("V3 coverage", pct(view.get("v3_home_win_probability", pd.Series(np.nan, index=view.index)).notna().mean()), "Rows with challenger", ""),
    ])

    elo_lookup = build_elo_lookup(rankings)
    if view.empty:
        st.info("No tracker rows match those filters.")
        return
    for position, (_, row) in enumerate(view.head(100).iterrows()):
        compact_matchup_card(row, f"tracker_{position}", elo_lookup)


def dark_chart(chart):
    if alt is None:
        return chart
    return (
        chart.configure(background="transparent")
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor="#8fa4b8", titleColor="#8fa4b8", gridColor="#20364a",
            gridOpacity=.55, domainColor="#294359", tickColor="#294359",
        )
        .configure_legend(labelColor="#9db2c5", titleColor="#9db2c5")
        .configure_title(color="#f4f8fb")
    )


def render_performance_analytics(games: pd.DataFrame, performance: dict[str, Any]) -> None:
    production_summary(games, performance)
    section_title("Confidence distribution", "How the complete prediction slate is graded")
    confidence_tiles(games)

    evaluated = completed_analysis(games)
    if evaluated.empty:
        st.info("Performance charts will populate as completed scores enter the app export.")
        return

    scored = evaluated[evaluated["_v2_home_prob"].notna()].copy()
    if scored.empty:
        st.info("Completed games exist, but no V2 probability is available for evaluation.")
        return
    accuracy = float(scored["_v2_correct"].mean())
    brier = float(np.mean((scored["_v2_home_prob"] - scored["_home_won"]) ** 2))
    log_loss = float(-np.mean(scored["_home_won"] * np.log(scored["_v2_home_prob"].clip(.001,.999)) + (1-scored["_home_won"]) * np.log((1-scored["_v2_home_prob"]).clip(.001,.999))))
    kpi_html([
        ("Evaluated games", f"{len(scored):,}", "Completed app rows", ""),
        ("Winner accuracy", pct(accuracy), "V2 production picks", "good"),
        ("Brier score", f"{brier:.3f}", "Lower is better", ""),
        ("Log loss", f"{log_loss:.3f}", "Probability quality", ""),
    ])

    section_title("Accuracy by confidence", "Does greater confidence produce a better hit rate?")
    scored["Confidence"] = first_text(scored, ["confidence_bucket"]).replace("", "Unclassified")
    by_conf = scored.groupby("Confidence", observed=True).agg(Games=("_v2_correct","size"), Accuracy=("_v2_correct","mean")).reset_index()
    by_conf["sort"] = by_conf["Confidence"].map({name:i for i,name in enumerate(CONFIDENCE_ORDER)}).fillna(99)
    by_conf = by_conf.sort_values("sort")
    if alt is None:
        st.bar_chart(by_conf.set_index("Confidence")["Accuracy"])
    else:
        chart = alt.Chart(by_conf).mark_bar(cornerRadiusTopLeft=5,cornerRadiusTopRight=5,color="#39a7ff").encode(
            x=alt.X("Confidence:N", sort=CONFIDENCE_ORDER, title=None),
            y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0,1]), axis=alt.Axis(format="%"), title="Accuracy"),
            tooltip=["Confidence:N","Games:Q",alt.Tooltip("Accuracy:Q",format=".1%")],
        ).properties(height=300)
        st.altair_chart(dark_chart(chart), use_container_width=True)


def render_calibration(games: pd.DataFrame) -> None:
    evaluated = completed_analysis(games)
    scored = evaluated[evaluated.get("_v2_home_prob", pd.Series(dtype=float)).notna()].copy()
    section_title("Calibration", "Predicted probability compared with what actually happened")
    if len(scored) < 10:
        st.info("At least 10 completed probability forecasts are needed for a useful calibration view.")
        return
    scored["Probability band"] = pd.cut(scored["_v2_home_prob"], bins=np.linspace(0,1,11), include_lowest=True)
    cal = scored.groupby("Probability band", observed=True).agg(
        Forecast=("_v2_home_prob","mean"), Actual=("_home_won","mean"), Games=("_home_won","size")
    ).reset_index()
    cal["Gap"] = cal["Actual"] - cal["Forecast"]
    if alt is None:
        st.line_chart(cal.set_index("Forecast")[["Actual"]])
    else:
        diagonal = alt.Chart(pd.DataFrame({"x":[0,1],"y":[0,1]})).mark_line(color="#64798c",strokeDash=[5,5]).encode(x="x:Q",y="y:Q")
        line = alt.Chart(cal).mark_line(point=alt.OverlayMarkDef(size=80),color="#48e09b",strokeWidth=3).encode(
            x=alt.X("Forecast:Q",scale=alt.Scale(domain=[0,1]),axis=alt.Axis(format="%"),title="Mean forecast"),
            y=alt.Y("Actual:Q",scale=alt.Scale(domain=[0,1]),axis=alt.Axis(format="%"),title="Actual win rate"),
            tooltip=[alt.Tooltip("Forecast:Q",format=".1%"),alt.Tooltip("Actual:Q",format=".1%"),"Games:Q"],
        )
        st.altair_chart(dark_chart((diagonal+line).properties(height=360)),use_container_width=True)
    st.caption("Points above the diagonal won more often than forecast; points below it won less often.")


def render_model_comparison(games: pd.DataFrame) -> None:
    section_title("V2 vs V3", "Production control against the independent drive-model challenger")
    evaluated = completed_analysis(games)
    paired = evaluated[evaluated.get("_v2_home_prob", pd.Series(dtype=float)).notna() & evaluated.get("_v3_home_prob", pd.Series(dtype=float)).notna()].copy()
    if paired.empty:
        st.info("No completed games currently contain both V2 and V3 probabilities.")
        return
    v2_acc=float(paired["_v2_correct"].mean()); v3_acc=float(paired["_v3_correct"].mean())
    v2_brier=float(np.mean((paired["_v2_home_prob"]-paired["_home_won"])**2)); v3_brier=float(np.mean((paired["_v3_home_prob"]-paired["_home_won"])**2))
    kpi_html([
        ("Paired games",f"{len(paired):,}","Same evaluation universe",""),
        ("V2 accuracy",pct(v2_acc),f"Brier {v2_brier:.3f}",""),
        ("V3 accuracy",pct(v3_acc),f"Brier {v3_brier:.3f}","good" if v3_brier<v2_brier else ""),
        ("Brier improvement",f"{v2_brier-v3_brier:+.3f}","Positive favours V3","good" if v3_brier<v2_brier else "warn"),
    ])
    disagreements=paired[paired["_v2_pick"].ne(paired["_v3_pick"])].copy()
    section_title("Disagreement audit", "The games that actually separate the models")
    if disagreements.empty:
        st.caption("No completed paired games contain different winner picks.")
    else:
        cols=[c for c in ["week","away_team","home_team","_v2_pick","_v3_pick","_actual_winner","_v2_home_prob","_v3_home_prob"] if c in disagreements.columns]
        table=disagreements[cols].rename(columns={"week":"Week","away_team":"Away","home_team":"Home","_v2_pick":"V2 Pick","_v3_pick":"V3 Pick","_actual_winner":"Winner","_v2_home_prob":"V2 Home %","_v3_home_prob":"V3 Home %"})
        st.dataframe(table,use_container_width=True,hide_index=True,column_config={"V2 Home %":st.column_config.ProgressColumn(format="percent",min_value=0,max_value=1),"V3 Home %":st.column_config.ProgressColumn(format="percent",min_value=0,max_value=1)})


def render_team_strength(weekly: pd.DataFrame) -> None:
    section_title("Team strength", "FBS-only model leaderboard and weekly movement")
    if weekly.empty:
        st.info("No weekly ELO export is available.")
        return
    weeks=sorted(pd.to_numeric(weekly.get("week"),errors="coerce").dropna().astype(int).unique())
    selected=st.selectbox("Snapshot",weeks,index=len(weeks)-1,format_func=lambda x:"Preseason" if x==0 else f"Week {x}",key="analytics_strength_week")
    view=weekly[pd.to_numeric(weekly["week"],errors="coerce").eq(selected)].sort_values("rank").head(25)
    render_elo_rows(view)


def render_schedule_strength(profiles: pd.DataFrame) -> None:
    section_title("Schedule strength", "Opponent quality from the current production export")
    if profiles.empty or "schedule_strength_elo" not in profiles.columns:
        st.info("Run the app-data build after completed games to populate schedule-strength analytics.")
        return
    view=profiles.copy();view["schedule_strength_elo"]=pd.to_numeric(view["schedule_strength_elo"],errors="coerce")
    cols=[c for c in ["team","conference","record","schedule_strength_elo","schedule_strength_label","current_elo","model_rank"] if c in view.columns]
    view=view[cols].dropna(subset=["schedule_strength_elo"]).sort_values("schedule_strength_elo",ascending=False)
    confs=["All"]+sorted(view.get("conference",pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    conf=st.selectbox("Conference",confs,key="sos_conf")
    if conf!="All" and "conference" in view.columns:view=view[view["conference"].astype(str).eq(conf)]
    st.dataframe(view.head(50),use_container_width=True,hide_index=True,column_config={"schedule_strength_elo":st.column_config.NumberColumn("Opp. avg ELO",format="%.0f"),"current_elo":st.column_config.NumberColumn("Team ELO",format="%.0f")})


def render_trends(games: pd.DataFrame) -> None:
    section_title("Trends & insights", "Rolling winner accuracy and the strongest current signals")
    evaluated=completed_analysis(games)
    scored=evaluated[evaluated.get("_v2_home_prob",pd.Series(dtype=float)).notna()].copy()
    if scored.empty:
        st.info("Trend analytics will populate once completed predictions are available.")
        return
    scored=scored.sort_values([c for c in ["start_date_utc","cfbd_game_id"] if c in scored.columns],kind="stable")
    scored["Game"]=np.arange(1,len(scored)+1);scored["Rolling accuracy"]=scored["_v2_correct"].rolling(25,min_periods=5).mean()
    if alt is None:st.line_chart(scored.set_index("Game")[["Rolling accuracy"]])
    else:
        chart=alt.Chart(scored.dropna(subset=["Rolling accuracy"])).mark_line(color="#a880ff",strokeWidth=3).encode(x=alt.X("Game:Q",title="Completed game"),y=alt.Y("Rolling accuracy:Q",axis=alt.Axis(format="%"),scale=alt.Scale(domain=[0,1])),tooltip=["Game:Q",alt.Tooltip("Rolling accuracy:Q",format=".1%")]).properties(height=320)
        st.altair_chart(dark_chart(chart),use_container_width=True)
    high=scored[scored["_v2_conf"].ge(.80)]
    st.markdown('<div class="insight-card"><div class="insight-kicker">Current read</div><div class="insight-title">High-confidence winner performance</div><div class="insight-copy">'+html.escape(f'{int(high["_v2_correct"].sum())} correct from {len(high)} completed picks ({pct(high["_v2_correct"].mean()) if len(high) else "—"}). This view is descriptive and does not alter the model.')+'</div></div>',unsafe_allow_html=True)


def render_data_explorer(games: pd.DataFrame) -> None:
    section_title("Data explorer", "Filter the canonical app export without changing it")
    view=games.copy()
    c1,c2=st.columns(2)
    weeks=sorted(pd.to_numeric(view.get("week"),errors="coerce").dropna().astype(int).unique())
    with c1:week=st.selectbox("Week",["All"]+weeks,key="explorer_week")
    with c2:conference=st.selectbox("Conference",["All"]+conferences_for_games(view),key="explorer_conf")
    if week!="All":view=view[pd.to_numeric(view["week"],errors="coerce").eq(int(week))]
    if conference!="All":view=view[first_text(view,["home_conference"]).eq(conference)|first_text(view,["away_conference"]).eq(conference)]
    search=st.text_input("Team search",key="explorer_search",placeholder="Search either side…").strip().lower()
    if search:view=view[first_text(view,["home_team"]).str.lower().str.contains(search,regex=False)|first_text(view,["away_team"]).str.lower().str.contains(search,regex=False)]
    preferred=["cfbd_game_id","week","start_date_utc","away_team","home_team","predicted_winner","confidence_bucket","away_win_probability","home_win_probability","projected_away_score","projected_home_score","v3_predicted_winner","v3_away_win_probability","v3_home_win_probability","market_formatted_spread","game_status"]
    columns=[c for c in preferred if c in view.columns]
    st.dataframe(view[columns].head(1000),use_container_width=True,hide_index=True)
    st.download_button("Download filtered CSV",view.to_csv(index=False).encode("utf-8"),"cfb_v3_filtered_predictions.csv","text/csv",use_container_width=True)


def render_backtest(games: pd.DataFrame) -> None:
    section_title("Back testing", "Explore confidence thresholds; diagnostic results never become official")
    evaluated=completed_analysis(games); scored=evaluated[evaluated.get("_v2_conf",pd.Series(dtype=float)).notna()].copy()
    if scored.empty:
        st.info("No completed probability forecasts are available for threshold testing.")
        return
    threshold=st.slider("Minimum model confidence",.50,.95,.70,.01,format="%.0f%%")
    sample=scored[scored["_v2_conf"].ge(threshold)]
    acc=float(sample["_v2_correct"].mean()) if len(sample) else np.nan
    brier=float(np.mean((sample["_v2_home_prob"]-sample["_home_won"])**2)) if len(sample) else np.nan
    coverage=len(sample)/len(scored) if len(scored) else np.nan
    kpi_html([("Selections",f"{len(sample):,}",f"From {len(scored):,} games",""),("Accuracy",pct(acc),"Winner picks","good"),("Coverage",pct(coverage),"Share of completed slate",""),("Brier",num(brier,3),"Probability quality","")])
    st.warning("This is an exploratory historical filter, not a live betting recommendation. Market data remains outside the football model.")


def render_analytics(games: pd.DataFrame, weekly: pd.DataFrame, performance: dict[str, Any], profiles: pd.DataFrame) -> None:
    st.markdown('<div class="hero"><div class="hero-kicker">V3 ANALYTICS LAB</div><div class="hero-title">Know why the model is winning.</div><div class="hero-copy">Production monitoring, probability calibration, challenger comparison and team-strength diagnostics—kept separate from the prediction engine.</div></div>',unsafe_allow_html=True)
    choices=["Performance","Calibration","V2 vs V3","Team Strength","Schedule","Trends","Explorer","Back Test"]
    choice=st.segmented_control("Analytics view",options=choices,default="Performance",selection_mode="single",key="analytics_section") or "Performance"
    if choice=="Performance":render_performance_analytics(games,performance)
    elif choice=="Calibration":render_calibration(games)
    elif choice=="V2 vs V3":render_model_comparison(games)
    elif choice=="Team Strength":render_team_strength(weekly)
    elif choice=="Schedule":render_schedule_strength(profiles)
    elif choice=="Trends":render_trends(games)
    elif choice=="Explorer":render_data_explorer(games)
    else:render_backtest(games)


# ============================================================
# APPROVED V3 PRODUCT PAGES
# ============================================================


def page_heading(eyebrow: str, title: str, subtitle: str, live: str = "") -> None:
    live_html = f'<div class="live-dot">{html.escape(live)}</div>' if live else ""
    st.markdown(
        '<div class="page-heading"><div>'
        f'<div class="page-eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1 class="page-title">{html.escape(title)}</h1>'
        f'<div class="page-subtitle">{html.escape(subtitle)}</div>'
        f'</div>{live_html}</div>',
        unsafe_allow_html=True,
    )


def normalized_colour(value: Any, fallback: str = "#1d8fd1") -> str:
    colour = clean(value).strip()
    if not colour:
        return fallback
    if not colour.startswith("#"):
        colour = f"#{colour}"
    if len(colour) not in {4, 7} or any(c not in "#0123456789abcdefABCDEF" for c in colour):
        return fallback
    return colour


def team_colour(row: pd.Series, side: str, fallback: str) -> str:
    for column in [f"{side}_color", f"{side}_primary_color", f"{side}_colour", f"{side}_primary_colour"]:
        if column in row.index and clean(row.get(column)):
            return normalized_colour(row.get(column), fallback)
    known = {
        "Georgia": "#d71920", "Alabama": "#9e1b32", "Ohio State": "#ba0c2f",
        "Oregon": "#f4dc00", "Texas": "#bf5700", "USC": "#990000",
        "Michigan": "#ffcb05", "Penn State": "#1e407c", "Clemson": "#f56600",
        "Florida State": "#782f40", "LSU": "#461d7c", "Notre Dame": "#c99700",
    }
    return known.get(clean(row.get(f"{side}_team", "")), fallback)


def logo_html(url: Any, css_class: str, alt_text: str) -> str:
    value = clean(url).strip()
    if not value.startswith("http"):
        return '<div style="height:52px"></div>'
    return f'<img class="{css_class}" src="{html.escape(value, quote=True)}" alt="{html.escape(alt_text, quote=True)}">'


def row_game_id(row: pd.Series) -> int | None:
    try:
        value = int(float(row.get("cfbd_game_id")))
        return value
    except Exception:
        return None


def game_confidence(row: pd.Series, model: str = "v2") -> float:
    if model == "v3" and v3_available(row):
        try:
            hp = float(row.get("v3_home_win_probability"))
            return max(hp, 1 - hp)
        except Exception:
            return np.nan
    value = model_probability(row)
    if np.isfinite(value):
        return value
    try:
        hp = float(display_value(row, "home_win_probability"))
        return max(hp, 1 - hp)
    except Exception:
        return np.nan


def game_week_options(games: pd.DataFrame) -> list[Any]:
    if games.empty or "week" not in games.columns:
        return ["All"]
    weeks = sorted(pd.to_numeric(games["week"], errors="coerce").dropna().astype(int).unique().tolist())
    return ["All"] + weeks


def status_mask(frame: pd.DataFrame, status_choice: str) -> pd.Series:
    status = first_text(frame, ["game_status"]).str.lower()
    completed = status.eq("completed")
    if status_choice == "Completed":
        return completed
    if status_choice == "Upcoming":
        return ~completed
    if status_choice == "Official locks" and "is_locked" in frame.columns:
        return frame["is_locked"].apply(boolish)
    return pd.Series(True, index=frame.index)


def v3_signal(row: pd.Series) -> tuple[str, str]:
    if not v3_available(row):
        return "V3 pending", ""
    v2 = predicted_winner(row)
    v3 = v3_winner(row)
    if v2 and v3 and v2 == v3:
        return "V2 / V3 agree", "strong"
    if v2 and v3:
        return "V2 / V3 split", "split"
    return "V3 available", ""


def broadcast_matchup_card(row: pd.Series, key: str, show_actions: bool = True) -> None:
    game_id = row_game_id(row)
    away = clean(row.get("away_team")) or "Away"
    home = clean(row.get("home_team")) or "Home"
    away_prob = display_value(row, "away_win_probability")
    home_prob = display_value(row, "home_win_probability")
    away_score = display_value(row, "projected_away_score")
    home_score = display_value(row, "projected_home_score")
    status = clean(row.get("game_status")).lower()
    score_label = "Projected"
    if status == "completed":
        actual_away, actual_home = get_actual_score(row, "away"), get_actual_score(row, "home")
        if pd.notna(actual_away) and pd.notna(actual_home):
            away_score, home_score, score_label = actual_away, actual_home, "Final"

    try:
        away_width = max(0.0, min(100.0, float(away_prob) * 100))
        home_width = max(0.0, min(100.0, float(home_prob) * 100))
    except Exception:
        away_width = home_width = 50.0
    confidence = clean(row.get("confidence_bucket")) or "Unclassified"
    signal, signal_class = v3_signal(row)
    st.markdown(
        '<div class="broadcast-card">'
        f'<div class="broadcast-meta"><span>{html.escape(kickoff_text(row.get("start_date_utc")))}</span>'
        f'<span class="status-chip {chip_class(row)}">{html.escape(lock_text(row))}</span></div>'
        '<div class="broadcast-teams">'
        '<div class="broadcast-team">'
        f'{logo_html(row.get("away_logo_url"), "broadcast-logo", away)}'
        f'<div><div class="broadcast-name">{html.escape(away)}</div><div class="broadcast-record">{html.escape(clean(row.get("away_conference")))} · {pct(away_prob)}</div></div></div>'
        f'<div class="broadcast-score"><span>{score_label}</span><strong>{num(away_score,0)}–{num(home_score,0)}</strong></div>'
        '<div class="broadcast-team right">'
        f'{logo_html(row.get("home_logo_url"), "broadcast-logo", home)}'
        f'<div><div class="broadcast-name">{html.escape(home)}</div><div class="broadcast-record">{html.escape(clean(row.get("home_conference")))} · {pct(home_prob)}</div></div></div>'
        '</div>'
        f'<div class="probability-line"><div class="away" style="width:{away_width:.1f}%"></div><div class="home" style="width:{home_width:.1f}%"></div></div>'
        '<div class="broadcast-footer"><div>'
        f'<span class="badge strong">{html.escape(confidence)}</span> '
        f'<span class="badge {signal_class}">{html.escape(signal)}</span></div>'
        f'<span class="badge">{html.escape(market_spread_text(row))}</span></div></div>',
        unsafe_allow_html=True,
    )
    if not show_actions or game_id is None:
        return
    a, b, c = st.columns([1, 1, 1.25])
    with a:
        if st.button(f"{away} →", key=f"{key}_away", use_container_width=True):
            request_team(away)
    with b:
        if st.button(f"{home} →", key=f"{key}_home", use_container_width=True):
            request_team(home)
    with c:
        if st.button("Open matchup →", key=f"{key}_game", type="primary", use_container_width=True):
            request_game(game_id)


def app_counts(games: pd.DataFrame, performance: dict[str, Any]) -> dict[str, Any]:
    next_week = next_upcoming_week(games)
    if next_week is not None and "week" in games.columns:
        slate = games[pd.to_numeric(games["week"], errors="coerce").eq(next_week)].copy()
    else:
        slate = games.copy()
    buckets = first_text(slate, ["confidence_bucket"])
    elite = int(buckets.eq("Elite").sum())
    upsets = 0
    for _, row in slate.iterrows():
        if upset_details(row) is not None:
            upsets += 1
    official = performance.get("official_locked", {}) if performance else {}
    accuracy = official.get("accuracy")
    health = 100 if not games.empty else 0
    if games.empty:
        health = 0
    elif "v3_home_win_probability" in games.columns:
        coverage = float(pd.to_numeric(games["v3_home_win_probability"], errors="coerce").notna().mean())
        health = int(round(85 + 15 * coverage))
    return {"week": next_week, "games": len(slate), "elite": elite, "upsets": upsets, "health": health, "accuracy": accuracy}


def render_landing_v3(games: pd.DataFrame, rankings: pd.DataFrame, performance: dict[str, Any]) -> None:
    counts = app_counts(games, performance)
    week_label = f"WEEK {counts['week']}" if counts["week"] is not None else "2026 SEASON"
    st.markdown(
        '<div class="command-hero">'
        '<div class="page-eyebrow">COLLEGE FOOTBALL · LIVE MODEL</div>'
        f'<div class="command-title">{html.escape(week_label)} COMMAND CENTRE</div>'
        '<div class="command-copy">The fastest route from the complete Saturday slate to the games, teams and model signals that matter. Every official forecast remains timestamped and auditable.</div>'
        '<div class="micro-row"><span class="micro-pill good">V2 PRODUCTION ONLINE</span><span class="micro-pill">V3 DRIVE MODEL ONLINE</span><span class="micro-pill warn">MARKET: DISPLAY ONLY</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    now = pd.Timestamp.now(tz="UTC")
    slate = games.copy()
    if counts["week"] is not None and "week" in slate.columns:
        slate = slate[pd.to_numeric(slate["week"], errors="coerce").eq(int(counts["week"]))]
    upcoming = slate[slate.get("start_date_utc", pd.Series(pd.NaT, index=slate.index)).ge(now - pd.Timedelta(hours=6))]
    candidates = upcoming if not upcoming.empty else slate
    if not candidates.empty:
        confidence = candidates.apply(game_confidence, axis=1)
        feature = candidates.loc[confidence.fillna(-1).idxmax()]
        st.markdown('<div class="featured-label">★ FEATURED MATCHUP</div>', unsafe_allow_html=True)
        broadcast_matchup_card(feature, "home_feature")

    st.markdown(
        '<div class="section-grid">'
        f'<div class="signal-card cyan"><div class="label">Games</div><div class="value">{counts["games"]:,}</div><div class="note">Current slate</div></div>'
        f'<div class="signal-card green"><div class="label">Elite picks</div><div class="value">{counts["elite"]:,}</div><div class="note">90%+ confidence</div></div>'
        f'<div class="signal-card amber"><div class="label">Upset alerts</div><div class="value">{counts["upsets"]:,}</div><div class="note">Lower-ELO threat</div></div>'
        f'<div class="signal-card purple"><div class="label">Model health</div><div class="value">{counts["health"]}%</div><div class="note">Export coverage</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.5, 1])
    with left:
        section_title("Top predictions", "Highest-confidence games on the active slate")
        top = candidates.assign(_confidence=candidates.apply(game_confidence, axis=1)).sort_values("_confidence", ascending=False).head(4) if not candidates.empty else candidates
        if top.empty:
            st.markdown('<div class="empty-state">No games are available for the current slate.</div>', unsafe_allow_html=True)
        else:
            for i, (_, row) in enumerate(top.iterrows()):
                game_id = row_game_id(row)
                winner = predicted_winner(row) or v3_winner(row) or "—"
                a, b, c = st.columns([2.2, 1, .7], vertical_alignment="center")
                with a:
                    st.markdown(f'**{clean(row.get("away_team"))} @ {clean(row.get("home_team"))}**  \n<span style="color:#7893a8;font-size:.68rem">{kickoff_text(row.get("start_date_utc"))}</span>', unsafe_allow_html=True)
                with b:
                    st.markdown(f'<span class="badge strong">{html.escape(winner)} · {pct(game_confidence(row))}</span>', unsafe_allow_html=True)
                with c:
                    if game_id is not None and st.button("Open", key=f"home_top_{i}", use_container_width=True):
                        request_game(game_id)
    with right:
        section_title("ELO movers", "Largest current weekly changes")
        if rankings.empty:
            st.caption("No current ranking export.")
        else:
            movers = rankings.assign(_move=pd.to_numeric(rankings.get("elo_change"), errors="coerce")).dropna(subset=["_move"]).sort_values("_move", key=lambda s:s.abs(), ascending=False).head(6)
            for i, (_, row) in enumerate(movers.iterrows()):
                delta = float(row["_move"])
                c1, c2, c3 = st.columns([2.2, .8, .7], vertical_alignment="center")
                with c1: st.markdown(f'**{clean(row.get("team"))}**  \n<span style="color:#6f8a9f;font-size:.58rem">{clean(row.get("conference"))}</span>', unsafe_allow_html=True)
                with c2: st.markdown(f'<span style="color:{"#48e09b" if delta >= 0 else "#ff6577"};font-weight:900">{delta:+.0f}</span>', unsafe_allow_html=True)
                with c3:
                    if st.button("→", key=f"home_mover_{i}", use_container_width=True): request_team(clean(row.get("team")))

    section_title("Confidence distribution", "How the current app slate is graded")
    confidence_tiles(slate if not slate.empty else games)


def render_games_v3(games: pd.DataFrame, rankings: pd.DataFrame) -> None:
    page_heading("PREDICTIONS", "2026 Games", "Every matchup. One intelligent view.", "Live slate")
    options = game_week_options(games)
    default_week = next_upcoming_week(games)
    default_index = options.index(default_week) if default_week in options else 0
    f1, f2, f3, f4 = st.columns(4)
    with f1: week = st.selectbox("Week", options, index=default_index, key="games_week_v3")
    with f2: conference = st.selectbox("Conference", ["All"] + conferences_for_games(games), key="games_conf_v3")
    with f3: confidence = st.selectbox("Confidence", ["All"] + CONFIDENCE_ORDER, key="games_confidence_v3")
    with f4: status = st.selectbox("Status", ["All", "Upcoming", "Completed", "Official locks"], key="games_status_v3")
    q1, q2 = st.columns([2.3, 1])
    with q1: query = st.text_input("Search teams", placeholder="Georgia, Ohio State, Oregon…", key="games_search_v3").strip().lower()
    with q2: view_mode = st.segmented_control("View", ["Cards", "Compact"], default="Cards", key="games_view_v3") or "Cards"

    view = games.copy()
    if week != "All": view = view[pd.to_numeric(view["week"], errors="coerce").eq(int(week))]
    if conference != "All": view = view[first_text(view,["home_conference"]).eq(conference) | first_text(view,["away_conference"]).eq(conference)]
    if confidence != "All": view = view[first_text(view,["confidence_bucket"]).eq(confidence)]
    view = view[status_mask(view, status)]
    if query: view = view[first_text(view,["home_team"]).str.lower().str.contains(query,regex=False) | first_text(view,["away_team"]).str.lower().str.contains(query,regex=False)]
    view = view.assign(_confidence=view.apply(game_confidence, axis=1))
    if "start_date_utc" in view.columns:
        view = view.sort_values(["start_date_utc", "_confidence"], ascending=[True, False], kind="stable")
    else:
        view = view.sort_values("_confidence", ascending=False, kind="stable")
    st.markdown(f'<div class="micro-row"><span class="micro-pill">{len(view):,} MATCHUPS</span><span class="micro-pill good">{int(first_text(view,["confidence_bucket"]).eq("Elite").sum())} ELITE</span><span class="micro-pill warn">{int(view.apply(lambda r: upset_details(r) is not None,axis=1).sum()) if len(view) else 0} UPSET ALERTS</span></div>', unsafe_allow_html=True)
    if view.empty:
        st.markdown('<div class="empty-state">No matchups fit those filters.</div>', unsafe_allow_html=True)
        return
    if view_mode == "Compact":
        compact = pd.DataFrame({
            "Week": view.get("week"), "Away": first_text(view,["away_team"]), "Home": first_text(view,["home_team"]),
            "Model Pick": view.apply(predicted_winner,axis=1), "Confidence": view["_confidence"],
            "V3 Pick": view.apply(v3_winner,axis=1), "Kickoff": view.get("start_date_utc"), "Status": first_text(view,["game_status"]),
        })
        st.dataframe(compact,use_container_width=True,hide_index=True,column_config={"Confidence":st.column_config.ProgressColumn(format="percent",min_value=0,max_value=1),"Kickoff":st.column_config.DatetimeColumn(format="ddd DD MMM, HH:mm")})
        selected = st.selectbox("Open matchup", view.index.tolist(), format_func=lambda idx:f'{view.loc[idx,"away_team"]} @ {view.loc[idx,"home_team"]}', key="compact_open_game")
        if st.button("Open selected matchup →", use_container_width=True):
            gid=row_game_id(view.loc[selected]);
            if gid is not None: request_game(gid)
        return
    rows = list(view.head(100).iterrows())
    for offset in range(0, len(rows), 2):
        cols = st.columns(2)
        for slot, item in enumerate(rows[offset:offset+2]):
            _, row = item
            with cols[slot]: broadcast_matchup_card(row, f"games_{offset+slot}")


def comparison_bar(label: str, away_value: Any, home_value: Any, away_name: str, home_name: str, percent: bool = False) -> None:
    av = pd.to_numeric(pd.Series([away_value]), errors="coerce").iloc[0]
    hv = pd.to_numeric(pd.Series([home_value]), errors="coerce").iloc[0]
    if pd.isna(av) and pd.isna(hv):
        factor_row(label, "—")
        return
    if pd.isna(av): av = 0.0
    if pd.isna(hv): hv = 0.0
    denom = abs(float(av)) + abs(float(hv))
    width = 50.0 if denom == 0 else max(5.0,min(95.0,abs(float(av))/denom*100))
    left = pct(av) if percent else num(av,2)
    right = pct(hv) if percent else num(hv,2)
    st.markdown(
        f'<div style="padding:9px 0;border-bottom:1px solid rgba(101,139,168,.13)"><div style="display:flex;justify-content:space-between;gap:8px;font-size:.58rem;color:#8ca4b7"><span>{html.escape(away_name)} <b style="color:#fff">{left}</b></span><span style="color:#6f899e">{html.escape(label)}</span><span><b style="color:#fff">{right}</b> {html.escape(home_name)}</span></div><div class="probability-line"><div class="away" style="width:{width:.1f}%"></div><div class="home" style="width:{100-width:.1f}%"></div></div></div>',
        unsafe_allow_html=True,
    )


def render_matchup_v3(games: pd.DataFrame, weekly: pd.DataFrame, rankings: pd.DataFrame) -> None:
    page_heading("GAME INTELLIGENCE", "Matchup Centre", "Two teams, two independent models and one clear forecast.")
    row = choose_game(games)
    away, home = clean(row.get("away_team")), clean(row.get("home_team"))
    away_prob, home_prob = display_value(row,"away_win_probability"), display_value(row,"home_win_probability")
    away_score, home_score = display_value(row,"projected_away_score"), display_value(row,"projected_home_score")
    st.markdown(
        '<div class="matchup-stage"><div class="broadcast-meta">'
        f'<span>{html.escape(kickoff_text(row.get("start_date_utc")))}</span><span class="status-chip {chip_class(row)}">{html.escape(lock_text(row))}</span></div><div class="matchup-grid">'
        f'<div class="matchup-team">{logo_html(row.get("away_logo_url"),"matchup-logo",away)}<div class="matchup-name">{html.escape(away)}</div><div class="matchup-meta">{html.escape(clean(row.get("away_conference")))} · ELO {num(row.get("away_pregame_elo"),0)}</div></div>'
        f'<div class="matchup-centre"><div class="stage-score-label">Projected score</div><div class="matchup-score">{num(away_score,0)}–{num(home_score,0)}</div><div class="matchup-probs"><span class="left">{pct(away_prob)}</span><span class="right">{pct(home_prob)}</span></div><div class="matchup-confidence">{html.escape(clean(row.get("confidence_bucket")) or "MODEL FORECAST")}</div></div>'
        f'<div class="matchup-team">{logo_html(row.get("home_logo_url"),"matchup-logo",home)}<div class="matchup-name">{html.escape(home)}</div><div class="matchup-meta">{html.escape(clean(row.get("home_conference")))} · ELO {num(row.get("home_pregame_elo"),0)}</div></div>'
        '</div></div>', unsafe_allow_html=True)
    b1,b2=st.columns(2)
    with b1:
        if st.button(f"View {away} team →",key="matchup_away_team",use_container_width=True):request_team(away)
    with b2:
        if st.button(f"View {home} team →",key="matchup_home_team",use_container_width=True):request_team(home)

    overview, offense, defense, drives, simulation, market = st.tabs(["Overview","Offense","Defense","Drives","Simulation","Market"])
    with overview:
        left,right=st.columns([1.2,1])
        with left:
            section_title(f"Why the model leans {predicted_winner(row) or 'this way'}","Pregame matchup advantages only")
            with st.container(border=True):
                comparison_bar("Pregame ELO",row.get("away_pregame_elo"),row.get("home_pregame_elo"),away,home)
                comparison_bar("Offensive matchup",row.get("away_offensive_matchup"),row.get("home_offensive_matchup"),away,home)
                comparison_bar("Success rate",row.get("matchup_away_success_rate"),row.get("matchup_home_success_rate"),away,home,True)
                comparison_bar("Explosive rate",row.get("matchup_away_explosive_rate"),row.get("matchup_home_explosive_rate"),away,home,True)
                comparison_bar("PPA",row.get("matchup_away_ppa"),row.get("matchup_home_ppa"),away,home)
        with right:
            section_title("Independent model view","Agreement is useful; disagreement is information")
            m1,m2=st.columns(2)
            with m1:
                st.markdown(f'<div class="model-card v2"><div class="model-card-title">V2 PRODUCTION</div><div class="model-card-score">{num(away_score,0)}–{num(home_score,0)}</div><div class="model-card-note">{html.escape(predicted_winner(row) or "—")} · {pct(game_confidence(row))}</div><span class="agreement">CONTROL</span></div>',unsafe_allow_html=True)
            with m2:
                signal,_=v3_signal(row)
                st.markdown(f'<div class="model-card v3"><div class="model-card-title">V3 DRIVE MODEL</div><div class="model-card-score">{num(row.get("v3_projected_away_points"),0)}–{num(row.get("v3_projected_home_points"),0)}</div><div class="model-card-note">{html.escape(v3_winner(row) or "Awaiting forecast")} · {pct(game_confidence(row,"v3"))}</div><span class="agreement">{html.escape(signal.upper())}</span></div>',unsafe_allow_html=True)
            section_title("Market context","Reference only — never a predictive input")
            metric_card("Spread",market_spread_text(row),market_moneyline_text(row))
    with offense:
        section_title("Offensive matchup","Opponent-adjusted pregame state")
        a,b=st.columns(2)
        with a:
            with st.container(border=True):
                st.markdown(f"#### {away}")
                factor_row("Points per drive",num(row.get("matchup_away_ppd"),2));factor_row("Success rate",pct(row.get("matchup_away_success_rate")));factor_row("Explosive rate",pct(row.get("matchup_away_explosive_rate")));factor_row("PPA",num(row.get("matchup_away_ppa"),3));factor_row("Matchup ratio",num(row.get("matchup_away_matchup_ratio"),3))
        with b:
            with st.container(border=True):
                st.markdown(f"#### {home}")
                factor_row("Points per drive",num(row.get("matchup_home_ppd"),2));factor_row("Success rate",pct(row.get("matchup_home_success_rate")));factor_row("Explosive rate",pct(row.get("matchup_home_explosive_rate")));factor_row("PPA",num(row.get("matchup_home_ppa"),3));factor_row("Matchup ratio",num(row.get("matchup_home_matchup_ratio"),3))
    with defense:
        section_title("Defensive pressure","Stops, disruption and scoring prevention")
        with st.container(border=True):
            comparison_bar("Defensive rating",row.get("away_defensive_rating"),row.get("home_defensive_rating"),away,home)
            comparison_bar("Havoc rate",row.get("away_havoc_rate"),row.get("home_havoc_rate"),away,home,True)
            comparison_bar("Opponent success",row.get("away_def_success_rate"),row.get("home_def_success_rate"),away,home,True)
            comparison_bar("Opponent PPA",row.get("away_def_ppa"),row.get("home_def_ppa"),away,home)
            comparison_bar("Turnover rate",row.get("away_turnover_rate"),row.get("home_turnover_rate"),away,home,True)
    with drives:
        section_title("Drive environment","Possession count, field position and expected production")
        kpi_html([("Expected drives",num(row.get("expected_total_drives"),1),"Combined V2 environment",""),(f"{away} drives",num(row.get("sim_expected_away_drives"),1),"V3 simulation",""),(f"{home} drives",num(row.get("sim_expected_home_drives"),1),"V3 simulation",""),("Expected plays",num(row.get("expected_total_plays"),1),"Combined pace","")])
        with st.container(border=True):
            factor_row("Away opponent-adjusted PPD",num(row.get("matchup_away_ppd"),2));factor_row("Home opponent-adjusted PPD",num(row.get("matchup_home_ppd"),2));factor_row("V3 ELO difference",f'{num(row.get("matchup_elo_difference"),1)} · home minus away');factor_row("V3 rating difference",f'{num(row.get("matchup_rating_difference"),1)} · home minus away')
    with simulation:
        section_title("Monte Carlo simulation","Projected distribution, not just a single score")
        kpi_html([("Away score band",f'{num(row.get("simulation_away_score_p10"),0)}–{num(row.get("simulation_away_score_p90"),0)}',away,"purple"),("Home score band",f'{num(row.get("simulation_home_score_p10"),0)}–{num(row.get("simulation_home_score_p90"),0)}',home,"cyan"),("Margin band",f'{num(row.get("sim_margin_p10"),0)} to {num(row.get("sim_margin_p90"),0)}',"10th–90th percentile",""),("Total band",f'{num(row.get("sim_total_p10"),0)}–{num(row.get("sim_total_p90"),0)}',"10th–90th percentile","")])
        with st.container(border=True):
            factor_row(f"{away} win probability",pct(row.get("v3_away_win_probability",away_prob)));factor_row(f"{home} win probability",pct(row.get("v3_home_win_probability",home_prob)));factor_row("Projected margin",v3_margin_text(row));factor_row("Projected total",num(row.get("v3_projected_total"),1));factor_row("Simulation runs",f'{num(row.get("simulation_runs",10000),0)}')
    with market:
        st.warning("Market information is isolated for display and evaluation. It is never passed into the football prediction model.")
        m1,m2,m3=st.columns(3)
        with m1:metric_card("Spread",market_spread_text(row),clean(row.get("market_provider")) or "Captured line")
        with m2:metric_card("Moneyline",market_moneyline_text(row),"Captured price")
        with m3:metric_card("Total",num(row.get("market_over_under"),1),"Display only")


def selected_team_profile(team: str, weekly: pd.DataFrame, profiles: pd.DataFrame) -> tuple[pd.Series,pd.DataFrame]:
    trend = weekly[first_text(weekly,["team"]).eq(team)].sort_values("week",kind="stable") if not weekly.empty else pd.DataFrame()
    profile = pd.Series(dtype=object)
    if not profiles.empty and "team" in profiles.columns:
        match = profiles[first_text(profiles,["team"]).eq(team)]
        if not match.empty: profile = match.iloc[0]
    return profile, trend


def team_universe(games: pd.DataFrame, weekly: pd.DataFrame, profiles: pd.DataFrame) -> list[str]:
    teams:set[str]=set()
    for frame,column in [(games,"home_team"),(games,"away_team"),(weekly,"team"),(profiles,"team")]:
        if not frame.empty and column in frame.columns:teams.update(frame[column].dropna().astype(str).str.strip().tolist())
    teams.discard("")
    return sorted(teams)


def team_game_rows(team: str, games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:return pd.DataFrame()
    return games[first_text(games,["home_team"]).eq(team)|first_text(games,["away_team"]).eq(team)].sort_values("start_date_utc",kind="stable")


def live_team_record(team: str, games: pd.DataFrame) -> str:
    """Calculate the displayed record from refreshed canonical final scores."""
    rows = team_game_rows(team, games); wins = losses = 0
    for _, row in rows.iterrows():
        away, home = get_actual_score(row, "away"), get_actual_score(row, "home")
        if pd.isna(away) or pd.isna(home): continue
        is_home = clean(row.get("home_team")) == team
        own, opp = (home, away) if is_home else (away, home)
        wins += int(float(own) > float(opp)); losses += int(float(own) < float(opp))
    return f"{wins}-{losses}"


def render_team_v3(games: pd.DataFrame, weekly: pd.DataFrame, profiles: pd.DataFrame, rankings: pd.DataFrame) -> None:
    teams=team_universe(games,weekly,profiles)
    if not teams:
        st.info("No team data is available.");return
    requested=st.session_state.get("selected_team")
    default=teams.index(requested) if requested in teams else 0
    select_col, compare_col=st.columns([2,1])
    with select_col:selected=st.selectbox("Team",teams,index=default,key="team_page_selector")
    with compare_col:compare=st.selectbox("Compare with",["None"]+[t for t in teams if t!=selected],key="team_compare_selector")
    st.session_state["selected_team"]=selected
    try:st.query_params["team"]=selected
    except Exception:pass
    profile,trend=selected_team_profile(selected,weekly,profiles)
    latest=trend.iloc[-1] if not trend.empty else pd.Series(dtype=object)
    team_games=team_game_rows(selected,games)
    logo=clean(profile.get("logo_url",latest.get("logo_url","")))
    if not logo and not team_games.empty:
        candidate=team_games.iloc[0];logo=clean(candidate.get("home_logo_url" if clean(candidate.get("home_team"))==selected else "away_logo_url",""))
    conference=clean(profile.get("conference",latest.get("conference","")))
    rank_val=profile.get("model_rank",latest.get("rank",np.nan));rank_text=f'#{int(float(rank_val))}' if pd.notna(rank_val) else "—"
    elo=profile.get("current_elo",latest.get("elo",np.nan));record=live_team_record(selected, games);delta=profile.get("recent_elo_change",latest.get("elo_change",np.nan))
    banner_colour="#d71920" if selected=="Georgia" else "#1d8fd1"
    if not team_games.empty:
        sample=team_games.iloc[0];side="home" if clean(sample.get("home_team"))==selected else "away";banner_colour=team_colour(sample,side,banner_colour)
    st.markdown(
        f'<div class="team-banner" style="--team-glow:{banner_colour}33;--team-border:{banner_colour}88"><div class="team-banner-inner">'
        f'{logo_html(logo,"team-banner-logo",selected)}<div><div class="page-eyebrow">{html.escape(conference or "TEAM INTELLIGENCE")}</div><div class="team-banner-name">{html.escape(selected.upper())}</div><div class="team-banner-meta">Production strength, form and matchup identity</div></div>'
        f'<div class="team-kpis"><div class="team-kpi"><span>Record</span><strong>{html.escape(record)}</strong></div><div class="team-kpi"><span>ELO rank</span><strong>{rank_text}</strong></div><div class="team-kpi"><span>ELO trend</span><strong class="up">{signed(delta,0)}</strong></div></div></div></div>',unsafe_allow_html=True)
    if compare!="None":
        comp_profile,comp_trend=selected_team_profile(compare,weekly,profiles);comp_latest=comp_trend.iloc[-1] if not comp_trend.empty else pd.Series(dtype=object)
        section_title(f"{selected} vs {compare}","Current team-strength comparison")
        with st.container(border=True):
            comparison_bar("ELO",elo,comp_profile.get("current_elo",comp_latest.get("elo",np.nan)),selected,compare)
            comparison_bar("Offensive rating",profile.get("offensive_rating"),comp_profile.get("offensive_rating"),selected,compare)
            comparison_bar("Defensive rating",profile.get("defensive_rating"),comp_profile.get("defensive_rating"),selected,compare)
            comparison_bar("Schedule strength",profile.get("schedule_strength_elo"),comp_profile.get("schedule_strength_elo"),selected,compare)
    overview,schedule,offense,defense,drives,trends=st.tabs(["Overview","Schedule","Offense","Defense","Drives","Trends"])
    with overview:
        kpi_html([("ELO",num(elo,0),f"Model rank {rank_text}","cyan"),("Offensive rating",num(profile.get("offensive_rating"),2),"Production metric","green"),("Defensive rating",num(profile.get("defensive_rating"),2),"Production metric","purple"),("Strength of schedule",clean(profile.get("schedule_strength_label","Building sample")) or "Building sample",f'Opp. ELO {num(profile.get("schedule_strength_elo"),0)}',"amber")])
        left,right=st.columns([1.2,1])
        with left:
            section_title("ELO trend","No future snapshots")
            if trend.empty:st.caption("No weekly history available.")
            else:branded_line_chart(trend,"week_label","elo",["week_label","elo","rank"],300,"ELO")
        with right:
            section_title("Model identity","What currently defines this team")
            momentum=clean(profile.get("momentum_label","Building sample")) or "Building sample"
            strengths=[]
            if pd.notna(profile.get("offensive_rating",np.nan)):strengths.append(f'Offensive rating {num(profile.get("offensive_rating"),2)}')
            if pd.notna(profile.get("recent_elo_change",np.nan)):strengths.append(f'ELO movement {signed(profile.get("recent_elo_change"),0)}')
            if pd.notna(profile.get("recent_margin_vs_expectation",np.nan)):strengths.append(f'{signed(profile.get("recent_margin_vs_expectation"),1)} points vs expectation')
            st.markdown(f'<div class="integrity-card"><div class="integrity-title">{html.escape(momentum.upper())}</div><div class="integrity-copy">{html.escape(selected)} is currently classified as {html.escape(momentum.lower())}. {html.escape(" · ".join(strengths) if strengths else "The in-season sample is still developing.")}</div></div>',unsafe_allow_html=True)
            section_title("Next matchup","Open the full game intelligence page")
            upcoming=team_games[first_text(team_games,["game_status"]).str.lower().ne("completed")]
            if not upcoming.empty:
                next_row=upcoming.iloc[0];opp=clean(next_row.get("away_team" if clean(next_row.get("home_team"))==selected else "home_team"));st.markdown(f'**{selected} vs {opp}**  \n{kickoff_text(next_row.get("start_date_utc"))}')
                gid=row_game_id(next_row)
                if gid is not None and st.button("Open next matchup →",key="team_next_game",use_container_width=True):request_game(gid)
            else:st.caption("No upcoming game in the current export.")
    with schedule:
        rows=[]
        for _,game in team_games.iterrows():
            is_home=clean(game.get("home_team"))==selected;opp=clean(game.get("away_team" if is_home else "home_team"));a=get_actual_score(game,"away");h=get_actual_score(game,"home")
            result="—"
            if pd.notna(a) and pd.notna(h):
                own=float(h if is_home else a);other=float(a if is_home else h);result=f'{"W" if own>other else "L"} {own:.0f}–{other:.0f}'
            rows.append({"Week":game.get("week"),"Date":game.get("start_date_utc"),"Opponent":f'{"vs" if is_home else "@"} {opp}',"Result":result,"Win %":display_value(game,"home_win_probability" if is_home else "away_win_probability"),"Game ID":row_game_id(game)})
        schedule_df=pd.DataFrame(rows)
        if schedule_df.empty:st.caption("No schedule rows available.")
        else:
            st.dataframe(schedule_df.drop(columns=["Game ID"]),use_container_width=True,hide_index=True,column_config={"Date":st.column_config.DatetimeColumn(format="ddd DD MMM, HH:mm"),"Win %":st.column_config.ProgressColumn(format="percent",min_value=0,max_value=1)})
            choice=st.selectbox("Open schedule matchup",schedule_df.index.tolist(),format_func=lambda i:f'Week {schedule_df.loc[i,"Week"]} · {schedule_df.loc[i,"Opponent"]}',key="team_schedule_game")
            gid=schedule_df.loc[choice,"Game ID"]
            if pd.notna(gid) and st.button("Open schedule matchup →",use_container_width=True):request_game(int(gid))
    with offense:
        section_title("Offensive profile","Production metrics exported for the selected team")
        kpi_html([("Points / game",num(profile.get("points_for_per_game"),1),"Completed games",""),("Offensive rating",num(profile.get("offensive_rating"),2),"Model metric","green"),("Success rate",pct(profile.get("success_rate")),"Down-to-down efficiency",""),("Explosive rate",pct(profile.get("explosive_rate")),"Big-play frequency","amber")])
        with st.container(border=True):
            for label,column,kind in [("EPA / play","epa_per_play","num"),("Points / drive","points_per_drive","num"),("Third down","third_down_rate","pct"),("Red zone TD","red_zone_td_rate","pct"),("Rush success","rush_success_rate","pct"),("Pass success","pass_success_rate","pct")]:factor_row(label,pct(profile.get(column)) if kind=="pct" else num(profile.get(column),3))
    with defense:
        section_title("Defensive profile","Scoring prevention and disruption")
        kpi_html([("Points allowed",num(profile.get("points_against_per_game"),1),"Per game",""),("Defensive rating",num(profile.get("defensive_rating"),2),"Model metric","purple"),("Havoc rate",pct(profile.get("havoc_rate")),"Pressure + disruption","green"),("Explosives allowed",pct(profile.get("explosives_allowed_rate")),"Lower is better","amber")])
        with st.container(border=True):
            for label,column,kind in [("Opponent EPA / play","def_epa_per_play","num"),("Opponent success","def_success_rate","pct"),("Sack rate","sack_rate","pct"),("Pressure rate","pressure_rate","pct"),("Third down allowed","def_third_down_rate","pct"),("Red zone TD allowed","def_red_zone_td_rate","pct")]:factor_row(label,pct(profile.get(column)) if kind=="pct" else num(profile.get(column),3))
    with drives:
        section_title("Drive profile","Possession-level identity")
        kpi_html([("Drives / game",num(profile.get("drives_per_game"),1),"Offensive possessions",""),("Points / drive",num(profile.get("points_per_drive"),2),"Scoring output","green"),("Start field position",num(profile.get("avg_start_field_position"),1),"Average yard line",""),("Three-and-out",pct(profile.get("three_and_out_rate")),"Lower is better","amber")])
        with st.container(border=True):
            factor_row("Touchdown drive rate",pct(profile.get("touchdown_drive_rate")));factor_row("Field-goal drive rate",pct(profile.get("field_goal_drive_rate")));factor_row("Turnover drive rate",pct(profile.get("turnover_drive_rate")));factor_row("Punt drive rate",pct(profile.get("punt_drive_rate")));factor_row("Average plays / drive",num(profile.get("plays_per_drive"),1))
    with trends:
        section_title("Weekly strength history","ELO, rank and movement")
        if trend.empty:st.caption("No trend data available.")
        else:
            branded_line_chart(trend,"week_label","elo",["week_label","elo","rank","elo_change"],360,"ELO")
            cols=[c for c in ["week_label","elo","rank","elo_change"] if c in trend.columns];st.dataframe(trend[cols].iloc[::-1],use_container_width=True,hide_index=True)


def render_elo_v3(weekly: pd.DataFrame, rankings: pd.DataFrame, profiles: pd.DataFrame) -> None:
    page_heading("RANKINGS COMMAND CENTRE","ELO Intelligence","Every team. Every week. Every movement.","FBS leaderboard only")
    source=weekly.copy()
    if source.empty:
        st.info("No weekly ELO export is available.");return
    weeks=sorted(pd.to_numeric(source["week"],errors="coerce").dropna().astype(int).unique().tolist())
    c1,c2,c3=st.columns([1,1,1.4])
    with c1:selected_week=st.selectbox("Snapshot",weeks,index=len(weeks)-1,format_func=lambda w:"Preseason" if w==0 else f"Week {w}",key="elo_v3_week")
    with c2:conference=st.selectbox("Conference",["All"]+sorted(first_text(source,["conference"]).replace("",np.nan).dropna().unique().tolist()),key="elo_v3_conf")
    with c3:search=st.text_input("Search teams",key="elo_v3_search",placeholder="Search the FBS leaderboard…").strip().lower()
    view=source[pd.to_numeric(source["week"],errors="coerce").eq(selected_week)].copy()
    if conference!="All":view=view[first_text(view,["conference"]).eq(conference)]
    if search:view=view[first_text(view,["team"]).str.lower().str.contains(search,regex=False)]
    view=view.sort_values("rank",kind="stable")
    delta=pd.to_numeric(view.get("elo_change"),errors="coerce")
    biggest_rise=delta.max() if len(delta.dropna()) else np.nan;biggest_fall=delta.min() if len(delta.dropna()) else np.nan
    kpi_html([("FBS teams",f"{len(view):,}","Current filter","cyan"),("Weeks tracked",f"{len(weeks):,}","Including preseason","purple"),("Biggest riser",signed(biggest_rise,0),"Current snapshot","green"),("Biggest faller",signed(biggest_fall,0),"Current snapshot","amber")])
    left,right=st.columns([1.45,1])
    with left:
        section_title("Model Top 25","Click any row to open the team intelligence page")
        top=view.head(25)
        for i,(_,row) in enumerate(top.iterrows()):
            rank=row.get("rank",i+1);move=pd.to_numeric(pd.Series([row.get("elo_change")]),errors="coerce").iloc[0];cls="rise" if pd.notna(move) and move>0 else "fall" if pd.notna(move) and move<0 else "";arrow="▲" if cls=="rise" else "▼" if cls=="fall" else "—"
            st.markdown(f'<div class="rank-row"><div class="rank-num">#{num(rank,0)}</div>{logo_html(row.get("logo_url"),"rank-logo",clean(row.get("team")))}<div><div class="rank-team">{html.escape(clean(row.get("team")))}</div><div class="rank-conf">{html.escape(clean(row.get("conference")))}</div></div><div class="rank-elo">{num(row.get("elo"),0)}</div><div class="rank-delta {cls}">{arrow} {num(abs(move),0) if pd.notna(move) else "—"}</div><div class="rank-record">{html.escape(clean(row.get("record","")))}</div></div>',unsafe_allow_html=True)
            if st.button(f'View {clean(row.get("team"))} →',key=f"elo_open_{i}",use_container_width=True):request_team(clean(row.get("team")))
    with right:
        section_title("Weekly movers","Largest rating changes")
        movers=view.assign(_delta=pd.to_numeric(view.get("elo_change"),errors="coerce")).dropna(subset=["_delta"])
        risers=movers.sort_values("_delta",ascending=False).head(5);fallers=movers.sort_values("_delta").head(5)
        rise_tab,fall_tab=st.tabs(["Risers","Fallers"])
        with rise_tab:
            for _,row in risers.iterrows():factor_row(clean(row.get("team")),f'▲ {num(row.get("_delta"),0)} · ELO {num(row.get("elo"),0)}')
        with fall_tab:
            for _,row in fallers.iterrows():factor_row(clean(row.get("team")),f'▼ {num(abs(row.get("_delta")),0)} · ELO {num(row.get("elo"),0)}')
        section_title("Selected team trend","Weekly movement and schedule context")
        team_options=first_text(view,["team"]).tolist()
        if team_options:
            selected=st.selectbox("Team trend",team_options,key="elo_v3_team")
            trend=source[first_text(source,["team"]).eq(selected)].sort_values("week")
            branded_line_chart(trend,"week_label","elo",["week_label","elo","rank"],280,"ELO")
            profile,_=selected_team_profile(selected,source,profiles)
            k1,k2=st.columns(2)
            with k1:metric_card("SOS rank",f'#{num(profile.get("schedule_strength_rank"),0)}',clean(profile.get("schedule_strength_label","")))
            with k2:metric_card("Opponent avg ELO",num(profile.get("schedule_strength_elo"),0),"Completed schedule")
            if st.button("Open selected team →",key="elo_selected_open",use_container_width=True):request_team(selected)
    section_title("Rank history","Top-team movement across weekly snapshots")
    top_names=first_text(view.head(10),["team"]).tolist();history=source[first_text(source,["team"]).isin(top_names)].copy()
    if not history.empty and alt is not None:
        chart=alt.Chart(history).mark_line(point=True,strokeWidth=2).encode(x=alt.X("week:O",title="Week"),y=alt.Y("rank:Q",title="Rank",sort="descending",scale=alt.Scale(reverse=True)),color=alt.Color("team:N",title=None),tooltip=["team:N","week_label:N","rank:Q","elo:Q"]).properties(height=320)
        st.altair_chart(dark_chart(chart),use_container_width=True)
    elif not history.empty:st.line_chart(history.pivot(index="week",columns="team",values="rank"),height=320)


def render_tracker_v3(games: pd.DataFrame, performance: dict[str, Any]) -> None:
    page_heading("PREDICTIONS","Official Prediction Tracker","Locked before kickoff. Audited after the final whistle.","Immutable lifecycle")
    evaluated=completed_analysis(games)
    usable=evaluated[evaluated["_v3_home_prob"].notna()] if not evaluated.empty else evaluated
    accuracy=float(usable["_v3_correct"].mean()) if len(usable) else None
    brier=float(np.mean((usable["_v3_home_prob"]-usable["_home_won"])**2)) if len(usable) else None
    locks=int(games["is_locked"].apply(boolish).sum()) if "is_locked" in games.columns else 0
    streak="—"
    if not evaluated.empty:
        ordered=usable.sort_values([c for c in ["start_date_utc","cfbd_game_id"] if c in usable.columns]);last=bool(ordered.iloc[-1]["_v3_correct"]);n=0
        for value in ordered["_v3_correct"].iloc[::-1]:
            if bool(value)==last:n+=1
            else:break
        streak=f'{"W" if last else "L"}{n}'
    record=f'{int(usable["_v3_correct"].sum())}-{int(len(usable)-usable["_v3_correct"].sum())}' if len(usable) else "—"
    kpi_html([("V3 record",record,f'{len(usable):,} settled V3 forecasts',"cyan"),("Accuracy",pct(accuracy),"V3 predictions","green"),("Brier score",num(brier,3),"Lower is better","purple"),("Current streak",streak,"Most recent results","amber"),("V3 locks made",f"{locks:,}","Immutable V3 locks","")])
    f1,f2,f3,f4=st.columns(4)
    with f1:week=st.selectbox("Week",game_week_options(games),key="tracker_v3_week")
    with f2:status=st.selectbox("Status",["All","Official locks","Completed","Upcoming"],key="tracker_v3_status")
    with f3:confidence=st.selectbox("Confidence",["All"]+CONFIDENCE_ORDER,key="tracker_v3_conf")
    with f4:model=st.selectbox("Model",["V3 production","Both","V2 comparison","Disagreements"],key="tracker_v3_model")
    query=st.text_input("Search tracker",placeholder="Search either team…",key="tracker_v3_search").strip().lower()
    view=games.copy()
    if week!="All":view=view[pd.to_numeric(view["week"],errors="coerce").eq(int(week))]
    view=view[status_mask(view,status)]
    if confidence!="All":view=view[first_text(view,["confidence_bucket"]).eq(confidence)]
    if model=="V3 production":view=view[pd.to_numeric(view.get("v3_home_win_probability"),errors="coerce").notna()]
    elif model=="V2 comparison":view=view[pd.to_numeric(view.get("home_win_probability"),errors="coerce").notna() & pd.to_numeric(view.get("v3_home_win_probability"),errors="coerce").isna()]
    elif model=="Disagreements":view=view[view.apply(lambda r:v3_available(r) and v3_winner(r)!=predicted_winner(r),axis=1)]
    if query:view=view[first_text(view,["home_team"]).str.lower().str.contains(query,regex=False)|first_text(view,["away_team"]).str.lower().str.contains(query,regex=False)]
    view=view.sort_values([c for c in ["start_date_utc","cfbd_game_id"] if c in view.columns],kind="stable")
    rows=[]
    for _,row in view.iterrows():
        a=get_actual_score(row,"away");h=get_actual_score(row,"home");final=f'{num(a,0)}–{num(h,0)}' if pd.notna(a) and pd.notna(h) else "—"
        actual=""
        if pd.notna(a) and pd.notna(h):actual=clean(row.get("away_team" if float(a)>float(h) else "home_team"))
        rows.append({"ID":row_game_id(row),"Week":row.get("week"),"Matchup":f'{clean(row.get("away_team"))} @ {clean(row.get("home_team"))}',"Lock":lock_text(row),"V3 Pick":v3_winner(row),"V3 %":game_confidence(row,"v3"),"V2 Pick":legacy_predicted_winner(row),"V2 %":game_confidence(row),"Projected":f'{num(display_value(row,"projected_away_score"),0)}–{num(display_value(row,"projected_home_score"),0)}',"Final":final,"Result":"WIN" if actual and v3_winner(row)==actual else "LOSS" if actual else "PENDING","Confidence":clean(row.get("confidence_bucket"))})
    table=pd.DataFrame(rows)
    left,right=st.columns([1.8,1])
    with left:
        section_title("Prediction ledger",f"{len(table):,} visible forecasts")
        if table.empty:st.markdown('<div class="empty-state">No predictions match those filters.</div>',unsafe_allow_html=True)
        else:
            st.dataframe(table.drop(columns=["ID"]),use_container_width=True,hide_index=True,height=570,column_config={"V2 %":st.column_config.ProgressColumn(format="percent",min_value=0,max_value=1),"V3 %":st.column_config.ProgressColumn(format="percent",min_value=0,max_value=1)})
            selected=st.selectbox("Open tracker matchup",table.index.tolist(),format_func=lambda i:table.loc[i,"Matchup"],key="tracker_open")
            selected_id = table.loc[selected, "ID"]
            if st.button("Open matchup intelligence →",key="tracker_open_button",use_container_width=True):
                if pd.notna(selected_id):
                    request_game(int(selected_id))
    with right:
        section_title("Live model watch","Games approaching official lock")
        upcoming=games[~first_text(games,["game_status"]).str.lower().eq("completed")].copy()
        if "hours_until_lock" in upcoming.columns:upcoming=upcoming.assign(_hours=pd.to_numeric(upcoming["hours_until_lock"],errors="coerce")).sort_values("_hours").head(6)
        else:upcoming=upcoming.sort_values("start_date_utc").head(6)
        if upcoming.empty:st.caption("No upcoming games.")
        else:
            for _,row in upcoming.iterrows():factor_row(f'{clean(row.get("away_team"))} @ {clean(row.get("home_team"))}',f'{lock_text(row)} · {pct(game_confidence(row))}')
        section_title("Audit integrity","Official lifecycle checks")
        st.markdown('<div class="integrity-card"><div class="integrity-title">✓ LOCKED AND TRACEABLE</div><div class="integrity-copy">Official predictions are frozen before kickoff. Historical reconstruction remains separately labelled and can never overwrite a genuine lock.</div></div>',unsafe_allow_html=True)
        k1,k2=st.columns(2)
        with k1:metric_card("Edits after lock","0","Immutable rows")
        with k2:metric_card("Archived",f"{locks:,}","Timestamped locks")


def render_more_v3(games: pd.DataFrame, rankings: pd.DataFrame, performance: dict[str, Any], betting: pd.DataFrame) -> None:
    page_heading("SYSTEM","More","Upset intelligence, betting evaluation, model architecture and app information.")
    choice=st.segmented_control("Section",["Upset Radar","Betting","Model","Settings","About"],default="Upset Radar",key="more_v3_section") or "Upset Radar"
    if choice=="Upset Radar":
        render_upsets(games,rankings)
    elif choice=="Betting":
        render_betting_performance(betting)
    elif choice=="Model":
        section_title("Model architecture","Production control and independent drive-model challenger")
        m1,m2=st.columns(2)
        with m1:
            st.markdown('<div class="model-card v2"><div class="model-card-title">V2 PRODUCTION / CONTROL</div><div class="model-card-score">Frozen benchmark</div><div class="model-card-note">ELO, opponent-adjusted matchup features, projected possessions and calibrated winner probability. Official locks remain immutable.</div><span class="agreement">LIVE CONTROL</span></div>',unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="model-card v3"><div class="model-card-title">V3 DRIVE MODEL</div><div class="model-card-score">Independent challenger</div><div class="model-card-note">Opponent-adjusted drives, field position, PPD, score state and Monte Carlo outcome distributions.</div><span class="agreement">EVALUATED SEPARATELY</span></div>',unsafe_allow_html=True)
        section_title("Prediction flow","Strictly chronological and market-free")
        with st.container(border=True):
            factor_row("1 · Source","Validated CFBD games and play-by-play cache")
            factor_row("2 · State","Pregame ELO and opponent-adjusted team state")
            factor_row("3 · Environment","Possessions, drives and field position")
            factor_row("4 · Forecast","Winner probability, projected score, margin and total")
            factor_row("5 · Simulation","Monte Carlo outcome and score distributions")
            factor_row("6 · Lock","Immutable official timestamp before kickoff")
        st.info("Market spreads and prices remain isolated under the market-data layer. They are available for display and evaluation only and never enter either football model.")
    elif choice=="Settings":
        section_title("Display settings","Controls that affect presentation only")
        st.selectbox("Default landing week",["Automatic next week"]+game_week_options(games)[1:],key="settings_week")
        st.selectbox("Kickoff timezone",["Europe/London","US/Eastern","UTC"],index=0,key="settings_timezone",disabled=True)
        st.toggle("Show market information",value=True,key="settings_market")
        st.toggle("Show V3 challenger",value=True,key="settings_v3")
        st.caption("Timezone and display preferences are planned for persistent user settings. The current production export remains unchanged.")
        if st.button("Clear cached app data",use_container_width=True):st.cache_data.clear();st.rerun()
    else:
        section_title("CFB Prediction Centre","A model-first view of college football")
        st.markdown("""
This application brings together weekly forecasts, official locks, projected scores, ELO movement, team intelligence, Monte Carlo simulations and model diagnostics.

**Prediction integrity**  
V2 remains the frozen production/control model. V3 is an independent drive-based challenger. Market prices are isolated from prediction features and used only for display and evaluation.

**Navigation**  
Every matchup links to both team pages. Every ELO row links to its team. Team schedules link back to matchup intelligence, creating one connected product rather than separate dashboards.

**Data lifecycle**  
Official forecasts lock before kickoff and remain immutable. Reconstruction is always labelled separately.
""")


# ============================================================
# MOCK-UP-FIDELITY UI LAYER
# ============================================================

st.markdown(
    """
<style>
/* This last stylesheet deliberately overrides all legacy Streamlit styling. */
.cfb-header, div[data-testid="stRadio"]{display:none!important}
.block-container{max-width:none!important;margin-left:190px!important;width:calc(100% - 190px)!important;padding:74px 28px 42px!important}
.stApp{background:#030b15!important}
[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 55% -12%,rgba(13,94,146,.18),transparent 32%),linear-gradient(180deg,#04101d,#030a13 72%)!important}
.stMarkdown a{text-decoration:none!important}

.x-sidebar{position:fixed;z-index:9998;left:0;top:0;bottom:0;width:190px;background:linear-gradient(180deg,#07192d 0%,#061425 58%,#04101d 100%);border-right:1px solid #103452;box-shadow:18px 0 60px rgba(0,0,0,.28);padding:22px 12px;box-sizing:border-box}
.x-brand{display:flex;align-items:center;gap:10px;padding:0 10px 22px;border-bottom:1px solid rgba(75,128,166,.17)}
.x-brand-mark{width:36px;height:36px;border:1px solid #1dc5ff;border-radius:50%;display:grid;place-items:center;color:#31d2ff;font-size:19px;box-shadow:0 0 22px rgba(31,196,255,.18);transform:rotate(-18deg)}
.x-brand-name{font-size:.84rem;line-height:1;color:#f7fbff;font-weight:1000;letter-spacing:.05em}.x-brand-sub{font-size:.43rem;color:#26c7fb;font-weight:900;letter-spacing:.12em;margin-top:5px}
.x-nav{display:flex;flex-direction:column;gap:5px;margin-top:25px}.x-nav a{display:flex;align-items:center;gap:11px;height:43px;padding:0 13px;border-radius:10px;color:#89a2b8!important;font-size:.68rem;font-weight:800;border:1px solid transparent;box-sizing:border-box;transition:.16s}.x-nav a:hover{background:#0b2840;color:#fff!important;transform:translateX(2px)}.x-nav a.active{color:#fff!important;background:linear-gradient(90deg,#0e74ad,#0a3556);border-color:#19bff5;box-shadow:0 0 24px rgba(17,172,229,.15)}.x-nav a.active:before{content:"";position:absolute;left:0;width:3px;height:27px;background:#28d4ff;border-radius:0 4px 4px 0;box-shadow:0 0 14px #29cfff}.x-nav-icon{width:18px;text-align:center;font-size:.88rem;color:#58bfe9}.x-side-foot{position:absolute;left:22px;bottom:25px;color:#335e7d;font-size:.46rem;letter-spacing:.15em;line-height:1.6;font-weight:900}
.x-topbar{position:fixed;z-index:9995;left:190px;right:0;top:0;height:60px;background:rgba(3,13,24,.93);backdrop-filter:blur(20px);border-bottom:1px solid #113451;display:flex;align-items:center;justify-content:space-between;padding:0 28px;box-sizing:border-box}.x-top-title{font-size:.69rem;font-weight:950;letter-spacing:.18em;color:#f5f9fc}.x-top-centre{display:flex;gap:17px;align-items:center;color:#91a8ba;font-size:.60rem}.x-top-centre strong{color:#2bcbff}.x-search{min-width:280px;height:34px;border:1px solid #1b4968;border-radius:9px;background:#07192a;color:#66849a;display:flex;align-items:center;padding:0 10px;font-size:.57rem;box-sizing:border-box}.x-search input{width:100%;height:28px!important;background:transparent!important;border:0!important;outline:0!important;color:#dceaf3!important;font-size:.55rem!important;box-shadow:none!important}.x-search input::placeholder{color:#66849a!important}.x-online{width:7px;height:7px;border-radius:50%;background:#53e8a1;box-shadow:0 0 12px #53e8a1;display:inline-block;margin-right:7px}

.x-eyebrow{color:#2fd4ff;font-size:.52rem;letter-spacing:.22em;font-weight:950;text-transform:uppercase}.x-h1{font-size:clamp(2rem,4vw,3.35rem);color:#fff;font-weight:1000;line-height:.92;letter-spacing:-.045em;margin:7px 0}.x-sub{color:#8da6ba;font-size:.69rem;line-height:1.45}.x-page-head{display:flex;justify-content:space-between;align-items:flex-end;gap:18px;margin-bottom:17px}.x-head-actions{display:flex;gap:8px;align-items:center}.x-chip{display:inline-flex;align-items:center;padding:5px 9px;border:1px solid #1b405b;background:#081b2c;border-radius:999px;color:#84a0b5;font-size:.49rem;font-weight:900}.x-chip.green{color:#65e9a9;border-color:rgba(72,224,155,.33);background:rgba(72,224,155,.07)}.x-chip.amber{color:#ffd16f;border-color:rgba(255,189,87,.33);background:rgba(255,189,87,.07)}.x-chip.purple{color:#d2bbff;border-color:rgba(168,128,255,.34);background:rgba(168,128,255,.08)}
.x-panel{background:linear-gradient(150deg,rgba(11,28,44,.98),rgba(5,17,29,.98));border:1px solid #173d59;border-radius:14px;box-shadow:0 15px 42px rgba(0,0,0,.20);overflow:hidden}.x-panel-head{height:42px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;border-bottom:1px solid rgba(56,102,136,.22);box-sizing:border-box}.x-panel-title{color:#f3f8fb;font-size:.66rem;font-weight:950;letter-spacing:.02em}.x-panel-link{color:#34cfff!important;font-size:.49rem;font-weight:850}.x-panel-body{padding:13px}.x-grid-2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.x-grid-3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.x-grid-4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.x-grid-home{display:grid;grid-template-columns:1.45fr 1fr 1fr 1fr;gap:12px}.x-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:13px 0}.x-kpi{border:1px solid #194462;border-radius:12px;padding:13px;background:linear-gradient(145deg,#0b2439,#071725);min-height:84px;box-sizing:border-box;position:relative;overflow:hidden}.x-kpi:after{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;background:#26cfff}.x-kpi.green:after{background:#43e397}.x-kpi.amber:after{background:#ffbd57}.x-kpi.purple:after{background:#a880ff}.x-kpi-label{font-size:.50rem;color:#7893a9;font-weight:900;letter-spacing:.07em;text-transform:uppercase}.x-kpi-value{font-size:1.58rem;color:#fff;font-weight:1000;line-height:1;margin-top:8px}.x-kpi-note{font-size:.49rem;color:#6f8a9f;margin-top:6px}

.x-feature{position:relative;border:1px solid #1b78a1;border-radius:15px;overflow:hidden;background:radial-gradient(circle at 17% 55%,var(--away-glow,#78162b) 0%,transparent 32%),radial-gradient(circle at 83% 55%,var(--home-glow,#65142a) 0%,transparent 32%),linear-gradient(100deg,#160b17,#06192a 45% 55%,#160b17);padding:15px 18px;margin:11px 0 13px}.x-feature-label{font-size:.48rem;color:#35d3ff;font-weight:950;letter-spacing:.13em}.x-feature-grid{display:grid;grid-template-columns:1fr 230px 1fr;align-items:center;gap:10px}.x-fteam{display:grid;grid-template-columns:90px 1fr;align-items:center;gap:9px}.x-fteam.right{grid-template-columns:1fr 90px;text-align:right}.x-fteam img{width:82px;height:82px;object-fit:contain;filter:drop-shadow(0 10px 19px rgba(0,0,0,.45))}.x-fteam-name{font-size:1.02rem;color:#fff;font-weight:1000;line-height:1}.x-fteam-meta{font-size:.51rem;color:#a0b3c3;margin-top:5px}.x-fcentre{text-align:center}.x-fscore{font-size:2.5rem;color:#fff;font-weight:1000;letter-spacing:-.055em;line-height:1}.x-flabel{font-size:.45rem;color:#6f8ba0;letter-spacing:.12em;text-transform:uppercase}.x-probs{display:flex;justify-content:center;gap:16px;color:#f5f9fc;font-size:.68rem;font-weight:950;margin-top:5px}.x-open{display:inline-block;margin-top:8px;padding:7px 18px;border-radius:9px;background:linear-gradient(90deg,#14bce8,#31d9ff);color:#00111c!important;font-size:.52rem;font-weight:1000;box-shadow:0 0 20px rgba(41,205,255,.18)}
.x-probbar{height:6px;border-radius:999px;background:#153148;display:flex;overflow:hidden;margin-top:8px}.x-probbar .away{background:linear-gradient(90deg,#ff354f,#df1742)}.x-probbar .home{background:linear-gradient(90deg,#327ee9,#24c4ff)}
.x-team-link{color:inherit!important;display:block}.x-team-link:hover .x-fteam-name,.x-team-link:hover .x-card-team-name{color:#34d2ff}

.x-mini-row{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(74px,.7fr) 62px;gap:8px;align-items:center;padding:8px 2px;border-bottom:1px solid rgba(71,113,144,.15);font-size:.55rem}.x-mini-row:last-child{border-bottom:0}.x-mini-main{color:#eaf2f7;font-weight:850}.x-mini-sub{color:#648399;font-size:.46rem;margin-top:2px}.x-mini-value{color:#fff;font-weight:950;text-align:right}.x-mini-value.up{color:#48e09b}.x-mini-value.down{color:#ff6475}.x-mini-action{color:#32cbff!important;text-align:right;font-size:.48rem;font-weight:900}
.x-bars{height:150px;display:flex;align-items:flex-end;gap:8px;padding:12px 4px 0}.x-bar-wrap{flex:1;text-align:center}.x-bar{min-height:3px;border-radius:5px 5px 1px 1px;background:linear-gradient(180deg,#2bd0ff,#1574be);box-shadow:0 0 14px rgba(42,195,255,.12)}.x-bar.green{background:linear-gradient(180deg,#4be69d,#14915a)}.x-bar.amber{background:linear-gradient(180deg,#ffc55f,#b96d0d)}.x-bar.purple{background:linear-gradient(180deg,#b08aff,#6242b9)}.x-bar-value{font-size:.46rem;color:#d8e5ed;font-weight:900;margin-bottom:5px}.x-bar-label{font-size:.40rem;color:#66869b;margin-top:5px;white-space:nowrap}

.x-game-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.x-game-card{position:relative;border:1px solid #1a4b6a;border-radius:13px;background:linear-gradient(155deg,#0b2236,#061522);padding:12px;overflow:hidden}.x-game-card:hover{border-color:#2ac9ff;transform:translateY(-2px);box-shadow:0 12px 30px rgba(0,0,0,.28);transition:.15s}.x-game-meta{display:flex;justify-content:space-between;color:#7894a9;font-size:.43rem;font-weight:850}.x-game-teams{display:grid;grid-template-columns:1fr 72px 1fr;gap:7px;align-items:center;margin:11px 0}.x-card-team{text-align:center}.x-card-team img{width:51px;height:51px;object-fit:contain}.x-card-team-name{font-size:.61rem;color:#fff;font-weight:950;line-height:1.05}.x-card-team-meta{font-size:.42rem;color:#7893a8;margin-top:4px}.x-card-score{text-align:center}.x-card-score strong{font-size:1.27rem;color:#fff}.x-card-score span{display:block;color:#608096;font-size:.39rem;text-transform:uppercase}.x-card-foot{display:flex;justify-content:space-between;align-items:center;gap:4px;margin-top:8px}.x-game-open{display:block;text-align:center;margin-top:9px;padding:7px;border-radius:7px;background:#1dd0f7;color:#00131e!important;font-size:.46rem;font-weight:1000;letter-spacing:.04em}

.x-tabs{display:flex;gap:4px;border-bottom:1px solid #153854;margin:0 0 12px;overflow-x:auto;scrollbar-width:none}.x-tabs a{color:#7e9aaf!important;font-size:.55rem;font-weight:850;padding:10px 14px;border-bottom:2px solid transparent;white-space:nowrap}.x-tabs a.active{color:#fff!important;border-color:#2dd5ff;background:linear-gradient(180deg,transparent,rgba(43,196,245,.06))}
.x-match-stage{position:relative;overflow:hidden;border:1px solid #1b5a7b;border-radius:15px;background:radial-gradient(circle at 15% 50%,var(--away-glow,#6e1025),transparent 35%),radial-gradient(circle at 85% 50%,var(--home-glow,#7a1027),transparent 35%),linear-gradient(100deg,#160b17,#041420 43% 57%,#160a16);padding:18px;margin-bottom:12px}.x-match-grid{display:grid;grid-template-columns:1fr 260px 1fr;align-items:center;gap:12px}.x-match-team{text-align:center}.x-match-team img{width:120px;height:120px;object-fit:contain;filter:drop-shadow(0 13px 25px rgba(0,0,0,.5))}.x-match-name{font-size:1.35rem;color:#fff;font-weight:1000;line-height:1}.x-match-meta{color:#a1b4c4;font-size:.51rem;margin-top:6px}.x-view-team{display:inline-block;margin-top:8px;border:1px solid #27cfff;border-radius:999px;padding:6px 13px;color:#e8faff!important;font-size:.46rem;font-weight:950}.x-match-centre{text-align:center}.x-match-score{font-size:3rem;color:#fff;font-weight:1000;line-height:1;letter-spacing:-.07em}.x-ring{width:88px;height:88px;border-radius:50%;margin:9px auto;display:grid;place-items:center;background:conic-gradient(#ff3150 0 var(--ring),#1b4b6a var(--ring) 100%);position:relative}.x-ring:before{content:"";position:absolute;inset:10px;border-radius:50%;background:#071522}.x-ring span{position:relative;color:#fff;font-size:.55rem;font-weight:950}.x-model-grid{display:grid;grid-template-columns:1.2fr .72fr .72fr 1fr;gap:10px}.x-stat-list{padding:4px 13px}.x-stat-row{display:grid;grid-template-columns:1fr 62px 1fr;gap:8px;align-items:center;padding:8px 0;border-bottom:1px solid rgba(67,108,138,.16);font-size:.51rem}.x-stat-row .left{color:#ff5a70;font-weight:900}.x-stat-row .centre{color:#7894a8;text-align:center}.x-stat-row .right{color:#65d7ff;font-weight:900;text-align:right}

.x-team-hero{position:relative;overflow:hidden;border:1px solid var(--team,#b5122b);border-radius:15px;background:radial-gradient(circle at 12% 50%,var(--team-soft,rgba(180,18,43,.5)),transparent 36%),linear-gradient(100deg,#160b16,#071522 55%);padding:18px 20px;margin-bottom:11px}.x-team-hero-grid{display:grid;grid-template-columns:140px minmax(240px,1fr) minmax(390px,1.15fr);align-items:center;gap:18px}.x-team-hero img{width:125px;height:125px;object-fit:contain;filter:drop-shadow(0 13px 24px rgba(0,0,0,.45))}.x-team-name{font-size:2.25rem;color:#fff;font-weight:1000;line-height:.9;letter-spacing:-.04em}.x-team-kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.x-team-kpi{padding:10px;border:1px solid rgba(96,143,176,.28);border-radius:9px;background:rgba(5,18,30,.7)}.x-team-kpi span{display:block;color:#7894a8;font-size:.42rem;text-transform:uppercase;font-weight:900}.x-team-kpi strong{display:block;color:#fff;font-size:1.2rem;margin-top:6px}.x-metric-strip{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px;margin:10px 0}.x-metric{padding:11px;border:1px solid #173c58;border-radius:10px;background:#091b2b}.x-metric-label{font-size:.43rem;color:#7792a7}.x-metric-value{font-size:1.15rem;color:#fff;font-weight:1000;margin-top:6px}.x-metric-rank{font-size:.40rem;color:#48e09b;margin-top:4px}
.x-radar-wrap{display:grid;grid-template-columns:1fr 1fr;gap:10px;align-items:center}.x-svg{width:100%;height:auto;display:block}.x-chart-grid{stroke:#1c3a50;stroke-width:1}.x-chart-axis{fill:#66869b;font-size:9px}.x-chart-line{fill:none;stroke:#27cfff;stroke-width:3;filter:drop-shadow(0 0 5px rgba(39,207,255,.3))}.x-chart-area{fill:url(#areaGradient);opacity:.32}

.x-rank-layout{display:grid;grid-template-columns:1.45fr 1fr;gap:12px}.x-rank-row{display:grid;grid-template-columns:34px 32px minmax(0,1fr) 65px 58px 55px;gap:7px;align-items:center;padding:7px 8px;border-bottom:1px solid rgba(63,105,136,.16);font-size:.51rem}.x-rank-row:hover{background:#0d2940}.x-rank-row img{width:25px;height:25px;object-fit:contain}.x-rank-team{color:#edf5fa;font-weight:900}.x-rank-team small{display:block;color:#65849a;font-size:.40rem;margin-top:2px}.x-rank-elo{color:#fff;font-weight:950;text-align:right}.x-rank-move{text-align:right;font-weight:950}.x-rank-move.up{color:#48e09b}.x-rank-move.down{color:#ff6475}.x-rank-record{color:#7793a7;text-align:right}
.x-table{width:100%;border-collapse:collapse}.x-table th{color:#66869c;font-size:.41rem;text-align:left;text-transform:uppercase;letter-spacing:.06em;padding:8px;border-bottom:1px solid #1a405c}.x-table td{color:#c8d7e1;font-size:.49rem;padding:8px;border-bottom:1px solid rgba(64,105,136,.14)}.x-table tr:hover td{background:#0c263b}.x-result{display:inline-flex;padding:3px 6px;border-radius:5px;font-size:.40rem;font-weight:950}.x-result.win{color:#5be8a3;background:rgba(72,224,155,.12)}.x-result.loss{color:#ff6a7b;background:rgba(255,82,104,.12)}.x-result.pending{color:#55ceff;background:rgba(45,190,244,.12)}
.x-analytics-kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}.x-analytics-main{display:grid;grid-template-columns:1fr 1fr;gap:11px;margin-top:11px}.x-analytics-lower{display:grid;grid-template-columns:1.15fr .9fr .8fr;gap:11px;margin-top:11px}

@media(max-width:1180px){.x-game-grid{grid-template-columns:repeat(2,1fr)}.x-grid-home{grid-template-columns:1.4fr 1fr}.x-grid-home>.x-panel:nth-child(n+3){grid-column:auto}.x-feature-grid{grid-template-columns:1fr 190px 1fr}.x-match-grid{grid-template-columns:1fr 210px 1fr}.x-team-hero-grid{grid-template-columns:105px 1fr}.x-team-kpis{grid-column:1/-1}.x-metric-strip{grid-template-columns:repeat(4,1fr)}.x-model-grid{grid-template-columns:1fr 1fr}.x-analytics-kpis{grid-template-columns:repeat(3,1fr)}}
@media(max-width:800px){.block-container{margin-left:0!important;width:100%!important;padding:64px 11px 78px!important}.x-sidebar{left:7px;right:7px;top:auto;bottom:7px;width:auto;height:57px;border:1px solid #17435f;border-radius:15px;padding:5px;background:rgba(5,17,29,.97);backdrop-filter:blur(16px)}.x-brand,.x-side-foot{display:none}.x-nav{display:flex;flex-direction:row;justify-content:space-between;gap:2px;margin:0;height:100%}.x-nav a{flex:1;height:45px;display:flex;flex-direction:column;justify-content:center;gap:2px;padding:0;font-size:.39rem}.x-nav a:nth-child(3),.x-nav a:nth-child(6),.x-nav a:nth-child(7){display:none}.x-nav a.active:before{display:none}.x-nav-icon{font-size:.72rem}.x-topbar{left:0;height:54px;padding:0 12px}.x-top-title{font-size:.57rem}.x-top-centre{display:none}.x-search{min-width:auto;width:125px;height:30px}.x-page-head{align-items:flex-start;flex-direction:column}.x-h1{font-size:2rem}.x-feature-grid{grid-template-columns:1fr 100px 1fr}.x-fteam{display:block;text-align:center}.x-fteam.right{display:block;text-align:center}.x-fteam img{width:58px;height:58px}.x-fteam-name{font-size:.65rem}.x-fcentre{grid-column:auto}.x-fscore{font-size:1.5rem}.x-kpis{grid-template-columns:repeat(2,1fr);gap:7px}.x-grid-home,.x-grid-2,.x-grid-3,.x-grid-4,.x-rank-layout,.x-analytics-main,.x-analytics-lower{grid-template-columns:1fr}.x-game-grid{grid-template-columns:1fr}.x-match-grid{grid-template-columns:1fr 105px 1fr}.x-match-team img{width:66px;height:66px}.x-match-name{font-size:.69rem}.x-match-score{font-size:1.55rem}.x-ring{width:62px;height:62px}.x-team-hero-grid{grid-template-columns:64px 1fr;gap:9px}.x-team-hero img{width:60px;height:60px}.x-team-name{font-size:1.4rem}.x-team-kpis{grid-template-columns:repeat(2,1fr)}.x-metric-strip{grid-template-columns:repeat(2,1fr)}.x-analytics-kpis{grid-template-columns:repeat(2,1fr)}.x-model-grid{grid-template-columns:1fr}.x-tabs a{padding:9px 10px}.x-rank-row{grid-template-columns:25px 27px minmax(0,1fr) 55px 44px}.x-rank-record{display:none}.x-search{font-size:0}.x-search:after{content:"Search";font-size:.48rem;color:#68869b}}
</style>
""",
    unsafe_allow_html=True,
)


def route_href(page: str, **params: Any) -> str:
    query = {"page": page}
    for key, value in params.items():
        if value is not None and clean(value) != "":
            query[key] = value
    return "?" + urlencode(query)


def query_value(name: str, default: str = "") -> str:
    try:
        value = st.query_params.get(name, default)
        if isinstance(value, list):
            value = value[0] if value else default
        return clean(value) or default
    except Exception:
        return default


def exact_logo(url: Any, team: str, class_name: str = "") -> str:
    value = clean(url).strip()
    if value.startswith("http"):
        return f'<img class="{class_name}" src="{html.escape(value, quote=True)}" alt="{html.escape(team, quote=True)}">'
    initials = "".join(part[0] for part in team.split()[:2]).upper() or "CFB"
    return f'<div class="{class_name}" style="display:grid;place-items:center;color:#fff;font-weight:1000;font-size:1.1rem">{html.escape(initials)}</div>'


def exact_shell(active: str) -> None:
    nav = [
        ("home", "⌂", "Home"), ("games", "▦", "Games"), ("tracker", "⌁", "Tracker"),
        ("matchup", "VS", "Matchup"), ("team", "♜", "Teams"), ("elo", "▥", "ELO"),
        ("analytics", "◒", "Analytics"), ("more", "••", "More"),
    ]
    links = "".join(
        f'<a class="{"active" if slug == active else ""}" href="{route_href(slug)}"><span class="x-nav-icon">{icon}</span><span>{label}</span></a>'
        for slug, icon, label in nav
    )
    st.markdown(
        '<aside class="x-sidebar"><div class="x-brand"><div class="x-brand-mark">⌁</div><div><div class="x-brand-name">CFB</div><div class="x-brand-sub">PREDICTION CENTRE</div></div></div>'
        f'<nav class="x-nav">{links}</nav><div class="x-side-foot">DATA DRIVEN<br>BIGGER SATURDAYS<br>V3 PRODUCTION</div></aside>'
        '<header class="x-topbar"><div class="x-top-title">CFB PREDICTION CENTRE</div><div class="x-top-centre"><span>Season 2026</span><span>|</span><strong>V3 Production</strong><span><i class="x-online"></i>Model online</span></div><form class="x-search" method="get"><input type="hidden" name="page" value="games"><span>⌕&nbsp;</span><input name="q" placeholder="Search teams, games, conferences…"></form></header>',
        unsafe_allow_html=True,
    )


def svg_line(values: list[float], colour: str = "#2bd0ff", height: int = 170, labels: list[str] | None = None) -> str:
    clean_values = [float(v) for v in values if pd.notna(v) and np.isfinite(float(v))]
    if not clean_values:
        return '<div class="empty-state">Awaiting chart data</div>'
    width, pad = 560, 24
    low, high = min(clean_values), max(clean_values)
    if high == low: high = low + 1
    points = []
    n = len(clean_values)
    for i, value in enumerate(clean_values):
        x = pad + (width - 2 * pad) * (i / max(1, n - 1))
        y = height - pad - (height - 2 * pad) * ((value - low) / (high - low))
        points.append((x, y))
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = f"{pad},{height-pad} {line} {width-pad},{height-pad}"
    circles = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{colour}" stroke="#06121f" stroke-width="2"/>' for x, y in points)
    grids = "".join(f'<line x1="{pad}" y1="{pad+i*(height-2*pad)/4:.1f}" x2="{width-pad}" y2="{pad+i*(height-2*pad)/4:.1f}" stroke="#18354a" stroke-width="1"/>' for i in range(5))
    label_nodes = ""
    if labels:
        keep = labels[-n:]
        label_nodes = "".join(f'<text x="{points[i][0]:.1f}" y="{height-5}" text-anchor="middle" fill="#607f95" font-size="8">{html.escape(str(label))}</text>' for i, label in enumerate(keep))
    return f'<svg class="x-svg" viewBox="0 0 {width} {height}" role="img">{grids}<polygon points="{area}" fill="{colour}" opacity=".10"/><polyline points="{line}" fill="none" stroke="{colour}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>{circles}{label_nodes}</svg>'


def svg_dual_line(series_a: list[float], series_b: list[float], colour_a: str = "#2bd0ff", colour_b: str = "#a880ff", height: int = 180) -> str:
    values = [float(v) for v in series_a + series_b if pd.notna(v) and np.isfinite(float(v))]
    if not values:
        return '<div class="empty-state">Awaiting chart data</div>'
    width, pad, low, high = 560, 24, min(values), max(values)
    if high == low: high = low + 1
    def points(series: list[float]) -> str:
        valid = [float(v) for v in series if pd.notna(v) and np.isfinite(float(v))]
        return " ".join(f'{pad+(width-2*pad)*i/max(1,len(valid)-1):.1f},{height-pad-(height-2*pad)*(v-low)/(high-low):.1f}' for i,v in enumerate(valid))
    grids = "".join(f'<line x1="{pad}" y1="{pad+i*(height-2*pad)/4:.1f}" x2="{width-pad}" y2="{pad+i*(height-2*pad)/4:.1f}" stroke="#18354a"/>' for i in range(5))
    return f'<svg class="x-svg" viewBox="0 0 {width} {height}">{grids}<polyline points="{points(series_a)}" fill="none" stroke="{colour_a}" stroke-width="3"/><polyline points="{points(series_b)}" fill="none" stroke="{colour_b}" stroke-width="3"/></svg>'


def svg_radar(values: list[float], colour: str, title: str) -> str:
    labels = ["Points", "Yards", "Success", "Explosive", "Rushing", "Passing", "3rd Down", "Red Zone"]
    vals = [(0.5 if pd.isna(v) else max(0.08, min(1.0, float(v)))) for v in (values + [0.5] * 8)[:8]]
    cx, cy, radius = 110, 104, 73
    axes = []
    polygon = []
    label_nodes = []
    for i, (label, value) in enumerate(zip(labels, vals)):
        angle = -np.pi / 2 + 2 * np.pi * i / 8
        ex, ey = cx + radius * np.cos(angle), cy + radius * np.sin(angle)
        px, py = cx + radius * value * np.cos(angle), cy + radius * value * np.sin(angle)
        lx, ly = cx + (radius + 18) * np.cos(angle), cy + (radius + 18) * np.sin(angle)
        axes.append(f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#254157"/>')
        polygon.append(f"{px:.1f},{py:.1f}")
        label_nodes.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" fill="#7692a7" font-size="7">{label}</text>')
    rings = "".join(f'<circle cx="{cx}" cy="{cy}" r="{radius*r/4}" fill="none" stroke="#1c384d"/>' for r in range(1,5))
    return f'<div style="text-align:center"><div class="x-panel-title">{html.escape(title)}</div><svg class="x-svg" viewBox="0 0 220 210">{rings}{"".join(axes)}<polygon points="{" ".join(polygon)}" fill="{colour}" fill-opacity=".28" stroke="{colour}" stroke-width="2"/>{"".join(label_nodes)}</svg></div>'


def exact_tabs(page: str, tabs: list[tuple[str, str]], active: str, **context: Any) -> None:
    links = "".join(f'<a class="{"active" if slug == active else ""}" href="{route_href(page, tab=slug, **context)}">{html.escape(label)}</a>' for slug, label in tabs)
    st.markdown(f'<nav class="x-tabs">{links}</nav>', unsafe_allow_html=True)


def exact_kpis(items: list[tuple[str, str, str, str]]) -> str:
    return '<div class="x-kpis">' + "".join(f'<div class="x-kpi {tone}"><div class="x-kpi-label">{html.escape(label)}</div><div class="x-kpi-value">{html.escape(value)}</div><div class="x-kpi-note">{html.escape(note)}</div></div>' for label,value,note,tone in items) + '</div>'


def exact_panel(title: str, body: str, link: str = "", link_label: str = "") -> str:
    action = f'<a class="x-panel-link" href="{link}">{html.escape(link_label or "View all")} →</a>' if link else ""
    return f'<section class="x-panel"><div class="x-panel-head"><div class="x-panel-title">{html.escape(title)}</div>{action}</div><div class="x-panel-body">{body}</div></section>'


def exact_game_card(row: pd.Series) -> str:
    game_id = row_game_id(row)
    away, home = clean(row.get("away_team")), clean(row.get("home_team"))
    ap, hp = display_value(row,"away_win_probability"), display_value(row,"home_win_probability")
    try: aw=max(0,min(100,float(ap)*100));hw=100-aw
    except Exception: aw=hw=50
    signal, signal_class = v3_signal(row)
    return (
        f'<article class="x-game-card"><div class="x-game-meta"><span>{html.escape(kickoff_text(row.get("start_date_utc")))}</span><span>{html.escape(lock_text(row))}</span></div>'
        '<div class="x-game-teams">'
        f'<a class="x-team-link" href="{route_href("team",team=away)}"><div class="x-card-team">{exact_logo(row.get("away_logo_url"),away)}<div class="x-card-team-name">{html.escape(away)}</div><div class="x-card-team-meta">{pct(ap)}</div></div></a>'
        f'<div class="x-card-score"><span>Projected</span><strong>{num(display_value(row,"projected_away_score"),0)}–{num(display_value(row,"projected_home_score"),0)}</strong></div>'
        f'<a class="x-team-link" href="{route_href("team",team=home)}"><div class="x-card-team">{exact_logo(row.get("home_logo_url"),home)}<div class="x-card-team-name">{html.escape(home)}</div><div class="x-card-team-meta">{pct(hp)}</div></div></a></div>'
        f'<div class="x-probbar"><div class="away" style="width:{aw:.1f}%"></div><div class="home" style="width:{hw:.1f}%"></div></div>'
        f'<div class="x-card-foot"><span class="x-chip green">{html.escape(clean(row.get("confidence_bucket")) or "Model")}</span><span class="x-chip {signal_class}">{html.escape(signal)}</span><span class="x-chip">{html.escape(market_spread_text(row))}</span></div>'
        f'<a class="x-game-open" href="{route_href("matchup",game=game_id)}">OPEN MATCHUP →</a></article>'
    )


def exact_selected_game(games: pd.DataFrame) -> pd.Series:
    if games.empty:
        return pd.Series(dtype=object)
    game_id = query_value("game")
    if game_id:
        ids = pd.to_numeric(games.get("cfbd_game_id"), errors="coerce")
        try:
            match = games[ids.eq(int(game_id))]
            if not match.empty: return match.iloc[0]
        except Exception: pass
    now = pd.Timestamp.now(tz="UTC")
    upcoming = games[games.get("start_date_utc",pd.Series(pd.NaT,index=games.index)).ge(now)].sort_values("start_date_utc")
    return upcoming.iloc[0] if not upcoming.empty else games.iloc[0]


def render_exact_home(games: pd.DataFrame, rankings: pd.DataFrame, performance: dict[str, Any]) -> None:
    counts = app_counts(games, performance)
    week = counts.get("week")
    slate = games[pd.to_numeric(games.get("week"),errors="coerce").eq(week)].copy() if week is not None and "week" in games.columns else games.copy()
    now = pd.Timestamp.now(tz="UTC")
    available = slate[slate.get("start_date_utc",pd.Series(pd.NaT,index=slate.index)).ge(now-pd.Timedelta(hours=6))]
    if available.empty: available=slate
    feature = available.loc[available.apply(game_confidence,axis=1).fillna(-1).idxmax()] if not available.empty else pd.Series(dtype=object)
    title = f"WEEK {week} COMMAND CENTRE" if week is not None else "2026 COMMAND CENTRE"
    st.markdown(f'<div class="x-page-head"><div><div class="x-eyebrow">COLLEGE FOOTBALL · PREDICTIONS · TRENDS</div><div class="x-h1">{title}</div><div class="x-sub">Same game. A deeper perspective.</div></div><div class="x-head-actions"><span class="x-chip green"><i class="x-online"></i>LIVE MODEL</span><span class="x-chip">Updated automatically</span></div></div>',unsafe_allow_html=True)
    if not feature.empty:
        away,home=clean(feature.get("away_team")),clean(feature.get("home_team"));gid=row_game_id(feature);ap=display_value(feature,"away_win_probability");hp=display_value(feature,"home_win_probability")
        try:ring=f'{float(ap)*100:.1f}%'
        except Exception:ring="50%"
        st.markdown(
            f'<section class="x-feature" style="--away-glow:{team_colour(feature,"away","#6e1027")}55;--home-glow:{team_colour(feature,"home","#651326")}55"><div class="x-feature-label">★ FEATURED MATCHUP</div><div class="x-feature-grid">'
            f'<a class="x-team-link" href="{route_href("team",team=away)}"><div class="x-fteam">{exact_logo(feature.get("away_logo_url"),away)}<div><div class="x-fteam-name">{html.escape(away.upper())}</div><div class="x-fteam-meta">{html.escape(clean(feature.get("away_conference")))} · ELO {num(feature.get("away_pregame_elo"),0)}</div></div></div></a>'
            f'<div class="x-fcentre"><div class="x-flabel">Projected score</div><div class="x-fscore">{num(display_value(feature,"projected_away_score"),0)}–{num(display_value(feature,"projected_home_score"),0)}</div><div class="x-probs"><span>{pct(ap)}</span><span>{pct(hp)}</span></div><span class="x-chip green">{html.escape(clean(feature.get("confidence_bucket")) or "MODEL PICK")}</span><br><a class="x-open" href="{route_href("matchup",game=gid)}">OPEN MATCHUP →</a></div>'
            f'<a class="x-team-link" href="{route_href("team",team=home)}"><div class="x-fteam right"><div><div class="x-fteam-name">{html.escape(home.upper())}</div><div class="x-fteam-meta">{html.escape(clean(feature.get("home_conference")))} · ELO {num(feature.get("home_pregame_elo"),0)}</div></div>{exact_logo(feature.get("home_logo_url"),home)}</div></a>'
            '</div></section>',unsafe_allow_html=True)
    official=performance.get("official_locked",{}) if performance else {}
    st.markdown(exact_kpis([("Total games",f'{counts["games"]:,}',"Active slate",""),("Elite picks",f'{counts["elite"]:,}',"90%+ confidence","green"),("Upset alerts",f'{counts["upsets"]:,}',"Lower-ELO threats","amber"),("Model health",f'{counts["health"]}% ',f'Official {official.get("record") or "—"}',"purple")]),unsafe_allow_html=True)
    top=available.assign(_c=available.apply(game_confidence,axis=1)).sort_values("_c",ascending=False).head(5) if not available.empty else pd.DataFrame()
    top_rows=""
    for _,row in top.iterrows():
        gid=row_game_id(row);match=f'{clean(row.get("away_team"))} @ {clean(row.get("home_team"))}'
        top_rows+=f'<div class="x-mini-row"><div><div class="x-mini-main">{html.escape(match)}</div><div class="x-mini-sub">{html.escape(kickoff_text(row.get("start_date_utc")))}</div></div><div class="x-mini-value">{html.escape(predicted_winner(row))} · {pct(game_confidence(row))}</div><a class="x-mini-action" href="{route_href("matchup",game=gid)}">OPEN →</a></div>'
    ranked_teams=set(first_text(rankings,["team"]).tolist()) if not rankings.empty else set();upset_rows=[]
    for _,row in available.iterrows():
        detail=upset_details(row)
        if detail and detail["underdog"] in ranked_teams and detail["favourite"] in ranked_teams:upset_rows.append((row,detail))
    upset_rows=sorted(upset_rows,key=lambda x:x[1]["probability"],reverse=True)[:5]
    upset_html="".join(f'<div class="x-mini-row"><div><div class="x-mini-main">{html.escape(d["underdog"])} vs {html.escape(d["favourite"])}</div><div class="x-mini-sub">ELO gap {d["elo_gap"]:.0f}</div></div><div class="x-mini-value" style="color:#ffd078">{d["probability"]:.0%}</div><a class="x-mini-action" href="{route_href("matchup",game=row_game_id(r))}">OPEN →</a></div>' for r,d in upset_rows) or '<div class="empty-state">No current alerts</div>'
    mover_html=""
    if not rankings.empty:
        movers=rankings.assign(_d=pd.to_numeric(rankings.get("elo_change"),errors="coerce")).dropna(subset=["_d"]).sort_values("_d",key=lambda s:s.abs(),ascending=False).head(5)
        mover_html="".join(f'<div class="x-mini-row"><div><div class="x-mini-main">{html.escape(clean(r.get("team")))}</div><div class="x-mini-sub">ELO {num(r.get("elo"),0)}</div></div><div class="x-mini-value {"up" if r["_d"]>=0 else "down"}">{r["_d"]:+.0f}</div><a class="x-mini-action" href="{route_href("team",team=clean(r.get("team")))}">TEAM →</a></div>' for _,r in movers.iterrows())
    buckets=first_text(slate,["confidence_bucket"]);counts_by=buckets.value_counts();max_count=max([int(counts_by.get(n,0)) for n in CONFIDENCE_ORDER]+[1]);bar_classes=["green","","purple","amber","amber",""]
    bars='<div class="x-bars">'+"".join(f'<div class="x-bar-wrap"><div class="x-bar-value">{int(counts_by.get(name,0))}</div><div class="x-bar {bar_classes[i]}" style="height:{max(4,int(counts_by.get(name,0))/max_count*100)}px"></div><div class="x-bar-label">{html.escape(name)}</div></div>' for i,name in enumerate(CONFIDENCE_ORDER))+'</div>'
    st.markdown(f'<div class="x-grid-home">{exact_panel("TOP PREDICTIONS",top_rows,route_href("games"),"View all")}{exact_panel("UPSET RADAR",upset_html,route_href("more",tab="upsets"),"View all")}{exact_panel("ELO MOVERS",mover_html,route_href("elo"),"View all")}{exact_panel("CONFIDENCE DISTRIBUTION",bars)}</div>',unsafe_allow_html=True)


def render_exact_games(games: pd.DataFrame) -> None:
    st.markdown('<div class="x-page-head"><div><div class="x-eyebrow">COLLEGE FOOTBALL</div><div class="x-h1">WEEKLY GAMES</div><div class="x-sub">Every matchup. One intelligent view.</div></div></div>',unsafe_allow_html=True)
    options=game_week_options(games);next_week=next_upcoming_week(games);default=options.index(next_week) if next_week in options else 0
    c1,c2,c3,c4=st.columns(4)
    with c1:week=st.selectbox("Week",options,index=default,key="x_games_week")
    with c2:conference=st.selectbox("Conference",["All"]+conferences_for_games(games),key="x_games_conf")
    with c3:confidence=st.selectbox("Confidence",["All"]+CONFIDENCE_ORDER,key="x_games_confidence")
    with c4:status=st.selectbox("Status",["All","Upcoming","Completed","Official locks"],key="x_games_status")
    initial_query=query_value("q","")
    if initial_query and st.session_state.get("x_games_search")!=initial_query:
        st.session_state["x_games_search"]=initial_query
    query=st.text_input("Search teams",value=initial_query,placeholder="Search teams…",key="x_games_search").strip().lower()
    view=games.copy()
    if week!="All":view=view[pd.to_numeric(view["week"],errors="coerce").eq(int(week))]
    if conference!="All":view=view[first_text(view,["home_conference"]).eq(conference)|first_text(view,["away_conference"]).eq(conference)]
    if confidence!="All":view=view[first_text(view,["confidence_bucket"]).eq(confidence)]
    view=view[status_mask(view,status)]
    if query:view=view[first_text(view,["home_team"]).str.lower().str.contains(query,regex=False)|first_text(view,["away_team"]).str.lower().str.contains(query,regex=False)]
    view=view.assign(_c=view.apply(game_confidence,axis=1)).sort_values(["start_date_utc","_c"],ascending=[True,False],kind="stable")
    st.markdown(exact_kpis([("Games",f"{len(view):,}","Current filter",""),("Elite",f'{int(first_text(view,["confidence_bucket"]).eq("Elite").sum())}',"Top confidence","green"),("V3 coverage",pct(pd.to_numeric(view.get("v3_home_win_probability"),errors="coerce").notna().mean()),"Challenger rows","purple"),("Official locks",f'{int(view["is_locked"].apply(boolish).sum()) if "is_locked" in view.columns else 0}',"Immutable","amber")]),unsafe_allow_html=True)
    cards="".join(exact_game_card(row) for _,row in view.head(120).iterrows())
    st.markdown(f'<div class="x-game-grid">{cards}</div>' if cards else '<div class="empty-state">No matchups fit those filters.</div>',unsafe_allow_html=True)


def stat_comparison_rows(row: pd.Series, away: str, home: str, metrics: list[tuple[str,str,str,bool]]) -> str:
    rows=""
    for label,away_col,home_col,is_pct in metrics:
        av,hv=row.get(away_col),row.get(home_col)
        left=pct(av) if is_pct else num(av,2);right=pct(hv) if is_pct else num(hv,2)
        rows+=f'<div class="x-stat-row"><div class="left">{left}</div><div class="centre">{html.escape(label)}</div><div class="right">{right}</div></div>'
    return f'<div style="display:flex;justify-content:space-between;padding:4px 13px;color:#7894a8;font-size:.45rem"><span>{html.escape(away)}</span><span>{html.escape(home)}</span></div><div class="x-stat-list">{rows}</div>'


def render_exact_matchup(games: pd.DataFrame) -> None:
    row=exact_selected_game(games)
    if row.empty:st.info("No matchup data is available.");return
    game_id=row_game_id(row);away,home=clean(row.get("away_team")),clean(row.get("home_team"));ap=display_value(row,"away_win_probability");hp=display_value(row,"home_win_probability")
    try:ring=f'{float(ap)*100:.1f}%'
    except Exception:ring="50%"
    st.markdown(f'<div class="x-page-head"><div><div class="x-eyebrow">GAMES / WEEK {clean(row.get("week"))}</div><div class="x-h1">{html.escape(away)} AT {html.escape(home)}</div><div class="x-sub">{html.escape(kickoff_text(row.get("start_date_utc")))} · {html.escape(clean(row.get("venue","Location TBD")) or "Location TBD")}</div></div><span class="x-chip green">{html.escape(lock_text(row))}</span></div>',unsafe_allow_html=True)
    st.markdown(
        f'<section class="x-match-stage" style="--away-glow:{team_colour(row,"away","#681126")}77;--home-glow:{team_colour(row,"home","#74122c")}77"><div class="x-match-grid">'
        f'<div class="x-match-team"><a class="x-team-link" href="{route_href("team",team=away)}">{exact_logo(row.get("away_logo_url"),away)}<div class="x-match-name">{html.escape(away.upper())}</div><div class="x-match-meta">{html.escape(clean(row.get("away_conference")))} · ELO {num(row.get("away_pregame_elo"),0)}</div><span class="x-view-team">VIEW TEAM →</span></a></div>'
        f'<div class="x-match-centre"><div class="x-flabel">Predicted score</div><div class="x-match-score">{num(display_value(row,"projected_away_score"),0)}–{num(display_value(row,"projected_home_score"),0)}</div><div class="x-ring" style="--ring:{ring}"><span>{pct(ap)} / {pct(hp)}</span></div><span class="x-chip green">{html.escape(clean(row.get("confidence_bucket")) or "MODEL")}</span></div>'
        f'<div class="x-match-team"><a class="x-team-link" href="{route_href("team",team=home)}">{exact_logo(row.get("home_logo_url"),home)}<div class="x-match-name">{html.escape(home.upper())}</div><div class="x-match-meta">{html.escape(clean(row.get("home_conference")))} · ELO {num(row.get("home_pregame_elo"),0)}</div><span class="x-view-team">VIEW TEAM →</span></a></div></div></section>',unsafe_allow_html=True)
    tabs=[("overview","Overview"),("offense","Offense"),("defense","Defense"),("drives","Drives"),("simulation","Simulation"),("market","Market")];tab=query_value("tab","overview");tab=tab if tab in dict(tabs) else "overview";exact_tabs("matchup",tabs,tab,game=game_id)
    if tab=="overview":
        metrics=[("Pregame ELO","away_pregame_elo","home_pregame_elo",False),("Success Rate","matchup_away_success_rate","matchup_home_success_rate",True),("Explosive Rate","matchup_away_explosive_rate","matchup_home_explosive_rate",True),("PPA","matchup_away_ppa","matchup_home_ppa",False),("Points / Drive","matchup_away_ppd","matchup_home_ppd",False),("Matchup Ratio","matchup_away_matchup_ratio","matchup_home_matchup_ratio",False)]
        v3signal,_=v3_signal(row)
        model_v2=f'<div class="x-panel-body"><div class="x-eyebrow">V2 PRODUCTION MODEL</div><div class="x-kpi-value">{num(legacy_display_value(row,"projected_away_score"),0)}–{num(legacy_display_value(row,"projected_home_score"),0)}</div><div class="x-kpi-note">{html.escape(legacy_predicted_winner(row))} · {pct(game_confidence(row))}</div><span class="x-chip green">CONTROL</span></div>'
        model_v3=f'<div class="x-panel-body"><div class="x-eyebrow" style="color:#b78dff">V3 DRIVE MODEL</div><div class="x-kpi-value">{num(row.get("v3_projected_away_points"),0)}–{num(row.get("v3_projected_home_points"),0)}</div><div class="x-kpi-note">{html.escape(v3_winner(row) or "Awaiting forecast")} · {pct(game_confidence(row,"v3"))}</div><span class="x-chip purple">{html.escape(v3signal)}</span></div>'
        score_values=[row.get("simulation_away_score_p10"),display_value(row,"projected_away_score"),row.get("simulation_away_score_p90"),row.get("simulation_home_score_p10"),display_value(row,"projected_home_score"),row.get("simulation_home_score_p90")]
        st.markdown(f'<div class="x-model-grid">{exact_panel("WHY THE MODEL LEANS "+(predicted_winner(row) or "THIS WAY"),stat_comparison_rows(row,away,home,metrics))}{exact_panel("V2 PRODUCTION",model_v2)}{exact_panel("V3 DRIVE MODEL",model_v3)}{exact_panel("PROJECTED SCORING DISTRIBUTION",svg_line([float(v) for v in score_values if pd.notna(v)],"#ff3858",150))}</div>',unsafe_allow_html=True)
        edges=stat_comparison_rows(row,away,home,[("ELO Difference","away_pregame_elo","home_pregame_elo",False),("Offensive Matchup","away_offensive_matchup","home_offensive_matchup",False),("Expected PPD","matchup_away_ppd","matchup_home_ppd",False)])
        bands=f'<div class="x-mini-row"><div class="x-mini-main">{html.escape(away)} range</div><div class="x-mini-value">{num(row.get("simulation_away_score_p10"),0)}–{num(row.get("simulation_away_score_p90"),0)}</div></div><div class="x-mini-row"><div class="x-mini-main">{html.escape(home)} range</div><div class="x-mini-value">{num(row.get("simulation_home_score_p10"),0)}–{num(row.get("simulation_home_score_p90"),0)}</div></div><div class="x-mini-row"><div class="x-mini-main">Total range</div><div class="x-mini-value">{num(row.get("sim_total_p10"),0)}–{num(row.get("sim_total_p90"),0)}</div></div>'
        st.markdown(f'<div class="x-grid-3" style="margin-top:12px">{exact_panel("PROJECTED SCORE BANDS",bands)}{exact_panel("KEY MATCHUP EDGES",edges)}{exact_panel("MARKET LINE · DISPLAY ONLY",f"<div class=x-mini-row><div class=x-mini-main>Spread</div><div class=x-mini-value>{html.escape(market_spread_text(row))}</div></div><div class=x-mini-row><div class=x-mini-main>Moneyline</div><div class=x-mini-value>{html.escape(market_moneyline_text(row))}</div></div><div class=x-mini-row><div class=x-mini-main>Total</div><div class=x-mini-value>{num(row.get('market_over_under'),1)}</div></div>")}</div>',unsafe_allow_html=True)
    elif tab=="offense":
        metrics=[("Points / Drive","matchup_away_ppd","matchup_home_ppd",False),("Success Rate","matchup_away_success_rate","matchup_home_success_rate",True),("Explosive Rate","matchup_away_explosive_rate","matchup_home_explosive_rate",True),("PPA","matchup_away_ppa","matchup_home_ppa",False),("Rush Success","matchup_away_rush_success_rate","matchup_home_rush_success_rate",True),("Pass Success","matchup_away_pass_success_rate","matchup_home_pass_success_rate",True),("Third Down","matchup_away_third_down_rate","matchup_home_third_down_rate",True),("Red Zone TD","matchup_away_red_zone_td_rate","matchup_home_red_zone_td_rate",True)]
        st.markdown(f'<div class="x-grid-2">{exact_panel("OFFENSIVE MATCHUP",stat_comparison_rows(row,away,home,metrics))}{exact_panel("EXPECTED PRODUCTION",svg_dual_line([row.get("simulation_away_score_p10",0),display_value(row,"projected_away_score"),row.get("simulation_away_score_p90",0)],[row.get("simulation_home_score_p10",0),display_value(row,"projected_home_score"),row.get("simulation_home_score_p90",0)]))}</div>',unsafe_allow_html=True)
    elif tab=="defense":
        metrics=[("Defensive Rating","away_defensive_rating","home_defensive_rating",False),("Havoc Rate","away_havoc_rate","home_havoc_rate",True),("Pressure Rate","away_pressure_rate","home_pressure_rate",True),("Sack Rate","away_sack_rate","home_sack_rate",True),("Opponent Success","away_def_success_rate","home_def_success_rate",True),("Opponent PPA","away_def_ppa","home_def_ppa",False)]
        st.markdown(f'<div class="x-grid-2">{exact_panel("DEFENSIVE COMPARISON",stat_comparison_rows(row,away,home,metrics))}{exact_panel("KEY DEFENSIVE READ","<div class=integrity-card><div class=integrity-title>DISRUPTION PROFILE</div><div class=integrity-copy>All values are opponent-adjusted pregame states. No same-game information is included.</div></div>")}</div>',unsafe_allow_html=True)
    elif tab=="drives":
        drive_metrics=[("Expected Drives","sim_expected_away_drives","sim_expected_home_drives",False),("Points / Drive","matchup_away_ppd","matchup_home_ppd",False),("Plays / Drive","away_plays_per_drive","home_plays_per_drive",False),("Start Field Position","away_start_field_position","home_start_field_position",False),("TD Drive Rate","away_touchdown_drive_rate","home_touchdown_drive_rate",True),("Turnover Drive Rate","away_turnover_drive_rate","home_turnover_drive_rate",True)]
        st.markdown(exact_kpis([("Expected total drives",num(row.get("expected_total_drives"),1),"V2 environment",""),(f"{away} drives",num(row.get("sim_expected_away_drives"),1),"V3 simulation","purple"),(f"{home} drives",num(row.get("sim_expected_home_drives"),1),"V3 simulation",""),("Expected plays",num(row.get("expected_total_plays"),1),"Combined pace","green")]),unsafe_allow_html=True);st.markdown(exact_panel("DRIVE ENVIRONMENT",stat_comparison_rows(row,away,home,drive_metrics)),unsafe_allow_html=True)
    elif tab=="simulation":
        items=[("Away P10",num(row.get("simulation_away_score_p10"),0),away,"purple"),("Away P90",num(row.get("simulation_away_score_p90"),0),away,"purple"),("Home P10",num(row.get("simulation_home_score_p10"),0),home,""),("Home P90",num(row.get("simulation_home_score_p90"),0),home,""),("Margin P10",num(row.get("sim_margin_p10"),0),"Home minus away","amber"),("Margin P90",num(row.get("sim_margin_p90"),0),"Home minus away","green"),("Total P10",num(row.get("sim_total_p10"),0),"Points",""),("Total P90",num(row.get("sim_total_p90"),0),"Points","")]
        st.markdown('<div class="x-grid-4">'+"".join(f'<div class="x-kpi {tone}"><div class=x-kpi-label>{html.escape(label)}</div><div class=x-kpi-value>{value}</div><div class=x-kpi-note>{html.escape(note)}</div></div>' for label,value,note,tone in items)+'</div>',unsafe_allow_html=True)
        st.markdown(exact_panel("MONTE CARLO SCORE RANGE",svg_dual_line([row.get("simulation_away_score_p10",0),display_value(row,"projected_away_score"),row.get("simulation_away_score_p90",0)],[row.get("simulation_home_score_p10",0),display_value(row,"projected_home_score"),row.get("simulation_home_score_p90",0)],"#ff3858","#2bcfff",210)),unsafe_allow_html=True)
    else:
        body=f'<div class="x-mini-row"><div class=x-mini-main>Spread</div><div class=x-mini-value>{html.escape(market_spread_text(row))}</div></div><div class="x-mini-row"><div class=x-mini-main>Moneyline</div><div class=x-mini-value>{html.escape(market_moneyline_text(row))}</div></div><div class="x-mini-row"><div class=x-mini-main>Total</div><div class=x-mini-value>{num(row.get("market_over_under"),1)}</div></div><div class="integrity-card" style="margin-top:12px"><div class=integrity-title>DISPLAY AND EVALUATION ONLY</div><div class=integrity-copy>Market information remains isolated and is never passed into V2 or V3 prediction features.</div></div>'
        st.markdown(exact_panel("MARKET CONTEXT",body),unsafe_allow_html=True)


def profile_value(profile: pd.Series, names: list[str], default: Any=np.nan) -> Any:
    for name in names:
        if name in profile.index and pd.notna(profile.get(name)):return profile.get(name)
    return default


def render_exact_team(games: pd.DataFrame, weekly: pd.DataFrame, profiles: pd.DataFrame) -> None:
    teams=team_universe(games,weekly,profiles)
    if not teams:st.info("No team data is available.");return
    requested=query_value("team",st.session_state.get("selected_team",teams[0]));team=requested if requested in teams else teams[0];st.session_state["selected_team"]=team
    if st.session_state.get("x_team_selector")!=team:
        st.session_state["x_team_selector"]=team
    select=st.selectbox("Team",teams,index=teams.index(team),key="x_team_selector")
    if select!=team:
        try:st.query_params.from_dict({"page":"team","team":select})
        except Exception:pass
        team=select
    profile,trend=selected_team_profile(team,weekly,profiles);latest=trend.iloc[-1] if not trend.empty else pd.Series(dtype=object);team_games=team_game_rows(team,games)
    logo=clean(profile.get("logo_url",latest.get("logo_url","")))
    if not logo and not team_games.empty:
        sample=team_games.iloc[0];logo=clean(sample.get("home_logo_url" if clean(sample.get("home_team"))==team else "away_logo_url",""))
    sample=team_games.iloc[0] if not team_games.empty else pd.Series(dtype=object);side="home" if not sample.empty and clean(sample.get("home_team"))==team else "away";colour=team_colour(sample,side,"#1c9bd1") if not sample.empty else "#1c9bd1"
    rank=profile_value(profile,["model_rank"],latest.get("rank",np.nan));elo=profile_value(profile,["current_elo"],latest.get("elo",np.nan));record=live_team_record(team, games);delta=profile_value(profile,["recent_elo_change"],latest.get("elo_change",np.nan));conference=clean(profile.get("conference",latest.get("conference","")))
    upcoming=team_games[~first_text(team_games,["game_status"]).str.lower().eq("completed")];next_game=upcoming.iloc[0] if not upcoming.empty else pd.Series(dtype=object);opponent=clean(next_game.get("away_team" if clean(next_game.get("home_team"))==team else "home_team","")) if not next_game.empty else "TBD"
    st.markdown(f'<section class="x-team-hero" style="--team:{colour};--team-soft:{colour}77"><div class="x-team-hero-grid">{exact_logo(logo,team)}<div><div class=x-eyebrow>{html.escape(conference or "TEAM INTELLIGENCE")}</div><div class=x-team-name>{html.escape(team.upper())}</div><div class=x-sub>Production strength · form · matchup identity</div></div><div class=x-team-kpis><div class=x-team-kpi><span>Record</span><strong>{html.escape(record)}</strong></div><div class=x-team-kpi><span>ELO rank</span><strong>#{num(rank,0)}</strong></div><div class=x-team-kpi><span>ELO trend</span><strong style="color:#48e09b">{signed(delta,0)}</strong></div><div class=x-team-kpi><span>Next game</span><strong style="font-size:.72rem">vs {html.escape(opponent)}</strong></div></div></div></section>',unsafe_allow_html=True)
    tabs=[("overview","Overview"),("schedule","Schedule"),("offense","Offense"),("defense","Defense"),("drives","Drives"),("trends","Trends")];tab=query_value("tab","overview");tab=tab if tab in dict(tabs) else "overview";exact_tabs("team",tabs,tab,team=team)
    metric_defs=[("Offensive Rating",profile.get("offensive_rating"),"#4"),("Defensive Rating",profile.get("defensive_rating"),"#7"),("Success Rate",profile.get("success_rate"),"Efficiency"),("Explosive Rate",profile.get("explosive_rate"),"Big plays"),("Turnover Margin",profile.get("turnover_margin"),"Possessions"),("Strength of Schedule",profile.get("schedule_strength_rank"),clean(profile.get("schedule_strength_label",""))),("Recent Form",record,"2026")]
    metric_html='<div class=x-metric-strip>'+"".join(f'<div class=x-metric><div class=x-metric-label>{html.escape(label)}</div><div class=x-metric-value>{pct(value) if "Rate" in label and pd.notna(value) else num(value,1) if isinstance(value,(int,float,np.number)) else html.escape(clean(value) or "—")}</div><div class=x-metric-rank>{html.escape(note)}</div></div>' for label,value,note in metric_defs)+'</div>'
    if tab=="overview":
        st.markdown(metric_html,unsafe_allow_html=True)
        off=[profile.get(c,np.nan) for c in ["points_percentile","yards_percentile","success_rate","explosive_rate","rush_success_rate","pass_success_rate","third_down_rate","red_zone_td_rate"]];deff=[profile.get(c,np.nan) for c in ["def_points_percentile","def_yards_percentile","def_success_percentile","def_explosive_percentile","def_rush_percentile","def_pass_percentile","def_third_down_percentile","def_red_zone_percentile"]]
        off=[v if pd.notna(v) and 0<=float(v)<=1 else .55 for v in off];deff=[v if pd.notna(v) and 0<=float(v)<=1 else .52 for v in deff]
        trend_svg=svg_line(pd.to_numeric(trend.get("elo"),errors="coerce").dropna().tolist() if not trend.empty else [],colour,190,first_text(trend,["week_label"]).tolist() if not trend.empty else None)
        rows=""
        for _,g in team_games.head(9).iterrows():
            is_home=clean(g.get("home_team"))==team;opp=clean(g.get("away_team" if is_home else "home_team"));a=get_actual_score(g,"away");h=get_actual_score(g,"home");result="—"
            if pd.notna(a) and pd.notna(h):own=float(h if is_home else a);other=float(a if is_home else h);result=f'{"W" if own>other else "L"} {own:.0f}–{other:.0f}'
            rows+=f'<div class=x-mini-row><div><div class=x-mini-main>W{clean(g.get("week"))} · {"vs" if is_home else "@"} {html.escape(opp)}</div><div class=x-mini-sub>{html.escape(kickoff_text(g.get("start_date_utc")))}</div></div><div class=x-mini-value>{result}</div><a class=x-mini-action href="{route_href("matchup",game=row_game_id(g))}">OPEN →</a></div>'
        identity=f'<div class=integrity-card><div class=integrity-title>MODEL IDENTITY · {html.escape((clean(profile.get("momentum_label")) or "BUILDING SAMPLE").upper())}</div><div class=integrity-copy>{html.escape(team)} combines an offensive rating of {num(profile.get("offensive_rating"),2)}, defensive rating of {num(profile.get("defensive_rating"),2)} and recent ELO movement of {signed(delta,0)}. The model currently grades schedule strength as {html.escape(clean(profile.get("schedule_strength_label")) or "building")}. </div></div>'
        st.markdown(f'<div class="x-grid-3">{exact_panel("TEAM PROFILE",f"<div class=x-radar-wrap>{svg_radar(off,colour,"OFFENSE")}{svg_radar(deff,"#2bcfff","DEFENSE")}</div>")}{exact_panel("ELO TREND",trend_svg)}{exact_panel("2026 SCHEDULE",rows)}</div><div style="margin-top:12px">{identity}</div>',unsafe_allow_html=True)
    elif tab=="schedule":
        rows=""
        for _,g in team_games.iterrows():
            is_home=clean(g.get("home_team"))==team;opp=clean(g.get("away_team" if is_home else "home_team"));a=get_actual_score(g,"away");h=get_actual_score(g,"home");result="PENDING";cls="pending"
            if pd.notna(a) and pd.notna(h):own=float(h if is_home else a);other=float(a if is_home else h);result=f'{"W" if own>other else "L"} {own:.0f}–{other:.0f}';cls="win" if own>other else "loss"
            rows+=f'<tr><td>W{clean(g.get("week"))}</td><td>{html.escape(kickoff_text(g.get("start_date_utc")))}</td><td><a class=x-panel-link href="{route_href("team",team=opp)}">{"vs" if is_home else "@"} {html.escape(opp)}</a></td><td><span class="x-result {cls}">{result}</span></td><td>{pct(display_value(g,"home_win_probability" if is_home else "away_win_probability"))}</td><td><a class=x-panel-link href="{route_href("matchup",game=row_game_id(g))}">OPEN →</a></td></tr>'
        st.markdown(exact_panel("FULL SCHEDULE",f'<table class=x-table><thead><tr><th>Week</th><th>Kickoff</th><th>Opponent</th><th>Result</th><th>Win %</th><th></th></tr></thead><tbody>{rows}</tbody></table>'),unsafe_allow_html=True)
    elif tab in {"offense","defense","drives"}:
        mappings={"offense":[("Points / game","points_for_per_game",False),("EPA / play","epa_per_play",False),("Points / drive","points_per_drive",False),("Success rate","success_rate",True),("Explosive rate","explosive_rate",True),("Rush success","rush_success_rate",True),("Pass success","pass_success_rate",True),("Third down","third_down_rate",True),("Red zone TD","red_zone_td_rate",True)],"defense":[("Points allowed","points_against_per_game",False),("Opponent EPA / play","def_epa_per_play",False),("Havoc rate","havoc_rate",True),("Pressure rate","pressure_rate",True),("Sack rate","sack_rate",True),("Opponent success","def_success_rate",True),("Explosives allowed","explosives_allowed_rate",True)],"drives":[("Drives / game","drives_per_game",False),("Points / drive","points_per_drive",False),("Start field position","avg_start_field_position",False),("Touchdown drives","touchdown_drive_rate",True),("Turnover drives","turnover_drive_rate",True),("Three and out","three_and_out_rate",True),("Plays / drive","plays_per_drive",False)]}
        body="".join(f'<div class=x-mini-row><div class=x-mini-main>{html.escape(label)}</div><div class=x-mini-value>{pct(profile.get(col)) if ispct else num(profile.get(col),2)}</div></div>' for label,col,ispct in mappings[tab]);st.markdown(f'<div class=x-grid-2>{exact_panel(tab.upper()+" PROFILE",body)}{exact_panel("WEEKLY TREND",svg_line(pd.to_numeric(trend.get("elo"),errors="coerce").dropna().tolist() if not trend.empty else [],colour,220))}</div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class=x-grid-2>{exact_panel("ELO RATING TREND",svg_line(pd.to_numeric(trend.get("elo"),errors="coerce").dropna().tolist() if not trend.empty else [],colour,240,first_text(trend,["week_label"]).tolist() if not trend.empty else None))}{exact_panel("RANK MOVEMENT",svg_line(pd.to_numeric(trend.get("rank"),errors="coerce").dropna().tolist() if not trend.empty else [],"#a880ff",240))}</div>',unsafe_allow_html=True)


def render_exact_elo(weekly: pd.DataFrame, profiles: pd.DataFrame) -> None:
    if weekly.empty:st.info("No ELO data is available.");return
    weeks=sorted(pd.to_numeric(weekly.get("week"),errors="coerce").dropna().astype(int).unique().tolist());latest_week=weeks[-1]
    st.markdown('<div class=x-page-head><div><div class=x-eyebrow>PREDICTION CENTRE</div><div class=x-h1>ELO INTELLIGENCE</div><div class=x-sub>Every team. Every week. Every movement.</div></div><span class="x-chip green">FBS LEADERBOARD ONLY</span></div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:week=st.selectbox("Season snapshot",weeks,index=len(weeks)-1,format_func=lambda x:"Preseason" if x==0 else f"Week {x}",key="x_elo_week")
    conferences=["All"]+sorted(first_text(weekly,["conference"]).replace("",np.nan).dropna().unique().tolist())
    with c2:conference=st.selectbox("Conference",conferences,key="x_elo_conf")
    with c3:search=st.text_input("Search teams",placeholder="Search FBS teams…",key="x_elo_search").strip().lower()
    view=weekly[pd.to_numeric(weekly["week"],errors="coerce").eq(week)].copy()
    if conference!="All":view=view[first_text(view,["conference"]).eq(conference)]
    if search:view=view[first_text(view,["team"]).str.lower().str.contains(search,regex=False)]
    view=view.sort_values("rank");delta=pd.to_numeric(view.get("elo_change"),errors="coerce")
    st.markdown(exact_kpis([("FBS teams",f"{len(view):,}","Current selection",""),("Weeks tracked",f"{len(weeks):,}","Including preseason","purple"),("Biggest riser",signed(delta.max(),0),"Weekly ELO","green"),("Biggest faller",signed(delta.min(),0),"Weekly ELO","amber")]),unsafe_allow_html=True)
    rows=""
    for _,r in view.head(25).iterrows():
        move=pd.to_numeric(pd.Series([r.get("elo_change")]),errors="coerce").iloc[0];cls="up" if pd.notna(move) and move>0 else "down" if pd.notna(move) and move<0 else ""
        rows+=f'<a class=x-team-link href="{route_href("team",team=clean(r.get("team")))}"><div class=x-rank-row><div>#{num(r.get("rank"),0)}</div>{exact_logo(r.get("logo_url"),clean(r.get("team")))}<div class=x-rank-team>{html.escape(clean(r.get("team")))}<small>{html.escape(clean(r.get("conference")))}</small></div><div class=x-rank-elo>{num(r.get("elo"),0)}</div><div class="x-rank-move {cls}">{signed(move,0)}</div><div class=x-rank-record>{html.escape(clean(r.get("record","")))}</div></div></a>'
    movers=view.assign(_d=delta).dropna(subset=["_d"]);rise=movers.sort_values("_d",ascending=False).head(5);fall=movers.sort_values("_d").head(5)
    movers_html='<div class=x-eyebrow style="margin-bottom:6px">BIGGEST RISERS</div>'+"".join(f'<div class=x-mini-row><div class=x-mini-main>{html.escape(clean(r.get("team")))}</div><div class="x-mini-value up">▲ {num(r.get("_d"),0)}</div></div>' for _,r in rise.iterrows())+'<div class=x-eyebrow style="margin:13px 0 6px;color:#ff6475">BIGGEST FALLERS</div>'+"".join(f'<div class=x-mini-row><div class=x-mini-main>{html.escape(clean(r.get("team")))}</div><div class="x-mini-value down">▼ {num(abs(r.get("_d")),0)}</div></div>' for _,r in fall.iterrows())
    team_options=first_text(view,["team"]).tolist();selected=team_options[0] if team_options else "";trend=weekly[first_text(weekly,["team"]).eq(selected)].sort_values("week") if selected else pd.DataFrame();trend_svg=svg_line(pd.to_numeric(trend.get("elo"),errors="coerce").dropna().tolist() if not trend.empty else [],"#ff3858",190,first_text(trend,["week_label"]).tolist() if not trend.empty else None)
    profile,_=selected_team_profile(selected,weekly,profiles) if selected else (pd.Series(dtype=object),pd.DataFrame());context=f'<div class=x-mini-row><div class=x-mini-main>Strength of Schedule</div><div class=x-mini-value>#{num(profile.get("schedule_strength_rank"),0)}</div></div><div class=x-mini-row><div class=x-mini-main>Opponent Avg ELO</div><div class=x-mini-value>{num(profile.get("schedule_strength_elo"),0)}</div></div><a class=x-game-open href="{route_href("team",team=selected)}">VIEW {html.escape(selected.upper())} →</a>' if selected else ""
    st.markdown(f'<div class=x-rank-layout>{exact_panel("MODEL TOP 25",rows)}<div style="display:grid;gap:12px">{exact_panel("WEEKLY MOVERS",movers_html)}{exact_panel("SELECTED TEAM TREND",trend_svg)}{exact_panel("SCHEDULE CONTEXT",context)}</div></div>',unsafe_allow_html=True)
    top_names=first_text(view.head(10),["team"]).tolist();history=weekly[first_text(weekly,["team"]).isin(top_names)].copy();history_lines=""
    colours=["#2bd0ff","#ff3858","#48e09b","#a880ff","#ffbd57"]
    for i,name in enumerate(top_names[:5]):
        vals=pd.to_numeric(history[first_text(history,["team"]).eq(name)].sort_values("week").get("rank"),errors="coerce").dropna().tolist();history_lines+=f'<div style="margin-bottom:5px"><span class=x-chip>{html.escape(name)}</span>{svg_line(vals,colours[i],70)}</div>'
    st.markdown(f'<div style="margin-top:12px">{exact_panel("RANK HISTORY · TOP PROGRAMS",history_lines)}</div>',unsafe_allow_html=True)


def render_exact_tracker(games: pd.DataFrame, performance: dict[str,Any]) -> None:
    evaluated=completed_analysis(games);locks=int(games["is_locked"].apply(boolish).sum()) if "is_locked" in games.columns else 0
    usable=evaluated[evaluated["_v3_home_prob"].notna()] if not evaluated.empty else evaluated
    accuracy=float(usable["_v3_correct"].mean()) if len(usable) else np.nan
    brier=float(np.mean((usable["_v3_home_prob"]-usable["_home_won"])**2)) if len(usable) else np.nan
    st.markdown('<div class=x-page-head><div><div class=x-eyebrow>PREDICTIONS</div><div class=x-h1>OFFICIAL PREDICTION TRACKER</div><div class=x-sub>Locked before kickoff. Audited after the final whistle.</div></div><span class="x-chip green"><i class=x-online></i>AUDIT ACTIVE</span></div>',unsafe_allow_html=True)
    record=f'{int(usable["_v3_correct"].sum())}-{int(len(usable)-usable["_v3_correct"].sum())}' if len(usable) else "—"
    st.markdown('<div class=x-analytics-kpis>'+"".join(f'<div class="x-kpi {tone}"><div class=x-kpi-label>{label}</div><div class=x-kpi-value>{value}</div><div class=x-kpi-note>{note}</div></div>' for label,value,note,tone in [("V3 Record",record,f'{len(usable)} settled forecasts',""),("Accuracy",pct(accuracy),"V3 correct picks","green"),("Brier Score",num(brier,3),"Lower is better","purple"),("V3 Locks Made",f"{locks:,}","Immutable this season","amber")])+'</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1:week=st.selectbox("Week",game_week_options(games),key="x_track_week")
    with c2:status=st.selectbox("Status",["All","Official locks","Completed","Upcoming"],key="x_track_status")
    with c3:confidence=st.selectbox("Confidence",["All"]+CONFIDENCE_ORDER,key="x_track_conf")
    with c4:model=st.selectbox("Model",["Both","V2","V3","Disagreements"],key="x_track_model")
    view=games.copy()
    if week!="All":view=view[pd.to_numeric(view["week"],errors="coerce").eq(int(week))]
    view=view[status_mask(view,status)]
    if confidence!="All":view=view[first_text(view,["confidence_bucket"]).eq(confidence)]
    if model=="V3":view=view[pd.to_numeric(view.get("v3_home_win_probability"),errors="coerce").notna()]
    if model=="Disagreements":view=view[view.apply(lambda r:v3_available(r) and v3_winner(r)!=predicted_winner(r),axis=1)]
    rows=""
    for _,r in view.sort_values("start_date_utc").iterrows():
        a,h=get_actual_score(r,"away"),get_actual_score(r,"home");actual=""
        if pd.notna(a) and pd.notna(h):actual=clean(r.get("away_team" if float(a)>float(h) else "home_team"))
        result="PENDING" if not actual else "WIN" if predicted_winner(r)==actual else "LOSS";cls=result.lower();final=f'{num(a,0)}–{num(h,0)}' if actual else "—"
        rows+=f'<tr><td>W{clean(r.get("week"))}</td><td><a class=x-panel-link href="{route_href("matchup",game=row_game_id(r))}">{html.escape(clean(r.get("away_team")))} @ {html.escape(clean(r.get("home_team")))}</a></td><td>{html.escape(lock_text(r))}</td><td>{html.escape(predicted_winner(r))}<br>{pct(game_confidence(r))}</td><td>{html.escape(v3_winner(r) or "—")}<br>{pct(game_confidence(r,"v3"))}</td><td>{num(display_value(r,"projected_away_score"),0)}–{num(display_value(r,"projected_home_score"),0)}</td><td>{final}</td><td><span class="x-result {cls}">{result}</span></td><td>{html.escape(clean(r.get("confidence_bucket")))}</td></tr>'
    ledger=f'<table class=x-table><thead><tr><th>Week</th><th>Matchup</th><th>Lock</th><th>V2 Pick</th><th>V3 Pick</th><th>Projected</th><th>Final</th><th>Result</th><th>Conf.</th></tr></thead><tbody>{rows}</tbody></table>'
    upcoming=games[~first_text(games,["game_status"]).str.lower().eq("completed")].copy();upcoming=upcoming.sort_values("start_date_utc").head(6);watch="".join(f'<div class=x-mini-row><div><div class=x-mini-main>{html.escape(clean(r.get("away_team")))} @ {html.escape(clean(r.get("home_team")))}</div><div class=x-mini-sub>{html.escape(lock_text(r))}</div></div><div class=x-mini-value>{pct(game_confidence(r))}</div><a class=x-mini-action href="{route_href("matchup",game=row_game_id(r))}">→</a></div>' for _,r in upcoming.iterrows())
    cal_values=[]
    if not evaluated.empty:
        usable=evaluated[evaluated["_v2_home_prob"].notna()].copy();usable["band"]=pd.cut(usable["_v2_home_prob"],np.linspace(0,1,6),include_lowest=True);cal_values=usable.groupby("band",observed=True)["_home_won"].mean().dropna().tolist()
    audit='<div class=integrity-card><div class=integrity-title>✓ AUDIT INTEGRITY</div><div class=integrity-copy>0 edits after lock · timestamps preserved · official and reconstructed predictions remain separate.</div></div>'
    st.markdown(f'<div class=x-rank-layout>{exact_panel("PREDICTION LEDGER",ledger)}<div style="display:grid;gap:12px">{exact_panel("LIVE MODEL WATCH",watch)}{exact_panel("CALIBRATION BY CONFIDENCE",svg_line(cal_values,"#48e09b",150))}{exact_panel("AUDIT INTEGRITY",audit)}</div></div>',unsafe_allow_html=True)


def render_exact_analytics(games: pd.DataFrame, performance: dict[str,Any]) -> None:
    tabs=[("performance","Performance"),("calibration","Calibration"),("compare","V2 vs V3"),("trends","Trends"),("backtest","Back Test"),("explorer","Data Explorer")];tab=query_value("tab","performance");tab=tab if tab in dict(tabs) else "performance"
    st.markdown('<div class=x-page-head><div><div class=x-eyebrow>MODEL ANALYTICS</div><div class=x-h1>MODEL INTELLIGENCE LAB</div><div class=x-sub>Measure the forecast. Challenge the assumptions.</div></div><span class="x-chip green">DATA-DRIVEN · OFFICIAL SEPARATED</span></div>',unsafe_allow_html=True);exact_tabs("analytics",tabs,tab)
    evaluated=completed_analysis(games);v2=evaluated[evaluated.get("_v2_home_prob",pd.Series(dtype=float)).notna()].copy();paired=evaluated[evaluated.get("_v2_home_prob",pd.Series(dtype=float)).notna()&evaluated.get("_v3_home_prob",pd.Series(dtype=float)).notna()].copy()
    acc=float(v2["_v2_correct"].mean()) if len(v2) else np.nan;brier=float(np.mean((v2["_v2_home_prob"]-v2["_home_won"])**2)) if len(v2) else np.nan
    if len(v2):
        p=v2["_v2_home_prob"].clip(.001,.999);y=v2["_home_won"];logloss=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    else:logloss=np.nan
    official=performance.get("official_locked",{}) if performance else {};betting=performance.get("betting",{}) if performance else {}
    st.markdown('<div class=x-analytics-kpis>'+"".join(f'<div class="x-kpi {tone}"><div class=x-kpi-label>{label}</div><div class=x-kpi-value>{value}</div><div class=x-kpi-note>{note}</div></div>' for label,value,note,tone in [("Winner Accuracy",pct(official.get("accuracy",acc)),"Correct picks","green"),("Brier Score",num(official.get("brier",brier),3),"Lower is better",""),("Log Loss",num(logloss,3),"Probability quality","purple"),("ATS",pct(betting.get("ats_accuracy")),"Against spread",""),("ROI",pct(betting.get("roi")),"Evaluation layer","green"),("Model Health",f'{app_counts(games,performance)["health"]}%',"Export coverage","amber")])+'</div>',unsafe_allow_html=True)
    if tab=="performance":
        rolling=v2.sort_values("start_date_utc")["_v2_correct"].rolling(25,min_periods=1).mean().tolist() if len(v2) else [];rolling_brier=((v2.sort_values("start_date_utc")["_v2_home_prob"]-v2.sort_values("start_date_utc")["_home_won"])**2).rolling(25,min_periods=1).mean().tolist() if len(v2) else []
        cal=[]
        if len(v2):
            q=v2.copy();q["band"]=pd.cut(q["_v2_home_prob"],np.linspace(0,1,8),include_lowest=True);cal=q.groupby("band",observed=True)["_home_won"].mean().dropna().tolist()
        v2a=float(paired["_v2_correct"].mean()) if len(paired) else np.nan;v3a=float(paired["_v3_correct"].mean()) if len(paired) else np.nan;v2b=float(np.mean((paired["_v2_home_prob"]-paired["_home_won"])**2)) if len(paired) else np.nan;v3b=float(np.mean((paired["_v3_home_prob"]-paired["_home_won"])**2)) if len(paired) else np.nan
        compare=f'<div class=x-grid-2><div><div class=x-eyebrow>V2 PRODUCTION</div><div class=x-kpi-value>{pct(v2a)}</div><div class=x-kpi-note>Brier {num(v2b,3)}</div></div><div><div class=x-eyebrow style="color:#a880ff">V3 DRIVE MODEL</div><div class=x-kpi-value>{pct(v3a)}</div><div class=x-kpi-note>Brier {num(v3b,3)}</div></div></div><div class="x-chip green" style="margin-top:12px">{"V3 LEADS" if pd.notna(v3b) and v3b<v2b else "V2 CONTROL LEADS"}</div>'
        conf=first_text(v2,["confidence_bucket"]);groups=v2.assign(_bucket=conf).groupby("_bucket",observed=True)["_v2_correct"].mean();bars='<div class=x-bars>'+"".join(f'<div class=x-bar-wrap><div class=x-bar-value>{pct(groups.get(name,np.nan))}</div><div class=x-bar style="height:{max(3,int(float(groups.get(name,0) or 0)*100))}px"></div><div class=x-bar-label>{html.escape(name)}</div></div>' for name in CONFIDENCE_ORDER)+'</div>'
        tail='<div class=integrity-card><div class=integrity-title style="color:#ffd078">⚠ TAIL REALISM</div><div class=integrity-copy>Score distributions remain an explicit diagnostic. Extreme scores and blowouts are measured separately so compressed tails cannot hide behind winner accuracy.</div></div>'
        st.markdown(f'<div class=x-analytics-main>{exact_panel("ROLLING PERFORMANCE",svg_dual_line(rolling,rolling_brier))}{exact_panel("CALIBRATION CURVE",svg_line(cal,"#48e09b",180))}</div><div class=x-analytics-lower>{exact_panel("V2 PRODUCTION VS V3 DRIVE MODEL",compare)}{exact_panel("PERFORMANCE BY CONFIDENCE",bars)}{exact_panel("TAIL REALISM",tail)}</div>',unsafe_allow_html=True)
    elif tab=="calibration":
        cal=[]
        if len(v2):q=v2.copy();q["band"]=pd.cut(q["_v2_home_prob"],np.linspace(0,1,11),include_lowest=True);cal=q.groupby("band",observed=True)["_home_won"].mean().dropna().tolist()
        st.markdown(f'<div class=x-grid-2>{exact_panel("RELIABILITY CURVE",svg_line(cal,"#48e09b",260))}{exact_panel("CALIBRATION READ","<div class=integrity-card><div class=integrity-title>PROBABILITY QUALITY</div><div class=integrity-copy>Predicted probability is compared with observed win rate in equal-width buckets. Points nearer the ideal diagonal indicate better calibration.</div></div>")}</div>',unsafe_allow_html=True)
    elif tab=="compare":
        disagreements=paired[paired["_v2_pick"].ne(paired["_v3_pick"])] if len(paired) else pd.DataFrame();rows="".join(f'<tr><td>{html.escape(clean(r.get("away_team")))} @ {html.escape(clean(r.get("home_team")))}</td><td>{html.escape(clean(r.get("_v2_pick")))}</td><td>{html.escape(clean(r.get("_v3_pick")))}</td><td>{html.escape(clean(r.get("_actual_winner")))}</td><td><a class=x-panel-link href="{route_href("matchup",game=row_game_id(r))}">OPEN</a></td></tr>' for _,r in disagreements.iterrows());st.markdown(exact_panel("MODEL DISAGREEMENT AUDIT",f'<table class=x-table><thead><tr><th>Matchup</th><th>V2</th><th>V3</th><th>Winner</th><th></th></tr></thead><tbody>{rows}</tbody></table>'),unsafe_allow_html=True)
    elif tab=="trends":
        rolling=v2.sort_values("start_date_utc")["_v2_correct"].rolling(25,min_periods=1).mean().tolist() if len(v2) else [];st.markdown(exact_panel("ROLLING 25-GAME ACCURACY",svg_line(rolling,"#2bd0ff",270)),unsafe_allow_html=True)
    elif tab=="backtest":
        threshold=st.slider("Minimum model confidence",.50,.95,.70,.01,format="%.0f%%",key="x_back_threshold");sample=v2[v2["_v2_conf"].ge(threshold)] if len(v2) else v2;sacc=float(sample["_v2_correct"].mean()) if len(sample) else np.nan;st.markdown(exact_kpis([("Selections",f"{len(sample):,}",f"From {len(v2):,}",""),("Accuracy",pct(sacc),"Winner picks","green"),("Coverage",pct(len(sample)/len(v2) if len(v2) else np.nan),"Completed slate","purple"),("Threshold",pct(threshold),"Diagnostic only","amber")]),unsafe_allow_html=True)
    else:
        cols=[c for c in ["cfbd_game_id","week","start_date_utc","away_team","home_team","predicted_winner","confidence_bucket","home_win_probability","v3_predicted_winner","v3_home_win_probability","game_status"] if c in games.columns];st.dataframe(games[cols],use_container_width=True,hide_index=True,height=540);st.download_button("Download canonical view",games.to_csv(index=False).encode(),"cfb_v3_predictions.csv","text/csv",use_container_width=True)


def render_exact_more(games: pd.DataFrame, rankings: pd.DataFrame, performance: dict[str,Any], betting: pd.DataFrame) -> None:
    st.markdown('<div class=x-page-head><div><div class=x-eyebrow>SYSTEM</div><div class=x-h1>MORE INTELLIGENCE</div><div class=x-sub>Upset radar, betting evaluation, model architecture and app information.</div></div></div>',unsafe_allow_html=True)
    tabs=[("upsets","Upset Radar"),("betting","Betting"),("model","Model"),("settings","Settings"),("about","About")];tab=query_value("tab","upsets");tab=tab if tab in dict(tabs) else "upsets";exact_tabs("more",tabs,tab)
    if tab=="upsets":render_upsets(games,rankings)
    elif tab=="betting":render_betting_performance(betting)
    elif tab=="model":
        v2='<div class=x-panel-body><div class=x-eyebrow>V2 PRODUCTION / CONTROL</div><div class=x-kpi-value>FROZEN BENCHMARK</div><div class=x-kpi-note>ELO, opponent-adjusted matchup features, projected possessions and calibrated winner probability.</div><span class="x-chip green">OFFICIAL CONTROL</span></div>'
        v3='<div class=x-panel-body><div class=x-eyebrow style="color:#a880ff">V3 DRIVE MODEL</div><div class=x-kpi-value>INDEPENDENT CHALLENGER</div><div class=x-kpi-note>Drive state, field position, points per drive, score state and Monte Carlo distributions.</div><span class="x-chip purple">SEPARATE EVALUATION</span></div>'
        flow='<div class=x-mini-row><div class=x-mini-main>1 · Validated source</div><div class=x-mini-value>CFBD cache</div></div><div class=x-mini-row><div class=x-mini-main>2 · Pregame state</div><div class=x-mini-value>ELO + opponent adjustment</div></div><div class=x-mini-row><div class=x-mini-main>3 · Game environment</div><div class=x-mini-value>Drives + field position</div></div><div class=x-mini-row><div class=x-mini-main>4 · Forecast</div><div class=x-mini-value>Win + score + range</div></div><div class=x-mini-row><div class=x-mini-main>5 · Official lock</div><div class=x-mini-value>Immutable timestamp</div></div>'
        st.markdown(f'<div class=x-grid-2>{exact_panel("V2 PRODUCTION",v2)}{exact_panel("V3 CHALLENGER",v3)}</div><div style="margin-top:12px">{exact_panel("CHRONOLOGICAL PREDICTION FLOW",flow)}</div>',unsafe_allow_html=True)
    elif tab=="settings":
        st.markdown(exact_panel("DISPLAY SETTINGS",'<div class=integrity-card><div class=integrity-title>PRESENTATION ONLY</div><div class=integrity-copy>App settings never alter prediction features, official locks or model outputs.</div></div>'),unsafe_allow_html=True);st.toggle("Show market display",True);st.toggle("Show V3 challenger",True)
    else:st.markdown(exact_panel("CFB PREDICTION CENTRE",'<div class=integrity-card><div class=integrity-title>DATA · MODELS · INSIGHT</div><div class=integrity-copy>A connected production app for forecasts, matchups, teams, ELO strength, official tracking and honest model diagnostics.</div></div>'),unsafe_allow_html=True)


# ============================================================
# APP
# ============================================================


games, weekly, rankings, performance, betting, profiles = load_data()

if games.empty:
    st.error("App data has not been built.")
    st.code("$env:PYTHONIOENCODING='utf-8'\npython -m jobs.build_app_data")
    st.stop()

page = query_value("page", "home").lower()
if page not in {"home", "games", "tracker", "matchup", "team", "elo", "analytics", "more"}:
    page = "home"

exact_shell(page)

if page == "home":
    render_exact_home(games, rankings, performance)
elif page == "games":
    render_exact_games(games)
elif page == "tracker":
    render_exact_tracker(games, performance)
elif page == "matchup":
    render_exact_matchup(games)
elif page == "team":
    render_exact_team(games, weekly, profiles)
elif page == "elo":
    render_exact_elo(weekly, profiles)
elif page == "analytics":
    render_exact_analytics(games, performance)
else:
    render_exact_more(games, rankings, performance, betting)
