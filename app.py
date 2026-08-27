"""
CineMatch — AI-Powered Movie Discovery
========================================
A content-based movie recommendation dashboard built on top of a
TF-IDF + cosine-similarity model trained on movie overviews.

Run with:
    streamlit run app.py

Requires `movies.csv` in the same directory, with at least the columns:
    id, title, overview, release_date, popularity, vote_average, vote_count
"""

import difflib
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# CONFIG
# ============================================================================

CSV_PATH = "movies.csv"
REQUIRED_COLUMNS = [
    "id", "title", "overview", "release_date",
    "popularity", "vote_average", "vote_count",
]
PERIOD_OPTIONS = ["All years", "2020s", "2010s", "2000s", "1990s", "Older"]
SORT_OPTIONS = ["Similarity", "Rating", "Popularity", "Release date"]

# TMDB is used purely as a presentation/enrichment layer for real poster
# artwork. It never feeds into the TF-IDF / cosine-similarity recommender.
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_REQUEST_TIMEOUT = 5  # seconds — never let a poster fetch hang the app
TMDB_POSTER_CACHE_TTL = 60 * 60 * 24  # 24h

# Duotone gradient palette used for the placeholder "poster" tiles.
# No poster URLs exist in the dataset, so cards use generated gradients
# + initials instead of fabricated artwork.
GRADIENT_PALETTE = [
    ("#3a0d3f", "#a8326e"),
    ("#0f1e3d", "#1f6f8b"),
    ("#3a1c2f", "#c94b4b"),
    ("#161327", "#4b3f72"),
    ("#0d2b1f", "#1f8a5f"),
    ("#2b1730", "#8a3ffc"),
    ("#211a12", "#c98a2c"),
    ("#111826", "#3d5a80"),
]

st.set_page_config(
    page_title="CineMatch — AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# STYLE
# ============================================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

        :root{
            --bg:#08080c;
            --bg-alt:#0d0d13;
            --panel:#131320;
            --panel-2:#181828;
            --border: rgba(255,255,255,0.08);
            --border-strong: rgba(255,255,255,0.16);
            --text:#f3f2f7;
            --text-dim:#9a97a8;
            --text-faint:#6b6878;
            --accent:#e6395f;
            --accent-2:#f5b942;
            --accent-glow: rgba(230,57,95,0.35);
            --good:#3fc98a;
        }

        html, body, [class*="css"]{
            font-family:'Inter', -apple-system, sans-serif;
        }

        .stApp{
            background:
                radial-gradient(1200px 600px at 10% -10%, rgba(230,57,95,0.10), transparent 60%),
                radial-gradient(1000px 500px at 100% 0%, rgba(245,185,66,0.06), transparent 55%),
                var(--bg);
            color: var(--text);
        }

        #MainMenu, footer, header[data-testid="stHeader"]{ background: transparent; }
        header[data-testid="stHeader"]{ background: rgba(8,8,12,0.0); }

        .block-container{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1300px; }

        h1, h2, h3, h4 { color: var(--text) !important; font-family:'Inter', sans-serif; letter-spacing:-0.01em; }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"]{
            background: linear-gradient(180deg, #0b0b12 0%, #08080c 100%);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] .block-container{ padding-top: 1.4rem; }
        .brand-mark{
            font-family:'Bebas Neue', sans-serif;
            font-size: 2.1rem;
            letter-spacing: 0.06em;
            background: linear-gradient(90deg, #ffffff 0%, var(--accent-2) 120%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height:1;
            margin-bottom: 0.1rem;
        }
        .brand-sub{ color: var(--text-dim); font-size:0.78rem; letter-spacing:0.04em; margin-bottom:1.3rem; }
        .side-divider{ height:1px; background: var(--border); margin: 1.1rem 0; border:none; }
        .side-label{ color: var(--text-faint); font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:0.15rem; }
        .side-value{ color: var(--text); font-size:0.86rem; font-weight:600; margin-bottom:0.9rem; }
        .side-value .mono{ font-family:'JetBrains Mono', monospace; color: var(--accent-2); }

        /* Radio nav styled as vertical tab list */
        section[data-testid="stSidebar"] div[role="radiogroup"]{ gap: 0.15rem; }
        section[data-testid="stSidebar"] div[role="radiogroup"] label{
            background: transparent;
            border-radius: 10px;
            padding: 0.5rem 0.7rem;
            border: 1px solid transparent;
            transition: all 0.15s ease;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
            background: rgba(255,255,255,0.04);
            border-color: var(--border);
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"]{
            background: linear-gradient(90deg, rgba(230,57,95,0.18), rgba(230,57,95,0.02));
            border-color: rgba(230,57,95,0.35);
        }

        /* ---------- Hero ---------- */
        .hero{
            position:relative;
            padding: 3.2rem 2.6rem;
            border-radius: 22px;
            background:
                radial-gradient(600px 260px at 85% 0%, rgba(230,57,95,0.22), transparent 60%),
                linear-gradient(135deg, #14121c 0%, #0c0b12 100%);
            border: 1px solid var(--border);
            overflow:hidden;
            margin-bottom: 1.8rem;
        }
        .hero::before{
            content:"";
            position:absolute; inset:0;
            background-image: repeating-linear-gradient(90deg, rgba(255,255,255,0.025) 0 1px, transparent 1px 34px);
            pointer-events:none;
        }
        .hero-eyebrow{
            display:inline-flex; align-items:center; gap:0.4rem;
            font-size:0.72rem; letter-spacing:0.14em; text-transform:uppercase;
            color: var(--accent-2); font-weight:700; margin-bottom:0.9rem;
        }
        .hero-title{
            font-family:'Bebas Neue', sans-serif;
            font-size: 3.6rem;
            line-height: 1.02;
            letter-spacing: 0.01em;
            color: #fff;
            margin: 0 0 0.7rem 0;
            max-width: 780px;
        }
        .hero-sub{ color: var(--text-dim); font-size: 1.05rem; max-width: 640px; margin-bottom: 0; }

        /* ---------- Stat strip ---------- */
        .stat-strip{ display:flex; gap:0.9rem; flex-wrap:wrap; margin: 1.6rem 0 0.4rem 0; }
        .stat-box{
            flex:1 1 160px;
            background: var(--panel);
            border:1px solid var(--border);
            border-radius:14px;
            padding: 1rem 1.2rem;
        }
        .stat-box .num{ font-family:'JetBrains Mono', monospace; font-size:1.55rem; font-weight:700; color:#fff; }
        .stat-box .lbl{ color:var(--text-faint); font-size:0.74rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:0.15rem; }
        .stat-box.accent .num{ color: var(--accent-2); }

        /* ---------- Section headers ---------- */
        .section-head{ display:flex; align-items:baseline; gap:0.6rem; margin: 2.2rem 0 0.9rem 0; }
        .section-head .tag{ color: var(--accent); font-family:'JetBrains Mono', monospace; font-size:0.75rem; }
        .section-head h3{ margin:0; font-size:1.35rem; }
        .section-sub{ color: var(--text-faint); font-size:0.88rem; margin-top:-0.5rem; margin-bottom:1rem;}

        /* ---------- Movie grid / cards ---------- */
        .movie-grid{
            display:grid;
            grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
            gap: 1rem;
            margin-bottom: 0.5rem;
        }
        .movie-card{
            position:relative;
            background: linear-gradient(180deg, var(--panel-2) 0%, var(--panel) 100%);
            border:1px solid var(--border);
            border-radius:16px;
            overflow:hidden;
            transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
            display:flex; flex-direction:column;
        }
        .movie-card:hover{
            transform: translateY(-4px);
            border-color: rgba(230,57,95,0.45);
            box-shadow: 0 14px 34px rgba(0,0,0,0.45), 0 0 0 1px rgba(230,57,95,0.12);
        }
        .card-poster{
            aspect-ratio: 2 / 3; position:relative;
            display:flex; align-items:center; justify-content:center;
            overflow:hidden;
        }
        .card-poster img{
            position:absolute; inset:0;
            width:100%; height:100%;
            object-fit:cover; object-position:center top;
            display:block;
        }
        .card-poster .poster-link{ position:absolute; inset:0; display:block; }
        .card-poster .initials{
            font-family:'Bebas Neue', sans-serif;
            font-size:2.4rem; color: rgba(255,255,255,0.92);
            letter-spacing:0.05em;
            text-shadow: 0 2px 10px rgba(0,0,0,0.35);
        }
        .rank-badge{
            position:absolute; top:9px; left:10px;
            background: rgba(8,8,12,0.55);
            border:1px solid rgba(255,255,255,0.25);
            backdrop-filter: blur(4px);
            color:#fff; font-family:'JetBrains Mono', monospace;
            font-size:0.72rem; font-weight:700;
            padding: 0.15rem 0.5rem; border-radius:20px;
        }
        .rating-badge{
            position:absolute; top:9px; right:10px;
            background: rgba(8,8,12,0.65);
            border:1px solid rgba(245,185,66,0.5);
            color: var(--accent-2); font-size:0.74rem; font-weight:700;
            padding: 0.15rem 0.5rem; border-radius:20px;
        }
        .card-body{ padding: 0.85rem 0.95rem 1rem 0.95rem; flex:1; display:flex; flex-direction:column; }
        .card-title{
            font-weight:700; font-size:0.96rem; color:#fff; line-height:1.25;
            margin-bottom:0.35rem;
            display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
            min-height: 2.4em;
        }
        .badge-row{ display:flex; flex-wrap:wrap; gap:0.35rem; margin-bottom:0.55rem; }
        .chip{
            font-size:0.68rem; font-weight:600; padding:0.14rem 0.5rem; border-radius:20px;
            border:1px solid var(--border-strong); color: var(--text-dim);
            font-family:'JetBrains Mono', monospace;
        }
        .chip.sim{ color: var(--accent); border-color: rgba(230,57,95,0.4); background: rgba(230,57,95,0.08); }
        .chip.year{ color: var(--text-dim); }
        .chip.pop{ color: #7ec8e3; border-color: rgba(126,200,227,0.3); }
        .card-overview{
            color: var(--text-faint); font-size:0.79rem; line-height:1.42;
            display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden;
            margin-top:auto;
        }

        /* ---------- KPI cards ---------- */
        .kpi-grid{ display:grid; grid-template-columns: repeat(auto-fit, minmax(170px,1fr)); gap:0.85rem; margin-bottom:0.6rem; }
        .kpi-card{
            background: var(--panel); border:1px solid var(--border); border-radius:14px;
            padding: 1.1rem 1.2rem;
        }
        .kpi-card .kpi-val{ font-family:'JetBrains Mono', monospace; font-size:1.5rem; font-weight:700; color:#fff; }
        .kpi-card .kpi-lbl{ color: var(--text-faint); font-size:0.72rem; text-transform:uppercase; letter-spacing:0.07em; margin-top:0.2rem;}
        .kpi-card .kpi-sub{ color: var(--text-dim); font-size:0.74rem; margin-top:0.3rem; }

        /* ---------- Detail card ---------- */
        .detail-card{
            background: linear-gradient(135deg, var(--panel-2), var(--panel));
            border:1px solid var(--border); border-radius:18px; padding:1.6rem 1.8rem;
            margin-bottom:1.2rem;
        }
        .detail-title{ font-family:'Bebas Neue', sans-serif; font-size:2rem; color:#fff; letter-spacing:0.02em; margin-bottom:0.3rem;}
        .detail-overview{ color: var(--text-dim); font-size:0.92rem; line-height:1.55; margin: 0.7rem 0 0.4rem 0; }

        .film-divider{
            height:22px; margin: 1.6rem 0 1.4rem 0;
            background-image: radial-gradient(circle, rgba(255,255,255,0.14) 3px, transparent 3.2px);
            background-size: 22px 22px; background-position: center;
            border-top:1px solid var(--border); border-bottom:1px solid var(--border);
        }

        .insight-box{
            background: var(--panel); border:1px solid var(--border); border-left:3px solid var(--accent);
            border-radius:12px; padding:1rem 1.3rem; font-size:0.86rem; color:var(--text-dim);
        }
        .insight-box b{ color:#fff; }
        .insight-row{ display:flex; justify-content:space-between; padding:0.3rem 0; border-bottom:1px dashed var(--border); }
        .insight-row:last-child{ border-bottom:none; }
        .insight-row .k{ color:var(--text-faint); }
        .insight-row .v{ color:#fff; font-weight:600; font-family:'JetBrains Mono', monospace; }

        .not-found-box{
            background: rgba(230,57,95,0.06); border:1px solid rgba(230,57,95,0.3);
            border-radius:14px; padding:1.4rem 1.6rem; color: var(--text);
        }

        .flow-step{
            background:var(--panel); border:1px solid var(--border); border-radius:12px;
            padding:0.6rem 1rem; text-align:center; font-family:'JetBrains Mono', monospace;
            font-size:0.82rem; color:#fff;
        }
        .flow-arrow{ text-align:center; color: var(--accent); font-size:1.1rem; margin: 0.15rem 0; }

        /* ---------- Streamlit widget overrides ---------- */
        .stTextInput input, .stNumberInput input{
            background: var(--panel) !important; color:#fff !important;
            border:1px solid var(--border-strong) !important; border-radius:10px !important;
        }
        .stSelectbox div[data-baseweb="select"] > div{
            background: var(--panel) !important; border-color: var(--border-strong) !important; border-radius:10px !important;
        }
        .stSlider [data-baseweb="slider"] div[role="slider"]{ background: var(--accent) !important; }
        .stButton button{
            background: linear-gradient(90deg, var(--accent), #b91d43) !important;
            color:#fff !important; border:none !important; border-radius:10px !important;
            font-weight:700 !important; padding:0.55rem 1.4rem !important;
            box-shadow: 0 6px 18px rgba(230,57,95,0.25) !important;
        }
        .stButton button:hover{ filter: brightness(1.08); }
        div[data-testid="stMetric"]{
            background: var(--panel); border:1px solid var(--border); border-radius:12px; padding:0.8rem 1rem;
        }
        div[data-testid="stMetricValue"]{ color:#fff !important; }
        .stTabs [data-baseweb="tab-list"]{ gap:0.4rem; border-bottom:1px solid var(--border); }
        .stTabs [data-baseweb="tab"]{
            background: transparent; color: var(--text-dim); border-radius:10px 10px 0 0; padding:0.5rem 1rem;
        }
        .stTabs [aria-selected="true"]{ color:#fff !important; border-bottom:2px solid var(--accent) !important; }
        .streamlit-expanderHeader{
            background: var(--panel) !important; border-radius:10px !important; color:#fff !important;
            border:1px solid var(--border) !important;
        }
        div[data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:12px; overflow:hidden; }
        hr{ border-color: var(--border) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

@st.cache_data(show_spinner=False)
def load_data(path: str = CSV_PATH) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load and clean movies.csv. Returns (dataframe, error_message)."""
    try:
        df = pd.read_csv(path, on_bad_lines="skip", skipinitialspace=True, engine="c")
    except FileNotFoundError:
        return None, f"Could not find `{path}`. Make sure it sits next to app.py."
    except Exception as exc:  # noqa: BLE001 - surface any parse failure to the UI
        return None, f"Failed to read `{path}`: {exc}"

    # Normalize column names and drop stray index columns from CSV export.
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, "Missing required column(s): " + ", ".join(missing)

    # Trim whitespace injected by the source export on every text column.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    df["title"] = df["title"].replace({"nan": np.nan, "": np.nan})
    df = df.dropna(subset=["title"]).reset_index(drop=True)

    df["overview"] = df["overview"].replace({"nan": ""}).fillna("")

    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(int)

    df["release_date_parsed"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date_parsed"].dt.year

    # Drop fully-duplicate rows (same id + title) to avoid double-counting.
    df = df.drop_duplicates(subset=["id", "title"], keep="first").reset_index(drop=True)

    return df, None


@st.cache_resource(show_spinner=False)
def build_recommender(df: pd.DataFrame):
    """
    Fit the TF-IDF vectorizer on movie overviews. This mirrors the notebook's
    working logic: `combined_features = movies_data['overview'].astype(str)`.

    NOTE on performance: rather than pre-computing and caching a full dense
    NxN cosine-similarity matrix (which would need >700MB for ~10k movies),
    we cache only the sparse TF-IDF feature matrix and compute similarity
    for a single query row on demand in `get_recommendations`. This keeps
    memory flat regardless of dataset size while still being pure
    TF-IDF + cosine-similarity under the hood.
    """
    corpus = df["overview"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(stop_words="english")
    feature_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, feature_matrix


# ============================================================================
# RECOMMENDATION LOGIC
# ============================================================================

def find_movie(query: str, titles: List[str], n: int = 6, cutoff: float = 0.5) -> List[str]:
    """
    Fuzzy-match a user query against known titles. Never raises on no match.

    Matching is done case-insensitively (mapping back to the original-cased
    title) since raw difflib.get_close_matches penalizes case differences
    heavily — e.g. without this, "batmn" would rank "Fatman" above "Batman"
    purely because of the capital "B" mismatch.
    """
    query = (query or "").strip()
    if not query:
        return []

    # Build a lowercase -> original-case lookup, deduping same-titled entries.
    lower_map = {}
    for t in titles:
        lower_map.setdefault(t.lower(), t)
    lower_titles = list(lower_map.keys())
    q_lower = query.lower()

    matches = difflib.get_close_matches(q_lower, lower_titles, n=n, cutoff=cutoff)
    if matches:
        return [lower_map[m] for m in matches]

    # Fallback: loosen the cutoff before giving up entirely.
    matches = difflib.get_close_matches(q_lower, lower_titles, n=n, cutoff=0.35)
    if matches:
        return [lower_map[m] for m in matches]

    # Final fallback: plain substring search (handles very short/odd queries).
    substring_hits = [orig for low, orig in lower_map.items() if q_lower in low]
    return substring_hits[:n]


def apply_period_filter(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "All years" or "release_year" not in df.columns:
        return df
    ranges = {"2020s": (2020, 2029), "2010s": (2010, 2019), "2000s": (2000, 2009), "1990s": (1990, 1999)}
    if period in ranges:
        lo, hi = ranges[period]
        return df[(df["release_year"] >= lo) & (df["release_year"] <= hi)]
    if period == "Older":
        return df[df["release_year"] < 1990]
    return df


def get_recommendations(
    selected_title: str,
    df: pd.DataFrame,
    feature_matrix,
    top_n: int = 10,
    min_rating: float = 0.0,
    period: str = "All years",
    sort_by: str = "Similarity",
) -> pd.DataFrame:
    """
    Core recommendation function.

    Ranking always originates from cosine similarity of TF-IDF vectors.
    `min_rating` / `period` / `sort_by` only reshape *which slice* of the
    similarity-ranked pool gets displayed and in what order — they never
    introduce movies that weren't already among the most similar.
    """
    matches = df.index[df["title"] == selected_title].tolist()
    if not matches:
        return pd.DataFrame()
    idx = matches[0]

    sims = cosine_similarity(feature_matrix[idx], feature_matrix).flatten()

    working = df.copy()
    working["similarity"] = sims
    working = working.drop(index=idx, errors="ignore")  # never recommend the movie to itself

    # 1) Build the similarity-ranked candidate pool (this is the real ranking).
    working = working.sort_values("similarity", ascending=False)

    # 2) Apply display filters on top of that pool.
    working = working[working["vote_average"] >= min_rating]
    working = apply_period_filter(working, period)

    # Keep a generous pool so re-sorting still has meaningful similar movies to choose from.
    pool_size = max(top_n * 4, 40)
    pool = working.head(pool_size).copy()

    # 3) Re-order the displayed pool per the user's chosen sort mode.
    sort_map = {
        "Similarity": ("similarity", False),
        "Rating": ("vote_average", False),
        "Popularity": ("popularity", False),
        "Release date": ("release_date_parsed", False),
    }
    sort_col, ascending = sort_map.get(sort_by, ("similarity", False))
    pool = pool.sort_values(sort_col, ascending=ascending, na_position="last")

    return pool.head(top_n)


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def fmt_count(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{int(n)}"


def fmt_year(row) -> str:
    y = row.get("release_year")
    if pd.isna(y):
        return "—"
    return str(int(y))


def get_gradient(title: str) -> Tuple[str, str]:
    idx = sum(ord(c) for c in str(title)) % len(GRADIENT_PALETTE)
    return GRADIENT_PALETTE[idx]


def get_initials(title: str) -> str:
    words = [w for w in str(title).split() if w[:1].isalnum()]
    if not words:
        return "??"
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[1][0]).upper()


def safe_overview(text: str, max_len: int = 150) -> str:
    text = (text or "").strip()
    if not text:
        return "No overview available for this title."
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "…"


def esc(text) -> str:
    """Minimal HTML escaping for user-facing / data-driven strings."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_html(html: str) -> None:
    """
    Render a raw HTML fragment via st.markdown safely.

    Streamlit's markdown parser treats <div>/<span>/etc. blocks as
    CommonMark "HTML block type 6", which terminates at the first blank
    line. Multi-line triple-quoted f-strings naturally pick up Python's
    source indentation, and the gaps between concatenated fragments can
    introduce a whitespace-only "blank" line — once that happens, every
    line after it (now indented 4+ spaces) gets reinterpreted as an
    indented Markdown code block and is shown as literal escaped text
    instead of being rendered.

    To avoid that entirely, this collapses the fragment to a single
    line with no blank lines and no leading indentation before handing
    it to st.markdown.
    """
    compact = "".join(line.strip() for line in html.strip().splitlines())
    st.markdown(compact, unsafe_allow_html=True)


# ============================================================================
# TMDB POSTER INTEGRATION
# ============================================================================
#
# This section is a presentation/enrichment layer ONLY. It never touches
# the TF-IDF vectorizer, the cosine-similarity computation, or the ranking
# logic above — those remain driven purely by `movies.csv`. If TMDB is
# unreachable, misconfigured, rate-limited, or simply has no match for a
# given movie, every function here degrades to returning None, and the UI
# falls back to the original gradient + initials placeholder.
#
# The dataset's `id` column is NOT assumed to be a TMDB ID. `get_tmdb_movie`
# tries a direct-by-ID lookup first (cheap, exact when it works) but only
# trusts it if the returned title is actually a plausible match for the
# row; otherwise it falls back to a title + release-year search. This
# protects against datasets where `id` is a different source's ID (e.g. a
# Kaggle/IMDb-style ID) as well as remakes, duplicate titles, and sequels
# sharing a name (e.g. "Batman" 1989 vs. "Batman" 2022 vs. "Batman Begins").

def get_tmdb_api_key() -> Optional[str]:
    """
    Safely read TMDB_API_KEY from Streamlit secrets.

    Never raises: returns None if `.streamlit/secrets.toml` doesn't exist,
    the key isn't set, or secrets access fails for any other reason.
    Callers must treat a None key as "TMDB is unavailable" and fall back.
    """
    try:
        key = st.secrets.get("TMDB_API_KEY")
    except Exception:  # noqa: BLE001 - e.g. no secrets.toml present at all
        return None
    key = str(key).strip() if key else ""
    return key or None


def _tmdb_get(path: str, params: dict) -> Optional[dict]:
    """
    Low-level TMDB GET request. Never raises and never hangs indefinitely.

    Returns None on: missing/invalid key, timeout, connection error, any
    non-2xx HTTP status (including 401 invalid-key and 429 rate-limited),
    or a malformed/non-JSON response.
    """
    api_key = get_tmdb_api_key()
    if not api_key:
        return None
    try:
        response = requests.get(
            f"{TMDB_API_BASE}{path}",
            params={**params, "api_key": api_key},
            timeout=TMDB_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        return None
    except requests.exceptions.RequestException:
        return None  # covers 401 / 404 / 429 / 5xx via raise_for_status()
    except ValueError:
        return None  # malformed JSON


@st.cache_data(show_spinner=False, ttl=TMDB_POSTER_CACHE_TTL)
def _tmdb_lookup_by_id(tmdb_id: int) -> Optional[dict]:
    """Direct TMDB movie lookup by ID. Cached per ID for 24h."""
    return _tmdb_get(f"/movie/{int(tmdb_id)}", {})


@st.cache_data(show_spinner=False, ttl=TMDB_POSTER_CACHE_TTL)
def _tmdb_search_by_title(title: str, year: Optional[int]) -> Optional[dict]:
    """
    Search TMDB by title (+ year when known) and pick the best candidate.

    Candidates are ranked by (1) title similarity, (2) exact release-year
    match, and (3) popularity only as a final tie-breaker — so a query for
    "Batman" (1989) won't resolve to "Batman" (2022) or an unrelated
    similarly-named film just because it's more popular today.
    """
    params = {"query": title, "include_adult": "false"}
    if year:
        params["year"] = int(year)
    data = _tmdb_get("/search/movie", params)
    results = (data or {}).get("results") or []

    if not results and year:
        # TMDB's `year` param can be overly strict (release vs. wide-release
        # dates); retry once without it before giving up.
        data = _tmdb_get("/search/movie", {"query": title, "include_adult": "false"})
        results = (data or {}).get("results") or []

    if not results:
        return None

    def _score(candidate: dict) -> Tuple[float, float, float]:
        cand_title = str(candidate.get("title") or candidate.get("original_title") or "")
        title_sim = difflib.SequenceMatcher(None, title.lower(), cand_title.lower()).ratio()
        cand_year = None
        release_date = candidate.get("release_date") or ""
        if len(release_date) >= 4 and release_date[:4].isdigit():
            cand_year = int(release_date[:4])
        year_match = 1.0 if (year and cand_year == year) else 0.0
        popularity = float(candidate.get("popularity") or 0.0)
        return (title_sim, year_match, popularity)

    return max(results, key=_score)


def get_tmdb_movie(row: pd.Series) -> Optional[dict]:
    """
    Resolve the best-matching TMDB movie record for a dataset row.

    Tries an ID-based lookup first, but only trusts it if the resulting
    title is plausibly the same movie; otherwise falls back to a
    title + release-year search.
    """
    title = str(row.get("title") or "").strip()
    if not title:
        return None

    year_val = row.get("release_year")
    year = int(year_val) if pd.notna(year_val) else None

    raw_id = row.get("id")
    if pd.notna(raw_id):
        try:
            tmdb_id = int(float(raw_id))
        except (TypeError, ValueError):
            tmdb_id = None
        if tmdb_id and tmdb_id > 0:
            candidate = _tmdb_lookup_by_id(tmdb_id)
            if candidate:
                cand_title = str(candidate.get("title") or candidate.get("original_title") or "")
                title_sim = difflib.SequenceMatcher(None, title.lower(), cand_title.lower()).ratio()
                if title_sim >= 0.6:  # guards against `id` not being a TMDB ID
                    return candidate

    return _tmdb_search_by_title(title, year)


def build_poster_url(poster_path: Optional[str]) -> Optional[str]:
    """TMDB image CDN URL for a poster path, or None if no path exists."""
    if not poster_path:
        return None
    return f"{TMDB_IMAGE_BASE}{poster_path}"


def get_poster_and_link(row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (poster_url, tmdb_page_url) for a row, or (None, None) if TMDB
    is unavailable, unconfigured, or no confident match/poster exists.
    This is the only entry point UI code should call for poster data.
    """
    if not get_tmdb_api_key():
        return None, None
    movie = get_tmdb_movie(row)
    if not movie:
        return None, None
    poster_url = build_poster_url(movie.get("poster_path"))
    if not poster_url:
        return None, None
    tmdb_id = movie.get("id")
    page_url = f"https://www.themoviedb.org/movie/{tmdb_id}" if tmdb_id else None
    return poster_url, page_url


def _poster_fragment_card(row: pd.Series) -> str:
    """
    Poster markup for a grid card: a real TMDB poster (linked to its TMDB
    page when available) if one can be resolved, otherwise the existing
    gradient + initials placeholder. Fills the `.card-poster` container.
    """
    poster_url, page_url = get_poster_and_link(row)
    if poster_url:
        img_html = f'<img src="{esc(poster_url)}" alt="{esc(row["title"])} poster" loading="lazy">'
        if page_url:
            return (
                f'<a class="poster-link" href="{esc(page_url)}" target="_blank" '
                f'rel="noopener noreferrer">{img_html}</a>'
            )
        return img_html

    g1, g2 = get_gradient(row["title"])
    return (
        f'<div style="position:absolute;inset:0;background:linear-gradient(135deg,{g1},{g2});'
        f'display:flex;align-items:center;justify-content:center;">'
        f'<div class="initials">{esc(get_initials(row["title"]))}</div></div>'
    )


def _poster_fragment_detail(row: pd.Series) -> str:
    """Poster markup for the selected-movie detail card's fixed-size box."""
    poster_url, page_url = get_poster_and_link(row)
    if poster_url:
        img_html = (
            f'<img src="{esc(poster_url)}" alt="{esc(row["title"])} poster" loading="lazy" '
            f'style="width:100%;height:100%;object-fit:cover;object-position:center top;'
            f'display:block;border-radius:12px;">'
        )
        if page_url:
            return (
                f'<a href="{esc(page_url)}" target="_blank" rel="noopener noreferrer" '
                f'style="display:block;width:100%;height:100%;">{img_html}</a>'
            )
        return img_html

    g1, g2 = get_gradient(row["title"])
    return (
        f'<div style="width:100%;height:100%;border-radius:12px;'
        f'background:linear-gradient(135deg,{g1},{g2});'
        f'display:flex;align-items:center;justify-content:center;">'
        f"<span style=\"font-family:'Bebas Neue',sans-serif;font-size:2.4rem;"
        f'color:rgba(255,255,255,0.9);">{esc(get_initials(row["title"]))}</span></div>'
    )


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_movie_card_html(row: pd.Series, rank: Optional[int] = None, show_similarity: bool = False) -> str:
    poster_html = _poster_fragment_card(row)
    rank_html = f'<div class="rank-badge">#{rank}</div>' if rank else ""
    rating = row.get("vote_average", 0) or 0
    year = fmt_year(row)
    pop = row.get("popularity", 0) or 0
    votes = fmt_count(row.get("vote_count", 0))

    sim_chip = ""
    if show_similarity and "similarity" in row and pd.notna(row["similarity"]):
        sim_pct = max(0.0, float(row["similarity"])) * 100
        sim_chip = f'<span class="chip sim">Similarity {sim_pct:.1f}%</span>'

    return f"""
    <div class="movie-card">
        <div class="card-poster">
            {poster_html}
            {rank_html}
            <div class="rating-badge">★ {rating:.1f}</div>
        </div>
        <div class="card-body">
            <div class="card-title">{esc(row['title'])}</div>
            <div class="badge-row">
                <span class="chip year">{year}</span>
                <span class="chip pop">Pop {pop:.1f}</span>
                <span class="chip">{votes} votes</span>
                {sim_chip}
            </div>
            <div class="card-overview">{esc(safe_overview(row.get('overview', '')))}</div>
        </div>
    </div>
    """


def render_movie_grid(rows: pd.DataFrame, ranked: bool = False, show_similarity: bool = False) -> None:
    if rows.empty:
        st.info("No movies match the current filters.")
        return
    cards = []
    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        cards.append(render_movie_card_html(row, rank=i if ranked else None, show_similarity=show_similarity))
    render_html(f'<div class="movie-grid">{"".join(cards)}</div>')


def render_movie_details(row: pd.Series) -> None:
    poster_html = _poster_fragment_detail(row)
    render_html(
        f"""
        <div class="detail-card">
            <div style="display:flex; gap:1.4rem; align-items:flex-start; flex-wrap:wrap;">
                <div style="width:110px; height:150px; border-radius:12px; flex-shrink:0;
                            overflow:hidden; border:1px solid var(--border);">
                    {poster_html}
                </div>
                <div style="flex:1; min-width:240px;">
                    <div class="detail-title">{esc(row['title'])}</div>
                    <div class="badge-row">
                        <span class="chip year">{fmt_year(row)}</span>
                        <span class="chip">★ {row.get('vote_average', 0):.1f} rating</span>
                        <span class="chip pop">Pop {row.get('popularity', 0):.1f}</span>
                        <span class="chip">{fmt_count(row.get('vote_count', 0))} votes</span>
                    </div>
                    <div class="detail-overview">{esc(row.get('overview', '') or 'No overview available.')}</div>
                </div>
            </div>
        </div>
        """
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rating", f"{row.get('vote_average', 0):.1f} / 10")
    c2.metric("Vote Count", fmt_count(row.get("vote_count", 0)))
    c3.metric("Popularity", f"{row.get('popularity', 0):.1f}")
    c4.metric("Movie ID", str(row.get("id", "—")))


def render_recommendation_insight(selected_title: str, total_movies: int, results: pd.DataFrame) -> None:
    top_sim = results["similarity"].max() * 100 if not results.empty and "similarity" in results else 0.0
    render_html(
        f"""
        <div class="insight-box">
            <div class="insight-row"><span class="k">Selected movie</span><span class="v">{esc(selected_title)}</span></div>
            <div class="insight-row"><span class="k">Movies analyzed</span><span class="v">{total_movies:,}</span></div>
            <div class="insight-row"><span class="k">Recommendation method</span><span class="v">Content-based filtering</span></div>
            <div class="insight-row"><span class="k">Text representation</span><span class="v">TF-IDF (overview)</span></div>
            <div class="insight-row"><span class="k">Similarity metric</span><span class="v">Cosine similarity</span></div>
            <div class="insight-row"><span class="k">Top similarity</span><span class="v">{top_sim:.1f}%</span></div>
        </div>
        """
    )


def render_why_this_movie() -> None:
    with st.expander("Why this movie? — how these recommendations are calculated"):
        st.markdown(
            """
            This recommendation is based on **textual similarity** between the selected
            movie's overview and every other movie's overview, measured with a
            **TF-IDF vector representation** and **cosine similarity**.

            The percentage shown on each card is the cosine similarity score between
            the selected movie's TF-IDF vector and that movie's TF-IDF vector,
            expressed as a percentage. It reflects how much overlapping, distinctively
            weighted vocabulary the two overviews share — not a learned understanding
            of taste, genre, or plot structure.
            """
        )


# ============================================================================
# PAGES
# ============================================================================

def render_sidebar(df: pd.DataFrame) -> str:
    render_html('<div class="brand-mark">CINEMATCH</div>')
    render_html('<div class="brand-sub">AI-powered movie discovery</div>')

    page = st.radio(
        "Navigate",
        ["Home", "Recommend", "Analytics", "Explore", "About"],
        label_visibility="collapsed",
        key="nav_radio",
    )

    render_html('<hr class="side-divider">')

    render_html('<div class="side-label">Dataset</div>')
    render_html(f'<div class="side-value"><span class="mono">{len(df):,}</span> movies</div>')

    render_html('<div class="side-label">Model</div>')
    render_html('<div class="side-value">TF-IDF + Cosine Similarity</div>')

    render_html('<div class="side-label">Search</div>')
    render_html('<div class="side-value">Fuzzy title matching</div>')

    return page


def render_home(df: pd.DataFrame) -> None:
    render_html(
        """
        <div class="hero">
            <div class="hero-eyebrow">🎬 Content-Based Recommender</div>
            <div class="hero-title">Discover Your Next<br>Favorite Movie</div>
            <div class="hero-sub">AI-powered recommendations based on movie content and similarity.</div>
        </div>
        """
    )

    with st.form("home_search_form"):
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "Search",
                placeholder="Search for a movie you like… e.g. The Godfather",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("🔎 Get Recommendations", width='stretch')

    if submitted and query.strip():
        st.session_state["pending_query"] = query.strip()
        st.session_state["nav_override"] = "Recommend"
        st.rerun()

    avg_rating = df["vote_average"].mean()
    avg_pop = df["popularity"].mean()
    rated_movies = df[df["vote_count"] >= 50]
    top_movie = rated_movies.sort_values("vote_average", ascending=False).iloc[0] if not rated_movies.empty else df.iloc[0]

    render_html(
        f"""
        <div class="stat-strip">
            <div class="stat-box"><div class="num">{len(df):,}</div><div class="lbl">Movies in Catalog</div></div>
            <div class="stat-box accent"><div class="num">{avg_rating:.2f}</div><div class="lbl">Average Rating</div></div>
            <div class="stat-box"><div class="num">{avg_pop:.1f}</div><div class="lbl">Average Popularity</div></div>
            <div class="stat-box accent"><div class="num">{top_movie['vote_average']:.1f}★</div><div class="lbl">Top Rated: {esc(top_movie['title'][:22])}</div></div>
        </div>
        """
    )

    tabs = st.tabs(["🏆 Top Rated", "🔥 Most Popular", "🆕 Recent Releases"])

    with tabs[0]:
        pool = df[df["vote_count"] >= max(50, int(df["vote_count"].quantile(0.5)))]
        top_rated = pool.sort_values(["vote_average", "vote_count"], ascending=[False, False]).head(5)
        st.caption(f"Ranked by rating among movies with at least {max(50, int(df['vote_count'].quantile(0.5))):,} votes.")
        render_movie_grid(top_rated)

    with tabs[1]:
        most_popular = df.sort_values("popularity", ascending=False).head(5)
        render_movie_grid(most_popular)

    with tabs[2]:
        recent = df.dropna(subset=["release_date_parsed"]).sort_values("release_date_parsed", ascending=False).head(5)
        render_movie_grid(recent)

    render_html('<div class="film-divider"></div>')
    render_model_explainer()


def render_model_explainer() -> None:
    render_html('<div class="section-head"><span class="tag">// how it works</span><h3>How does the recommender work?</h3></div>')
    with st.expander("See the recommendation pipeline", expanded=False):
        steps = [
            "Movie Overview", "Text preprocessing", "TF-IDF Vectorization",
            "Movie Feature Vectors", "Cosine Similarity", "Similarity Ranking",
            "Top Recommendations",
        ]
        for i, step in enumerate(steps):
            render_html(f'<div class="flow-step">{esc(step)}</div>')
            if i < len(steps) - 1:
                render_html('<div class="flow-arrow">↓</div>')

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                **TF-IDF (Term Frequency – Inverse Document Frequency)**
                - **TF** — how important a word is *within one* movie's overview.
                - **IDF** — how rare that word is *across all* movie overviews.
                - **TF-IDF = TF × IDF** — words that are frequent in one overview
                  but rare across the whole dataset get the highest weight.
                """
            )
        with c2:
            st.markdown(
                """
                **Cosine Similarity**

                Every overview becomes a high-dimensional vector of TF-IDF
                weights. Cosine similarity measures the angle between two
                vectors — a score near **1.0** means the two overviews share
                very similar weighted vocabulary; a score near **0** means
                they share almost none.
                """
            )


def render_recommend(df: pd.DataFrame, feature_matrix) -> None:
    render_html('<div class="section-head"><span class="tag">// recommend</span><h3>What movie do you like?</h3></div>')

    default_query = st.session_state.pop("pending_query", "")
    query = st.text_input(
        "Movie title",
        value=default_query,
        placeholder="Type a movie title — typos are okay, try 'batmn'…",
        label_visibility="collapsed",
    )

    with st.expander("⚙️ Recommendation controls", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            top_n = st.slider("Number of recommendations", 5, 30, 10, step=1)
        with c2:
            min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, step=0.5)
        with c3:
            period = st.selectbox("Release period", PERIOD_OPTIONS, index=0)
        with c4:
            sort_by = st.selectbox("Sort recommendations by", SORT_OPTIONS, index=0)

    if not query.strip():
        st.info("Start typing a movie title above to get personalized, content-based recommendations.")
        return

    titles = df["title"].tolist()
    close_matches = find_movie(query, titles, n=6)

    if not close_matches:
        render_html(
            """
            <div class="not-found-box">
                <strong>We couldn't find that movie. Try a different title.</strong><br>
                <span style="color:var(--text-dim); font-size:0.88rem;">
                    Double-check the spelling, or try just the first word or two of the title.
                </span>
            </div>
            """
        )
        sample = df.sort_values("popularity", ascending=False).head(6)["title"].tolist()
        st.caption("A few popular titles you could try instead:")
        st.write(", ".join(sample))
        return

    selected_title = close_matches[0]
    if len(close_matches) > 1 or selected_title.lower() != query.strip().lower():
        alt = ", ".join(close_matches[1:5])
        note = f"Showing results for **{selected_title}**"
        if alt:
            note += f" — other close matches: {alt}"
        st.caption(note)

    selected_row = df[df["title"] == selected_title].iloc[0]

    render_html('<div class="section-head"><span class="tag">// selected</span><h3>Selected Movie</h3></div>')
    render_movie_details(selected_row)

    results = get_recommendations(
        selected_title, df, feature_matrix,
        top_n=top_n, min_rating=min_rating, period=period, sort_by=sort_by,
    )

    render_html('<div class="section-head"><span class="tag">// results</span><h3>Because You Watched…</h3></div>')

    if results.empty:
        st.warning("No similar movies matched your current filters. Try loosening the rating or period filter.")
    else:
        render_movie_grid(results, ranked=True, show_similarity=True)

    render_why_this_movie()

    render_html('<div class="section-head"><span class="tag">// insight</span><h3>Recommendation Insight</h3></div>')
    render_recommendation_insight(selected_title, len(df), results)


def render_analytics(df: pd.DataFrame) -> None:
    render_html('<div class="section-head"><span class="tag">// analytics</span><h3>Dataset Analytics</h3></div>')

    total = len(df)
    avg_rating = df["vote_average"].mean()
    highest_rated = df.loc[df["vote_average"].idxmax()] if total else None
    avg_pop = df["popularity"].mean()
    high_rated_count = (df["vote_average"] >= 8).sum()

    render_html(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-val">{total:,}</div><div class="kpi-lbl">Total Movies</div></div>
            <div class="kpi-card"><div class="kpi-val">{avg_rating:.2f}</div><div class="kpi-lbl">Average Rating</div></div>
            <div class="kpi-card"><div class="kpi-val">{highest_rated['vote_average']:.1f}★</div><div class="kpi-lbl">Highest Rated</div><div class="kpi-sub">{esc(highest_rated['title'][:26])}</div></div>
            <div class="kpi-card"><div class="kpi-val">{avg_pop:.1f}</div><div class="kpi-lbl">Avg Popularity</div></div>
            <div class="kpi-card"><div class="kpi-val">{high_rated_count:,}</div><div class="kpi-lbl">Movies Rated 8+</div></div>
        </div>
        """
    )

    plot_template = _plotly_dark_template()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Rating Distribution")
        fig = px.histogram(df, x="vote_average", nbins=30, color_discrete_sequence=["#e6395f"])
        fig.update_layout(**plot_template, xaxis_title="Vote Average", yaxis_title="Number of Movies")
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("##### Movies by Release Year")
        by_year = df.dropna(subset=["release_year"]).copy()
        by_year["release_year"] = by_year["release_year"].astype(int)
        by_year = by_year[(by_year["release_year"] >= 1900) & (by_year["release_year"] <= datetime.now().year)]
        year_counts = by_year.groupby("release_year").size().reset_index(name="count")
        if year_counts.empty:
            st.info("No valid release-year data available.")
        else:
            fig = px.bar(year_counts, x="release_year", y="count", color_discrete_sequence=["#f5b942"])
            fig.update_layout(**plot_template, xaxis_title="Release Year", yaxis_title="Number of Movies")
            st.plotly_chart(fig, width='stretch')

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("##### Rating vs. Popularity")
        sample = df if len(df) <= 4000 else df.sample(4000, random_state=42)
        fig = px.scatter(
            sample, x="vote_average", y="popularity",
            hover_data={"title": True, "vote_average": True, "popularity": True, "release_date": True},
            color_discrete_sequence=["#7ec8e3"], opacity=0.6,
        )
        fig.update_layout(**plot_template, xaxis_title="Vote Average", yaxis_title="Popularity")
        st.plotly_chart(fig, width='stretch')

    with col4:
        st.markdown("##### Vote Count vs. Rating")
        sample2 = df[df["vote_count"] > 0]
        sample2 = sample2 if len(sample2) <= 4000 else sample2.sample(4000, random_state=42)
        fig = px.scatter(
            sample2, x="vote_count", y="vote_average", log_x=True,
            hover_data={"title": True, "vote_count": True, "vote_average": True},
            color_discrete_sequence=["#a8326e"], opacity=0.6,
        )
        fig.update_layout(**plot_template, xaxis_title="Vote Count (log scale)", yaxis_title="Vote Average")
        st.plotly_chart(fig, width='stretch')

    st.markdown("##### Top Rated Movies")
    vote_threshold = max(50, int(df["vote_count"].quantile(0.80)))
    qualified = df[df["vote_count"] >= vote_threshold]
    st.caption(f"Ranking restricted to movies with at least **{vote_threshold:,} votes** so low-vote outliers can't dominate.")
    top10 = qualified.sort_values("vote_average", ascending=False).head(10).iloc[::-1]
    if top10.empty:
        st.info("No movies meet the vote-count threshold.")
    else:
        fig = go.Figure(go.Bar(
            x=top10["vote_average"], y=top10["title"], orientation="h",
            marker_color="#e6395f", text=top10["vote_average"].round(1), textposition="outside",
        ))
        fig.update_layout(**plot_template, xaxis_title="Rating", yaxis_title="", height=420)
        st.plotly_chart(fig, width='stretch')


def _plotly_dark_template() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9c7d4", family="Inter, sans-serif"),
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )


def render_explore(df: pd.DataFrame) -> None:
    render_html('<div class="section-head"><span class="tag">// explore</span><h3>Explore Movies</h3></div>')

    with st.expander("🎯 View full movie details", expanded=False):
        pick = st.selectbox("Choose a movie", options=df["title"].sort_values().tolist(), index=None, placeholder="Select a movie…")
        if pick:
            row = df[df["title"] == pick].iloc[0]
            render_movie_details(row)

    st.markdown("##### Filters")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        search_text = st.text_input("Title contains", placeholder="e.g. love, war, alien")
    with f2:
        rating_range = st.slider("Rating range", 0.0, 10.0, (0.0, 10.0), step=0.5)
    with f3:
        year_min = int(df["release_year"].min()) if df["release_year"].notna().any() else 1900
        year_max = int(df["release_year"].max()) if df["release_year"].notna().any() else datetime.now().year
        year_range = st.slider("Release year range", year_min, year_max, (year_min, year_max))
    with f4:
        min_votes = st.number_input("Minimum vote count", min_value=0, value=0, step=100)

    pop_min, pop_max = float(df["popularity"].min()), float(df["popularity"].max())
    pop_range = st.slider("Popularity range", pop_min, pop_max, (pop_min, pop_max))

    filtered = df.copy()
    if search_text.strip():
        filtered = filtered[filtered["title"].str.contains(search_text.strip(), case=False, na=False)]
    filtered = filtered[
        (filtered["vote_average"] >= rating_range[0]) & (filtered["vote_average"] <= rating_range[1])
        & (filtered["popularity"] >= pop_range[0]) & (filtered["popularity"] <= pop_range[1])
        & (filtered["vote_count"] >= min_votes)
    ]
    filtered = filtered[
        filtered["release_year"].isna()
        | ((filtered["release_year"] >= year_range[0]) & (filtered["release_year"] <= year_range[1]))
    ]

    sort_col = st.selectbox("Sort table by", ["Rating", "Popularity", "Vote Count", "Release Date", "Title"], index=0)
    sort_map = {
        "Rating": "vote_average", "Popularity": "popularity",
        "Vote Count": "vote_count", "Release Date": "release_date_parsed", "Title": "title",
    }
    ascending = sort_col == "Title"
    filtered = filtered.sort_values(sort_map[sort_col], ascending=ascending, na_position="last")

    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} movies**")

    display_df = filtered[["title", "release_date", "vote_average", "vote_count", "popularity"]].rename(
        columns={
            "title": "Title", "release_date": "Release Date", "vote_average": "Rating",
            "vote_count": "Vote Count", "popularity": "Popularity",
        }
    )
    st.dataframe(display_df, width='stretch', hide_index=True, height=440)


def render_about(df: pd.DataFrame) -> None:
    render_html('<div class="section-head"><span class="tag">// about</span><h3>About CineMatch</h3></div>')

    st.markdown(
        """
        CineMatch is a **content-based movie recommender**. It does not use
        collaborative filtering, deep learning, embeddings, or any trained
        neural recommender model — it works purely from the text of each
        movie's overview.
        """
    )

    render_html('<div class="section-head"><span class="tag">// architecture</span><h3>System Architecture</h3></div>')
    arch_steps = [
        "movies.csv", "Pandas DataFrame", "Overview Text", "TF-IDF Vectorizer",
        "Feature Matrix", "Cosine Similarity (per query)", "Fuzzy Movie Search",
        "Recommendation Ranking", "Streamlit UI",
    ]
    for i, step in enumerate(arch_steps):
        render_html(f'<div class="flow-step">{esc(step)}</div>')
        if i < len(arch_steps) - 1:
            render_html('<div class="flow-arrow">↓</div>')

    render_html('<div class="section-head"><span class="tag">// honesty</span><h3>Model Limitations</h3></div>')
    st.markdown(
        """
        - TF-IDF depends entirely on the **vocabulary actually present** in each overview.
        - **Synonyms and paraphrases** aren't recognized — "cop" and "police officer" are unrelated words to this model.
        - Recommendations are **purely content-based**; no user history, ratings, or behavior is used.
        - **Popularity and user behavior are not learned** by the model — they're only available as separate display/sort options.
        - Very **short or sparse overviews** produce weaker, less reliable similarity scores.
        - Cosine similarity measures **vocabulary overlap**, not genuine semantic or narrative understanding.
        """
    )

    render_model_explainer()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    inject_css()

    df, error = load_data(CSV_PATH)

    if error:
        st.error(f"🚫 Dataset validation failed: {error}")
        st.caption("Expected core columns: " + ", ".join(REQUIRED_COLUMNS))
        st.stop()

    if df is None or df.empty:
        st.error("🚫 The dataset loaded but contains no usable rows.")
        st.stop()

    vectorizer, feature_matrix = build_recommender(df)

    # Allow the Home page's search box to jump straight into Recommend
    # (must be applied before the radio widget is instantiated).
    if st.session_state.get("nav_override"):
        st.session_state["nav_radio"] = st.session_state.pop("nav_override")

    with st.sidebar:
        page = render_sidebar(df)
        if not get_tmdb_api_key():
            st.caption("🎬 Add `TMDB_API_KEY` in Streamlit secrets to show real posters.")

    if page == "Home":
        render_home(df)
    elif page == "Recommend":
        render_recommend(df, feature_matrix)
    elif page == "Analytics":
        render_analytics(df)
    elif page == "Explore":
        render_explore(df)
    elif page == "About":
        render_about(df)


if __name__ == "__main__":
    main()