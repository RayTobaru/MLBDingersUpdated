from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

try:
    from .compat import ROOT
except Exception:
    ROOT = Path.cwd()


ODDS_CANDIDATE_PATHS = [
    "FDodds.csv",
    "FDOdds.csv",
    "fdodds.csv",
    "data/FDodds.csv",
    "data/FDOdds.csv",
    "outputs/FDodds.csv",
]


def _norm_name(x) -> str:
    return (
        str(x or "")
        .lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .replace("’", "")
        .strip()
    )


def _compact_name(x) -> str:
    s = _norm_name(x)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Normalize common suffix differences between sportsbook names and MLB names.
    # Example: "Jazz Chisholm" vs "Jazz Chisholm Jr."
    parts = s.split()
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    while parts and parts[-1] in suffixes:
        parts = parts[:-1]

    return " ".join(parts)


def _parse_american_odds(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan

    s = str(x).strip()
    if not s:
        return np.nan

    s = s.replace("+", "")
    s = re.sub(r"[^0-9\-.]", "", s)

    if not s or s in {"-", ".", "-."}:
        return np.nan

    try:
        val = float(s)
    except Exception:
        return np.nan

    if not np.isfinite(val) or abs(val) < 100:
        return np.nan

    return int(round(val))


def _american_to_implied_prob(odds):
    odds = _parse_american_odds(odds)
    if not np.isfinite(odds):
        return np.nan

    if odds > 0:
        return 100.0 / (odds + 100.0)

    return abs(odds) / (abs(odds) + 100.0)


def _prob_to_american(p):
    try:
        p = float(p)
    except Exception:
        return np.nan

    if not np.isfinite(p) or p <= 0 or p >= 1:
        return np.nan

    if p >= 0.5:
        return int(round(-100.0 * p / (1.0 - p)))

    return int(round(100.0 * (1.0 - p) / p))


def _profit_on_100(odds):
    odds = _parse_american_odds(odds)
    if not np.isfinite(odds):
        return np.nan

    if odds > 0:
        return float(odds)

    return 10000.0 / abs(float(odds))


def find_fd_odds_file(base_dir: str | Path | None = None) -> Path | None:
    base = Path(base_dir) if base_dir else Path.cwd()

    candidates = []
    for rel in ODDS_CANDIDATE_PATHS:
        candidates.append(base / rel)

    try:
        candidates.append(Path(ROOT) / "FDodds.csv")
        candidates.append(Path(ROOT) / "FDOdds.csv")
    except Exception:
        pass

    for p in candidates:
        try:
            if p.exists() and p.stat().st_size > 0:
                return p
        except Exception:
            continue

    return None


def _looks_like_junk_line(x: str) -> bool:
    """
    Ignore text that can appear when copying FanDuel cards/images/buttons.
    """
    t = str(x or "").strip()
    if not t:
        return True

    low = t.lower().strip()

    junk_exact = {
        "image", "player image", "photo", "logo", "team logo", "fanduel",
        "fan duel", "sgp", "same game parlay", "quickbet", "quick bet",
        "add", "add selection", "bet now", "home run", "to hit a home run",
        "hr", "odds", "player", "market", "more wagers", "see more",
    }

    if low in junk_exact:
        return True

    junk_contains = [
        "data:image",
        "https://",
        "http://",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        "svg",
        "boost",
        "promo",
        "suspended",
        "locked",
        "unavailable",
        "cash out",
    ]

    if any(j in low for j in junk_contains):
        return True

    # If it has no letters and is not odds, ignore.
    if not re.search(r"[A-Za-z]", t) and not np.isfinite(_parse_american_odds(t)):
        return True

    return False


def _looks_like_player_name(x: str) -> bool:
    t = str(x or "").strip()
    if _looks_like_junk_line(t):
        return False

    # Odds are not names.
    if np.isfinite(_parse_american_odds(t)):
        return False

    # Player names should contain letters.
    if not re.search(r"[A-Za-z]", t):
        return False

    # Avoid obvious stat/market rows.
    low = t.lower()
    bad_fragments = [
        "over ", "under ", "total", "hits", "runs", "rbi", "strikeouts",
        "bases", "stolen", "walks", "alternate", "line", "odds",
    ]
    if any(b in low for b in bad_fragments):
        return False

    # Names are usually short. This prevents paragraphs/card labels.
    if len(t) > 45:
        return False

    return True


def _read_one_column_alternating(path: Path) -> pd.DataFrame:
    """
    Parse messy FanDuel copy/paste odds.

    Supports:
      Matt Olson
      350

      Michael Harris II
      +360

    Also tolerates blank lines and many copy-pasted image/card artifacts.
    """
    try:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        vals = [line.strip() for line in text.splitlines()]
    except Exception:
        raw = pd.read_csv(path, header=None)
        vals = []
        for col in raw.columns:
            vals.extend(raw[col].dropna().astype(str).tolist())

    rows = []
    pending_name = None
    ignored = []

    for raw_v in vals:
        v = str(raw_v or "").strip()
        if not v:
            continue

        odds = _parse_american_odds(v)

        if np.isfinite(odds):
            if pending_name:
                rows.append({"player_name": pending_name, "fd_odds": int(odds)})
                pending_name = None
            else:
                ignored.append(v)
            continue

        if _looks_like_player_name(v):
            pending_name = v
        else:
            ignored.append(v)

    out = pd.DataFrame(rows)

    if not out.empty:
        out["player_name"] = out["player_name"].astype(str).str.strip()
        out["fd_odds"] = out["fd_odds"].map(_parse_american_odds)
        out = out.dropna(subset=["player_name", "fd_odds"]).copy()
        out["fd_odds"] = out["fd_odds"].astype(int)
        out = out.drop_duplicates("player_name", keep="last").reset_index(drop=True)

    return out


def read_fd_odds(path: str | Path | None = None) -> pd.DataFrame:
    odds_path = Path(path) if path else find_fd_odds_file()

    if odds_path is None or not odds_path.exists():
        return pd.DataFrame(columns=["player_name", "fd_odds", "name_key"])

    # Try normal two-column file first.
    try:
        raw = pd.read_csv(odds_path)
    except Exception:
        raw = pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["player_name", "fd_odds", "name_key"])

    lower = {str(c).lower().strip(): c for c in raw.columns}

    name_col = None
    odds_col = None

    for k in ["player_name", "name", "athlete", "batter", "player"]:
        if k in lower:
            name_col = lower[k]
            break

    for k in ["fd_odds", "odds", "fanduel_odds", "hr_odds", "american_odds"]:
        if k in lower:
            odds_col = lower[k]
            break

    if name_col is None or odds_col is None:
        # Supports your screenshot-style one-column alternating name/odds paste.
        parsed = _read_one_column_alternating(odds_path)
    else:
        parsed = raw[[name_col, odds_col]].copy()
        parsed.columns = ["player_name", "fd_odds"]

        # Preserve optional metadata columns if present.
        for c in raw.columns:
            lc = str(c).lower().strip()
            if lc in {"date", "matchup", "book", "market", "updated_at"} and c not in parsed.columns:
                parsed[lc] = raw[c]

    if parsed.empty:
        return pd.DataFrame(columns=["player_name", "fd_odds", "name_key"])

    parsed["player_name"] = parsed["player_name"].astype(str).str.strip()
    parsed["fd_odds"] = parsed["fd_odds"].map(_parse_american_odds)
    parsed = parsed.dropna(subset=["player_name", "fd_odds"]).copy()
    parsed["fd_odds"] = parsed["fd_odds"].astype(int)
    parsed["name_key"] = parsed["player_name"].map(_compact_name)

    parsed = parsed[parsed["name_key"].ne("")].copy()
    parsed = parsed.drop_duplicates("name_key", keep="last").reset_index(drop=True)

    return parsed


def _pick_prob_col(df: pd.DataFrame) -> str:
    for c in ["model_prob_hr", "P(HR>=1)", "p_hr", "prob_hr"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find HR probability column. Expected model_prob_hr or P(HR>=1).")


def _value_tag(edge, ev):
    try:
        edge = float(edge)
        ev = float(ev)
    except Exception:
        return "no_odds"

    if not np.isfinite(edge) or not np.isfinite(ev):
        return "no_odds"

    if edge >= 0.04 and ev >= 20:
        return "elite_value"
    if edge >= 0.025 and ev >= 10:
        return "strong_value"
    if edge >= 0.010 and ev >= 3:
        return "lean_value"
    if edge >= 0.000 and ev >= 0:
        return "fair"
    return "pass"


def build_hr_value_board(
    batter_df: pd.DataFrame,
    odds_path: str | Path | None = None,
    *,
    include_no_odds: bool = True,
) -> pd.DataFrame:
    if batter_df is None or batter_df.empty:
        return pd.DataFrame()

    df = batter_df.copy()

    prob_col = _pick_prob_col(df)
    df["model_prob_hr"] = pd.to_numeric(df[prob_col], errors="coerce")

    # Handle accidental percent format.
    if df["model_prob_hr"].max(skipna=True) > 1.0:
        df["model_prob_hr"] = df["model_prob_hr"] / 100.0

    df["model_prob_hr"] = df["model_prob_hr"].clip(0.000001, 0.999999)

    if "player_name" not in df.columns:
        raise ValueError("Batter output missing player_name column.")

    odds = read_fd_odds(odds_path)

    df["name_key"] = df["player_name"].map(_compact_name)

    # Avoid collisions with older odds columns already present in batter outputs.
    stale_odds_cols = [
        "fd_odds",
        "fd_implied_prob",
        "fd_implied_pct",
        "fair_odds_american",
        "edge_prob",
        "edge_pct_pts",
        "ev_per_100",
        "value_tag",
        "odds_match_status",
    ]
    df = df.drop(columns=[c for c in stale_odds_cols if c in df.columns], errors="ignore")

    if odds.empty:
        df["fd_odds"] = np.nan
        df["odds_match_status"] = "no_odds_file"
    else:
        odds_clean = odds[["name_key", "fd_odds"]].copy()
        odds_clean["fd_odds"] = pd.to_numeric(odds_clean["fd_odds"], errors="coerce")
        odds_clean = odds_clean.dropna(subset=["name_key", "fd_odds"]).drop_duplicates("name_key", keep="last")

        df = df.merge(
            odds_clean,
            on="name_key",
            how="left",
            validate="many_to_one",
        )
        df["odds_match_status"] = np.where(df["fd_odds"].notna(), "matched", "no_odds")

    df["fd_implied_prob"] = df["fd_odds"].map(_american_to_implied_prob)
    df["fd_implied_pct"] = 100.0 * df["fd_implied_prob"]
    df["model_prob_pct"] = 100.0 * df["model_prob_hr"]
    df["fair_odds_american"] = df["model_prob_hr"].map(_prob_to_american)

    df["edge_prob"] = df["model_prob_hr"] - df["fd_implied_prob"]
    df["edge_pct_pts"] = 100.0 * df["edge_prob"]

    profit = df["fd_odds"].map(_profit_on_100)
    df["ev_per_100"] = df["model_prob_hr"] * profit - (1.0 - df["model_prob_hr"]) * 100.0

    df["value_tag"] = [
        _value_tag(e, ev)
        for e, ev in zip(df["edge_prob"], df["ev_per_100"])
    ]

    if not include_no_odds:
        df = df[df["fd_odds"].notna()].copy()

    keep = [
        "date", "matchup", "team", "batting_order", "player_name", "player_id",
        "opp_pitcher", "opp_pitcher_id", "opp_pitcher_hand",
        "exp_hr", "P(HR>=1)", "model_prob_hr", "hr_pa",
        "fd_odds", "fd_implied_prob", "fd_implied_pct",
        "fair_odds_american", "edge_prob", "edge_pct_pts", "ev_per_100", "value_tag",
        "odds_match_status",
        "batter_hr_mult", "batter_form_hr_mult", "pitcher_hr_danger_mult",
        "zone_hr_overall_mult", "zone_hr_damage_mult", "zone_hr_barrel_mult",
        "zone_hr_air_mult", "zone_hr_confidence", "zone_hr_candidate_source",
        "zone_hr_sample_regime", "zone_hr_batter_archetype", "zone_hr_pitcher_arch",
        "zone_hr_exact_batter_rows", "zone_hr_batter_archetype_rows",
        "zone_hr_pitcher_archetype_rows", "zone_hr_hand_rows", "zone_hr_league_rows",
        "status",
    ]

    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    for c in ["fd_implied_prob", "edge_prob", "model_prob_hr"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)

    for c in ["fd_implied_pct", "model_prob_pct", "edge_pct_pts", "ev_per_100"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    sort_cols = []
    ascending = []

    if "ev_per_100" in out.columns:
        sort_cols.append("ev_per_100")
        ascending.append(False)

    if "edge_prob" in out.columns:
        sort_cols.append("edge_prob")
        ascending.append(False)

    if "model_prob_hr" in out.columns:
        sort_cols.append("model_prob_hr")
        ascending.append(False)

    if sort_cols:
        out = out.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)

    return out


def build_and_save_hr_values(
    batter_df: pd.DataFrame,
    out_path: str | Path,
    odds_path: str | Path | None = None,
    *,
    include_no_odds: bool = True,
) -> pd.DataFrame:
    board = build_hr_value_board(batter_df, odds_path=odds_path, include_no_odds=include_no_odds)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(out, index=False)

    return board


def write_fd_odds_template(path: str | Path = "FDodds.csv") -> Path:
    p = Path(path)
    if p.exists():
        return p

    template = pd.DataFrame([
        {"player_name": "Kerry Carpenter", "fd_odds": 255},
        {"player_name": "Byron Buxton", "fd_odds": 290},
        {"player_name": "Matt Wallner", "fd_odds": 300},
        {"player_name": "Spencer Torkelson", "fd_odds": 340},
    ])

    template.to_csv(p, index=False)
    return p
