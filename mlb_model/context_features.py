from __future__ import annotations

import io
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .compat import LEGACY  # noqa: F401

import fetch  # type: ignore
from fetch import (
    YEAR,
    ABBR2ID,
    safe_get,
    fetch_statcast_raw,
    pitcher_mix_last_starts,
    batter_xiso_by_pitch,
    fetch_boxscore_officials,
    load_umpire_network_stats,
    get_game_weather,
)


@lru_cache(maxsize=2048)
def recent_batter_count_profile(batter_pid: int, pitcher_hand: str, days: int = 365) -> dict[str, float]:
    end = datetime.today()
    start = end - timedelta(days=days)
    df = fetch_statcast_raw(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if df is None or df.empty or "batter" not in df.columns:
        return {"ahead": 1.0, "even": 1.0, "behind": 1.0}
    df = df[df["batter"] == batter_pid].copy()
    if df.empty:
        return {"ahead": 1.0, "even": 1.0, "behind": 1.0}
    if pitcher_hand and "p_throws" in df.columns:
        hand_df = df[df["p_throws"].astype(str).str.upper().str.startswith(str(pitcher_hand).upper()[:1])]
        if len(hand_df) >= 40:
            df = hand_df
    if not {"balls", "strikes", "events"}.issubset(df.columns):
        return {"ahead": 1.0, "even": 1.0, "behind": 1.0}

    balls = pd.to_numeric(df["balls"], errors="coerce").fillna(0)
    strikes = pd.to_numeric(df["strikes"], errors="coerce").fillna(0)
    events = df["events"].astype(str)
    xb = events.isin(["double", "triple", "home_run"]).astype(float)

    def _bucket(b: float, s: float) -> str:
        if b > s:
            return "ahead"
        if s > b:
            return "behind"
        return "even"

    grp = pd.DataFrame({"bucket": [_bucket(b, s) for b, s in zip(balls, strikes)], "xb": xb})
    base = float(grp["xb"].mean()) if len(grp) else 0.08
    if base <= 0:
        base = 0.08
    out: dict[str, float] = {}
    for bucket in ("ahead", "even", "behind"):
        sub = grp[grp["bucket"] == bucket]
        rate = float(sub["xb"].mean()) if len(sub) >= 10 else base
        out[bucket] = float(np.clip(rate / base, 0.85, 1.20))
    return out


@lru_cache(maxsize=2048)
def pitch_type_matchup_multiplier(batter_pid: int, pitcher_pid: int | None, pitcher_hand: str) -> dict[str, float]:
    """
    Lightweight matchup layer:
    - combines batter xISO by pitch type with opponent recent pitch mix
    - applies hand-specific count-state boosts to extra-base outcomes
    Returns bounded multipliers for hit quality.
    """
    if not batter_pid or not pitcher_pid:
        return {"hit_mult": 1.0, "xb_mult": 1.0, "hr_mult": 1.0}

    mix = pitcher_mix_last_starts(int(pitcher_pid), days=35, max_games=4) or {}
    xiso = batter_xiso_by_pitch(int(batter_pid), days=220) or {}
    if not mix or not xiso:
        count_mult = recent_batter_count_profile(int(batter_pid), pitcher_hand)
        xb = 0.30 * count_mult["ahead"] + 0.50 * count_mult["even"] + 0.20 * count_mult["behind"]
        xb = float(np.clip(xb, 0.88, 1.15))
        return {"hit_mult": 1.0, "xb_mult": xb, "hr_mult": float(np.clip(0.96 + 0.35 * (xb - 1.0), 0.90, 1.12))}

    league_xiso = np.mean(list(xiso.values())) if len(xiso) else 0.18
    if not np.isfinite(league_xiso) or league_xiso <= 0:
        league_xiso = 0.18
    weighted_xiso = 0.0
    total_w = 0.0
    for pt, w in mix.items():
        weighted_xiso += float(w) * float(xiso.get(pt, league_xiso))
        total_w += float(w)
    if total_w <= 0:
        weighted_xiso = league_xiso
    else:
        weighted_xiso /= total_w
    raw_xb = float(np.clip(weighted_xiso / league_xiso, 0.85, 1.20))
    count_mult = recent_batter_count_profile(int(batter_pid), pitcher_hand)
    count_xb = 0.30 * count_mult["ahead"] + 0.50 * count_mult["even"] + 0.20 * count_mult["behind"]
    xb = float(np.clip(0.65 * raw_xb + 0.35 * count_xb, 0.88, 1.18))
    hit_mult = float(np.clip(0.985 + 0.20 * (xb - 1.0), 0.95, 1.05))
    hr_mult = float(np.clip(0.95 + 0.55 * (xb - 1.0), 0.88, 1.18))
    return {"hit_mult": hit_mult, "xb_mult": xb, "hr_mult": hr_mult}


@lru_cache(maxsize=512)
def umpire_zone_multipliers(game_pk: int | None) -> dict[str, Any]:
    if not game_pk:
        return {"umpire": "", "k_mult": 1.0, "bb_mult": 1.0, "zone_mult": 1.0}
    try:
        df = load_umpire_network_stats()
    except Exception:
        df = pd.DataFrame()
    name = ""
    try:
        offs = fetch_boxscore_officials(int(game_pk))
        if offs:
            name = offs[0]
    except Exception:
        pass
    if df is None or df.empty:
        return {"umpire": name, "k_mult": 1.0, "bb_mult": 1.0, "zone_mult": 1.0}
    name_col = "Umpire" if "Umpire" in df.columns else df.columns[0]
    rec = df[df[name_col].astype(str).str.upper() == str(name).upper()]
    if rec.empty:
        return {"umpire": name, "k_mult": 1.0, "bb_mult": 1.0, "zone_mult": 1.0}
    row = rec.iloc[0]
    k_mult = float(row.get("k_rate", 1.0)) if pd.notna(row.get("k_rate", 1.0)) else 1.0
    bb_mult = float(row.get("bb_rate", 1.0)) if pd.notna(row.get("bb_rate", 1.0)) else 1.0
    zone_mult = float(np.clip(1.0 + 0.40 * (k_mult - 1.0) - 0.15 * (bb_mult - 1.0), 0.94, 1.06))
    return {"umpire": name, "k_mult": k_mult, "bb_mult": bb_mult, "zone_mult": zone_mult}


@lru_cache(maxsize=512)
def weather_multipliers(game_pk: int | None) -> dict[str, float | str]:
    if not game_pk:
        return {"hit_mult": 1.0, "xb_mult": 1.0, "hr_mult": 1.0, "roof_state": "unknown"}
    w = get_game_weather(int(game_pk)) or {}
    temp = w.get("temp_f")
    wind_dir = str(w.get("wind_dir") or "").lower()
    wind_raw = w.get("wind_mph")
    cond = str(w.get("conditions") or "").lower()
    roof_state = "indoor" if ("roof" in cond or "dome" in cond or "indoors" in cond) else "outdoor"
    try:
        wind = float(wind_raw)
    except Exception:
        import re
        m = re.search(r"(\d+(?:\.\d+)?)", str(wind_raw))
        wind = float(m.group(1)) if m else 0.0
    if roof_state == "indoor":
        wind = 0.0
    temp_mult = 1.0 if temp is None else float(np.clip(1.0 + 0.0025 * (float(temp) - 70.0), 0.94, 1.07))
    out_mult = 1.0 + min(max(wind, 0.0), 20.0) * 0.004 if "out" in wind_dir else 1.0
    in_mult = 1.0 - min(max(wind, 0.0), 20.0) * 0.0035 if "in" in wind_dir else 1.0
    hr_mult = float(np.clip(temp_mult * out_mult * in_mult, 0.88, 1.14))
    xb_mult = float(np.clip(1.0 + 0.55 * (hr_mult - 1.0), 0.93, 1.09))
    hit_mult = float(np.clip(1.0 + 0.20 * (xb_mult - 1.0), 0.97, 1.03))
    return {"hit_mult": hit_mult, "xb_mult": xb_mult, "hr_mult": hr_mult, "roof_state": roof_state}


# ---------------------------------------------------------------------------
# Defense-behind-pitcher layer
# ---------------------------------------------------------------------------

def _defense_cache_path(path: str | None = None) -> Path:
    if path:
        return Path(path)
    return LEGACY.parent / "data" / "team_defense_proxies.csv"


def _score_to_factor(score: float) -> float:
    return float(np.clip(1.0 - 0.012 * score, 0.94, 1.06))


def _safe_float(d: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in d and d.get(k) not in (None, ""):
            try:
                return float(d.get(k))
            except Exception:
                continue
    return float(default)


def _extract_stat_dict(js: dict[str, Any]) -> dict[str, Any]:
    stats = js.get("stats", []) if isinstance(js, dict) else []
    for blk in stats:
        splits = blk.get("splits", []) if isinstance(blk, dict) else []
        if splits:
            stat = splits[0].get("stat", {})
            if isinstance(stat, dict):
                return stat
    return {}


def _fetch_team_fielding_stats(team_abbr: str, season: int) -> dict[str, Any]:
    team_id = ABBR2ID.get(team_abbr)
    if not team_id:
        return {}

    endpoints = [
        (f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats", {"stats": "season", "group": "fielding", "season": season}),
        ("https://statsapi.mlb.com/api/v1/teams/stats", {"teamIds": team_id, "stats": "season", "group": "fielding", "season": season, "sportIds": 1}),
    ]
    for url, params in endpoints:
        try:
            r = safe_get(url, params)
            if not r:
                continue
            js = r.json()
            stat = _extract_stat_dict(js)
            if stat:
                return stat
        except Exception:
            continue
    return {}


def auto_build_team_defense_proxies(path: str | None = None, season: int | None = None, force: bool = False) -> pd.DataFrame:
    """
    Auto-build a team defense proxy table from fetchable public stats.

    Priority:
    1) use an existing fresh local cache
    2) fetch current season team fielding stats from MLB StatsAPI and derive a bounded defense factor

    The output table still supports manual overrides later, but manual entry is no longer required.
    """
    season = int(season or YEAR)
    cache_path = _defense_cache_path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force:
        age_days = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days
        if age_days <= 3:
            try:
                existing = pd.read_csv(cache_path)
                if not existing.empty:
                    return existing
            except Exception:
                pass

    rows: list[dict[str, Any]] = []
    for team in sorted(ABBR2ID):
        stat = _fetch_team_fielding_stats(team, season)
        if not stat:
            rows.append({"team": team, "fielding_pct": np.nan, "errors": np.nan, "double_plays": np.nan, "caught_stealing": np.nan, "stolen_bases": np.nan})
            continue
        rows.append(
            {
                "team": team,
                "fielding_pct": _safe_float(stat, "fielding", "fieldingPercentage", default=np.nan),
                "errors": _safe_float(stat, "errors", default=np.nan),
                "double_plays": _safe_float(stat, "doublePlays", default=np.nan),
                "caught_stealing": _safe_float(stat, "caughtStealing", default=np.nan),
                "stolen_bases": _safe_float(stat, "stolenBases", default=np.nan),
                "games": _safe_float(stat, "gamesPlayed", "games", default=np.nan),
                "assists": _safe_float(stat, "assists", default=np.nan),
                "putouts": _safe_float(stat, "putOuts", "putouts", default=np.nan),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["team", "oaa", "positioning", "defense_factor"])
        df.to_csv(cache_path, index=False)
        return df

    # Build robust proxies from available team fielding stats.
    g = df["games"].replace(0, np.nan)
    df["errors_pg"] = df["errors"] / g
    df["dp_pg"] = df["double_plays"] / g
    attempts = df["caught_stealing"] + df["stolen_bases"]
    df["cs_rate"] = np.where(attempts > 0, df["caught_stealing"] / attempts, np.nan)

    def z(s: pd.Series, invert: bool = False) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True)
        if not np.isfinite(sd) or sd == 0:
            out = pd.Series(0.0, index=s.index)
        else:
            out = (s - mu) / sd
        return -out if invert else out

    pct_z = z(df["fielding_pct"], invert=False)
    err_z = z(df["errors_pg"], invert=True)
    dp_z = z(df["dp_pg"], invert=False)
    cs_z = z(df["cs_rate"], invert=False)

    total_score = 0.45 * pct_z + 0.25 * err_z + 0.20 * dp_z + 0.10 * cs_z.fillna(0.0)
    total_score = total_score.fillna(0.0)

    # Store OAA / positioning as proxy columns so downstream code stays stable.
    df["oaa"] = (2.2 * total_score).round(3)
    df["positioning"] = (0.9 * pct_z + 0.6 * dp_z).fillna(0.0).round(3)
    df["defense_factor"] = total_score.map(_score_to_factor).round(4)
    keep = ["team", "oaa", "positioning", "defense_factor", "fielding_pct", "errors_pg", "dp_pg", "cs_rate"]
    out = df[keep].sort_values("team").reset_index(drop=True)
    out.to_csv(cache_path, index=False)
    return out


def load_team_defense_proxies(path: str | None = None, season: int | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns a defense proxy table.

    Behavior:
    - loads an existing table if present
    - auto-builds one from fetched team fielding stats when the file is missing/empty/stale
    - manual CSV/parquet overrides are still supported, but no longer required
    """
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    candidates.extend([
        LEGACY.parent / "data" / "team_defense_proxies.parquet",
        LEGACY.parent / "data" / "team_defense_proxies.csv",
    ])

    if force_refresh and not path:
        return auto_build_team_defense_proxies(season=season, force=True)

    for p in candidates:
        try:
            if p.suffix.lower() == ".parquet" and p.exists():
                df = pd.read_parquet(p)
            elif p.exists():
                df = pd.read_csv(p)
            else:
                continue
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                if {"team", "defense_factor"}.issubset(df.columns):
                    return df
        except Exception:
            continue

    return auto_build_team_defense_proxies(path=str(_defense_cache_path(path)), season=season, force=force_refresh)


def defense_multipliers(fielding_team: str, defense_df: pd.DataFrame | None = None) -> dict[str, float]:
    df = defense_df if defense_df is not None else load_team_defense_proxies()
    if df is None or df.empty or "team" not in df.columns:
        return {"hit_mult": 1.0, "xb_mult": 1.0, "defense_factor": 1.0}
    rec = df[df["team"].astype(str).str.upper() == str(fielding_team).upper()]
    if rec.empty:
        return {"hit_mult": 1.0, "xb_mult": 1.0, "defense_factor": 1.0}
    row = rec.iloc[0]
    oaa = float(row.get("oaa", 0.0) or 0.0)
    pos = float(row.get("positioning", 0.0) or 0.0)
    explicit = row.get("defense_factor", np.nan)
    if pd.notna(explicit):
        fac = float(explicit)
    else:
        fac = float(np.clip(1.0 - 0.0035 * oaa - 0.0020 * pos, 0.93, 1.07))
    hit_mult = float(np.clip(fac, 0.94, 1.06))
    xb_mult = float(np.clip(1.0 + 0.45 * (fac - 1.0), 0.96, 1.04))
    return {"hit_mult": hit_mult, "xb_mult": xb_mult, "defense_factor": fac}


def apply_context_stack(
    hit_pa: float,
    p1b: float,
    p2b: float,
    p3b: float,
    phr: float,
    *,
    batter_pid: int,
    pitcher_pid: int | None,
    pitcher_hand: str,
    game_pk: int | None,
    fielding_team: str,
    defense_df: pd.DataFrame | None = None,
) -> dict[str, float | str]:
    pt = pitch_type_matchup_multiplier(batter_pid, pitcher_pid, pitcher_hand)
    ump = umpire_zone_multipliers(game_pk)
    weather = weather_multipliers(game_pk)
    defense = defense_multipliers(fielding_team, defense_df=defense_df)

    hit_mult = float(np.clip(pt["hit_mult"] * ump["zone_mult"] * weather["hit_mult"] * defense["hit_mult"], 0.90, 1.10))
    xb_mult = float(np.clip(pt["xb_mult"] * weather["xb_mult"] * defense["xb_mult"], 0.85, 1.18))
    hr_mult = float(np.clip(pt["hr_mult"] * weather["hr_mult"] * (1.0 + 0.25 * (ump["zone_mult"] - 1.0)), 0.82, 1.24))

    phr2 = float(np.clip(phr * hr_mult, 0.0, 0.24))
    p2b2 = float(np.clip(p2b * xb_mult, 0.0, 0.20))
    p3b2 = float(np.clip(p3b * xb_mult, 0.0, 0.05))
    p1b_pool = max(hit_pa * hit_mult - (phr2 + p2b2 + p3b2), 0.0)
    p1b2 = float(np.clip(p1b_pool, 0.0, 0.35))
    hit2 = min(p1b2 + p2b2 + p3b2 + phr2, 0.55)

    return {
        "hit_pa": hit2,
        "1b_pa": p1b2,
        "2b_pa": p2b2,
        "3b_pa": p3b2,
        "hr_pa": phr2,
        "pitch_type_mult": pt["xb_mult"],
        "zone_mult": ump["zone_mult"],
        "weather_hr_mult": weather["hr_mult"],
        "defense_factor": defense["defense_factor"],
        "umpire": str(ump.get("umpire", "")),
        "roof_state": str(weather.get("roof_state", "unknown")),
    }
