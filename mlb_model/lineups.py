from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .compat import ROOT

DEFAULT_OVERRIDE_PATH = ROOT / "lineup_overrides.json"

import fetch  # type: ignore
from fetch import (
    safe_get,
    clean_name,
    strip_pos,
    batting_stats,
    fetch_statcast_raw,
    roster_map,
    ABBR2ID,
    YEAR,
    name_to_mlbam_id,
)  # type: ignore


TEAM_ABBR_ALIASES = {
    "KC": "KCR",
    "AZ": "ARI",
    "CWS": "CHW",
    "WSH": "WSN",
    "TB": "TBR",
    "SF": "SFG",
    "SD": "SDP",
    "ATH": "OAK",
}
DEFAULT_LINEUP = ["TBD"] * 9


def _normalize_matchup(matchup: str) -> str:
    return str(matchup or "").upper().replace(" ", "").strip()


def _canonical_team_abbr(team_abbr: str) -> str:
    ab = str(team_abbr or "").strip().upper()
    if ab in ABBR2ID:
        return ab
    return TEAM_ABBR_ALIASES.get(ab, ab)


def load_overrides(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_OVERRIDE_PATH
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_override_for_game(date_str: str, matchup: str, path: str | None = None) -> dict[str, Any] | None:
    data = load_overrides(path)
    key = _normalize_matchup(matchup)

    day = data.get(date_str, {}) if isinstance(data, dict) else {}
    if isinstance(day, dict):
        rec = day.get(key) or day.get(key.replace("@", " @ "))
        if isinstance(rec, dict):
            return rec

    if isinstance(data, dict):
        rec = data.get(key) or data.get(key.replace("@", " @ "))
        if isinstance(rec, dict):
            return rec

    games = data.get("games", []) if isinstance(data, dict) else []
    if isinstance(games, list):
        for rec in games:
            if not isinstance(rec, dict):
                continue
            if str(rec.get("date", "")) == str(date_str) and _normalize_matchup(rec.get("matchup", "")) == key:
                return rec
    return None


def lineup_source_weight(source: str | None) -> float:
    src = str(source or "").strip().lower()
    mapping = {
        "confirmed": 1.00,
        "official": 1.00,
        "manual": 1.00,
        "override": 1.00,
        "hybrid": 0.97,
        "hybrid_projected": 0.96,
        "projected_history": 0.96,
        "projected_roster": 0.95,
        "projected": 0.94,
        "unconfirmed": 0.94,
    }
    return float(mapping.get(src, 0.95))


def lineup_certainty_score(source: str | None, has_full_lineups: bool) -> float:
    src = str(source or "").strip().lower()
    base = {
        "confirmed": 1.00,
        "official": 1.00,
        "manual": 0.99,
        "override": 0.99,
        "hybrid": 0.92,
        "hybrid_projected": 0.90,
        "projected_history": 0.90,
        "projected_roster": 0.88,
        "projected": 0.82,
        "unconfirmed": 0.78,
    }.get(src, 0.80)
    completeness = 1.0 if bool(has_full_lineups) else 0.90
    return round(max(0.0, min(1.0, base * completeness)), 3)


def batting_order_pa(slot: int, is_home: bool, lineup_weight: float = 1.0) -> float:
    base = {
        1: 4.75,
        2: 4.62,
        3: 4.54,
        4: 4.44,
        5: 4.31,
        6: 4.16,
        7: 4.02,
        8: 3.89,
        9: 3.78,
    }.get(int(slot), 4.00)
    home_adj = 0.985 if is_home else 1.0
    certainty_adj = 0.96 + 0.04 * float(lineup_weight)
    return round(base * home_adj * certainty_adj, 3)


@lru_cache(maxsize=32)
def _season_batting_table() -> pd.DataFrame:
    try:
        df = batting_stats(YEAR, qual=0)
    except Exception:
        df = pd.DataFrame()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "Name" in out.columns:
        out["Name_norm"] = out["Name"].map(lambda x: clean_name(strip_pos(x)))
    return out


@lru_cache(maxsize=8)
def _statcast_window(days: int) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=int(days))
    raw = fetch_statcast_raw(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    return raw if isinstance(raw, pd.DataFrame) else pd.DataFrame()


@lru_cache(maxsize=64)
def _team_roster(team_abbr: str) -> list[dict[str, Any]]:
    ab = _canonical_team_abbr(team_abbr)
    tid = ABBR2ID.get(ab)
    out: dict[str, dict[str, Any]] = {}

    try:
        rm = roster_map(ab) or {}
        for nm, pid in rm.items():
            if pid:
                out[clean_name(nm)] = {"name": nm, "id": int(pid), "pos": ""}
    except Exception:
        pass

    if tid:
        for suffix in ("active", "40Man"):
            try:
                resp = safe_get(f"https://statsapi.mlb.com/api/v1/teams/{tid}/roster/{suffix}")
                js = resp.json() if resp else {}
                for rec in js.get("roster", []) or []:
                    person = rec.get("person", {}) or {}
                    nm = person.get("fullName") or ""
                    pid = person.get("id")
                    pos = (rec.get("position", {}) or {}).get("abbreviation") or ""
                    if nm and pid:
                        out[clean_name(nm)] = {"name": nm, "id": int(pid), "pos": str(pos).upper()}
            except Exception:
                continue

    return list(out.values())


@lru_cache(maxsize=512)
def _recent_team_lineup_history(team_abbr: str, lookback_days: int = 14) -> list[list[str]]:
    ab = _canonical_team_abbr(team_abbr)
    tid = ABBR2ID.get(ab)
    if not tid:
        return []

    today = datetime.today()
    histories: list[list[str]] = []
    for d in range(1, lookback_days + 1):
        date_str = (today - timedelta(days=d)).strftime("%Y-%m-%d")
        try:
            resp = safe_get("https://statsapi.mlb.com/api/v1/schedule", {"sportId": 1, "teamId": tid, "date": date_str})
            js = resp.json() if resp else {}
        except Exception:
            continue
        for day in js.get("dates", []) or []:
            for game in day.get("games", []) or []:
                away_id = (game.get("teams", {}).get("away", {}).get("team", {}) or {}).get("id")
                home_id = (game.get("teams", {}).get("home", {}).get("team", {}) or {}).get("id")
                side = "away" if away_id == tid else ("home" if home_id == tid else None)
                if not side:
                    continue
                pk = game.get("gamePk")
                if not pk:
                    continue
                try:
                    bx = safe_get(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore")
                    bjs = bx.json() if bx else {}
                    teams = (((bjs.get("liveData", {}) or {}).get("boxscore", {}) or {}).get("teams", {}) or {})
                    players = teams.get(side, {}).get("players", {}) or {}
                    bats = (teams.get(side, {}) or {}).get("batters", [])[:9]
                    lineup = []
                    for pid in bats:
                        rec = players.get(str(pid), {}) or {}
                        nm = ((rec.get("person", {}) or {}).get("fullName") or "").strip()
                        if nm:
                            lineup.append(strip_pos(nm))
                    if len(lineup) == 9:
                        histories.append(lineup)
                except Exception:
                    continue
    return histories


def _weighted_history_scores(team_abbr: str, pitcher_hand: str = "R") -> dict[str, dict[str, float]]:
    histories = _recent_team_lineup_history(team_abbr, lookback_days=14)
    if not histories:
        return {}

    scores: dict[str, dict[str, float]] = defaultdict(lambda: {"slot_score": 0.0, "appear_score": 0.0, "lead": 0.0, "second": 0.0, "clean": 0.0})
    n_hist = len(histories)
    for idx, lineup in enumerate(histories):
        recency = 1.0 - 0.05 * idx
        recency = max(recency, 0.35)
        for slot, nm in enumerate(lineup, 1):
            key = clean_name(strip_pos(nm))
            if not key:
                continue
            slot_bonus = max(0.0, 10.0 - slot)
            scores[key]["slot_score"] += recency * slot_bonus
            scores[key]["appear_score"] += recency
            if slot == 1:
                scores[key]["lead"] += recency
            elif slot == 2:
                scores[key]["second"] += recency
            elif slot in (3, 4, 5):
                scores[key]["clean"] += recency
    return scores


@lru_cache(maxsize=4096)
def _recent_batter_profile(pid: int, pitcher_hand: str = "R") -> dict[str, float]:
    stats_30 = _statcast_window(30)
    stats_365 = _statcast_window(365)

    def _slice(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "batter" not in df.columns:
            return pd.DataFrame()
        out = df[df["batter"] == int(pid)].copy()
        if out.empty:
            return out
        if pitcher_hand and "p_throws" in out.columns:
            hand = str(pitcher_hand).upper()[:1]
            sub = out[out["p_throws"].astype(str).str.upper().str.startswith(hand)]
            if len(sub) >= 10:
                out = sub
        return out

    def _profile(df: pd.DataFrame) -> dict[str, float]:
        if df.empty or "events" not in df.columns:
            return {"pa": 0.0, "obp": 0.315, "slg": 0.390, "iso": 0.140, "k_rate": 0.22}
        ev = df["events"].astype(str)
        pa = max(int(ev.notna().sum()), len(df))
        hits = ev.isin(["single", "double", "triple", "home_run"]).sum()
        bb = ev.isin(["walk", "hit_by_pitch"]).sum()
        tb = (
            (ev == "single").sum()
            + 2 * (ev == "double").sum()
            + 3 * (ev == "triple").sum()
            + 4 * (ev == "home_run").sum()
        )
        k = ev.isin(["strikeout", "strikeout_double_play"]).sum()
        obp = (hits + bb) / max(pa, 1)
        slg = tb / max(pa, 1)
        avg = hits / max(pa, 1)
        iso = max(slg - avg, 0.0)
        return {"pa": float(pa), "obp": float(obp), "slg": float(slg), "iso": float(iso), "k_rate": float(k / max(pa, 1))}

    p30 = _profile(_slice(stats_30))
    p365 = _profile(_slice(stats_365))
    return {
        "pa30": p30["pa"],
        "pa365": p365["pa"],
        "obp30": p30["obp"],
        "iso30": p30["iso"],
        "k30": p30["k_rate"],
        "obp365": p365["obp"],
        "iso365": p365["iso"],
        "k365": p365["k_rate"],
    }


@lru_cache(maxsize=2048)
def _season_row_by_pid(pid: int) -> dict[str, Any]:
    bat = _season_batting_table()
    if bat.empty:
        return {}
    id_col = next((c for c in bat.columns if str(c).lower() in ("playerid", "player_id", "pid", "mlbam_id")), None)
    if id_col is not None:
        try:
            hit = bat[pd.to_numeric(bat[id_col], errors="coerce") == int(pid)]
            if not hit.empty:
                return hit.iloc[0].to_dict()
        except Exception:
            pass
    return {}


@lru_cache(maxsize=2048)
def _season_row_by_name(name: str) -> dict[str, Any]:
    bat = _season_batting_table()
    if bat.empty or "Name_norm" not in bat.columns:
        return {}
    key = clean_name(strip_pos(name))
    hit = bat[bat["Name_norm"] == key]
    if hit.empty:
        return {}
    return hit.iloc[0].to_dict()


@lru_cache(maxsize=4096)
def _player_projection_features(team_abbr: str, player_name: str, pid: int, pitcher_hand: str = "R") -> dict[str, Any]:
    season = _season_row_by_pid(pid) or _season_row_by_name(player_name) or {}
    recent = _recent_batter_profile(pid, pitcher_hand)

    def _pct(v: Any, default: float) -> float:
        try:
            x = float(v)
        except Exception:
            return default
        return x / 100.0 if x > 1.0 else x

    pa_season = float(season.get("PA", 0.0) or 0.0)
    obp_season = float(season.get("OBP", 0.315) or 0.315)
    iso_season = float(season.get("ISO", 0.140) or 0.140)
    k_season = _pct(season.get("K%", 0.22), 0.22)
    sb = float(season.get("SB", 0.0) or 0.0)

    obp = 0.55 * recent["obp30"] + 0.25 * recent["obp365"] + 0.20 * obp_season
    iso = 0.45 * recent["iso30"] + 0.25 * recent["iso365"] + 0.30 * iso_season
    k_rate = 0.35 * recent["k30"] + 0.25 * recent["k365"] + 0.40 * k_season
    pa_signal = 0.55 * min(recent["pa30"], 120.0) + 0.25 * min(recent["pa365"], 600.0) / 5.0 + 0.20 * min(pa_season, 650.0) / 5.0

    bats = str(season.get("Bats", "") or season.get("bats", "") or "").upper()[:1]
    platoon = 0.0
    if bats in {"L", "R"} and pitcher_hand in {"L", "R"}:
        platoon = 0.02 if bats != pitcher_hand else -0.01
    elif bats == "S":
        platoon = 0.015

    speed = min(sb / max(pa_season, 150.0), 0.10) if pa_season > 0 else 0.01

    return {
        "name": player_name,
        "id": int(pid),
        "bats": bats or "",
        "obp": float(np.clip(obp, 0.250, 0.430)),
        "iso": float(np.clip(iso, 0.060, 0.340)),
        "k_rate": float(np.clip(k_rate, 0.08, 0.38)),
        "contact": float(np.clip(1.0 - k_rate, 0.60, 0.92)),
        "pa_signal": float(pa_signal),
        "speed": float(np.clip(speed, 0.0, 0.10)),
        "platoon": platoon,
    }


def _non_pitcher_roster(team_abbr: str) -> list[dict[str, Any]]:
    roster = _team_roster(team_abbr)
    out = []
    for rec in roster:
        pos = str(rec.get("pos", "")).upper()
        if pos.startswith("P") or pos in {"TWP"}:
            continue
        out.append(rec)
    return out


def _rank_candidates(team_abbr: str, pitcher_hand: str = "R") -> list[dict[str, Any]]:
    history_scores = _weighted_history_scores(team_abbr, pitcher_hand)
    candidates = []
    for rec in _non_pitcher_roster(team_abbr):
        name = strip_pos(rec.get("name", ""))
        pid = rec.get("id")
        if not name or not pid:
            continue
        feats = _player_projection_features(team_abbr, name, int(pid), pitcher_hand)
        h = history_scores.get(clean_name(strip_pos(name)), {})
        overall = (
            2.8 * feats["obp"]
            + 2.2 * feats["iso"]
            + 0.18 * feats["contact"]
            + 0.0016 * feats["pa_signal"]
            + 0.8 * feats["platoon"]
            + 0.055 * h.get("slot_score", 0.0)
            + 0.11 * h.get("appear_score", 0.0)
        )
        lead = 3.1 * feats["obp"] + 0.30 * feats["contact"] + 0.85 * feats["speed"] + 0.8 * feats["platoon"] + 0.35 * h.get("lead", 0.0)
        second = 3.0 * feats["obp"] + 0.25 * feats["contact"] + 0.45 * feats["iso"] + 0.5 * feats["platoon"] + 0.28 * h.get("second", 0.0)
        clean = 2.0 * feats["iso"] + 1.2 * feats["obp"] + 0.15 * feats["contact"] + 0.5 * feats["platoon"] + 0.30 * h.get("clean", 0.0)
        feats.update({"overall_score": overall, "lead_score": lead, "second_score": second, "clean_score": clean})
        candidates.append(feats)
    candidates.sort(key=lambda x: x["overall_score"], reverse=True)
    return candidates


def build_projected_lineup(team_abbr: str, pitcher_hand: str = "R", n: int = 9) -> list[str]:
    cands = _rank_candidates(team_abbr, pitcher_hand)
    if not cands:
        return DEFAULT_LINEUP[:n]

    remaining = cands.copy()
    lineup: list[dict[str, Any]] = []

    def pick(key: str) -> dict[str, Any]:
        remaining.sort(key=lambda x: x.get(key, x["overall_score"]), reverse=True)
        return remaining.pop(0)

    lineup.append(pick("lead_score"))
    lineup.append(pick("second_score"))
    remaining.sort(key=lambda x: x["overall_score"], reverse=True)
    lineup.append(remaining.pop(0))
    lineup.append(pick("clean_score"))
    remaining.sort(key=lambda x: x["clean_score"], reverse=True)
    lineup.append(remaining.pop(0))
    remaining.sort(key=lambda x: x["overall_score"], reverse=True)
    while remaining and len(lineup) < n:
        lineup.append(remaining.pop(0))

    names = [str(x["name"]) for x in lineup[:n]]
    if len(names) < n:
        names.extend(["TBD"] * (n - len(names)))
    return names


def fill_tbd_lineup(lineup: list[str] | None, team_abbr: str, pitcher_hand: str = "R", n: int = 9) -> list[str]:
    base = list(lineup or [])[:n]
    while len(base) < n:
        base.append("TBD")
    projected = build_projected_lineup(team_abbr, pitcher_hand=pitcher_hand, n=n)
    used = {clean_name(strip_pos(x)) for x in base if str(x).strip() and str(x).strip().upper() != "TBD"}
    proj_iter = [p for p in projected if clean_name(strip_pos(p)) not in used]
    out = []
    j = 0
    for nm in base:
        raw = strip_pos(str(nm or "")).strip()
        if not raw or raw.upper() == "TBD":
            if j < len(proj_iter):
                out.append(proj_iter[j])
                j += 1
            else:
                out.append("TBD")
        else:
            out.append(raw)
    return out


def lineup_needs_projection(lineup: list[str] | None) -> bool:
    vals = [strip_pos(str(x or "")).strip() for x in list(lineup or [])[:9]]
    if len([x for x in vals if x]) < 7:
        return True
    tbd = sum(1 for x in vals if (not x) or x.upper() == "TBD")
    return tbd >= 2


def finalize_lineup(lineup: list[str] | None, team_abbr: str, pitcher_hand: str = "R", n: int = 9) -> list[str]:
    if lineup_needs_projection(lineup):
        return fill_tbd_lineup(lineup, team_abbr, pitcher_hand=pitcher_hand, n=n)
    vals = [strip_pos(str(x or "")).strip() for x in list(lineup or [])[:n]]
    vals = [v if v else "TBD" for v in vals]
    if len(vals) < n:
        vals.extend(["TBD"] * (n - len(vals)))
    return vals[:n]


def projected_lineup_package(team_abbr: str, pitcher_hand: str = "R", source: str = "projected_history") -> dict[str, Any]:
    lineup = build_projected_lineup(team_abbr, pitcher_hand=pitcher_hand, n=9)
    weight = lineup_source_weight(source)
    certainty = lineup_certainty_score(source, has_full_lineups=all(x != "TBD" for x in lineup))
    return {
        "team": _canonical_team_abbr(team_abbr),
        "lineup": lineup,
        "source": source,
        "lineup_weight": weight,
        "lineup_certainty": certainty,
    }


def lineup_with_ids(lineup: list[str], team_abbr: str) -> list[dict[str, Any]]:
    roster = _team_roster(team_abbr)
    by_name = {clean_name(strip_pos(r.get("name", ""))): r for r in roster}
    out = []
    for slot, nm in enumerate(list(lineup or [])[:9], 1):
        key = clean_name(strip_pos(nm))
        rec = by_name.get(key, {})
        pid = rec.get("id")
        if not pid and key and key != "TBD":
            try:
                pid = name_to_mlbam_id(strip_pos(nm))
            except Exception:
                pid = None
        out.append({"batting_order": slot, "player_name": strip_pos(nm), "player_id": pid})
    return out


__all__ = [
    "DEFAULT_OVERRIDE_PATH",
    "DEFAULT_LINEUP",
    "TEAM_ABBR_ALIASES",
    "load_overrides",
    "get_override_for_game",
    "lineup_source_weight",
    "lineup_certainty_score",
    "batting_order_pa",
    "build_projected_lineup",
    "fill_tbd_lineup",
    "lineup_needs_projection",
    "finalize_lineup",
    "projected_lineup_package",
    "lineup_with_ids",
]
