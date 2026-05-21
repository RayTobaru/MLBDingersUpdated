from __future__ import annotations

from functools import lru_cache
from datetime import timedelta
import numpy as np
import pandas as pd

from .compat import ROOT, LEGACY, CACHE_DIR  # noqa: F401
import pickle

def _zone_hr_cache_path(kind: str, as_of_date: str, days: int) -> Path:
    safe_date = str(as_of_date).replace('/', '-').replace(' ', '_')
    return CACHE_DIR / f"zone_hr_{kind}_{safe_date}_{int(days)}.pkl"

def _read_pickle(path: Path):
    try:
        if path.exists() and path.stat().st_size > 0:
            with path.open("rb") as f:
                return pickle.load(f)
    except Exception:
        return None
    return None

def _write_pickle(path: Path, obj) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(obj, f)
    except Exception:
        pass


import fetch  # type: ignore
from fetch import fetch_statcast_raw, batter_bats, pitcher_throws

try:
    from .zone_pitch_features import _pitch_family, _find_pitcher_arch
except Exception:
    _find_pitcher_arch = None

    PITCH_FAMILY_MAP = {
        "FF": "FF_CT", "FC": "FF_CT", "FA": "FF_CT",
        "SI": "SI_FT", "FT": "SI_FT",
        "SL": "SL_SW", "ST": "SL_SW", "SV": "SL_SW",
        "CH": "CH_SP", "FS": "CH_SP", "FO": "CH_SP", "SC": "CH_SP",
        "CU": "CB_KC", "KC": "CB_KC", "CS": "CB_KC",
    }

    def _pitch_family(pt: object) -> str:
        return PITCH_FAMILY_MAP.get(str(pt or "").upper().strip(), "OTHER")


HR_EVENTS = {"home_run"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}
BBE_EVENTS = {
    "field_out", "force_out", "grounded_into_double_play", "double_play",
    "fielders_choice", "fielders_choice_out", "single", "double", "triple",
    "home_run", "sac_fly", "sac_bunt", "field_error",
}
SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "foul_bunt", "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "missed_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
CONTACT_DESCRIPTIONS = {
    "foul", "foul_tip", "foul_bunt",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
}

DEFAULTS = {
    "zone_hr_overall_mult": 1.0,
    "zone_hr_contact_mult": 1.0,
    "zone_hr_damage_mult": 1.0,
    "zone_hr_barrel_mult": 1.0,
    "zone_hr_air_mult": 1.0,
    "zone_hr_pull_air_mult": 1.0,
    "zone_hr_k_suppress_mult": 1.0,
    "zone_hr_confidence": 0.0,
    "zone_hr_candidate_source": "default",
    "zone_hr_sample_regime": "default",
    "zone_hr_batter_archetype": "",
    "zone_hr_pitcher_arch": "",
    "zone_hr_exact_batter_rows": 0.0,
    "zone_hr_batter_archetype_rows": 0.0,
    "zone_hr_pitcher_archetype_rows": 0.0,
    "zone_hr_hand_rows": 0.0,
    "zone_hr_league_rows": 0.0,
    "zone_hr_early_season_shrink": 0.0,
}


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _zone_cell_from_row(row) -> str:
    try:
        z = int(float(row.get("zone")))
        if 1 <= z <= 9:
            return f"Z{z}"
    except Exception:
        pass

    # Fallback 3x3 approximation from plate location.
    x = _safe_float(row.get("plate_x"), np.nan)
    z = _safe_float(row.get("plate_z"), np.nan)
    if not np.isfinite(x) or not np.isfinite(z):
        return "Z0"

    col = 0 if x < -0.28 else 1 if x <= 0.28 else 2
    rown = 0 if z > 3.0 else 1 if z >= 2.1 else 2
    return f"Z{rown * 3 + col + 1}"


def _count_bucket(strikes) -> int:
    try:
        s = int(float(strikes))
    except Exception:
        s = 0
    return int(np.clip(s, 0, 2))


def _tb_from_events(events: pd.Series) -> pd.Series:
    ev = events.astype(str).str.lower()
    return pd.Series(
        np.select(
            [ev.eq("single"), ev.eq("double"), ev.eq("triple"), ev.eq("home_run")],
            [1.0, 2.0, 3.0, 4.0],
            default=0.0,
        ),
        index=events.index,
    )


@lru_cache(maxsize=32)
def _hr_prepared_window(as_of_date: str, days: int = 365) -> pd.DataFrame:
    end = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(end):
        end = pd.Timestamp.today()
    start = end - timedelta(days=int(days))

    cache_path = _zone_hr_cache_path("prepared", end.strftime("%Y-%m-%d"), int(days))
    cached = _read_pickle(cache_path)
    if isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached.copy()

    try:
        raw = fetch_statcast_raw(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    except Exception:
        raw = pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    for c in ["pitcher", "batter"]:
        if c not in df.columns:
            return pd.DataFrame()
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["pitcher", "batter"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["pitcher"] = df["pitcher"].astype(int)
    df["batter"] = df["batter"].astype(int)

    if "pitch_type" not in df.columns:
        df["pitch_type"] = "OTHER"
    if "p_throws" not in df.columns:
        df["p_throws"] = "R"
    if "stand" not in df.columns:
        df["stand"] = "R"
    if "strikes" not in df.columns:
        df["strikes"] = 0
    if "description" not in df.columns:
        df["description"] = ""
    if "events" not in df.columns:
        df["events"] = ""

    df["pitch_type"] = df["pitch_type"].astype(str).str.upper().str.strip().replace({"": "OTHER"})
    df["pitch_family"] = df["pitch_type"].map(_pitch_family)
    df["zone_cell"] = df.apply(_zone_cell_from_row, axis=1)
    df["count_bucket"] = df["strikes"].map(_count_bucket)

    p_hand = df["p_throws"].astype(str).str.upper().str[:1].where(df["p_throws"].notna(), "R")
    b_side = df["stand"].astype(str).str.upper().str[:1].where(df["stand"].notna(), "R")
    df["hand_matchup"] = p_hand + b_side

    desc = df["description"].astype(str).str.lower()
    ev = df["events"].astype(str).str.lower()

    df["swing"] = desc.isin(SWING_DESCRIPTIONS).astype(int)
    df["whiff"] = desc.isin(WHIFF_DESCRIPTIONS).astype(int)
    df["contact"] = desc.isin(CONTACT_DESCRIPTIONS).astype(int)
    df["in_play"] = desc.str.startswith("hit_into_play").astype(int)
    df["hr"] = ev.eq("home_run").astype(int)
    df["hit"] = ev.isin(HIT_EVENTS).astype(int)

    ls = pd.to_numeric(df.get("launch_speed", pd.Series(index=df.index, dtype=float)), errors="coerce")
    la = pd.to_numeric(df.get("launch_angle", pd.Series(index=df.index, dtype=float)), errors="coerce")

    df["bbe"] = ((ls.notna()) | ev.isin(BBE_EVENTS) | df["in_play"].eq(1)).fillna(False).astype("int8")

    hard_mask = (ls >= 95).fillna(False)
    barrel_mask = ((ls >= 98).fillna(False) & la.between(24, 34).fillna(False))
    air_mask = ((la >= 10).fillna(False) & (la <= 50).fillna(False))

    df["hard"] = hard_mask.astype("int8")
    df["barrel"] = barrel_mask.astype("int8")
    df["air"] = air_mask.astype("int8")

    # Rough pull-air proxy from hit coordinate if available. Safe fallback otherwise.
    hc_x = pd.to_numeric(df.get("hc_x", pd.Series(index=df.index, dtype=float)), errors="coerce")
    pulled = pd.Series(False, index=df.index)
    pulled = pulled | ((b_side == "R") & (hc_x < 125))
    pulled = pulled | ((b_side == "L") & (hc_x > 125))
    df["pull_air"] = (df["air"].eq(1) & pulled.fillna(False)).astype(int)

    df["tb"] = _tb_from_events(df["events"])
    df["damage_contact"] = np.where(df["bbe"].eq(1), df["tb"], 0.0)

    # Assign batter archetypes from the same window.
    df["batter_archetype"] = _assign_batter_archetypes(df)

    _write_pickle(cache_path, df)
    return df


def _assign_batter_archetypes(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "batter" not in df.columns:
        return pd.Series("", index=df.index, dtype=object)

    tmp = df.copy()
    rows = []
    for bid, g in tmp.groupby("batter"):
        bside = str(g.get("stand", pd.Series(["R"])).mode().iloc[0])[:1].upper() if "stand" in g.columns and not g.empty else "R"
        bbe = max(float(pd.to_numeric(g.get("bbe"), errors="coerce").fillna(0).sum()), 1.0)
        swing = max(float(pd.to_numeric(g.get("swing"), errors="coerce").fillna(0).sum()), 1.0)

        hard = float(pd.to_numeric(g.get("hard"), errors="coerce").fillna(0).sum()) / bbe
        barrel = float(pd.to_numeric(g.get("barrel"), errors="coerce").fillna(0).sum()) / bbe
        air = float(pd.to_numeric(g.get("air"), errors="coerce").fillna(0).sum()) / bbe
        whiff = float(pd.to_numeric(g.get("whiff"), errors="coerce").fillna(0).sum()) / swing

        if barrel >= 0.105 or hard >= 0.44:
            power = "POWER"
        elif whiff <= 0.19 and hard < 0.38:
            power = "CONTACT"
        else:
            power = "BAL"

        if air >= 0.47:
            shape = "AIR"
        elif air <= 0.28:
            shape = "GB"
        else:
            shape = "MIX"

        rows.append((int(bid), f"{bside}|{power}|{shape}"))

    mp = dict(rows)
    return df["batter"].map(mp).fillna("R|BAL|MIX")


def _group_rates(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    metrics = ["swing", "whiff", "contact", "bbe", "hard", "barrel", "air", "pull_air", "hr", "tb", "damage_contact"]
    g = df.groupby(keys, dropna=False)[metrics].sum().reset_index()
    g["pitch_count"] = df.groupby(keys, dropna=False).size().values

    swing = pd.to_numeric(g["swing"], errors="coerce").fillna(0).clip(lower=1.0)
    bbe = pd.to_numeric(g["bbe"], errors="coerce").fillna(0).clip(lower=1.0)
    air = pd.to_numeric(g["air"], errors="coerce").fillna(0).clip(lower=1.0)

    g["contact_rate"] = pd.to_numeric(g["contact"], errors="coerce").fillna(0) / swing
    g["whiff_rate"] = pd.to_numeric(g["whiff"], errors="coerce").fillna(0) / swing
    g["hard_rate"] = pd.to_numeric(g["hard"], errors="coerce").fillna(0) / bbe
    g["barrel_rate"] = pd.to_numeric(g["barrel"], errors="coerce").fillna(0) / bbe
    g["air_rate"] = pd.to_numeric(g["air"], errors="coerce").fillna(0) / bbe
    g["pull_air_rate"] = pd.to_numeric(g["pull_air"], errors="coerce").fillna(0) / air
    g["hr_bbe_rate"] = pd.to_numeric(g["hr"], errors="coerce").fillna(0) / bbe
    g["damage_per_bbe"] = pd.to_numeric(g["damage_contact"], errors="coerce").fillna(0) / bbe

    return g


@lru_cache(maxsize=32)
def _hr_tables(as_of_date: str, days: int = 365) -> dict[str, pd.DataFrame]:
    cache_path = _zone_hr_cache_path("tables", str(as_of_date), int(days))
    cached = _read_pickle(cache_path)
    if isinstance(cached, dict) and cached:
        return cached

    df = _hr_prepared_window(as_of_date, days)
    if df is None or df.empty:
        return {}

    tabs = {
        "league_family": _group_rates(df, ["pitch_family", "zone_cell", "count_bucket"]),
        "hand_family": _group_rates(df, ["hand_matchup", "pitch_family", "zone_cell", "count_bucket"]),
        "pitcher_family": _group_rates(df, ["pitcher", "hand_matchup", "pitch_family", "zone_cell", "count_bucket"]),
        "batter_family": _group_rates(df, ["batter", "hand_matchup", "pitch_family", "zone_cell", "count_bucket"]),
        "batter_arch_family": _group_rates(df, ["batter_archetype", "hand_matchup", "pitch_family", "zone_cell", "count_bucket"]),
    }

    # Pitcher archetype table, using the derived pitcher arch helper when available.
    try:
        pt_arch = []
        pids = sorted(df["pitcher"].dropna().astype(int).unique().tolist())
        for pid in pids:
            try:
                hand = pitcher_throws(int(pid))
            except Exception:
                hand = "R"
            if callable(_find_pitcher_arch):
                arch = _find_pitcher_arch(tabs, int(pid), hand)
            else:
                arch = f"{str(hand or 'R')[:1].upper()}|OTHER|NONE"
            pt_arch.append((int(pid), arch))
        arch_map = dict(pt_arch)
        dfa = df.copy()
        dfa["pitcher_archetype"] = dfa["pitcher"].map(arch_map).fillna("R|OTHER|NONE")
        tabs["pitcher_arch_family"] = _group_rates(dfa, ["pitcher_archetype", "hand_matchup", "pitch_family", "zone_cell", "count_bucket"])
    except Exception:
        tabs["pitcher_arch_family"] = pd.DataFrame()

    _write_pickle(cache_path, tabs)
    return tabs


def _subset_one(df: pd.DataFrame, filt: dict) -> pd.Series | None:
    if df is None or df.empty:
        return None
    s = df
    for k, v in filt.items():
        if k not in s.columns:
            return None
        s = s[s[k].astype(str) == str(v)]
        if s.empty:
            return None
    return s.iloc[0] if not s.empty else None


def _row_get(row, col: str, default: float) -> float:
    if row is None:
        return float(default)
    return _safe_float(row.get(col, default), default)


def _find_batter_arch(tabs: dict[str, pd.DataFrame], batter_id: int, batter_stand: str | None = None) -> str:
    bf = tabs.get("batter_family", pd.DataFrame())
    hand = str(batter_stand or "R").upper()[:1]
    if hand not in {"L", "R"}:
        hand = "R"

    if bf is not None and not bf.empty:
        s = bf[pd.to_numeric(bf.get("batter"), errors="coerce") == int(batter_id)].copy()
        if not s.empty:
            # Reconstruct a conservative archetype from damage rates.
            bbe = max(float(pd.to_numeric(s.get("bbe"), errors="coerce").fillna(0).sum()), 1.0)
            swing = max(float(pd.to_numeric(s.get("swing"), errors="coerce").fillna(0).sum()), 1.0)
            barrel = float(pd.to_numeric(s.get("barrel"), errors="coerce").fillna(0).sum()) / bbe
            hard = float(pd.to_numeric(s.get("hard"), errors="coerce").fillna(0).sum()) / bbe
            air = float(pd.to_numeric(s.get("air"), errors="coerce").fillna(0).sum()) / bbe
            whiff = float(pd.to_numeric(s.get("whiff"), errors="coerce").fillna(0).sum()) / swing

            if barrel >= 0.105 or hard >= 0.44:
                power = "POWER"
            elif whiff <= 0.19 and hard < 0.38:
                power = "CONTACT"
            else:
                power = "BAL"

            shape = "AIR" if air >= 0.47 else "GB" if air <= 0.28 else "MIX"
            return f"{hand}|{power}|{shape}"

    return f"{hand}|BAL|MIX"


def _usage_candidates(tabs: dict[str, pd.DataFrame], pitcher_id: int, hand_matchup: str, pitcher_arch: str) -> pd.DataFrame:
    pf = tabs.get("pitcher_family", pd.DataFrame())
    paf = tabs.get("pitcher_arch_family", pd.DataFrame())
    hf = tabs.get("hand_family", pd.DataFrame())
    lf = tabs.get("league_family", pd.DataFrame())

    out = []

    if pf is not None and not pf.empty:
        s = pf[
            (pd.to_numeric(pf.get("pitcher"), errors="coerce") == int(pitcher_id)) &
            (pf.get("hand_matchup").astype(str) == str(hand_matchup))
        ].copy()
        if not s.empty:
            s["candidate_type"] = "pitcher_family"
            s["source_weight"] = 0.78
            out.append(s)

    if paf is not None and not paf.empty:
        s = paf[
            (paf.get("pitcher_archetype").astype(str) == str(pitcher_arch)) &
            (paf.get("hand_matchup").astype(str) == str(hand_matchup))
        ].copy()
        if not s.empty:
            s["candidate_type"] = "pitcher_archetype"
            s["source_weight"] = 0.56
            out.append(s)

    if hf is not None and not hf.empty:
        s = hf[hf.get("hand_matchup").astype(str) == str(hand_matchup)].copy()
        if not s.empty:
            s["candidate_type"] = "hand_matchup"
            s["source_weight"] = 0.36
            out.append(s)

    if lf is not None and not lf.empty:
        s = lf.copy()
        if not s.empty:
            s["candidate_type"] = "league"
            s["source_weight"] = 0.24
            out.append(s)

    if not out:
        return pd.DataFrame()

    c = pd.concat(out, ignore_index=True, sort=False)
    c["pitch_count"] = pd.to_numeric(c.get("pitch_count"), errors="coerce").fillna(0.0)
    c["source_weight"] = pd.to_numeric(c.get("source_weight"), errors="coerce").fillna(1.0)
    c = c[c["pitch_count"] > 0].copy()
    return c


def _context_rows(tabs: dict[str, pd.DataFrame], batter_id: int, batter_arch: str, pitcher_arch: str, hand_matchup: str, pitch_family: str, zone_cell: str, count_bucket: int) -> dict:
    return {
        "batter_family": _subset_one(tabs.get("batter_family", pd.DataFrame()), {
            "batter": int(batter_id), "hand_matchup": hand_matchup, "pitch_family": pitch_family,
            "zone_cell": zone_cell, "count_bucket": count_bucket,
        }),
        "batter_arch": _subset_one(tabs.get("batter_arch_family", pd.DataFrame()), {
            "batter_archetype": batter_arch, "hand_matchup": hand_matchup, "pitch_family": pitch_family,
            "zone_cell": zone_cell, "count_bucket": count_bucket,
        }),
        "pitcher_arch": _subset_one(tabs.get("pitcher_arch_family", pd.DataFrame()), {
            "pitcher_archetype": pitcher_arch, "hand_matchup": hand_matchup, "pitch_family": pitch_family,
            "zone_cell": zone_cell, "count_bucket": count_bucket,
        }),
        "hand": _subset_one(tabs.get("hand_family", pd.DataFrame()), {
            "hand_matchup": hand_matchup, "pitch_family": pitch_family,
            "zone_cell": zone_cell, "count_bucket": count_bucket,
        }),
        "league": _subset_one(tabs.get("league_family", pd.DataFrame()), {
            "pitch_family": pitch_family, "zone_cell": zone_cell, "count_bucket": count_bucket,
        }),
    }


def _source_counts(candidates: pd.DataFrame, sampled_rows: list[dict]) -> dict[str, float]:
    out = {
        "exact_batter": 0.0,
        "batter_archetype": 0.0,
        "pitcher_archetype": 0.0,
        "hand_matchup": 0.0,
        "league": 0.0,
    }

    for r in sampled_rows:
        for k in out:
            out[k] += float(r.get(k + "_n", 0.0) or 0.0)

    if out["pitcher_archetype"] <= 0 and candidates is not None and not candidates.empty:
        ctype = candidates.get("candidate_type", pd.Series("", index=candidates.index)).astype(str)
        pc = pd.to_numeric(candidates.get("pitch_count"), errors="coerce").fillna(0.0)
        out["pitcher_archetype"] += float(pc[ctype.eq("pitcher_archetype")].sum())
        out["hand_matchup"] += float(pc[ctype.eq("hand_matchup")].sum())
        out["league"] += float(pc[ctype.eq("league")].sum())

    return out


def _top_source(counts: dict[str, float]) -> str:
    for k in ["exact_batter", "batter_archetype", "pitcher_archetype", "hand_matchup", "league"]:
        if float(counts.get(k, 0.0) or 0.0) > 0:
            return k
    return "default"


def _season_elapsed_factor(as_of_date: str | None) -> float:
    dt = pd.to_datetime(as_of_date, errors="coerce") if as_of_date else pd.Timestamp.today()
    if pd.isna(dt):
        dt = pd.Timestamp.today()
    start = pd.Timestamp(year=int(dt.year), month=3, day=20)
    mature = pd.Timestamp(year=int(dt.year), month=7, day=15)
    return float(np.clip(max((dt - start).days, 0) / max((mature - start).days, 1), 0.0, 1.0))


def _hr_shrink(as_of_date: str | None, counts: dict[str, float], conf: float) -> float:
    elapsed = _season_elapsed_factor(as_of_date)
    exact = min(float(counts.get("exact_batter", 0.0) or 0.0) / 80.0, 1.0)
    barch = min(float(counts.get("batter_archetype", 0.0) or 0.0) / 240.0, 1.0)
    parch = min(float(counts.get("pitcher_archetype", 0.0) or 0.0) / 360.0, 1.0)
    evidence = 0.45 * exact + 0.30 * barch + 0.25 * parch
    return float(np.clip(0.38 + 0.25 * elapsed + 0.28 * evidence + 0.09 * conf, 0.30, 1.0))


def _sample_regime(counts: dict[str, float], conf: float, as_of_date: str | None) -> str:
    src = _top_source(counts)
    prefix = "early_season_" if _season_elapsed_factor(as_of_date) < 0.80 else ""
    if src == "exact_batter" and conf >= 0.45:
        return prefix + "exact_batter"
    if src == "batter_archetype" and conf >= 0.36:
        return prefix + "batter_archetype"
    if src == "pitcher_archetype" and conf >= 0.30:
        return prefix + "pitcher_archetype"
    if src == "hand_matchup":
        return prefix + "hand_matchup"
    return prefix + "league"


def summarize_zone_hr_context(
    batter_id: int | None,
    pitcher_id: int | None,
    *,
    batter_stand: str | None = None,
    pitcher_hand: str | None = None,
    as_of_date: str | None = None,
    days: int = 365,
) -> dict:
    out = dict(DEFAULTS)

    if not batter_id or not pitcher_id:
        return out

    as_of = str(as_of_date or pd.Timestamp.today().strftime("%Y-%m-%d"))
    tabs = _hr_tables(as_of, int(days))
    if not tabs:
        return out

    try:
        bstand = str(batter_stand or batter_bats(int(batter_id)) or "R").upper()[:1]
    except Exception:
        bstand = str(batter_stand or "R").upper()[:1]
    if bstand not in {"L", "R"}:
        bstand = "R"

    try:
        phand = str(pitcher_hand or pitcher_throws(int(pitcher_id)) or "R").upper()[:1]
    except Exception:
        phand = str(pitcher_hand or "R").upper()[:1]
    if phand not in {"L", "R"}:
        phand = "R"

    hand_matchup = phand + bstand
    batter_arch = _find_batter_arch(tabs, int(batter_id), bstand)

    if callable(_find_pitcher_arch):
        try:
            pitcher_arch = _find_pitcher_arch(tabs, int(pitcher_id), phand)
        except Exception:
            pitcher_arch = f"{phand}|OTHER|NONE"
    else:
        pitcher_arch = f"{phand}|OTHER|NONE"

    candidates = _usage_candidates(tabs, int(pitcher_id), hand_matchup, pitcher_arch)
    if candidates.empty:
        return out

    candidates = candidates.copy()
    candidates["raw_weight"] = (
        pd.to_numeric(candidates.get("pitch_count"), errors="coerce").fillna(0.0)
        * pd.to_numeric(candidates.get("source_weight"), errors="coerce").fillna(1.0)
    )
    candidates = candidates[candidates["raw_weight"] > 0].copy()
    if candidates.empty:
        return out

    # Keep runtime controlled.
    candidates = candidates.groupby(["pitch_family", "zone_cell", "count_bucket"], dropna=False, as_index=False)["raw_weight"].sum()
    total_w = float(candidates["raw_weight"].sum())
    if total_w <= 0:
        return out
    candidates["w"] = candidates["raw_weight"] / total_w

    parts = []
    sampled_rows = []

    for row in candidates.itertuples(index=False):
        fam = str(row.pitch_family)
        zone = str(row.zone_cell)
        count = int(row.count_bucket)
        w = float(row.w)

        rows = _context_rows(tabs, int(batter_id), batter_arch, pitcher_arch, hand_matchup, fam, zone, count)
        league = rows["league"]
        hand = rows["hand"]

        base_contact = _row_get(hand, "contact_rate", _row_get(league, "contact_rate", 0.74))
        base_whiff = _row_get(hand, "whiff_rate", _row_get(league, "whiff_rate", 0.24))
        base_damage = _row_get(hand, "damage_per_bbe", _row_get(league, "damage_per_bbe", 0.34))
        base_barrel = _row_get(hand, "barrel_rate", _row_get(league, "barrel_rate", 0.075))
        base_air = _row_get(hand, "air_rate", _row_get(league, "air_rate", 0.38))
        base_pull_air = _row_get(hand, "pull_air_rate", _row_get(league, "pull_air_rate", 0.32))

        # Evidence rows.
        bf = rows["batter_family"]
        ba = rows["batter_arch"]
        pa = rows["pitcher_arch"]

        def n_of(r):
            return _row_get(r, "pitch_count", 0.0)

        exact_n = n_of(bf)
        barch_n = n_of(ba)
        parch_n = n_of(pa)
        hand_n = n_of(hand)
        league_n = n_of(league)

        sampled_rows.append({
            "exact_batter_n": exact_n,
            "batter_archetype_n": barch_n,
            "pitcher_archetype_n": parch_n,
            "hand_matchup_n": hand_n,
            "league_n": league_n,
        })

        def mix_metric(metric: str, base: float) -> float:
            # Batter exact and batter archetype carry most of the HR signal.
            weights = []
            vals = []

            for src_row, max_w, sample_div in [
                (bf, 0.42, 70.0),
                (ba, 0.28, 220.0),
                (pa, 0.18, 320.0),
                (hand, 0.08, 600.0),
            ]:
                n = n_of(src_row)
                if src_row is not None and n > 0:
                    ww = min(n / sample_div, max_w)
                    weights.append(ww)
                    vals.append(_row_get(src_row, metric, base))

            used = min(sum(weights), 0.86)
            val = (1.0 - used) * base
            for ww, vv in zip(weights, vals):
                val += ww * vv
            return float(val)

        contact = mix_metric("contact_rate", base_contact)
        whiff = mix_metric("whiff_rate", base_whiff)
        damage = mix_metric("damage_per_bbe", base_damage)
        barrel = mix_metric("barrel_rate", base_barrel)
        air = mix_metric("air_rate", base_air)
        pull_air = mix_metric("pull_air_rate", base_pull_air)

        contact_mult = np.clip(contact / max(base_contact, 1e-6), 0.82, 1.18)
        k_suppress_mult = np.clip(base_whiff / max(whiff, 1e-6), 0.82, 1.20)
        damage_mult = np.clip(damage / max(base_damage, 1e-6), 0.72, 1.45)
        barrel_mult = np.clip(barrel / max(base_barrel, 1e-6), 0.65, 1.55)
        air_mult = np.clip(air / max(base_air, 1e-6), 0.78, 1.30)
        pull_air_mult = np.clip(pull_air / max(base_pull_air, 1e-6), 0.80, 1.25)

        overall = (
            0.15 * contact_mult
            + 0.15 * k_suppress_mult
            + 0.30 * damage_mult
            + 0.25 * barrel_mult
            + 0.10 * air_mult
            + 0.05 * pull_air_mult
        )

        parts.append({
            "w": w,
            "contact": w * float(contact_mult),
            "k_suppress": w * float(k_suppress_mult),
            "damage": w * float(damage_mult),
            "barrel": w * float(barrel_mult),
            "air": w * float(air_mult),
            "pull_air": w * float(pull_air_mult),
            "overall": w * float(overall),
        })

    if not parts:
        return out

    arr = pd.DataFrame(parts)
    counts = _source_counts(candidates, sampled_rows)
    source = _top_source(counts)

    conf = float(np.clip(
        0.16
        + 0.14 * min(float(counts.get("exact_batter", 0.0)) / 80.0, 1.0)
        + 0.13 * min(float(counts.get("batter_archetype", 0.0)) / 240.0, 1.0)
        + 0.12 * min(float(counts.get("pitcher_archetype", 0.0)) / 360.0, 1.0)
        + 0.05 * min(float(counts.get("hand_matchup", 0.0)) / 700.0, 1.0),
        0.05,
        0.72,
    ))

    shrink = _hr_shrink(as_of, counts, conf)

    def blend(raw: float, lo: float, hi: float) -> float:
        return float(np.clip(1.0 + conf * shrink * (_safe_float(raw, 1.0) - 1.0), lo, hi))

    raw_contact = float(arr["contact"].sum())
    raw_ksup = float(arr["k_suppress"].sum())
    raw_damage = float(arr["damage"].sum())
    raw_barrel = float(arr["barrel"].sum())
    raw_air = float(arr["air"].sum())
    raw_pull = float(arr["pull_air"].sum())
    raw_overall = float(arr["overall"].sum())

    out.update({
        "zone_hr_overall_mult": blend(raw_overall, 0.88, 1.16),
        "zone_hr_contact_mult": blend(raw_contact, 0.92, 1.10),
        "zone_hr_damage_mult": blend(raw_damage, 0.86, 1.18),
        "zone_hr_barrel_mult": blend(raw_barrel, 0.84, 1.20),
        "zone_hr_air_mult": blend(raw_air, 0.90, 1.12),
        "zone_hr_pull_air_mult": blend(raw_pull, 0.92, 1.10),
        "zone_hr_k_suppress_mult": blend(raw_ksup, 0.92, 1.10),
        "zone_hr_confidence": round(conf, 3),
        "zone_hr_candidate_source": source,
        "zone_hr_sample_regime": _sample_regime(counts, conf, as_of),
        "zone_hr_batter_archetype": batter_arch,
        "zone_hr_pitcher_arch": pitcher_arch,
        "zone_hr_exact_batter_rows": round(float(counts.get("exact_batter", 0.0)), 1),
        "zone_hr_batter_archetype_rows": round(float(counts.get("batter_archetype", 0.0)), 1),
        "zone_hr_pitcher_archetype_rows": round(float(counts.get("pitcher_archetype", 0.0)), 1),
        "zone_hr_hand_rows": round(float(counts.get("hand_matchup", 0.0)), 1),
        "zone_hr_league_rows": round(float(counts.get("league", 0.0)), 1),
        "zone_hr_early_season_shrink": round(float(shrink), 3),
    })

    return out
