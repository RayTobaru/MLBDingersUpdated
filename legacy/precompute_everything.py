#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
precompute_everything.py

Nightly job that:
  • Caches park factors
  • Builds RE24
  • Trains K/9 ensemble + intervals
  • Fits starter IP regression
  • Negative-binomial dispersions
  • Batter career rates
  • Per-PA isotonic calibrators for H/TB/HR (90d)
  • HR per-game calibrators by hitter archetype (90d/180d)

Exports:
  predict_k9, predict_k9_interval, sample_starter_ip, sample_pa_outcome, shrink_vs_pitcher,
  kt_montecarlo, simulate_full_game, H_theta/H_p/TB_theta/TB_p, NB_K_theta/NB_K_p,
  iso_hit_calibrators, iso_pa_calibrators, hr_game_cals, HR_LAMBDA_SCALE, infer_hr_archetype,
  expected_matchup_xiso, features_base, league_feature_means, LEAGUE_BAT, IP_REG, etc.
"""

import json
import io, pickle, cloudpickle, logging, functools
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from sklearn.linear_model import Ridge, Lasso, PoissonRegressor, QuantileRegressor, LogisticRegression, Ridge as RidgeReg
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
import math
from functools import lru_cache
import re

from fetch import (
    ID2ABBR, YEAR,
    disk_cache_h2h, fetch_retrosheet_pbp,
    get_recent_bb_stats, get_statcast_batter_features, get_statcast_pitcher_features,
    pitching_stats, batting_stats, clean_name, name_to_mlbam_id, fetch_statcast_raw,
    get_statcast_pitch_data, get_statcast_batter_data, get_batted_ball_profile,
    fetch_yearly_park_factors, fetch_monthly_park_factors,
    get_recent_pitcher_k9, get_recent_pitcher_era, select_high_leverage_reliever,
    empirical_bayes_shrink, empirical_bayes_shrink_era, get_game_weather,
    fetch_boxscore_officials, load_umpire_network_stats, get_catcher_framing_leaderboard,
    framing_runs_for, get_pitch_type_profile, pitcher_mix_last_starts, batter_xiso_by_pitch,
    team_bullpen_hrpa, tail_wind_pct, full_pitching_staff
)

try:
    from fetch import (
        build_pitcher_count_hand_state_table,
        build_batter_count_hand_state_table,
        build_pitcher_pitchtype_count_zone_state_table,
        build_batter_pitchtype_count_zone_state_table,
        get_matchup_tensor_features,
        get_starter_survival_context,
        get_hr_contact_context_prob,
    )
except Exception:
    build_pitcher_count_hand_state_table = None  # type: ignore
    build_batter_count_hand_state_table = None  # type: ignore
    build_pitcher_pitchtype_count_zone_state_table = None  # type: ignore
    build_batter_pitchtype_count_zone_state_table = None  # type: ignore
    get_matchup_tensor_features = None  # type: ignore
    get_starter_survival_context = None  # type: ignore
    get_hr_contact_context_prob = None  # type: ignore


# --- Caching paths (top of file is fine) ---
B14_PATH = "cache/b14_feats.parquet"
SEASON_SPLITS_PATH = "cache/season_hr_splits.json"
PITCH_SIDE_PATH = "cache/pitcher_vs_side_hr.json"


# --- Robust imports so Pylance never flags "undefined" ---
try:
    from multiprocessing.pool import ThreadPool as _ThreadPool
    ThreadPool = _ThreadPool
except Exception:
    ThreadPool = None  # fallback to sequential map below

try:
    from sklearn.feature_selection import RFE as _RFE
    RFE = _RFE
except Exception:
    RFE = None  # fallback: keep lasso-selected features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
CACHE_DIR = Path("./cache"); CACHE_DIR.mkdir(exist_ok=True)
print("CACHE_DIR is:", CACHE_DIR.resolve())

SIMS_K = 5000
LEAGUE_K_RATE = 0.22
LEAGUE_BF_PER_IP = 4.15
# Dirichlet smoothing strengths (small = light, big = heavy)
COUNT_PT_ALPHA_MIX = 12.0   # prior mass for P(pitch_type | count, side)
COUNT_PT_ALPHA_END = 60.0   # prior mass for P(end-with-contact at count)


# --- Season ISO lookup (cached) ---
@lru_cache(None)
def _season_iso_map(year: int = YEAR) -> dict:
    try:
        df = batting_stats(year, qual=0).copy()
        df["Name_norm"] = df.Name.apply(clean_name)
        df["pid"] = df["Name_norm"].apply(name_to_mlbam_id)
        return df.set_index("pid")["ISO"].to_dict()
    except Exception:
        return {}

def get_season_iso(pid: int | None) -> float:
    if not pid:
        return float("nan")
    try:
        return float(_season_iso_map().get(int(pid), float("nan")))
    except Exception:
        return float("nan")

# --- Temperature → HR multiplier (small, capped) ---
def _temp_multiplier(temp_f) -> float:
    # Accept int/float or strings like "78 F"
    if temp_f is None:
        return 1.0
    try:
        t = float(temp_f)
    except Exception:
        m = re.search(r"(-?\d+(\.\d+)?)", str(temp_f))
        t = float(m.group(1)) if m else float("nan")
    if not np.isfinite(t):
        return 1.0
    # ~+1% HR per +5°F above 70, capped ±7%
    return float(np.clip(1.0 + 0.01 * ((t - 70.0) / 5.0), 0.93, 1.07))

# ------------------------ cache helpers ------------------------
def disk_cache(filename):
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*a, **k):
            p = CACHE_DIR / filename
            if p.exists():
                try:
                    return cloudpickle.loads(p.read_bytes())
                except Exception:
                    pass
            res = fn(*a, **k)
            try:
                p.write_bytes(cloudpickle.dumps(res))
            except Exception:
                pass
            return res
        return wrapped
    return deco

def parquet_cache_df(key_fmt: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*a, **k):
            fname = key_fmt.format(*a, **k) if "{" in key_fmt else key_fmt
            pq = CACHE_DIR / fname
            pkl = CACHE_DIR / (fname.replace(".parquet", "") + ".pkl")
            if pq.exists():
                try:
                    return pd.read_parquet(pq)
                except Exception:
                    try: pq.unlink()
                    except Exception: pass
            if pkl.exists():
                try:
                    return pickle.loads(pkl.read_bytes())
                except Exception:
                    try: pkl.unlink()
                    except Exception: pass
            df = fn(*a, **k)
            try:
                if isinstance(df, pd.DataFrame):
                    try:
                        df.to_parquet(pq, index=False)
                    except Exception:
                        pkl.write_bytes(pickle.dumps(df))
                else:
                    pkl.write_bytes(pickle.dumps(df))
            except Exception:
                pass
            return df
        return wrapped
    return deco

def _nz_df(df):
    """Return df if it's a non-empty DataFrame, else an empty DataFrame."""
    return df if (isinstance(df, pd.DataFrame) and not df.empty) else pd.DataFrame()

# ===== NEW: lightweight caches the simulator can read quickly =====
from pathlib import Path
import os

_B14_PATH             = Path("cache/b14_feats.parquet")
_SEASON_SPLITS_PATH   = Path("cache/season_hr_splits.pkl")
_PITCHER_SIDE_HR_PATH = Path("cache/pitcher_vs_side_hr.pkl")

def build_b14_feats(days_window=180, days_recent=14, out_path=B14_PATH):
    """
    Precompute last-14-day batter features:
      - barrel_14d_pct
      - hardhit_14d_pct (EV >= 95 mph)
      - ev_mean_14d
    Writes a parquet (or csv fallback) and returns a DataFrame indexed by batter.
    """
    end = datetime.today().date()
    start = end - timedelta(days=days_window)

    df = fetch_statcast_raw(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Empty / missing case -> write empty frame with schema
    if df is None or df.empty or "batter" not in df.columns:
        out = pd.DataFrame(columns=["batter","barrel_14d_pct","hardhit_14d_pct","ev_mean_14d"]).set_index("batter")
        try: out.to_parquet(out_path)
        except Exception: out.to_csv(out_path.replace(".parquet",".csv"))
        return out

    # Filter to last N days
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df[df["game_date"] >= pd.Timestamp(end - timedelta(days=days_recent))]

    # Keep valid batters
    df = df[pd.to_numeric(df.get("batter"), errors="coerce").notna()].copy()
    if df.empty:
        out = pd.DataFrame(columns=["batter","barrel_14d_pct","hardhit_14d_pct","ev_mean_14d"]).set_index("batter")
        try:
            out.to_parquet(out_path)
        except Exception:
            out.to_csv(out_path.replace(".parquet", ".csv"))
        return out

    df["batter"] = df["batter"].astype(int)

    # Exit velocity (ls) – robust to missing column
    if "launch_speed" in df.columns:
        df["ls"] = pd.to_numeric(df["launch_speed"], errors="coerce")
    else:
        df["ls"] = np.nan  # broadcast NaN

    # Barrel flag – prefer explicit flag, fallback proxy
    if "is_barrel" not in df.columns:
        if "barrel" in df.columns:
            df["is_barrel"] = (pd.to_numeric(df["barrel"], errors="coerce") == 1)
        else:
            if "launch_angle" in df.columns:
                la = pd.to_numeric(df["launch_angle"], errors="coerce")
            else:
                la = pd.Series(np.nan, index=df.index)
            df["is_barrel"] = la.between(26, 30) & (df["ls"] >= 98)

    # ensure boolean has no NA so .mean() is safe
    df["is_barrel"] = df["is_barrel"].fillna(False)

   # --- helpers (place them just above the groupby) ---
    def _safe_mean_numeric(s):
        v = pd.to_numeric(s, errors="coerce")
        m = v.mean(skipna=True)
        return float(m) if pd.notna(m) else np.nan

    def _frac_ge_numeric(s, thresh):
        v = pd.to_numeric(s, errors="coerce")
        n = int(v.notna().sum())
        if n == 0:
            return 0.0
        return float((v[v.notna()] >= thresh).sum()) / n

    # --- aggregate robustly ---
    agg = df.groupby("batter").agg(
        barrel_14d_pct=("is_barrel", lambda s: float(s.astype("float").mean())),
        hardhit_14d_pct=("ls",       lambda s: _frac_ge_numeric(s, 95)),
        ev_mean_14d    =("ls",       _safe_mean_numeric),
    ).reset_index()

    # clean, index, save once
    agg = agg.replace([np.inf, -np.inf], np.nan).set_index("batter")
    try:
        agg.to_parquet(out_path)
    except Exception:
        agg.to_csv(out_path.replace(".parquet", ".csv"))

    return agg

def load_b14_feats(path=B14_PATH):
    """Load the 14d batter feats cache (parquet or csv). Returns DataFrame indexed by batter or empty DF."""
    try:
        return pd.read_parquet(path)
    except Exception:
        try:
            return pd.read_csv(path.replace(".parquet",".csv"), index_col=0)
        except Exception:
            return pd.DataFrame()

@disk_cache("count_pt_tables.pkl")
def build_count_pitchtype_tables(days: int = 365):
    """
    Returns (already SMOOTHED):
      {
        "PITCH": {pitcher_id: {"L": {"ALL":{pt:prob}, "0-0":{pt:prob}, ...},
                               "R": {...}}},
        "BAT":   {batter_id: {"0-0":prob, "1-0":prob, ...}},   # P(AB ends with CONTACT at that count)
        "PRIORS": {
            "end_at_count": {"0-0":p, "1-0":p, ...},          # league contact-ending distribution
            "mix": {"L": {"ALL":{pt:prob}, "0-0":{pt:prob}, ...},
                    "R": {"ALL":{pt:prob}, "0-0":{pt:prob}, ...}}
        }
      }
    (All maps are already Dirichlet-smoothed toward league priors.)
    """
    end = datetime.today()
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    df = fetch_statcast_raw(start, end.strftime("%Y-%m-%d"))

    out = {"PITCH": {}, "BAT": {}, "PRIORS": {"end_at_count": {}, "mix": {"L": {}, "R": {}}}}
    if df is None or df.empty:
        return out

    df = df.copy()
    # normalize basics
    for c in ("balls", "strikes"):
        if c not in df.columns:
            df[c] = 0
    df["count"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)

    if "stand" not in df.columns:
        df["stand"] = "R"
    df["side_bat"] = df["stand"].astype(str).str.upper().str[0].where(df["stand"].isin(["L","R"]), "R")

    if "pitch_type" not in df.columns:
        df["pitch_type"] = "FF"  # safe default

    # Contact-ending PA proxy
    contact_events = {
        "single","double","triple","home_run","field_out","force_out",
        "grounded_into_double_play","double_play","fielders_choice_out"
    }
    if "events" not in df.columns:
        df["events"] = ""
    df["is_contact_end"] = df["events"].astype(str).isin(contact_events)

    # ---------- League priors ----------
    # End-at-count prior
    end_contact = df[df["is_contact_end"]]
    if not end_contact.empty:
        lec = end_contact["count"].value_counts(normalize=True)
        out["PRIORS"]["end_at_count"] = {k: float(v) for k, v in lec.items()}
    else:
        out["PRIORS"]["end_at_count"] = {"0-0": 1.0}

    # Pitch-type prior by side and count (and ALL)
    for side in ("L","R"):
        gs = df[df["side_bat"] == side]
        if gs.empty:
            # minimal fallback
            out["PRIORS"]["mix"][side]["ALL"] = {"FF": 1.0}
            continue
        all_mix = gs["pitch_type"].value_counts(normalize=True)
        out["PRIORS"]["mix"][side]["ALL"] = {pt: float(p) for pt, p in all_mix.items()}
        for c, gc in gs.groupby("count"):
            vc = gc["pitch_type"].value_counts(normalize=True)
            out["PRIORS"]["mix"][side][c] = {pt: float(p) for pt, p in vc.items()}

    # helper: Dirichlet smooth counts -> probs using a prior dict
    def _smooth(counts: pd.Series, prior: dict[str,float], alpha: float) -> dict[str,float]:
        counts = counts.copy()
        counts = counts[counts > 0]
        N = float(counts.sum())
        # ensure the support is union of observed and prior keys
        keys = set(counts.index.tolist()) | set(prior.keys())
        # normalized prior over support
        p_prior = np.array([prior.get(k, 0.0) for k in keys], dtype=float)
        p_prior = p_prior / p_prior.sum() if p_prior.sum() > 0 else np.ones(len(keys)) / max(len(keys), 1)
        c_obs = np.array([float(counts.get(k, 0.0)) for k in keys], dtype=float)
        num = c_obs + alpha * p_prior
        den = N + alpha
        probs = num / den if den > 0 else p_prior
        return {k: float(p) for k, p in zip(keys, probs)}

    # ---------- Pitcher-specific: P(pt | count, side), smoothed toward league prior ----------
    if "pitcher" in df.columns:
        for pid, g in df.groupby("pitcher"):
            side_map = {}
            for side, gs in g.groupby("side_bat"):
                cm = {}

                # choose priors
                prior_all = out["PRIORS"]["mix"].get(side, {}).get("ALL", {"FF": 1.0})

                # ALL
                vc_all = gs["pitch_type"].value_counts()
                cm["ALL"] = _smooth(vc_all, prior_all, COUNT_PT_ALPHA_MIX)

                # per-count
                for c, gc in gs.groupby("count"):
                    vc = gc["pitch_type"].value_counts()
                    prior_c = out["PRIORS"]["mix"].get(side, {}).get(c, prior_all)
                    cm[c] = _smooth(vc, prior_c, COUNT_PT_ALPHA_MIX)

                side_map[side] = cm
            out["PITCH"][int(pid)] = side_map

    # ---------- Batter-specific: P(end-with-contact at count), smoothed toward league prior ----------
    if "batter" in df.columns:
        bg = df[df["is_contact_end"]]
        lec = out["PRIORS"]["end_at_count"]
        for bid, g in bg.groupby("batter"):
            vc = g["count"].value_counts()
            out["BAT"][int(bid)] = _smooth(vc, lec, COUNT_PT_ALPHA_END)

    return out

@disk_cache("heart_rate_60d.pkl")
def build_heart_rate_table(days: int = 60):
    """
    Returns {"league": heart_pct_league, "pitcher": {pid: heart_pct}}
    'Heart' is approximated by Statcast zone in {5, 2, 8, 4, 6} when 'zone' exists;
    else fallback to a tight box with plate_x/plate_z.
    """
    end = datetime.today()
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    df = fetch_statcast_raw(start, end.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        return {"league": 0.15, "pitcher": {}}

    df = df.copy()
    heart_mask = pd.Series(False, index=df.index)

    if "zone" in df.columns:
        # strike-zone 1..9 grid; treat crosshair + center as "heart"
        heart_z = {2,4,5,6,8}  # conservative 'heart' set
        heart_mask = df["zone"].isin(list(heart_z))
    else:
        # fallback rectangle for heart: middle 1/3 both ways (roughly)
        if {"plate_x","plate_z"}.issubset(df.columns):
            px = pd.to_numeric(df["plate_x"], errors="coerce")
            pz = pd.to_numeric(df["plate_z"], errors="coerce")
            # strike zone ~ [-0.83, 0.83] x [1.5, 3.5]. Middle third tighter:
            heart_mask = (px.abs() <= 0.28) & (pz.between(2.3, 3.0))
        else:
            return {"league": 0.15, "pitcher": {}}

    heart = heart_mask.astype(int)
    league = float(heart.mean()) if len(heart) else 0.15

    out = {"league": league, "pitcher": {}}
    if "pitcher" in df.columns:
        for pid, g in df.groupby("pitcher"):
            out["pitcher"][int(pid)] = float(heart.loc[g.index].mean()) if len(g) else league
    return out

def build_pitcher_vs_side_hr(days=365, out_path=PITCH_SIDE_PATH):
    """
    Build {pitcher_id: {"vs_L_HR_rate": float, "vs_R_HR_rate": float}}
    using last `days` of Statcast. Writes JSON and returns dict.
    HR_rate = HR / PA (where PA ~= events non-null).
    """
    end   = datetime.today()
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    sc    = fetch_statcast_raw(start, end.strftime("%Y-%m-%d"))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if sc is None or sc.empty or "pitcher" not in sc.columns:
        with open(out_path, "w") as f: json.dump({}, f)
        return {}

    g = sc.copy()
    g["pitcher"] = pd.to_numeric(g["pitcher"], errors="coerce")
    g = g.dropna(subset=["pitcher"]).copy()
    if g.empty:
        with open(out_path, "w") as f: json.dump({}, f)
        return {}

    g["pitcher"] = g["pitcher"].astype(int)
    g["is_pa"]   = g["events"].notna().astype(int)
    g["is_hr"]   = g["events"].astype(str).str.lower().eq("home_run").astype(int)

    if "stand" not in g.columns:
        with open(out_path, "w") as f: json.dump({}, f)
        return {}

    g["stand"] = g["stand"].astype(str).str.upper().str[0].where(g["stand"].isin(["L","R"]))
    g = g.dropna(subset=["stand"])

    out = {}
    for pid, sub in g.groupby("pitcher"):
        rec = {}
        for side in ("L","R"):
            ss = sub[sub["stand"] == side]
            pa = int(ss["is_pa"].sum())
            hr = int(ss["is_hr"].sum())
            rec[f"vs_{side}_HR_rate"] = (hr / max(pa, 1))
        out[int(pid)] = rec

    with open(out_path, "w") as f:
        json.dump(out, f)
    return out


def load_pitcher_vs_side_hr(path=PITCH_SIDE_PATH):
    """Load pitcher vs-side HR rates dict; return {} if missing."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}



def expected_sp_ip_simple(starter_pid: int) -> float:
    """
    Very stable SP IP expectation using season IP/GS with a small pull to league.
    Falls back to 5.4 IP if we can't see the starter in season stats.
    """
    try:
        ps = pitching_stats(YEAR, qual=0).copy()
        ps["Name_norm"] = ps.Name.apply(clean_name)
        ps["pid"] = ps["Name_norm"].apply(name_to_mlbam_id)
        rec = ps[ps["pid"] == starter_pid]
        if rec.empty:
            return 5.4
        ip = float(rec["IP"].iloc[0] or 0.0)
        gs = float(rec["GS"].iloc[0] or 1.0)
        ip_per_start = ip / max(gs, 1.0)
        # mild shrink to a leaguey 5.4 IP for robustness
        return float(0.8 * ip_per_start + 0.2 * 5.4)
    except Exception:
        return 5.4

def sp_pa_share(starter_pid: int) -> float:
    """
    Approx fraction of a batting team's PAs that occur vs the starter.
    Total team PA per game ~ 38; ~4.2 PA/inning → share ≈ IP/9.
    Clamp to reasonable bounds (15%..95%).
    """
    m = expected_sp_ip_simple(starter_pid)
    return float(np.clip(m / 9.0, 0.15, 0.95))

@disk_cache("recent_evt_feats.pkl")
def build_recent_event_features(days_short:int=1, days_med:int=7):
    """
    Returns {batter_pid: {hr_1d, pa_1d, hr_7d, pa_7d, bbe95_7d, pulled_fly_7d, xwobacon_7d}}
    pulled_fly_7d ≈ pulled + airborne (using bb_type + coarse spray proxy).
    """
    end = datetime.today()
    s1  = (end - timedelta(days=days_short)).strftime("%Y-%m-%d")
    s7  = (end - timedelta(days=days_med)).strftime("%Y-%m-%d")
    e   = end.strftime("%Y-%m-%d")

    # yesterday (or last 1d window)
    d1 = _nz_df(fetch_statcast_raw(s1, e))
    d7 = _nz_df(fetch_statcast_raw(s7, e))

    out = {}
    # --- 1d ---
    if not d1.empty:
        d1["is_pa"] = d1["events"].notna().astype(int)
        g1 = d1.groupby("batter")
        hr1 = g1["events"].apply(lambda s: (s=="home_run").sum()).rename("hr_1d")
        pa1 = g1["is_pa"].sum().rename("pa_1d")
        df1 = pd.concat([hr1, pa1], axis=1).reset_index()
    else:
        df1 = pd.DataFrame(columns=["batter","hr_1d","pa_1d"])

    # --- 7d ---
    if not d7.empty:
        d7 = d7.copy()
        d7["is_pa"] = d7["events"].notna().astype(int)
        d7 = d7.replace([np.inf,-np.inf], np.nan)
        # ensure numeric
        for c in ("launch_speed","launch_angle"):
            if c in d7.columns:
                d7[c] = pd.to_numeric(d7[c], errors="coerce")

        g7   = d7.groupby("batter")
        hr7  = g7["events"].apply(lambda s: (s=="home_run").sum()).rename("hr_7d")
        pa7  = g7["is_pa"].sum().rename("pa_7d")

        # quality contact
        bbe95 = g7.apply(lambda g: (pd.to_numeric(g.launch_speed, errors="coerce")>=95).sum()).rename("bbe95_7d")

        # pulled airborne proxy (coarse but robust to missing spray): pull≈ launch_angle within [-20,20] + fly_ball
        def _pulled_fly(g):
            la_ok = pd.to_numeric(g.launch_angle, errors="coerce").between(-20,20)
            fb_ok = (g.bb_type=="fly_ball")
            return int((la_ok & fb_ok).sum())
        pfly = g7.apply(_pulled_fly).rename("pulled_fly_7d")

        # xwOBA on contact
        xw = g7.apply(lambda g: pd.to_numeric(g.get("estimated_woba_using_speedangle", pd.Series(dtype=float)), errors="coerce").mean()) \
               .rename("xwobacon_7d")

        df7 = pd.concat([hr7, pa7, bbe95, pfly, xw], axis=1).reset_index()
    else:
        df7 = pd.DataFrame(columns=["batter","hr_7d","pa_7d","bbe95_7d","pulled_fly_7d","xwobacon_7d"])

    # merge
    df = pd.merge(df1, df7, how="outer", on="batter").fillna(0)
    for _, r in df.iterrows():
        out[int(r["batter"])] = {
            "hr_1d":        int(r.get("hr_1d", 0)),
            "pa_1d":        int(r.get("pa_1d", 0)),
            "hr_7d":        int(r.get("hr_7d", 0)),
            "pa_7d":        int(r.get("pa_7d", 0)),
            "bbe95_7d":     int(r.get("bbe95_7d", 0)),
            "pulled_fly_7d":int(r.get("pulled_fly_7d", 0)),
            "xwobacon_7d":  float(r.get("xwobacon_7d", 0.0)) if pd.notna(r.get("xwobacon_7d", np.nan)) else 0.0
        }
    return out

RECENT_EVT = build_recent_event_features()

def get_recent_evt_feats(batter_pid) -> dict:
    default = {"hr_1d":0,"pa_1d":0,"hr_7d":0,"pa_7d":0,"bbe95_7d":0,"pulled_fly_7d":0,"xwobacon_7d":0.0}
    try:
        pid = int(batter_pid)
    except Exception:
        return default
    return RECENT_EVT.get(pid, default)

@disk_cache("recent_pitcher_hr_allow.pkl")
def build_pitcher_recent_hr_allow(days:int=30):
    """
    Returns {pitcher_id: {pa_30d, hr_30d, hrpa_30d, bbe95_allowed, pulled_fly_allowed}}
    """
    end = datetime.today()
    s   = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    e   = end.strftime("%Y-%m-%d")
    df  = _nz_df(fetch_statcast_raw(s, e))
    if df.empty:
        return {}

    df = df.copy()
    df["is_pa"] = df["events"].notna().astype(int)
    for c in ("launch_speed","launch_angle"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    g   = df.groupby("pitcher")
    pa  = g["is_pa"].sum().rename("pa")
    hr  = g["events"].apply(lambda s: (s=="home_run").sum()).rename("hr")
    b95 = g.apply(lambda g2: (pd.to_numeric(g2.launch_speed, errors="coerce")>=95).sum()).rename("bbe95")

    # pulled airborne proxy per pitcher
    def _pulled_fly(g2: pd.DataFrame) -> int:
        la = pd.to_numeric(g2.get("launch_angle"), errors="coerce")
        bt = g2.get("bb_type")
        if la is None or bt is None:
            return 0
        # pulled ≈ launch_angle in [-20,20]; airborne = fly_ball
        return int(((bt == "fly_ball") & la.between(-20, 20)).sum())

    pf  = g.apply(lambda g2: _pulled_fly(g2)).rename("pulled_fly")
    dd  = pd.concat([pa,hr,b95,pf], axis=1).reset_index()
    dd["hrpa"] = dd["hr"]/dd["pa"].clip(lower=1)
    return {int(r.pitcher): {"pa_30d": int(r.pa), "hr_30d": int(r.hr),
                             "hrpa_30d": float(r.hrpa), "bbe95_allowed": int(r.bbe95),
                             "pulled_fly_allowed": int(r.pulled_fly)} for _, r in dd.iterrows()}

PIT_RECENT = build_pitcher_recent_hr_allow()

def get_pitcher_recent(pid) -> dict:
    default = {"pa_30d":0,"hr_30d":0,"hrpa_30d":0.0,"bbe95_allowed":0,"pulled_fly_allowed":0}
    try:
        pid = int(pid)
    except Exception:
        return default
    return PIT_RECENT.get(pid, default)

def batter_recent_multiplier(evt: dict) -> float:
    """
    Conservative 'hot-contact' bump:
      • HR yesterday: +3%
      • Hard-hit per PA (≥95 mph) last 7d: up to ±10%
      • Pulled airborne contact per PA last 7d: up to ±12%
      • xwOBAcon last 7d: up to ±12%
    """
    m = 1.0
    if evt.get("hr_1d", 0) > 0:
        m *= 1.03
    pa7 = max(float(evt.get("pa_7d", 0)), 1.0)
    hh_rate   = float(evt.get("bbe95_7d", 0)) / pa7
    pfly_rate = float(evt.get("pulled_fly_7d", 0)) / pa7
    xw = float(evt.get("xwobacon_7d", 0.0))

    # league-ish anchors: hh/PA≈0.10, pulled_fly/PA≈0.035, xwOBAcon≈0.360
    m *= float(np.clip(1.0 + 0.60*(hh_rate - 0.10), 0.90, 1.10))
    m *= float(np.clip(1.0 + 0.70*(pfly_rate - 0.035), 0.90, 1.12))
    if math.isfinite(xw) and xw > 0:
        m *= float(np.clip(1.0 + 0.80*(xw - 0.360), 0.90, 1.12))
    return float(np.clip(m, 0.85, 1.20))

def pitcher_recent_multiplier(rec: dict, league_hrpa: float) -> float:
    """
    If a pitcher has been allowing more HR/PA over last 30d than league, nudge up to ±20%.
    """
    hrpa = float(rec.get("hrpa_30d", 0.0))
    if hrpa <= 0 or league_hrpa <= 0:
        return 1.0
    rel = (hrpa/league_hrpa) - 1.0
    return float(np.clip(1.0 + 0.50*rel, 0.80, 1.20))


# ------------------------ league context ------------------------
_LEAGUE_RE24 = {
    (0,0):0.485,(0,1):0.856,(0,2):1.089,(0,3):1.391,(0,4):1.397,(0,5):1.761,(0,6):2.004,(0,7):2.390,
    (1,0):0.243,(1,1):0.533,(1,2):0.781,(1,3):1.109,(1,4):1.129,(1,5):1.485,(1,6):1.744,(1,7):2.071,
    (2,0):0.109,(2,1):0.294,(2,2):0.453,(2,3):0.709,(2,4):0.719,(2,5):1.075,(2,6):1.315,(2,7):1.676
}

@disk_cache("league_bat_rates.pkl")
def compute_league_bat_rates(year: int) -> dict:
    df = batting_stats(year, qual=0).copy()
    df["1B_only"] = df.H - df["2B"] - df["3B"] - df.HR
    totPA = max(float(df.PA.sum()), 1.0)
    rates = {
        "1B": df["1B_only"].sum()/totPA,
        "2B": df["2B"].sum()/totPA,
        "3B": df["3B"].sum()/totPA,
        "HR": df.HR.sum()/totPA,
        "RBI": df.RBI.sum()/totPA,
        "R": df.R.sum()/totPA
    }
    rates["TB"] = rates["1B"] + 2*rates["2B"] + 3*rates["3B"] + 4*rates["HR"]
    return rates

LEAGUE_BAT = compute_league_bat_rates(YEAR)

def _fit_nb_dispersion(arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.inf, 0.0

    mu = float(np.mean(arr))
    var = float(np.var(arr))

    if not np.isfinite(mu) or not np.isfinite(var) or mu <= 0:
        return np.inf, 0.0

    gap = var - mu
    if gap > 1e-12:
        theta = max((mu ** 2) / gap, 0.1)
        denom = theta + mu
        p = (theta / denom) if denom > 0 else 0.0
    else:
        theta, p = np.inf, 0.0

    return theta, p

@disk_cache("nb_dispersion_bat.pkl")
def compute_batter_dispersions(year: int):
    df = batting_stats(year, qual=0).copy()
    df['1B'] = df.H - df['2B'] - df['3B'] - df.HR
    df['TB'] = df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df.HR
    H_theta, H_p = _fit_nb_dispersion(df.H.values)
    TB_theta, TB_p = _fit_nb_dispersion(df.TB.values)
    return H_theta, H_p, TB_theta, TB_p

H_theta, H_p, TB_theta, TB_p = compute_batter_dispersions(YEAR)

@disk_cache("nb_dispersion_k.pkl")
def compute_strikeout_dispersion(year: int):
    df = pitching_stats(year, qual=0)

    if df is None or df.empty:
        return np.inf, 0.0

    if "SO" in df.columns:
        so = pd.to_numeric(df["SO"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).values
        return _fit_nb_dispersion(so)

    # fallback: derive SO from IP and K/9 if SO is missing
    if "IP" in df.columns and "K/9" in df.columns:
        ip = pd.to_numeric(df["IP"], errors="coerce").fillna(0.0)
        k9 = pd.to_numeric(df["K/9"], errors="coerce").fillna(0.0)
        so = (ip * k9 / 9.0).replace([np.inf, -np.inf], np.nan).fillna(0).values
        return _fit_nb_dispersion(so)

    return np.inf, 0.0

NB_K_theta, NB_K_p = compute_strikeout_dispersion(YEAR)
NB_K_theta = NB_K_theta * 0.5  # mild inflation

# ------------------------ RE24 ------------------------
def compute_re24_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    df["PA_ID"] = df.groupby("game_pk").cumcount()
    def runs_after(g):
        s = 0
        for _, r in g.iterrows():
            s += r.runs_scored
            if r.outs_before >= 3:
                break
        return s
    df["inning_id"] = df.game_pk.astype(str) + "_" + df.inning.astype(str)
    rows = []
    for (_, pa), g in df.groupby(["inning_id", "PA_ID"]):
        rows.append((int(g.outs_before.iloc[0]), int(g.bases_before.iloc[0]), runs_after(g)))
    re = pd.DataFrame(rows, columns=["outs", "bases", "runs24"])
    return re.groupby(["outs", "bases"]).runs24.mean().reset_index().rename(columns={"runs24": "RE24"})

@disk_cache("run_exp_matrix.pkl")
def build_run_expectancy_matrix():
    try:
        pbp = fetch_retrosheet_pbp(YEAR)
    except Exception as e:
        logging.warning(f"RE24 fetch error: {e}")
        pbp = pd.DataFrame()
    if not pbp.empty:
        re = compute_re24_from_pbp(pbp)
        M = np.zeros((25, 25))
        for _, r in re.iterrows():
            M[int(r.outs)*8 + int(r.bases), :] = r.RE24
        return M
    M = np.zeros((25, 25))
    for (o, b), v in _LEAGUE_RE24.items():
        M[o*8 + b, :] = v
    return M

RUN_EXP_MATRIX = build_run_expectancy_matrix()

# ------------------------ K/9 ensemble ------------------------
@disk_cache("k9_ensemble_v2.pkl")
def train_ensemble_k9_model():
    dfs = []
    for y in range(YEAR - 4, YEAR + 1):
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                df = pitching_stats(y, qual=0)
        except Exception:
            continue

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue

        needed = {"Name", "IP", "GS", "K/9", "ERA", "Team"}
        if not needed.issubset(df.columns):
            continue

        df = df.copy()
        df["Season"] = y
        df["IP_per_start"] = (pd.to_numeric(df["IP"], errors="coerce") / pd.to_numeric(df["GS"], errors="coerce").replace(0, np.nan)).fillna(0)
        df["Name_norm"] = df["Name"].apply(clean_name)
        df["pid"] = df["Name_norm"].apply(name_to_mlbam_id)
        dfs.append(df)
    if not dfs:
        return (lambda r: 8.0), (lambda r: (8.0, 8.0)), [], {}
    LP_full = pd.concat(dfs, ignore_index=True)
    seasons = sorted(LP_full.Season.unique())
    if len(seasons) < 2:
        means = {f: LP_full[f].mean() for f in LP_full.columns if f not in ("Season","Name_norm","pid")}
        return (lambda row: means.get("K%", 8.0)), (lambda row: (8.0,8.0)), [], means

    pids = [int(x) for x in LP_full.pid.dropna().unique()]
    if ThreadPool:
        with ThreadPool(8) as pool:
            sc_feats = pool.map(get_statcast_pitcher_features, pids)
            pt_feats = pool.map(get_pitch_type_profile,      pids)
    else:
        sc_feats = [get_statcast_pitcher_features(pid) for pid in pids]
        pt_feats = [get_pitch_type_profile(pid) for pid in pids]

    df_sc = pd.DataFrame(sc_feats, index=pids).reset_index().rename(columns={"index": "pid"})
    df_pt = pd.DataFrame(pt_feats, index=pids).reset_index().rename(columns={"index": "pid"})
    bb7   = [get_recent_bb_stats(pid,  7) or {} for pid in pids]
    bb14  = [get_recent_bb_stats(pid, 14) or {} for pid in pids]
    df_bb7  = pd.DataFrame(bb7,  index=pids).reset_index().rename(columns={"index": "pid"})
    df_bb14 = pd.DataFrame(bb14, index=pids).reset_index().rename(columns={"index": "pid"})

    LP_full = (LP_full.merge(df_sc,  on="pid", how="left")
                      .merge(df_pt,  on="pid", how="left")
                      .merge(df_bb7, on="pid", how="left")
                      .merge(df_bb14,on="pid", how="left"))

    LP_full["Shrunk_K9"]  = empirical_bayes_shrink(    LP_full["K/9"], LP_full["IP"])
    LP_full["Shrunk_ERA"] = empirical_bayes_shrink_era(LP_full["ERA"], LP_full["IP"])

    for c in ["ev_mean_7d","ev_std_7d","la_mean_7d","barrel_pct_7d",
          "ev_mean_14d","ev_std_14d","la_mean_14d","barrel_pct_14d"]:
        if c in LP_full.columns:
            LP_full[c] = LP_full[c].fillna(0.0)
        else:
            LP_full[c] = 0.0

    mpf = fetch_monthly_park_factors(YEAR); m = datetime.today().month
    LP_full["park_k_factor"] = LP_full["Team"].map(
        lambda t: mpf.get(t, {}).get(m, {}).get("SO", 100) / 100.0
    )

    base_feats = ["SwStr%","K%","FIP","BB%","WHIP","IP_per_start","Shrunk_K9","Shrunk_ERA",
              "FB_pct","OS_pct","SwStr_FB","SwStr_OS",
              "ev_mean_7d","ev_std_7d","la_mean_7d","barrel_pct_7d",
              "ev_mean_14d","ev_std_14d","la_mean_14d","barrel_pct_14d",
              "park_k_factor"]

    for c in ("FB_pct","OS_pct","SwStr_FB","SwStr_OS"):
        if c in LP_full.columns:
            LP_full[c] = LP_full[c].fillna(0.0)
        else:
            LP_full[c] = 0.0

    X = LP_full[base_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = pd.to_numeric(LP_full["K/9"], errors="coerce").fillna(0)

    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    if X.empty or len(X) < 5:
        league_feature_means = {f: 0.0 for f in base_feats}
        return (lambda row: 8.0), (lambda row: (6.0, 10.0)), base_feats, league_feature_means

    lasso = Lasso(alpha=0.01, max_iter=5000).fit(X, y)
    sel   = [f for f,coef in zip(base_feats, lasso.coef_) if abs(coef) > 1e-4] or base_feats

    if RFE:
        rfe = RFE(RandomForestRegressor(n_estimators=100, random_state=0),
                  n_features_to_select=min(30, len(sel))).fit(X[sel], y)
        final_feats = [f for f,s in zip(sel, rfe.support_) if s]
    else:
        final_feats = sel

    yrs = sorted(LP_full.Season.unique()); splits = []
    for i in range(len(yrs)-1):
        tr = LP_full[LP_full.Season.isin(yrs[:i+1])].index
        va = LP_full[LP_full.Season==yrs[i+1]].index
        if len(tr) > 0 and len(va) > 0:
            splits.append((tr,va))

    if len(splits) == 0:
        mean_k9 = float(y.mean()) if len(y) else 8.0
        league_feature_means = {
            f: float(X[f].mean()) if f in X.columns else 0.0
            for f in final_feats
        }
        return (
            lambda row: mean_k9,
            lambda row: (max(mean_k9 - 1.5, 0.0), mean_k9 + 1.5),
            final_feats,
            league_feature_means,
        )

    def rs(mdl,params):
        return RandomizedSearchCV(mdl, params, n_iter=10, cv=splits,
                                  scoring="neg_mean_squared_error", random_state=0).fit(X[final_feats], y)

    learners = [
        GradientBoostingRegressor(random_state=0),
        xgb.XGBRegressor(tree_method="hist", objective="reg:squarederror", random_state=0, verbosity=0),
        RandomForestRegressor(random_state=0),
        PoissonRegressor(alpha=1e-3, max_iter=1000),
        lgb.LGBMRegressor(random_state=0, verbosity=-1)
    ]
    params = [
        {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.01,0.1]},
        {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.01,0.1]},
        {"n_estimators":[100,200],"max_depth":[None,5]},
        {"alpha":[1e-3,1e-2,1e-1]},
        {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.01,0.1]}
    ]

    base = []
    for mdl, p in zip(learners, params):
        best = rs(mdl, p).best_estimator_
        best.fit(X[final_feats], y)
        base.append(best)

    oof = np.zeros((len(X), len(base)))
    for tr, va in splits:
        clones = [type(m)(**m.get_params()) for m in base]
        for i, mdl in enumerate(clones):
            mdl.fit(X.iloc[tr][final_feats], y.iloc[tr])
            oof[va, i] = mdl.predict(X.iloc[va][final_feats])

    meta_mean = Ridge(alpha=1.0).fit(oof, y)
    meta_q10  = QuantileRegressor(quantile=0.10, alpha=0).fit(oof, y)
    meta_q90  = QuantileRegressor(quantile=0.90, alpha=0).fit(oof, y)
    iso_k9    = IsotonicRegression(out_of_bounds="clip").fit(meta_mean.predict(oof), y)

    league_feature_means = {f: X[f].mean() for f in final_feats}
    logging.info("▶ K/9 ensemble ready.")

    def predict_k9(row: pd.Series) -> float:
        v = row[final_feats].fillna(0).values.reshape(1, -1)
        p = np.array([m.predict(v)[0] for m in base]).reshape(1, -1)
        return float(iso_k9.predict([float(meta_mean.predict(p)[0])])[0])

    def predict_k9_interval(row: pd.Series) -> tuple:
        v = row[final_feats].fillna(0).values.reshape(1, -1)
        p = np.array([m.predict(v)[0] for m in base]).reshape(1, -1)
        return float(meta_q10.predict(p)[0]), float(meta_q90.predict(p)[0])

    return predict_k9, predict_k9_interval, final_feats, league_feature_means

# ------------------------ Hits/TB ensemble ------------------------
@disk_cache("hit_ensemble_v2.pkl")
def train_ensemble_hit_model():
    dfs = []
    for y in range(YEAR - 4, YEAR + 1):
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                df = batting_stats(y, qual=0)
        except Exception:
            continue

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue

        needed = {"Name", "H", "2B", "3B", "HR", "PA", "BB", "HBP", "SB", "ISO"}
        if not needed.issubset(df.columns):
            continue

        df = df.copy()
        df["Season"] = y
        df["Name_norm"] = df["Name"].apply(clean_name)
        df["pid"] = df["Name_norm"].apply(name_to_mlbam_id)
        dfs.append(df)
    if not dfs:
        return (lambda r: 0.0), (lambda r: (0.0,0.0)), [], {}

    HB = pd.concat(dfs, ignore_index=True)
    HB['1B'] = HB.H - HB['2B'] - HB['3B'] - HB.HR
    HB['TB'] = HB['1B'] + 2*HB['2B'] + 3*HB['3B'] + 4*HB.HR
    HB[['1B','2B','3B','HR','TB']] = HB[['1B','2B','3B','HR','TB']].fillna(0)

    pids = [int(x) for x in HB.pid.dropna().unique()]
    if ThreadPool:
        with ThreadPool(8) as pool:
            feats = pool.map(get_statcast_batter_features, pids)
    else:
        feats = [get_statcast_batter_features(pid) for pid in pids]
    df_sc = pd.DataFrame(feats, index=pids).reset_index().rename(columns={"index":"pid"})
    HB = HB.merge(df_sc, on="pid", how="left")

    base_feats = ["PA","BB","HBP","SB","ISO"] + [c for c in df_sc.columns if c!="pid"]
    Xh = HB[base_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    yh = pd.to_numeric(HB["H"], errors="coerce").fillna(0)

    valid = Xh.notna().all(axis=1) & yh.notna()
    Xh = Xh.loc[valid].reset_index(drop=True)
    yh = yh.loc[valid].reset_index(drop=True)
    HB = HB.loc[valid].reset_index(drop=True)

    if Xh.empty or len(Xh) < 5:
        league_hit_means = {f: 0.0 for f in base_feats}
        iso_cal = {}
        return (lambda row: 0.75), (lambda row: (0.0, 2.0)), base_feats, league_hit_means, iso_cal

    lasso = Lasso(alpha=0.01, max_iter=5000).fit(Xh, yh)
    sel   = [f for f,coef in zip(base_feats, lasso.coef_) if abs(coef) > 1e-4] or base_feats

    if RFE:
        rfe = RFE(RandomForestRegressor(n_estimators=100, random_state=0),
                  n_features_to_select=min(30, len(sel))).fit(Xh[sel], yh)
        feats_fin = [f for f,s in zip(sel, rfe.support_) if s]
    else:
        feats_fin = sel

    yrs = sorted(HB.Season.unique())
    splits = []
    for i in range(len(yrs)-1):
        tr = HB[HB.Season.isin(yrs[:i+1])].index
        va = HB[HB.Season==yrs[i+1]].index
        if len(tr) > 0 and len(va) > 0:
            splits.append((tr,va))

    if len(splits) == 0:
        mean_hits = float(yh.mean()) if len(yh) else 0.75
        league_hit_means = {
            f: float(Xh[f].mean()) if f in Xh.columns else 0.0
            for f in feats_fin
        }
        iso_cal = {}
        return (
            lambda row: mean_hits,
            lambda row: (max(mean_hits - 1.0, 0.0), mean_hits + 1.0),
            feats_fin,
            league_hit_means,
            iso_cal,
        )

    def rs(mdl, params):
        return RandomizedSearchCV(mdl, params, n_iter=10, cv=splits,
                                  scoring="neg_mean_squared_error", random_state=0).fit(Xh[feats_fin], yh)

    learners = [
        GradientBoostingRegressor(random_state=0),
        xgb.XGBRegressor(tree_method="hist", objective="reg:squarederror", random_state=0, verbosity=0),
        RandomForestRegressor(random_state=0),
        PoissonRegressor(alpha=1e-3, max_iter=1000),
        lgb.LGBMRegressor(random_state=0, verbosity=-1)
    ]
    params = [
        {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.01,0.1]},
        {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.01,0.1]},
        {"n_estimators":[100,200],"max_depth":[None,5]},
        {"alpha":[1e-3,1e-2,1e-1]},
        {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.01,0.1]}
    ]

    base = []
    for mdl, p in zip(learners, params):
        best = rs(mdl, p).best_estimator_
        best.fit(Xh[feats_fin], yh)
        base.append(best)

    oofh = np.zeros((len(Xh), len(base)))
    for tr, va in splits:
        clones = [type(m)(**m.get_params()) for m in base]
        for i, mdl in enumerate(clones):
            mdl.fit(Xh.iloc[tr][feats_fin], yh.iloc[tr])
            oofh[va, i] = mdl.predict(Xh.iloc[va][feats_fin])

    mr_mean = Ridge(alpha=1.0).fit(oofh, yh)
    mr_q10  = QuantileRegressor(quantile=0.10, alpha=0).fit(oofh, yh)
    mr_q90  = QuantileRegressor(quantile=0.90, alpha=0).fit(oofh, yh)

    iso_cal = {}
    raw = mr_mean.predict(oofh)
    thresholds = {
        "P(Hits≥1)":("H",1),"P(Hits≥2)":("H",2),"P(Hits≥3)":("H",3),"P(Hits≥4)":("H",4),
        "P(1B≥1)":("1B",1),"P(2B≥1)":("2B",1),"P(3B≥1)":("3B",1),"P(HR≥1)":("HR",1),
        "P(TB≥1)":("TB",1),"P(TB≥2)":("TB",2),"P(TB≥3)":("TB",3),"P(TB≥4)":("TB",4),
        "P(RBI≥1)":("RBI",1),"P(Run≥1)":("R",1)
    }
    for ev, (col, cut) in thresholds.items():
        ybin = (HB[col] >= cut).astype(int).values
        iso_cal[ev] = IsotonicRegression(out_of_bounds="clip").fit(raw, ybin)

    league_hit_means = {f: Xh[f].mean() for f in feats_fin}
    logging.info("▶ Hit/TB ensemble ready.")

    def predict_hits(row: pd.Series) -> float:
        v = row[feats_fin].fillna(0).values.reshape(1, -1)
        p = np.array([m.predict(v)[0] for m in base]).reshape(1, -1)
        return float(mr_mean.predict(p)[0])

    def predict_hits_interval(row: pd.Series) -> tuple:
        v = row[feats_fin].fillna(0).values.reshape(1, -1)
        p = np.array([m.predict(v)[0] for m in base]).reshape(1, -1)
        return float(mr_q10.predict(p)[0]), float(mr_q90.predict(p)[0])

    return predict_hits, predict_hits_interval, feats_fin, league_hit_means, iso_cal

# ------------------------ HR models (stack + micros) ------------------------
@disk_cache("hr_feat_matrix.pkl")
def _build_hr_matrix():
    Xs, ys = [], []
    for y in range(YEAR-3, YEAR+1):
        sb = fetch_statcast_raw(f"{y}-03-01", f"{y}-11-01")
        if sb is None or sb.empty:
            continue

        sb = sb.rename(columns={"launch_speed":"exit_velocity"})
        # robust against weird NaNs / infs (keep bb_type & events non-null)
        sb = sb.replace([np.inf, -np.inf], np.nan)
        sb[["exit_velocity","launch_angle"]] = sb[["exit_velocity","launch_angle"]].fillna(0.0).astype(float)
        sb = sb.dropna(subset=["bb_type","events"])

        sb["HR"] = (sb.events=="home_run").astype(int)
        sb["barrel_pct"] = sb.get("barrel", 0.0)

        hr_sum  = sb.groupby("game_pk")["HR"].transform("sum")
        fly_sum = sb.groupby("game_pk").apply(lambda g: (g.bb_type=="fly_ball").sum()).reindex(sb.index).fillna(0)
        sb["HR_FB_rate"] = (hr_sum / fly_sum.replace(0, np.nan)).fillna(0.0)

        sb["pull_pct"]   = sb.launch_angle.between(-20,20).astype(int)
        sb["park_hr_factor"] = 1.0

        feats = ["exit_velocity","launch_angle","barrel_pct","HR_FB_rate","pull_pct","park_hr_factor"]
        Xs.append(sb[feats].fillna(0.0))
        ys.append(sb["HR"])
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)

@disk_cache("hr_base_models.pkl")
def _train_hr_bases():
    X, y = _build_hr_matrix()
    learners = [
        GradientBoostingClassifier(n_estimators=120, random_state=0),
        RandomForestClassifier(n_estimators=300, random_state=0),
        LogisticRegression(max_iter=2500),
        lgb.LGBMClassifier(n_estimators=250, random_state=0),
        xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=250, random_state=0)
    ]
    return [m.fit(X, y) for m in learners]

@disk_cache("hr_feat_cols.pkl")
def _get_hr_feat_cols():
    X, _ = _build_hr_matrix()
    return tuple(X.columns)

@disk_cache("hr_stack_meta.pkl")
def _train_hr_stacker():
    X, y = _build_hr_matrix()
    base = _train_hr_bases()
    tscv = TimeSeriesSplit(n_splits=4)
    M = np.zeros((len(X), len(base)))
    for tr, va in tscv.split(X):
        for i, mdl in enumerate(base):
            m2 = clone(mdl)
            m2.fit(X.iloc[tr], y.iloc[tr])
            M[va, i] = m2.predict_proba(X.iloc[va])[:, 1]
    meta = RidgeReg(alpha=1.0).fit(M, y)
    raw  = meta.predict(M)
    iso  = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
    return base, meta, iso

def predict_hr_proba(feat_row: pd.Series) -> float:
    base, m, iso = _train_hr_stacker()
    cols = _get_hr_feat_cols()
    xi = feat_row.reindex(cols, fill_value=0).values.reshape(1, -1)
    preds = np.array([mdl.predict_proba(xi)[:, 1][0] for mdl in base]).reshape(1, -1)
    return float(iso.predict([m.predict(preds)[0]])[0])

@disk_cache("hr_count_pt_model.pkl")
def train_hr_count_pt_model():
    sb = fetch_statcast_raw(f"{YEAR-1}-03-01", f"{YEAR-1}-11-01").dropna(subset=["balls","strikes","pitch_type","events"])
    sb["HR"] = (sb.events=="home_run").astype(int)
    sb["count"] = sb.balls.astype(str) + "-" + sb.strikes.astype(str)
    X = pd.get_dummies(sb[["count","pitch_type"]], drop_first=True)
    y = sb.HR
    return LogisticRegression(max_iter=1500).fit(X, y), X.columns.tolist()

def build_hr_count_pt_onehot(count: str, pitch_type: str) -> pd.Series:
    """
    Returns a Series aligned to the trained logistic's columns for (count, pitch_type).
    Unknown dummies are silently ignored.
    """
    row = {col: 0 for col in _hr_count_pt_cols}
    ckey = f"count_{count}"
    pkey = f"pitch_type_{pitch_type}"
    if ckey in row: row[ckey] = 1
    if pkey in row: row[pkey] = 1
    return pd.Series(row)


_hr_count_pt_model, _hr_count_pt_cols = train_hr_count_pt_model()
def predict_hr_count_pt(feat_row: pd.Series) -> float:
    X = feat_row.reindex(_hr_count_pt_cols, fill_value=0).values.reshape(1, -1)
    return float(_hr_count_pt_model.predict_proba(X)[:,1][0])

@disk_cache("hr_hardhit_model.pkl")
def train_hr_hardhit_model():
    df = fetch_statcast_raw(f"{YEAR-1}-03-01", f"{YEAR-1}-11-01").dropna(subset=["launch_speed","events"])
    df["HR"] = (df.events=="home_run").astype(int)
    return LogisticRegression(max_iter=600).fit(df[["launch_speed"]], df.HR)
_hr_hardhit_model = train_hr_hardhit_model()
def predict_hr_hardhit(feat_row: pd.Series) -> float:
    return float(_hr_hardhit_model.predict_proba([[feat_row.get("exit_velocity", 0.0)]])[:,1][0])

@disk_cache("hr_2swhiff_model.pkl")
def train_hr_2swhiff_model():
    df = fetch_statcast_raw(f"{YEAR-1}-03-01", f"{YEAR-1}-11-01")
    df = df[(df.strikes==2) & df.description.notnull()]
    df["whiff"] = df.description.str.contains("swinging_strike").astype(int)
    df["HR"]    = (df.events=="home_run").astype(int)
    return LogisticRegression(max_iter=600).fit(df[["whiff"]], df.HR)
_hr_2swhiff_model = train_hr_2swhiff_model()
def predict_hr_2swhiff(feat_row: pd.Series) -> float:
    return float(_hr_2swhiff_model.predict_proba([[feat_row.get("hr_rate_2S", 0.0)]])[:,1][0])

@disk_cache("hr_pullangle_model.pkl")
def train_hr_pullangle_model():
    df = fetch_statcast_raw(f"{YEAR-1}-03-01", f"{YEAR-1}-11-01").dropna(subset=["launch_angle","events"])
    df["pull"] = df.launch_angle.between(-20,20).astype(int)
    df["HR"]   = (df.events=="home_run").astype(int)
    return LogisticRegression(max_iter=600).fit(df[["pull"]], df.HR)
_hr_pullangle_model = train_hr_pullangle_model()
def predict_hr_pullangle(feat_row: pd.Series) -> float:
    return float(_hr_pullangle_model.predict_proba([[feat_row.get("pull_pct", 0.0)]])[:,1][0])

# ------------------------ materialize main models ------------------------
predict_k9,  predict_k9_interval,  k9_model_feats,  league_feature_means  = train_ensemble_k9_model()
predict_hits,predict_hits_interval,hit_model_feats, league_hit_means, iso_hit_calibrators = train_ensemble_hit_model()

# ------------------------ extra helpers ------------------------
features_base     = k9_model_feats  + ["Recent_K9_7d","Recent_K9_14d","Recent_ERA_7d","Recent_ERA_14d","Days_Rest","Travel_Days","IP_7d","IP_14d","Apps_7d","Apps_14d"]
features_hit_base = hit_model_feats + ["IP_7d","Apps_7d","IP_14d","Apps_14d"]

@disk_cache("h2h_lr_splits_filtered.pkl")
def compute_h2h_splits(): return {}

def expected_matchup_xiso(pit_pid:int, bat_pid:int) -> float:
    mix = pitcher_mix_last_starts(pit_pid) or {}
    bx  = batter_xiso_by_pitch(bat_pid)    or {}
    if not mix or not bx:
        return 0.0
    return float(sum(w * bx.get(pt, 0.0) for pt, w in mix.items()))

def batter_game_hr_prob(
    batter_pid: int,
    batter_stand: str | None,
    batting_team: str,         # e.g., "SEA"
    opp_team: str,             # the fielding team (their SP/bullpen)
    opp_starter_pid: int,
    exp_pa: float,
    game_pk: int | None = None
) -> float:
    """
    Estimate P(HR≥1) for a batter in a game by blending vs-starter and vs-bullpen
    HR/PA, then applying matchup xISO, park+wind+temperature, and finally calibrating
    with the archetype calibrator (if available) → per-PA isotonic → Poisson fallback.
    """

    # --- league baseline (HR per PA) ---
    league_hrpa = max(LEAGUE_BAT.get("HR", 0.025), 1e-6)

    # --- batter baseline HR/PA over 180d (shrunk to league) ---
    end = datetime.today()
    start = (end - timedelta(days=180)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    sb = fetch_statcast_raw(start, end_s)
    if sb is None or sb.empty:
        base_hrpa = league_hrpa
    else:
        b = sb[sb["batter"] == batter_pid]
        pa_b = int(b["events"].notna().sum())
        hr_b = int((b["events"] == "home_run").sum())
        prior_pa = 120
        base_hrpa = ((hr_b + league_hrpa * prior_pa) / (pa_b + prior_pa)) if (pa_b + prior_pa) > 0 else league_hrpa

    # --- opponent SP vs-side allowance; bullpen HR/PA ---
    sp_feats = get_statcast_pitcher_features(opp_starter_pid) or {}
    side = (batter_stand or "R").upper()
    if side.startswith("L"):
        sp_hr = float(sp_feats.get("vs_L_HR_rate", np.nan))
    else:
        sp_hr = float(sp_feats.get("vs_R_HR_rate", np.nan))
    f_sp = (sp_hr / league_hrpa) if (np.isfinite(sp_hr) and sp_hr > 0) else 1.0

    bp_hr, _ = team_bullpen_hrpa(opp_team, exclude_pid=opp_starter_pid)
    f_bp = (bp_hr / league_hrpa) if bp_hr > 0 else 1.0

    share_sp = sp_pa_share(opp_starter_pid)
    p_pitch = base_hrpa * (share_sp * f_sp + (1.0 - share_sp) * f_bp)

    # --- batter hot/cold bump (1d/7d) ---
    p_pitch *= batter_recent_multiplier(get_recent_evt_feats(batter_pid))

    # --- pitcher recent HR/PA allowed bump (30d) ---
    p_pitch *= pitcher_recent_multiplier(get_pitcher_recent(opp_starter_pid), league_hrpa)

    # --- matchup xISO bump (pitch mix × batter-by-pitch ISO) ---
    try:
        xiso = expected_matchup_xiso(opp_starter_pid, batter_pid)  # 0..~.4+
        # elasticity ~ ±40% for roughly ±0.08 ISO around ~.160 (clamped)
        p_pitch *= float(np.clip(1.0 + 0.4 * ((xiso - 0.160) / 0.160), 0.7, 1.3))
    except Exception:
        pass

    # --- park + wind + temperature ---
    mon = datetime.today().month
    pf = fetch_monthly_park_factors(YEAR) or {}
    pf_hr = (pf.get(opp_team, {}).get(mon, {}).get("HR", 100) / 100.0)

    wind_mul = 1.0 + float(tail_wind_pct(game_pk or 0))  # 0..~0.3
    temp_mul = 1.0
    if game_pk:
        try:
            w = get_game_weather(game_pk) or {}
            temp_mul = _temp_multiplier(w.get("temp_F"))
        except Exception:
            pass

    p_adj = p_pitch * pf_hr * wind_mul * temp_mul

    # --- per-game λ and calibration ---
    lam = float(exp_pa) * float(p_adj) * float(HR_LAMBDA_SCALE)
    lam = max(0.0, min(lam, 5.0))

    # prefer archetype calibrator if present; else per-PA iso; else Poisson
    try:
        scb = get_statcast_batter_features(batter_pid) or {}
        arch = infer_hr_archetype(
            None,
            scb.get("stand", batter_stand or "R"),
            get_season_iso(batter_pid),
            scb.get("pull_pct", None)
        )
    except Exception:
        arch = None

    if arch and ("HR", 1, arch) in hr_game_cals:
        p_game = float(hr_game_cals[("HR", 1, arch)].predict([lam])[0])
    elif ("HR", 1) in iso_pa_calibrators:
        p_game = float(iso_pa_calibrators[("HR", 1)].predict([lam])[0])
    else:
        p_game = float(1.0 - np.exp(-lam))

    return float(np.clip(p_game, 0.0, 0.999))

# ------------------------ IP regression & sampler ------------------------
@disk_cache("ip_regression.pkl")
def fit_ip_regression(year:int):
    df = pitching_stats(year, qual=0).copy()
    need_cols = ["IP", "K/9"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df.dropna(subset=need_cols)
    if df.empty:
        X = pd.DataFrame([[5.5, 8.0, 8.0]], columns=["IP", "Recent_K9_5", "Recent_K9_10"])
        y = pd.Series([5.5])
        return Ridge(alpha=1.0).fit(X, y)
    df["Recent_K9_5"]  = df["K/9"].rolling(5,  min_periods=1).mean()
    df["Recent_K9_10"] = df["K/9"].rolling(10, min_periods=1).mean()
    return Ridge(alpha=1.0).fit(df[["IP","Recent_K9_5","Recent_K9_10"]], df.IP)

IP_REG = fit_ip_regression(YEAR)

def sample_starter_ip(sip: float, rec_k9: float) -> float:
    mu = IP_REG.predict([[sip, rec_k9, rec_k9]])[0]
    return float(np.clip(np.random.normal(mu, 0.8), 0.1, sip))

# ------------------------ batter career + recent splits ------------------------
@disk_cache("batter_career_rates.pkl")
def build_batter_career_rates(years:list, weights:list=None) -> pd.DataFrame:
    if weights is None:
        weights = np.linspace(0.1, 0.4, len(years))
    frames = []
    for y, w in zip(years, weights):
        try:
            bat = batting_stats(y, qual=0)
        except Exception:
            continue
        bat = bat.copy()
        req = ["Name","PA","H","2B","3B","HR","BB","HBP","RBI","R"]
        missing = [c for c in req if c not in bat.columns]
        if missing or bat.empty:
            continue
        bat["Name_norm"] = bat.Name.apply(clean_name)
        bat["1B"] = bat.H - bat["2B"] - bat["3B"] - bat.HR
        bat["TB"] = bat["1B"] + 2*bat["2B"] + 3*bat["3B"] + 4*bat.HR
        bat["weight"] = w
        frames.append(bat[["Name_norm","PA","H","1B","2B","3B","HR","TB","BB","HBP","RBI","R","weight"]])
    if not frames:
        return pd.DataFrame(columns=["Name_norm", "PA", "H_rate", "1B_rate", "2B_rate", "3B_rate", "HR_rate", "TB_rate", "RBI_rate", "R_rate"])
    df = pd.concat(frames, ignore_index=True)
    df["wPA"] = df.PA * df.weight
    agg = df.groupby("Name_norm").apply(lambda g: pd.Series({
        "PA":       g.wPA.sum(),
        "H_rate":   (g.H   * g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
        "1B_rate":  (g["1B"]* g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
        "2B_rate":  (g["2B"]* g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
        "3B_rate":  (g["3B"]* g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
        "HR_rate":  (g.HR  * g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
        "TB_rate":  (g.TB  * g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
        "RBI_rate": (g.RBI * g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
        "R_rate":   (g.R   * g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
    })).reset_index()
    return agg

# ------------------------ Season HR splits vs R/L (cached) ------------------------
@disk_cache("season_hr_splits.pkl")
def load_season_hr_splits(year: int = YEAR, days_back: int = 365) -> dict[int, dict[str, int]]:
    """
    Returns {batter_id: {"HR_vs_R": int, "HR_vs_L": int}} using Statcast (season-to-date by default).
    Cached to avoid re-pulling each run.
    """
    try:
        end   = datetime.today()
        start = datetime(year, 3, 1) if days_back <= 0 else (end - timedelta(days=days_back))
        df    = fetch_statcast_raw(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if df is None or df.empty or "events" not in df.columns or "batter" not in df.columns:
            return {}

        ev = df.copy()
        ev["events"] = ev["events"].astype(str).str.lower()

        # Make sure we have pitcher handedness
        if "p_throws" not in ev.columns and "pitcher" in ev.columns:
            pids = ev["pitcher"].dropna().astype(int).unique().tolist()
            map_throws = {}
            for pid in pids:
                feats = get_statcast_pitcher_features(pid) or {}
                th = str(feats.get("throws", feats.get("p_throws", "R"))).upper()[:1]
                map_throws[pid] = th if th in ("R","L") else "R"
            ev["p_throws"] = ev["pitcher"].map(map_throws).fillna("R")

        if "p_throws" not in ev.columns:
            return {}

        use = ev.loc[ev["events"].eq("home_run"), ["batter", "p_throws"]].copy()
        use["p_throws"] = use["p_throws"].astype(str).str.upper().str[0]
        agg = use.groupby(["batter", "p_throws"]).size().unstack(fill_value=0)

        out = {}
        for bid, row in agg.iterrows():
            out[int(bid)] = {"HR_vs_R": int(row.get("R", 0)), "HR_vs_L": int(row.get("L", 0))}
        return out
    except Exception:
        return {}

BATTER_CAREER = build_batter_career_rates([YEAR-3, YEAR-2, YEAR-1, YEAR])

@disk_cache("recent_splits.pkl")
def build_recent_splits(days:int=30) -> dict:
    return {}

RECENT_SPLITS = build_recent_splits()

def shrink_vs_pitcher(obs_rate, obs_pa, fallback_rate, prior_pa:int=2) -> float:
    pa = max(float(obs_pa), 0.0)
    prior = max(float(prior_pa), 0.0)
    if pa <= 0:
        return float(fallback_rate)
    return float((float(obs_rate)*pa + float(fallback_rate)*prior) / (pa + prior))

# ------------------------ K Monte Carlo helpers ------------------------
def sample_pa_outcome(rate_dict: dict) -> str:
    allowed = ("1B","2B","3B","HR","BB","HBP")
    probs = {k: float(rate_dict.get(k, 0.0)) for k in allowed}
    s = float(sum(v for v in probs.values() if np.isfinite(v)))
    s = float(np.clip(s, 0.0, 0.999))  # ceiling to avoid negative OUT mass
    p_out = 1.0 - s

    events  = list(probs.keys()) + ["OUT"]
    weights = np.array(list(probs.values()) + [p_out], dtype=float)
    weights /= weights.sum() if weights.sum() > 0 else 1.0
    return np.random.choice(events, p=weights)

def kt_montecarlo(r:dict, lineup:list) -> dict:
    base   = r["Pred_K9"]
    recent = get_recent_pitcher_k9(r["Pitcher_ID"]) or base
    pred   = 0.5*base + 0.5*recent
    if r.get("FF_vel", 0) >= 95:
        pred *= 1.03
    cntL = sum("(L)" in h for h in lineup)
    cntR = sum("(R)" in h or "(S)" in h for h in lineup)
    if cntL + cntR:
        pw = ((r.get("whiff_L",0)*cntL) + (r.get("whiff_R",0)*cntR)) / (cntL + cntR)
        if pw > 0.30:
            pred *= 1.02
    mu = pred * r.get("IP_per_start", 5.0) / 9.0
    pf = fetch_monthly_park_factors(YEAR).get(r["Team"], {})
    mu *= pf.get(datetime.today().month, {}).get("SO", 100)/100.0
    if np.isfinite(NB_K_theta) and NB_K_p > 0:
        p_i = NB_K_theta/(NB_K_theta + mu) if (NB_K_theta + mu) > 0 else 1.0
        samp = nbinom(n=NB_K_theta, p=p_i).rvs(size=SIMS_K)
    else:
        samp = np.random.poisson(mu, size=SIMS_K)
    mean_k = samp.mean()
    med    = int(np.percentile(samp, 50))
    probs  = {f"P(K≥{k})": float((samp>=k).mean()) for k in range(2,11)}
    lo, hi = predict_k9_interval(pd.Series(r))
    return {"Mean_K_start": mean_k, "Median_K_start": med, **probs, "Pred_K9": round(pred,3), "90% CI": f"{int(lo)}–{int(hi)}"}

# ------------------------ HR per-game calibrators by archetype ------------------------
def infer_hr_archetype(name_norm:str|None, stand:str|None, iso:float|None, pull_pct:float|None) -> str:
    s = (stand or "R").upper()
    side = "L" if s.startswith("L") else "R"
    power = False
    try:
        if iso is not None and np.isfinite(iso) and float(iso) >= 0.200:
            power = True
    except Exception:
        pass
    try:
        if not power and pull_pct is not None and np.isfinite(pull_pct) and float(pull_pct) >= 0.45:
            power = True
    except Exception:
        pass
    return f"{side}-Power" if power else f"{side}-Contact"

@disk_cache("hr_game_cals.pkl")
def build_hr_game_calibrators(days_recent:int=90, days_rate:int=180):
    end = datetime.today()
    sr  = (end - timedelta(days=days_recent)).strftime("%Y-%m-%d")
    sr2 = (end - timedelta(days=days_rate)).strftime("%Y-%m-%d")
    e   = end.strftime("%Y-%m-%d")

    df_recent = fetch_statcast_raw(sr, e)
    if df_recent is None or df_recent.empty:
        return {}

    df_recent["is_pa"] = df_recent["events"].notna().astype(int)
    gb = df_recent.groupby(["batter","game_pk"])
    PAg = gb["is_pa"].sum().rename("PA_game")
    HRg = gb["events"].apply(lambda s: (s=="home_run").sum()).rename("C_HR")
    recent = pd.concat([PAg, HRg], axis=1).reset_index()

    df_rate = fetch_statcast_raw(sr2, e)
    if df_rate is None or df_rate.empty:
        return {}

    df_rate["is_pa"] = df_rate["events"].notna().astype(int)
    gb2 = df_rate.groupby("batter")
    PA  = gb2["is_pa"].sum().rename("PA")
    HR  = gb2["events"].apply(lambda s: (s=="home_run").sum()).rename("HR")
    rate = pd.concat([PA, HR], axis=1).reset_index()
    rate["R_HR_rate"] = rate["HR"] / rate["PA"].clip(lower=1)

    pids = [int(x) for x in recent["batter"].unique().tolist()]
    if not pids:
        return {}

    if ThreadPool:
        with ThreadPool(8) as pool:
            feats = pool.map(get_statcast_batter_features, pids)
        df_feat = pd.DataFrame(feats, index=pids).reset_index().rename(columns={"index":"batter"})
    else:
        feats = [get_statcast_batter_features(pid) for pid in pids]
        df_feat = pd.DataFrame(feats, index=pids).reset_index().rename(columns={"index":"batter"})

    # --- NEW: guarantee expected columns before the merge ---
    if "stand" not in df_feat.columns:
        df_feat["stand"] = "R"
    if "pull_pct" not in df_feat.columns:
        df_feat["pull_pct"] = 0.40
    # normalize types/sanity
    df_feat["stand"] = df_feat["stand"].astype(str).str.upper().str[0].fillna("R")
    df_feat["pull_pct"] = pd.to_numeric(df_feat["pull_pct"], errors="coerce").clip(0, 1).fillna(0.40)

    # ISO for the season (ok if this fails; we backfill NaN)
    try:
        bat = batting_stats(YEAR, qual=0).copy()
        bat["Name_norm"] = bat.Name.apply(clean_name)
        bat["pid"] = bat["Name_norm"].apply(name_to_mlbam_id)
        iso_map = bat.set_index("pid")["ISO"].to_dict()
        df_feat["ISO_season"] = df_feat["batter"].map(iso_map)
    except Exception:
        df_feat["ISO_season"] = np.nan

    data = (
        recent.merge(rate[["batter","R_HR_rate"]], on="batter", how="left")
              .merge(df_feat[["batter","stand","pull_pct","ISO_season"]], on="batter", how="left")
    )

    data["R_HR_rate"] = data["R_HR_rate"].fillna(0.0)
    data["PA_game"]   = data["PA_game"].fillna(0)
    data["lam_hat"]   = data["PA_game"] * data["R_HR_rate"]
    data["y"]         = (data["C_HR"] >= 1).astype(int)

    cals = {}
    # guard: need some variation to fit isotonic
    if data["lam_hat"].nunique() >= 2 and data["y"].nunique() >= 2:
        try:
            iso_all = IsotonicRegression(out_of_bounds="clip").fit(
                data["lam_hat"].to_numpy(dtype=float),
                data["y"].to_numpy(dtype=int)
            )
            cals[("HR", 1, "ALL")] = iso_all
            cals[("HR", 1)]        = iso_all
        except Exception:
            pass

    def _arch(row):
        return infer_hr_archetype(
            None,
            row.get("stand","R"),
            row.get("ISO_season", np.nan),
            row.get("pull_pct", np.nan)
        )

    try:
        data["arch"] = data.apply(_arch, axis=1)
        for arch, g in data.groupby("arch"):
            if len(g) < 800 or g["lam_hat"].nunique() < 2 or g["y"].nunique() < 2:
                continue
            cals[("HR", 1, arch)] = IsotonicRegression(out_of_bounds="clip").fit(
                g["lam_hat"].to_numpy(dtype=float),
                g["y"].to_numpy(dtype=int)
            )
    except Exception:
        pass

    return cals

hr_game_cals = build_hr_game_calibrators()

# Global λ scale (tunable; rough market tilt)
@disk_cache("hr_lambda_scale.pkl")
def _estimate_hr_lambda_scale() -> float:
    end = datetime.today()
    sr  = (end - timedelta(days=90)).strftime("%Y-%m-%d")
    e   = end.strftime("%Y-%m-%d")
    df  = fetch_statcast_raw(sr, e)
    if df is None or df.empty:
        return 1.15
    df["is_pa"] = df["events"].notna().astype(int)
    g   = df.groupby(["batter","game_pk"])
    pag = g["is_pa"].sum().rename("PA_game")
    y   = (g["events"].apply(lambda s: (s=="home_run").sum()) >= 1).astype(int).rename("Y")
    recent = pd.concat([pag, y], axis=1).reset_index()

    sr2 = (end - timedelta(days=180)).strftime("%Y-%m-%d")
    r   = fetch_statcast_raw(sr2, e)
    if r is None or r.empty:
        return 1.15
    r["is_pa"] = r["events"].notna().astype(int)
    b  = r.groupby("batter")
    pa = b["is_pa"].sum()
    hr = b["events"].apply(lambda s: (s=="home_run").sum())
    rate = (hr / pa.clip(lower=1)).rename("pHR")
    base = pd.concat([pa.rename("PA"), rate], axis=1).reset_index()

    dfm = recent.merge(base[["batter","pHR"]], on="batter", how="left").fillna({"pHR": 0.0})
    lam = dfm["PA_game"] * dfm["pHR"]
    y   = dfm["Y"].values.astype(int)
    grid = np.linspace(0.9, 1.35, 10)
    best_s, best_ll = 1.15, -1e18
    for s in grid:
        p = 1.0 - np.exp(-s * lam.values)
        p = np.clip(p, 1e-6, 1-1e-6)
        ll = np.sum(y*np.log(p) + (1-y)*np.log(1-p))
        if ll > best_ll:
            best_ll, best_s = ll, s
    return float(best_s)

HR_LAMBDA_SCALE = _estimate_hr_lambda_scale()

# ------------------------ Per-PA isotonic (H/TB/HR) ------------------------
@disk_cache("iso_pa_calibrators.pkl")
def build_pa_isotonic_calibrators(days_recent:int=90, days_rate:int=180):
    end = datetime.today()
    sr  = (end - timedelta(days=days_recent)).strftime("%Y-%m-%d")
    sr2 = (end - timedelta(days=days_rate)).strftime("%Y-%m-%d")
    e   = end.strftime("%Y-%m-%d")

    df_recent = fetch_statcast_raw(sr, e)
    if df_recent is None or df_recent.empty:
        return {}
    df_recent["is_pa"] = df_recent["events"].notna().astype(int)
    gb = df_recent.groupby(["batter","game_pk"])
    pa = gb["is_pa"].sum().rename("PA_game")
    h1 = gb["events"].apply(lambda s: (s=="single").sum()).rename("C_1B")
    h2 = gb["events"].apply(lambda s: (s=="double").sum()).rename("C_2B")
    h3 = gb["events"].apply(lambda s: (s=="triple").sum()).rename("C_3B")
    hr = gb["events"].apply(lambda s: (s=="home_run").sum()).rename("C_HR")
    tb = (h1 + 2*h2 + 3*h3 + 4*hr).rename("C_TB")
    recent = pd.concat([pa,h1,h2,h3,hr,tb], axis=1).reset_index()

    df_rate = fetch_statcast_raw(sr2, e)
    if df_rate is None or df_rate.empty:
        return {}
    df_rate["is_pa"] = df_rate["events"].notna().astype(int)
    gb2 = df_rate.groupby("batter")
    PA  = gb2["is_pa"].sum().rename("PA")
    R1  = gb2["events"].apply(lambda s: (s=="single").sum()).rename("R_1B")
    R2  = gb2["events"].apply(lambda s: (s=="double").sum()).rename("R_2B")
    R3  = gb2["events"].apply(lambda s: (s=="triple").sum()).rename("R_3B")
    RR  = gb2["events"].apply(lambda s: (s=="home_run").sum()).rename("R_HR")
    RTB = (R1 + 2*R2 + 3*R3 + 4*RR).rename("R_TB")
    rate = pd.concat([PA,R1,R2,R3,RR,RTB], axis=1).reset_index()
    for c in ["R_1B","R_2B","R_3B","R_HR","R_TB"]:
        rate[c+"_rate"] = rate[c] / rate["PA"].clip(lower=1)

    data = (recent.merge(rate[["batter","R_1B_rate","R_2B_rate","R_3B_rate","R_HR_rate","R_TB_rate"]],
                         on="batter", how="left")).fillna(0.0)
    data["lam_H"]  = data["PA_game"] * (data["R_1B_rate"] + data["R_2B_rate"] + data["R_3B_rate"] + data["R_HR_rate"])
    data["lam_TB"] = data["PA_game"] * data["R_TB_rate"]
    data["lam_HR"] = data["PA_game"] * data["R_HR_rate"]
    data["H_count"] = data["C_1B"] + data["C_2B"] + data["C_3B"] + data["C_HR"]

    labs = {
        ("H",1):(data["H_count"]>=1).astype(int),
        ("H",2):(data["H_count"]>=2).astype(int),
        ("H",3):(data["H_count"]>=3).astype(int),
        ("H",4):(data["H_count"]>=4).astype(int),
        ("TB",1):(data["C_TB"]>=1).astype(int),
        ("TB",2):(data["C_TB"]>=2).astype(int),
        ("TB",3):(data["C_TB"]>=3).astype(int),
        ("TB",4):(data["C_TB"]>=4).astype(int),
        ("HR",1):(data["C_HR"]>=1).astype(int),
    }
    xs = {("H",k): data["lam_H"]  for k in (1,2,3,4)} \
       | {("TB",k): data["lam_TB"] for k in (1,2,3,4)} \
       | {("HR",1): data["lam_HR"]}

    cals = {}
    for key in labs:
        cals[key] = IsotonicRegression(out_of_bounds="clip").fit(xs[key].values.astype(float),
                                                                 labs[key].values.astype(int))
    return cals

iso_pa_calibrators = build_pa_isotonic_calibrators()

def _sanity_check_pa_calibrators(cals: dict) -> None:
    try:
        keys = list(cals.keys())
        logging.info(f"iso_pa_calibrators: {len(keys)} keys; sample {keys[:6]}")
        test = np.array([0.0, 0.25, 0.5, 1.0], dtype=float)
        for k in [("HR",1), ("H",1), ("TB",2)]:
            if k in cals:
                y = np.asarray(cals[k].predict(test))
                if np.any(np.diff(y) < -1e-8):
                    logging.warning(f"Non-monotone calibrator {k} over test points.")
    except Exception as e:
        logging.warning(f"Calibrator sanity check skipped: {e}")

_sanity_check_pa_calibrators(iso_pa_calibrators)

# ------------------------ Simulation helpers (added back) ------------------------
def simulate_markov_inning(pitcher_profile: dict,
                           batter_rates: list[dict],
                           transition_matrix=None) -> int:
    """
    Very simple Markov inning using event rates in batter_rates.
    State = outs*8 + bases_bitmask(1B,2B,3B).
    """
    state, runs, outs = 0, 0, 0
    idx = 0
    while outs < 3:
        rates = batter_rates[idx % len(batter_rates)]
        evt = sample_pa_outcome(rates)

        if evt == 'HR':
            b = state % 8
            runs += (b & 1) + ((b & 2) >> 1) + ((b & 4) >> 2) + 1
            state = outs*8 + 0

        elif evt == '3B':
            b = state % 8
            runs += (b & 1) + ((b & 2) >> 1) + ((b & 4) >> 2)
            state = outs*8 + 4

        elif evt == '2B':
            b = state % 8
            runs += ((b & 2) >> 1) + ((b & 4) >> 2)
            state = outs*8 + (2 | ((b & 1) << 1))

        elif evt == '1B':
            b = state % 8
            runs += ((b & 4) >> 2)
            new_b = ((b << 1) & 7) | 1
            state = outs*8 + new_b

        elif evt in ('BB','HBP'):
            b = state % 8
            if b == 7:
                runs += 1
            state = outs*8 + ((b | 1) & 7)

        else:  # OUT
            outs += 1
            state = outs*8 + (state % 8)

        idx += 1

    return runs

def simulate_reliever(rel: dict,
                      batter_rates: list[dict],
                      bullpen_profile: dict,
                      base_out_state: int) -> dict:
    if not rel:
        ip = np.random.uniform(0.1, 1.0)
        return {"IP": ip, "Runs_Allowed": poisson.rvs(1.5), "Pitcher_ID": None}

    season_ip = rel.get("Season_IP", 10.0)
    rec_k9    = rel.get("Recent_K9_30d", rel.get("Season_K9", 8.5))
    ip        = sample_starter_ip(season_ip, rec_k9)
    ip        = float(np.clip(ip, 0.1, 2.0))

    runs = 0
    outs_int = int(np.floor(ip))
    for _ in range(outs_int):
        runs += simulate_markov_inning(rel, batter_rates, RUN_EXP_MATRIX)

    frac = ip - outs_int
    if frac > 0:
        runs += int(round(frac * simulate_markov_inning(rel, batter_rates, RUN_EXP_MATRIX)))

    return {"IP": ip, "Runs_Allowed": runs, "Pitcher_ID": rel.get("Pitcher_ID")}

def simulate_full_game(team_abbr: str,
                       lineup: list,
                       starter_stats: dict,
                       df_rel: pd.DataFrame,
                       bullpen_profile: dict,
                       batter_rates: list[dict]) -> dict:
    ip_allowed = min(
        starter_stats.get("IP_per_start", 5.0),
        sample_starter_ip(starter_stats.get("IP_per_start", 5.0), starter_stats.get("Pred_K9", 8.0))
    )

    runs = 0
    outs_int = int(np.floor(ip_allowed))
    for _ in range(outs_int):
        runs += simulate_markov_inning(starter_stats, batter_rates, RUN_EXP_MATRIX)

    frac = ip_allowed - outs_int
    if frac > 0:
        runs += int(round(frac * simulate_markov_inning(starter_stats, batter_rates, RUN_EXP_MATRIX)))

    usage, used, inning = [], set(), outs_int + 1
    while inning <= 9:
        rel = select_high_leverage_reliever(df_rel, used, inning)
        if not rel:
            break
        used.add(rel["Pitcher_ID"])
        out = simulate_reliever(rel, batter_rates, bullpen_profile, 0)
        runs += out["Runs_Allowed"]
        usage.append((rel["Name"], out))
        inning += int(np.ceil(out["IP"]))

    return {"Runs": runs, "Usage": usage}



# ------------------------ K decomposition + matchup scoring ------------------------
def _derive_pitching_bf(df: pd.DataFrame) -> pd.Series:
    """
    Robust batters-faced estimate.
    Prefer BF when available; otherwise derive from outs + baserunners.
    """
    if "BF" in df.columns:
        bf = pd.to_numeric(df["BF"], errors="coerce")
        if bf.notna().any():
            return bf.fillna(0.0)

    ip = pd.to_numeric(df.get("IP", 0.0), errors="coerce").fillna(0.0)
    h  = pd.to_numeric(df.get("H", 0.0), errors="coerce").fillna(0.0)
    bb = pd.to_numeric(df.get("BB", 0.0), errors="coerce").fillna(0.0)
    hbp = pd.to_numeric(df.get("HBP", 0.0), errors="coerce").fillna(0.0)
    # BF ~= outs recorded + reached-base events
    return (ip * 3.0 + h + bb + hbp).fillna(0.0)


def _safe_pitching_target_frame() -> pd.DataFrame:
    """
    Season-stacked pitcher frame used for decomposition models:
      y_ip_start, y_bf_per_ip, y_k_per_bf
    """
    dfs = []
    for y in range(YEAR - 4, YEAR + 1):
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                df = pitching_stats(y, qual=0)
        except Exception:
            continue
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue

        df = df.copy()
        need_any = {"Name", "IP", "GS"}
        if not need_any.issubset(df.columns):
            continue

        if "SO" not in df.columns:
            if "K/9" in df.columns:
                ip_num = pd.to_numeric(df["IP"], errors="coerce").fillna(0.0)
                k9_num = pd.to_numeric(df["K/9"], errors="coerce").fillna(0.0)
                df["SO"] = (ip_num * k9_num / 9.0).fillna(0.0)
            else:
                df["SO"] = 0.0

        df["Season"] = y
        df["Name_norm"] = df["Name"].apply(clean_name)
        df["pid"] = df["Name_norm"].apply(name_to_mlbam_id)
        df["BF_est"] = _derive_pitching_bf(df)

        ip = pd.to_numeric(df["IP"], errors="coerce").fillna(0.0)
        gs = pd.to_numeric(df["GS"], errors="coerce").replace(0, np.nan)
        so = pd.to_numeric(df["SO"], errors="coerce").fillna(0.0)
        bf = pd.to_numeric(df["BF_est"], errors="coerce").fillna(0.0)

        df["y_ip_start"] = (ip / gs).clip(lower=0.5, upper=9.0).fillna(0.0)
        df["y_bf_per_ip"] = (bf / ip.replace(0, np.nan)).clip(lower=2.2, upper=6.0).fillna(0.0)
        df["y_k_per_bf"] = (so / bf.replace(0, np.nan)).clip(lower=0.03, upper=0.50).fillna(0.0)

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


@disk_cache("k_decomposition_models_v1.pkl")
def train_k_decomposition_models():
    LP = _safe_pitching_target_frame()
    if LP.empty:
        feat_cols = ["Shrunk_K9", "Shrunk_ERA", "IP_per_start", "park_k_factor"]
        means = {f: 0.0 for f in feat_cols}
        return (
            (lambda row: 5.4),
            (lambda row: 4.20),
            (lambda row: 0.235),
            (lambda row: 9.0 * 4.20 * 0.235),
            (lambda row: 5.4 * 4.20 * 0.235),
            feat_cols,
            means,
        )

    pids = [int(x) for x in LP.pid.dropna().unique()]
    if ThreadPool:
        with ThreadPool(8) as pool:
            sc_feats = pool.map(get_statcast_pitcher_features, pids)
            pt_feats = pool.map(get_pitch_type_profile, pids)
    else:
        sc_feats = [get_statcast_pitcher_features(pid) for pid in pids]
        pt_feats = [get_pitch_type_profile(pid) for pid in pids]

    df_sc = pd.DataFrame(sc_feats, index=pids).reset_index().rename(columns={"index": "pid"})
    df_pt = pd.DataFrame(pt_feats, index=pids).reset_index().rename(columns={"index": "pid"})
    bb7   = [get_recent_bb_stats(pid, 7) or {} for pid in pids]
    bb14  = [get_recent_bb_stats(pid, 14) or {} for pid in pids]
    df_bb7  = pd.DataFrame(bb7,  index=pids).reset_index().rename(columns={"index": "pid"})
    df_bb14 = pd.DataFrame(bb14, index=pids).reset_index().rename(columns={"index": "pid"})

    LP = (
        LP.merge(df_sc, on="pid", how="left")
          .merge(df_pt, on="pid", how="left")
          .merge(df_bb7, on="pid", how="left")
          .merge(df_bb14, on="pid", how="left")
    )

    LP["Shrunk_K9"]  = empirical_bayes_shrink(LP["K/9"], LP["IP"]) if "K/9" in LP.columns else 8.5
    LP["Shrunk_ERA"] = empirical_bayes_shrink_era(LP["ERA"], LP["IP"]) if "ERA" in LP.columns else 4.2
    LP["IP_per_start"] = pd.to_numeric(LP["y_ip_start"], errors="coerce").fillna(0.0)

    for c in [
        "SwStr%","K%","FIP","BB%","WHIP","FB_pct","OS_pct","SwStr_FB","SwStr_OS",
        "ev_mean_7d","ev_std_7d","la_mean_7d","barrel_pct_7d",
        "ev_mean_14d","ev_std_14d","la_mean_14d","barrel_pct_14d"
    ]:
        if c not in LP.columns:
            LP[c] = 0.0
        LP[c] = pd.to_numeric(LP[c], errors="coerce").fillna(0.0)

    mpf = fetch_monthly_park_factors(YEAR)
    m = datetime.today().month
    LP["park_k_factor"] = LP["Team"].map(
        lambda t: (mpf.get(t, {}).get(m, {}).get("SO", 100) / 100.0) if pd.notna(t) else 1.0
    ).fillna(1.0)

    feat_cols = [
        "SwStr%","K%","FIP","BB%","WHIP","IP_per_start","Shrunk_K9","Shrunk_ERA",
        "FB_pct","OS_pct","SwStr_FB","SwStr_OS",
        "ev_mean_7d","ev_std_7d","la_mean_7d","barrel_pct_7d",
        "ev_mean_14d","ev_std_14d","la_mean_14d","barrel_pct_14d",
        "park_k_factor"
    ]
    X = LP[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y_ip   = pd.to_numeric(LP["y_ip_start"], errors="coerce").fillna(0.0)
    y_bfip = pd.to_numeric(LP["y_bf_per_ip"], errors="coerce").fillna(0.0)
    y_kbf  = pd.to_numeric(LP["y_k_per_bf"], errors="coerce").fillna(0.0)

    valid = X.notna().all(axis=1) & y_ip.notna() & y_bfip.notna() & y_kbf.notna()
    X = X.loc[valid].reset_index(drop=True)
    y_ip = y_ip.loc[valid].reset_index(drop=True)
    y_bfip = y_bfip.loc[valid].reset_index(drop=True)
    y_kbf = y_kbf.loc[valid].reset_index(drop=True)

    if X.empty or len(X) < 20:
        means = {f: float(X[f].mean()) if f in X.columns and len(X) else 0.0 for f in feat_cols}
        return (
            (lambda row: 5.4),
            (lambda row: 4.20),
            (lambda row: 0.235),
            (lambda row: 9.0 * 4.20 * 0.235),
            (lambda row: 5.4 * 4.20 * 0.235),
            feat_cols,
            means,
        )

    def _fit_reg(y: pd.Series, clip_lo: float, clip_hi: float):
        gb = GradientBoostingRegressor(random_state=0, n_estimators=200, learning_rate=0.05, max_depth=3)
        rf = RandomForestRegressor(random_state=0, n_estimators=300, max_depth=6, min_samples_leaf=3)
        rd = Ridge(alpha=1.0)
        gb.fit(X, y)
        rf.fit(X, y)
        oof = np.column_stack([gb.predict(X), rf.predict(X)])
        rd.fit(oof, y)

        def _pred(row: pd.Series) -> float:
            vals = row.reindex(feat_cols).fillna(pd.Series(means)).astype(float).values.reshape(1, -1)
            base = np.column_stack([gb.predict(vals), rf.predict(vals)])
            return float(np.clip(rd.predict(base)[0], clip_lo, clip_hi))
        return _pred

    means = {f: float(X[f].mean()) for f in feat_cols}
    pred_ip = _fit_reg(y_ip.clip(0.5, 9.0), 0.5, 9.0)
    pred_bfip = _fit_reg(y_bfip.clip(2.2, 6.0), 2.2, 6.0)
    pred_kbf = _fit_reg(y_kbf.clip(0.03, 0.50), 0.03, 0.50)

    def pred_k9_decomp(row: pd.Series) -> float:
        return float(np.clip(9.0 * pred_bfip(row) * pred_kbf(row), 3.0, 16.0))

    def pred_k_start_decomp(row: pd.Series) -> float:
        return float(np.clip(pred_ip(row) * pred_bfip(row) * pred_kbf(row), 0.0, 15.0))

    return pred_ip, pred_bfip, pred_kbf, pred_k9_decomp, pred_k_start_decomp, feat_cols, means


(
    predict_ip_start,
    predict_bf_per_ip,
    predict_k_per_bf,
    predict_k9_decomp,
    predict_k_start_decomp,
    k_decomp_feature_cols,
    league_decomp_feature_means,
) = train_k_decomposition_models()


# ------------------------ Pitch-type / count matchup tables ------------------------
@disk_cache("pitchtype_matchup_tables_v1.pkl")
def build_pitchtype_matchup_tables(days: int = 365):
    """
    Precompute batter and pitcher pitch-type tendency tables for matchup-aware K/HR logic.
    """
    end = datetime.today()
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    df = fetch_statcast_raw(start, end.strftime("%Y-%m-%d"))
    out = {"BATTER_PT": {}, "PITCHER_PT": {}, "BATTER_COUNT": {}, "PITCHER_COUNT": {}}
    if df is None or df.empty:
        return out

    g = df.copy()
    if "pitch_type" not in g.columns:
        g["pitch_type"] = "FF"
    if "balls" not in g.columns:
        g["balls"] = 0
    if "strikes" not in g.columns:
        g["strikes"] = 0
    if "description" not in g.columns:
        g["description"] = ""

    g["count"] = g["balls"].astype(int).astype(str) + "-" + g["strikes"].astype(int).astype(str)
    desc = g["description"].astype(str).str.lower()
    g["is_whiff"] = desc.str.contains("swinging_strike", na=False).astype(int)
    g["is_contact"] = g["events"].astype(str).isin(["single","double","triple","home_run","field_out","force_out"]).astype(int)

    if "batter" in g.columns:
        for bid, sub in g.groupby("batter"):
            pt = sub.groupby("pitch_type").agg(
                pitches=("pitch_type", "size"),
                whiff_rate=("is_whiff", "mean"),
                contact_rate=("is_contact", "mean"),
            )
            out["BATTER_PT"][int(bid)] = {
                str(k): {
                    "pitches": int(v["pitches"]),
                    "whiff_rate": float(v["whiff_rate"]),
                    "contact_rate": float(v["contact_rate"]),
                }
                for k, v in pt.to_dict("index").items()
            }
            ct = sub.groupby("count").agg(
                pitches=("count", "size"),
                whiff_rate=("is_whiff", "mean"),
                contact_rate=("is_contact", "mean"),
            )
            out["BATTER_COUNT"][int(bid)] = {
                str(k): {
                    "pitches": int(v["pitches"]),
                    "whiff_rate": float(v["whiff_rate"]),
                    "contact_rate": float(v["contact_rate"]),
                }
                for k, v in ct.to_dict("index").items()
            }

    if "pitcher" in g.columns:
        for pid, sub in g.groupby("pitcher"):
            pt = sub.groupby("pitch_type").agg(
                pitches=("pitch_type", "size"),
                whiff_rate=("is_whiff", "mean"),
                contact_rate=("is_contact", "mean"),
            )
            out["PITCHER_PT"][int(pid)] = {
                str(k): {
                    "pitches": int(v["pitches"]),
                    "whiff_rate": float(v["whiff_rate"]),
                    "contact_rate": float(v["contact_rate"]),
                }
                for k, v in pt.to_dict("index").items()
            }
            ct = sub.groupby("count").agg(
                pitches=("count", "size"),
                whiff_rate=("is_whiff", "mean"),
                contact_rate=("is_contact", "mean"),
            )
            out["PITCHER_COUNT"][int(pid)] = {
                str(k): {
                    "pitches": int(v["pitches"]),
                    "whiff_rate": float(v["whiff_rate"]),
                    "contact_rate": float(v["contact_rate"]),
                }
                for k, v in ct.to_dict("index").items()
            }

    return out


PITCHTYPE_MATCHUP_TABLES = build_pitchtype_matchup_tables()


def expected_pitchtype_k_boost(pit_pid: int, bat_pid: int) -> float:
    """
    Simple matchup-specific K boost based on pitcher recent pitch mix and batter weakness by pitch type.
    > 1.0 favors strikeouts, < 1.0 suppresses strikeouts.
    """
    try:
        mix = pitcher_mix_last_starts(pit_pid) or {}
        bat = PITCHTYPE_MATCHUP_TABLES.get("BATTER_PT", {}).get(int(bat_pid), {})
        pit = PITCHTYPE_MATCHUP_TABLES.get("PITCHER_PT", {}).get(int(pit_pid), {})
        if not mix:
            return 1.0

        contrib = 0.0
        total_w = 0.0
        for pt, w in mix.items():
            b_row = bat.get(str(pt), {})
            p_row = pit.get(str(pt), {})
            b_contact = float(b_row.get("contact_rate", 0.72))
            p_whiff = float(p_row.get("whiff_rate", 0.10))
            # baseline ~ contact 0.72, whiff 0.10
            score = 1.0 + 0.8 * (p_whiff - 0.10) - 0.6 * (b_contact - 0.72)
            contrib += w * score
            total_w += w
        if total_w <= 0:
            return 1.0
        return float(np.clip(contrib / total_w, 0.85, 1.15))
    except Exception:
        return 1.0


def lineup_pitchtype_k_boost(pit_pid: int, batter_ids: list[int]) -> float:
    vals = [expected_pitchtype_k_boost(pit_pid, bid) for bid in batter_ids if bid]
    if not vals:
        return 1.0
    return float(np.clip(np.mean(vals), 0.88, 1.12))


# ------------------------ Automatic scoring / closing-line benchmarking ------------------------
def _load_any_table_local(pathlike) -> pd.DataFrame:
    p = Path(pathlike)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(p)
    return pd.read_csv(p)


def _normalize_date_cols_local(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in list(out.columns):
        name = str(c).lower()
        if name in {"date", "game_date", "as_of_date"}:
            dt = pd.to_datetime(out[c], errors="coerce")
            out[c] = dt.dt.strftime("%Y-%m-%d").where(dt.notna(), out[c].astype(str))
    return out


def auto_score_prediction_files(
    preds_path,
    actuals_path,
    join_keys=("date", "matchup", "player_name"),
    pred_col="P(HR>=1)",
    actual_col="actual_hr",
):
    """
    Load saved prediction/actual files and compute compact scoring metrics.
    """
    preds = _normalize_date_cols_local(_load_any_table_local(preds_path))
    acts = _normalize_date_cols_local(_load_any_table_local(actuals_path))

    keys = [k for k in join_keys if k in preds.columns and k in acts.columns]
    if not keys:
        raise ValueError("No overlapping join keys found between prediction and actual files.")

    use_pred = preds[keys + [pred_col]].copy()
    use_act = acts[keys + [actual_col]].copy()
    use_pred[pred_col] = pd.to_numeric(use_pred[pred_col], errors="coerce")
    use_act[actual_col] = pd.to_numeric(use_act[actual_col], errors="coerce")

    merged = use_pred.merge(use_act, on=keys, how="inner").dropna(subset=[pred_col, actual_col]).copy()
    if merged.empty:
        return pd.DataFrame([{
            "rows": 0,
            "brier": np.nan,
            "logloss": np.nan,
            "avg_pred": np.nan,
            "avg_actual": np.nan,
            "pred_col": pred_col,
            "actual_col": actual_col,
        }])

    p = merged[pred_col].clip(1e-6, 1 - 1e-6).astype(float).values
    y = merged[actual_col].astype(float).values
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    out = pd.DataFrame([{
        "rows": int(len(merged)),
        "brier": brier,
        "logloss": logloss,
        "avg_pred": float(np.mean(p)),
        "avg_actual": float(np.mean(y)),
        "pred_col": pred_col,
        "actual_col": actual_col,
    }])

    # Optional bucket calibration summary
    try:
        merged["bucket"] = pd.cut(merged[pred_col], bins=np.linspace(0, 1, 11), include_lowest=True)
        calib = (
            merged.groupby("bucket")
            .agg(n=(actual_col, "size"), pred_mean=(pred_col, "mean"), actual_mean=(actual_col, "mean"))
            .reset_index()
        )
        out.attrs["calibration"] = calib
    except Exception:
        pass

    return out


def auto_benchmark_closing_lines(
    preds_path,
    lines_path,
    join_keys=("date", "matchup", "player_name"),
    pred_prob_col="P(HR>=1)",
    book_prob_col="book_prob",
):
    """
    Join saved predictions to a closing-line table and compare model vs book probabilities.
    """
    preds = _normalize_date_cols_local(_load_any_table_local(preds_path))
    lines = _normalize_date_cols_local(_load_any_table_local(lines_path))

    keys = [k for k in join_keys if k in preds.columns and k in lines.columns]
    if not keys:
        raise ValueError("No overlapping join keys found between prediction and closing-line files.")

    use_pred = preds[keys + [pred_prob_col]].copy()
    use_line = lines[keys + [book_prob_col]].copy()
    use_pred[pred_prob_col] = pd.to_numeric(use_pred[pred_prob_col], errors="coerce")
    use_line[book_prob_col] = pd.to_numeric(use_line[book_prob_col], errors="coerce")

    merged = use_pred.merge(use_line, on=keys, how="inner").dropna(subset=[pred_prob_col, book_prob_col]).copy()
    if merged.empty:
        return pd.DataFrame([{
            "rows": 0,
            "model_mean_prob": np.nan,
            "book_mean_prob": np.nan,
            "avg_edge": np.nan,
            "positive_edge_rows": 0,
        }])

    merged["edge"] = merged[pred_prob_col] - merged[book_prob_col]
    merged["abs_edge"] = merged["edge"].abs()

    return pd.DataFrame([{
        "rows": int(len(merged)),
        "model_mean_prob": float(merged[pred_prob_col].mean()),
        "book_mean_prob": float(merged[book_prob_col].mean()),
        "avg_edge": float(merged["edge"].mean()),
        "avg_abs_edge": float(merged["abs_edge"].mean()),
        "positive_edge_rows": int((merged["edge"] > 0).sum()),
        "positive_edge_rate": float((merged["edge"] > 0).mean()),
    }])



# ------------------------ hierarchical shrinkage / two-stage HR / context / uncertainty ------------------------
def hierarchical_shrink_rate(
    obs_rate: float,
    obs_n: float,
    group_rate: float,
    league_rate: float,
    prior_group_n: float = 50.0,
    prior_league_n: float = 150.0,
) -> float:
    obs_rate = float(obs_rate) if np.isfinite(obs_rate) else float(league_rate)
    obs_n = max(float(obs_n) if np.isfinite(obs_n) else 0.0, 0.0)
    group_rate = float(group_rate) if np.isfinite(group_rate) else float(league_rate)
    league_rate = float(league_rate) if np.isfinite(league_rate) else float(obs_rate)
    group_post = (obs_rate * obs_n + group_rate * prior_group_n) / max(obs_n + prior_group_n, 1e-9)
    return float((group_post * (obs_n + prior_group_n) + league_rate * prior_league_n) / max(obs_n + prior_group_n + prior_league_n, 1e-9))


@disk_cache("hierarchical_priors_v1.pkl")
def build_hierarchical_priors():
    priors = {
        "league": {"pitcher_kbf": LEAGUE_K_RATE / LEAGUE_BF_PER_IP, "batter_k_rate": LEAGUE_K_RATE, "pitcher_hr_danger": LEAGUE_BAT.get("HR", 0.025), "batter_hr_skill": LEAGUE_BAT.get("HR", 0.025)},
        "pitcher": {},
        "batter": {},
    }
    try:
        pit = pitching_stats(YEAR, qual=0).copy()
        pit["Name_norm"] = pit["Name"].apply(clean_name)
        pit["pid"] = pit["Name_norm"].apply(name_to_mlbam_id)
        pit["BF_est"] = _derive_pitching_bf(pit)
        pit["SO_rate"] = pd.to_numeric(pit.get("SO", 0), errors="coerce").fillna(0.0) / pit["BF_est"].replace(0, np.nan)
        pit["HR_rate"] = pd.to_numeric(pit.get("HR", 0), errors="coerce").fillna(0.0) / pit["BF_est"].replace(0, np.nan)
        for _, r in pit.iterrows():
            pid = r.get("pid")
            if pd.isna(pid):
                continue
            bf = float(pd.to_numeric(r.get("BF_est"), errors="coerce"))
            priors["pitcher"][int(pid)] = {
                "n_bf": bf if np.isfinite(bf) else 0.0,
                "k_per_bf": float(pd.to_numeric(r.get("SO_rate"), errors="coerce")) if pd.notna(r.get("SO_rate")) else np.nan,
                "hr_per_bf": float(pd.to_numeric(r.get("HR_rate"), errors="coerce")) if pd.notna(r.get("HR_rate")) else np.nan,
            }
    except Exception:
        pass
    try:
        bat = batting_stats(YEAR, qual=0).copy()
        bat["Name_norm"] = bat["Name"].apply(clean_name)
        bat["pid"] = bat["Name_norm"].apply(name_to_mlbam_id)
        so_col = "SO" if "SO" in bat.columns else None
        if so_col:
            bat["K_rate"] = pd.to_numeric(bat[so_col], errors="coerce").fillna(0.0) / pd.to_numeric(bat["PA"], errors="coerce").replace(0, np.nan)
        else:
            bat["K_rate"] = np.nan
        bat["HR_rate"] = pd.to_numeric(bat.get("HR", 0), errors="coerce").fillna(0.0) / pd.to_numeric(bat["PA"], errors="coerce").replace(0, np.nan)
        for _, r in bat.iterrows():
            pid = r.get("pid")
            if pd.isna(pid):
                continue
            pa = float(pd.to_numeric(r.get("PA"), errors="coerce"))
            priors["batter"][int(pid)] = {
                "n_pa": pa if np.isfinite(pa) else 0.0,
                "k_rate": float(pd.to_numeric(r.get("K_rate"), errors="coerce")) if pd.notna(r.get("K_rate")) else np.nan,
                "hr_rate": float(pd.to_numeric(r.get("HR_rate"), errors="coerce")) if pd.notna(r.get("HR_rate")) else np.nan,
            }
    except Exception:
        pass
    return priors


HIERARCHICAL_PRIORS = build_hierarchical_priors()


def get_pitcher_kbf_prior(pid: int | None) -> float:
    if not pid:
        return float(HIERARCHICAL_PRIORS["league"]["pitcher_kbf"])
    rec = HIERARCHICAL_PRIORS.get("pitcher", {}).get(int(pid), {})
    return hierarchical_shrink_rate(
        rec.get("k_per_bf", HIERARCHICAL_PRIORS["league"]["pitcher_kbf"]),
        rec.get("n_bf", 0.0),
        HIERARCHICAL_PRIORS["league"]["pitcher_kbf"],
        HIERARCHICAL_PRIORS["league"]["pitcher_kbf"],
        prior_group_n=75.0,
        prior_league_n=200.0,
    )


def get_batter_k_susceptibility_prior(pid: int | None) -> float:
    if not pid:
        return float(HIERARCHICAL_PRIORS["league"]["batter_k_rate"])
    rec = HIERARCHICAL_PRIORS.get("batter", {}).get(int(pid), {})
    return hierarchical_shrink_rate(
        rec.get("k_rate", HIERARCHICAL_PRIORS["league"]["batter_k_rate"]),
        rec.get("n_pa", 0.0),
        HIERARCHICAL_PRIORS["league"]["batter_k_rate"],
        HIERARCHICAL_PRIORS["league"]["batter_k_rate"],
        prior_group_n=60.0,
        prior_league_n=180.0,
    )


def get_pitcher_hr_danger_prior(pid: int | None) -> float:
    if not pid:
        return float(HIERARCHICAL_PRIORS["league"]["pitcher_hr_danger"])
    rec = HIERARCHICAL_PRIORS.get("pitcher", {}).get(int(pid), {})
    return hierarchical_shrink_rate(
        rec.get("hr_per_bf", HIERARCHICAL_PRIORS["league"]["pitcher_hr_danger"]),
        rec.get("n_bf", 0.0),
        HIERARCHICAL_PRIORS["league"]["pitcher_hr_danger"],
        HIERARCHICAL_PRIORS["league"]["pitcher_hr_danger"],
        prior_group_n=90.0,
        prior_league_n=240.0,
    )


def get_batter_hr_skill_prior(pid: int | None) -> float:
    if not pid:
        return float(HIERARCHICAL_PRIORS["league"]["batter_hr_skill"])
    rec = HIERARCHICAL_PRIORS.get("batter", {}).get(int(pid), {})
    return hierarchical_shrink_rate(
        rec.get("hr_rate", HIERARCHICAL_PRIORS["league"]["batter_hr_skill"]),
        rec.get("n_pa", 0.0),
        HIERARCHICAL_PRIORS["league"]["batter_hr_skill"],
        HIERARCHICAL_PRIORS["league"]["batter_hr_skill"],
        prior_group_n=75.0,
        prior_league_n=220.0,
    )


def _zone_bucket_from_raw(df: pd.DataFrame) -> pd.Series:
    zone = pd.to_numeric(df.get("zone", pd.Series(index=df.index, dtype=float)), errors="coerce")
    if zone.notna().any():
        return pd.Series(np.where(zone.isin([5]), "heart", np.where(zone.isin([1,2,3,4,6,7,8,9]), "shadow", "chase")), index=df.index)
    px = pd.to_numeric(df.get("plate_x", pd.Series(index=df.index, dtype=float)), errors="coerce")
    pz = pd.to_numeric(df.get("plate_z", pd.Series(index=df.index, dtype=float)), errors="coerce")
    return pd.Series(
        np.where((px.abs() <= 0.28) & (pz.between(2.3, 3.0)), "heart",
                 np.where((px.abs() <= 0.85) & (pz.between(1.6, 3.5)), "shadow", "chase")),
        index=df.index,
    )


@disk_cache("damage_contact_model_v1.pkl")
def train_damage_contact_model(days: int = 540):
    end = datetime.today()
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    df = fetch_statcast_raw(start, end.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        return None, []
    g = df.copy().replace([np.inf, -np.inf], np.nan)
    g["count"] = g.get("balls", 0).fillna(0).astype(int).astype(str) + "-" + g.get("strikes", 0).fillna(0).astype(int).astype(str)
    g["pitch_type_norm"] = g.get("pitch_type", pd.Series(index=g.index, dtype=object)).astype(str).str.upper().replace({"":"FF"}).fillna("FF")
    g["zone_bucket"] = _zone_bucket_from_raw(g)
    g["stand"] = g.get("stand", pd.Series(index=g.index, dtype=object)).astype(str).str.upper().str[0].replace({"":"R"}).fillna("R")
    g["p_throws"] = g.get("p_throws", pd.Series(index=g.index, dtype=object)).astype(str).str.upper().str[0].replace({"":"R"}).fillna("R")
    g["exit_velocity"] = (pd.to_numeric(g["launch_speed"], errors="coerce").fillna(0.0) if "launch_speed" in g.columns else pd.Series(0.0, index=g.index, dtype="float64"))
    g["launch_angle"] = (pd.to_numeric(g["launch_angle"], errors="coerce").fillna(0.0) if "launch_angle" in g.columns else pd.Series(0.0, index=g.index, dtype="float64"))
    g["barrel_pct"] = (pd.to_numeric(g["barrel"], errors="coerce").fillna(0.0) if "barrel" in g.columns else pd.Series(0.0, index=g.index, dtype="float64"))
    g["pull_pct"] = g["launch_angle"].between(-20, 20).astype(float)
    g["is_damage_contact"] = (((g["exit_velocity"] >= 98) & g["launch_angle"].between(24, 35)) | (g["barrel_pct"] == 1)).astype(int)
    X = pd.get_dummies(g[["count","pitch_type_norm","zone_bucket","stand","p_throws"]], drop_first=False)
    X["exit_velocity"] = g["exit_velocity"]
    X["launch_angle"] = g["launch_angle"]
    X["barrel_pct"] = g["barrel_pct"]
    X["pull_pct"] = g["pull_pct"]
    y = g["is_damage_contact"].astype(int)
    if len(y.unique()) < 2:
        return None, list(X.columns)
    mdl = GradientBoostingClassifier(random_state=0, n_estimators=150, learning_rate=0.05, max_depth=3)
    mdl.fit(X, y)
    return mdl, list(X.columns)


DAMAGE_CONTACT_MODEL, DAMAGE_CONTACT_COLS = train_damage_contact_model()


def predict_damage_contact_prob(feat_row: pd.Series) -> float:
    if DAMAGE_CONTACT_MODEL is None:
        return 0.18
    base = {c: 0.0 for c in DAMAGE_CONTACT_COLS}
    for key in ("count","pitch_type_norm","zone_bucket","stand","p_throws"):
        if key in feat_row.index:
            col = f"{key}_{feat_row.get(key)}"
            if col in base:
                base[col] = 1.0
    for key in ("exit_velocity","launch_angle","barrel_pct","pull_pct"):
        if key in DAMAGE_CONTACT_COLS:
            base[key] = float(pd.to_numeric(feat_row.get(key, 0.0), errors="coerce"))
    X = pd.DataFrame([base], columns=DAMAGE_CONTACT_COLS)
    return float(np.clip(DAMAGE_CONTACT_MODEL.predict_proba(X)[:, 1][0], 0.01, 0.95))


@disk_cache("hr_given_damage_model_v1.pkl")
def train_hr_given_damage_model(days: int = 540):
    end = datetime.today()
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    df = fetch_statcast_raw(start, end.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        return None, []
    g = df.copy().replace([np.inf, -np.inf], np.nan)
    g["events"] = g.get("events", pd.Series(index=g.index, dtype=object)).astype(str).str.lower()
    g["count"] = g.get("balls", 0).fillna(0).astype(int).astype(str) + "-" + g.get("strikes", 0).fillna(0).astype(int).astype(str)
    g["pitch_type_norm"] = g.get("pitch_type", pd.Series(index=g.index, dtype=object)).astype(str).str.upper().replace({"":"FF"}).fillna("FF")
    g["zone_bucket"] = _zone_bucket_from_raw(g)
    g["stand"] = g.get("stand", pd.Series(index=g.index, dtype=object)).astype(str).str.upper().str[0].replace({"":"R"}).fillna("R")
    g["p_throws"] = g.get("p_throws", pd.Series(index=g.index, dtype=object)).astype(str).str.upper().str[0].replace({"":"R"}).fillna("R")
    g["exit_velocity"] = (pd.to_numeric(g["launch_speed"], errors="coerce").fillna(0.0) if "launch_speed" in g.columns else pd.Series(0.0, index=g.index, dtype="float64"))
    g["launch_angle"] = (pd.to_numeric(g["launch_angle"], errors="coerce").fillna(0.0) if "launch_angle" in g.columns else pd.Series(0.0, index=g.index, dtype="float64"))
    g["barrel_pct"] = (pd.to_numeric(g["barrel"], errors="coerce").fillna(0.0) if "barrel" in g.columns else pd.Series(0.0, index=g.index, dtype="float64"))
    g["pull_pct"] = g["launch_angle"].between(-20, 20).astype(float)
    damage = (((g["exit_velocity"] >= 98) & g["launch_angle"].between(24, 35)) | (g["barrel_pct"] == 1))
    g = g[damage.fillna(False)].copy()
    if g.empty:
        return None, []
    g["is_hr"] = g["events"].eq("home_run").astype(int)
    X = pd.get_dummies(g[["count","pitch_type_norm","zone_bucket","stand","p_throws"]], drop_first=False)
    X["exit_velocity"] = g["exit_velocity"]
    X["launch_angle"] = g["launch_angle"]
    X["barrel_pct"] = g["barrel_pct"]
    X["pull_pct"] = g["pull_pct"]
    park = (pd.to_numeric(g["park_hr_factor"], errors="coerce").fillna(1.0) if "park_hr_factor" in g.columns else pd.Series(1.0, index=g.index, dtype="float64"))
    X["park_hr_factor"] = park
    y = g["is_hr"].astype(int)
    if len(y.unique()) < 2:
        return None, list(X.columns)
    mdl = GradientBoostingClassifier(random_state=0, n_estimators=120, learning_rate=0.05, max_depth=3)
    mdl.fit(X, y)
    return mdl, list(X.columns)


HR_GIVEN_DAMAGE_MODEL, HR_GIVEN_DAMAGE_COLS = train_hr_given_damage_model()


def predict_hr_given_damage_prob(feat_row: pd.Series) -> float:
    if HR_GIVEN_DAMAGE_MODEL is None:
        return 0.08
    base = {c: 0.0 for c in HR_GIVEN_DAMAGE_COLS}
    for key in ("count","pitch_type_norm","zone_bucket","stand","p_throws"):
        if key in feat_row.index:
            col = f"{key}_{feat_row.get(key)}"
            if col in base:
                base[col] = 1.0
    for key in ("exit_velocity","launch_angle","barrel_pct","pull_pct","park_hr_factor"):
        if key in HR_GIVEN_DAMAGE_COLS:
            base[key] = float(pd.to_numeric(feat_row.get(key, 0.0), errors="coerce"))
    X = pd.DataFrame([base], columns=HR_GIVEN_DAMAGE_COLS)
    return float(np.clip(HR_GIVEN_DAMAGE_MODEL.predict_proba(X)[:, 1][0], 0.005, 0.95))


def predict_two_stage_hr_pa(feat_row: pd.Series) -> dict:
    damage_p = predict_damage_contact_prob(feat_row)
    hr_given_damage = predict_hr_given_damage_prob(feat_row)
    context_p = 1.0
    if callable(get_hr_contact_context_prob):
        try:
            context_p = float(np.clip(get_hr_contact_context_prob(
                feat_row.get("pitcher_id"),
                feat_row.get("batter_id"),
                park_factor=float(feat_row.get("park_hr_factor", 1.0) or 1.0),
                weather_factor=float(feat_row.get("weather_hr_mult", 1.0) or 1.0),
                launch_bucket=str(feat_row.get("launch_bucket", "mid")),
            ), 0.70, 1.35))
        except Exception:
            context_p = 1.0
    hr_pa = float(np.clip(damage_p * hr_given_damage * context_p, 0.0005, 0.25))
    return {
        "p_damage_contact": float(np.clip(damage_p, 0.01, 0.95)),
        "p_hr_given_damage": float(np.clip(hr_given_damage, 0.005, 0.95)),
        "p_hr_pa_two_stage": hr_pa,
        "hr_context_mult_two_stage": float(context_p),
    }


@disk_cache("markov_context_defaults_v1.pkl")
def build_markov_context_defaults():
    slot_weights = np.array([1.12, 1.08, 1.04, 1.00, 0.98, 0.96, 0.94, 0.91, 0.87], dtype=float)
    slot_weights = slot_weights / slot_weights.mean()
    team_exp_pa = 38.0
    exp_pa_by_slot = {i + 1: float(team_exp_pa / 9.0 * slot_weights[i]) for i in range(9)}
    run_bias = {1:1.06,2:1.04,3:1.01,4:0.98,5:0.96,6:0.97,7:1.00,8:1.02,9:1.03}
    rbi_bias = {1:0.90,2:0.96,3:1.05,4:1.08,5:1.07,6:1.02,7:0.97,8:0.93,9:0.90}
    hr_bias = {slot: float(np.clip(0.94 + 0.02 * exp_pa_by_slot[slot], 0.88, 1.12)) for slot in exp_pa_by_slot}
    return {
        "expected_pa_by_slot": exp_pa_by_slot,
        "expected_runs_by_team": 4.4,
        "starter_pa_share": 0.68,
        "bullpen_pa_share": 0.32,
        "rbi_context_by_slot": rbi_bias,
        "hr_context_by_slot": hr_bias,
        "run_context_by_slot": run_bias,
    }


MARKOV_CONTEXT_DEFAULTS = build_markov_context_defaults()


def export_markov_context_for_lineup(lineup_len: int = 9, team_runs: float | None = None, starter_pa_share: float | None = None) -> pd.DataFrame:
    ctx = MARKOV_CONTEXT_DEFAULTS.copy()
    team_runs = float(team_runs) if team_runs is not None and np.isfinite(team_runs) else float(ctx["expected_runs_by_team"])
    starter_pa_share = float(starter_pa_share) if starter_pa_share is not None and np.isfinite(starter_pa_share) else float(ctx["starter_pa_share"])
    rows = []
    for slot in range(1, int(max(lineup_len, 1)) + 1):
        rows.append({
            "batting_order": slot,
            "expected_pa_by_slot": float(ctx["expected_pa_by_slot"].get(slot, 4.2)),
            "expected_runs_by_team": team_runs,
            "starter_pa_share": starter_pa_share,
            "bullpen_pa_share": float(1.0 - starter_pa_share),
            "rbi_context_by_slot": float(ctx["rbi_context_by_slot"].get(slot, 1.0)),
            "hr_context_by_slot": float(ctx["hr_context_by_slot"].get(slot, 1.0)),
            "run_context_by_slot": float(ctx["run_context_by_slot"].get(slot, 1.0)),
        })
    return pd.DataFrame(rows)


def summarize_binary_prob_uncertainty(prob: float, n_eff: float = 150.0) -> dict:
    p = float(np.clip(prob, 1e-6, 1 - 1e-6))
    n_eff = max(float(n_eff), 1.0)
    alpha = 1.0 + p * n_eff
    beta = 1.0 + (1.0 - p) * n_eff
    mean = alpha / (alpha + beta)
    var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
    sd = float(np.sqrt(max(var, 0.0)))
    lo = float(np.clip(mean - 1.645 * sd, 0.0, 1.0))
    hi = float(np.clip(mean + 1.645 * sd, 0.0, 1.0))
    entropy = float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))
    return {"mean": float(mean), "variance": float(var), "sd": sd, "lo90": lo, "hi90": hi, "entropy": entropy}


# ------------------------ exports & cache_all ------------------------
__all__ = [
    'predict_k9','predict_k9_interval','features_base','league_feature_means',
    'kt_montecarlo','sample_starter_ip','simulate_full_game','shrink_vs_pitcher','sample_pa_outcome',
    'H_theta','H_p','TB_theta','TB_p','NB_K_theta','NB_K_p','LEAGUE_BAT',
    'predict_hits','predict_hits_interval','features_hit_base','league_hit_means','iso_hit_calibrators',
    'predict_hr_proba','predict_hr_count_pt','predict_hr_hardhit','predict_hr_2swhiff','predict_hr_pullangle',
    'iso_pa_calibrators','expected_matchup_xiso','hr_game_cals','HR_LAMBDA_SCALE','infer_hr_archetype','IP_REG','batter_game_hr_prob',
    'get_recent_evt_feats','batter_recent_multiplier','get_pitcher_recent','pitcher_recent_multiplier', 'get_season_iso',
    'load_b14_feats','load_season_hr_splits','load_pitcher_vs_side_hr','load_season_hr_splits', 'build_count_pitchtype_tables','build_hr_count_pt_onehot','build_heart_rate_table',
    'predict_ip_start','predict_bf_per_ip','predict_k_per_bf','predict_k9_decomp','predict_k_start_decomp',
    'k_decomp_feature_cols','league_decomp_feature_means','build_pitchtype_matchup_tables',
    'expected_pitchtype_k_boost','lineup_pitchtype_k_boost','auto_score_prediction_files','auto_benchmark_closing_lines',
    'hierarchical_shrink_rate','build_hierarchical_priors','HIERARCHICAL_PRIORS','get_pitcher_kbf_prior','get_batter_k_susceptibility_prior','get_pitcher_hr_danger_prior','get_batter_hr_skill_prior',
    'train_damage_contact_model','predict_damage_contact_prob','train_hr_given_damage_model','predict_hr_given_damage_prob','predict_two_stage_hr_pa',
    'build_markov_context_defaults','MARKOV_CONTEXT_DEFAULTS','export_markov_context_for_lineup','summarize_binary_prob_uncertainty',
]

def cache_all():
    logging.info("1) park factors"); fetch_yearly_park_factors(YEAR); fetch_monthly_park_factors(YEAR)
    logging.info("2) run-exp"); build_run_expectancy_matrix()
    logging.info("3) train K9"); train_ensemble_k9_model()
    logging.info("3b) train Hits"); train_ensemble_hit_model()
    logging.info("4) NB dispersions"); compute_batter_dispersions(YEAR); compute_strikeout_dispersion(YEAR)
    logging.info("5) IP regression"); fit_ip_regression(YEAR)
    logging.info("6) batter career"); build_batter_career_rates([YEAR-3,YEAR-2,YEAR-1,YEAR])
    logging.info("7) per-PA calibrators"); build_pa_isotonic_calibrators()
    logging.info("8) HR archetype calibrators"); build_hr_game_calibrators()
    logging.info("9) HR λ-scale"); _estimate_hr_lambda_scale()
    logging.info("10) 14d batter feats");            build_b14_feats(days_window=180, days_recent=14)
    logging.info("11) season HR splits");            load_season_hr_splits(YEAR)
    logging.info("12) pitcher vs-side HR (365d)");   build_pitcher_vs_side_hr(days=365)
    logging.info("13) season HR R/L splits");         load_season_hr_splits()
    logging.info("14) K decomposition models");      train_k_decomposition_models()
    logging.info("15) pitch-type matchup tables");   build_pitchtype_matchup_tables()


if __name__=="__main__":
    cache_all()
    predict_k9,  predict_k9_interval,  _, league_feature_means  = train_ensemble_k9_model()
    predict_hits,predict_hits_interval, _, league_hit_means, iso_hit_calibrators = train_ensemble_hit_model()
    print("precompute_everything complete!")
