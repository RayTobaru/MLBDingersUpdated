from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

try:
    from scipy.stats import nbinom
except Exception:  # pragma: no cover
    nbinom = None

from .compat import ROOT, LEGACY  # noqa: F401
from .predict_batter_outcomes import fetch_matchups_for_date, _resolve_lineups, _strict_resolve_batter  # type: ignore
from .zone_pitch_features import summarize_zone_pitch_ko_context

import fetch  # type: ignore
from fetch import (
    pitching_stats,
    batting_stats,
    clean_name,
    strip_pos,
    fetch_statcast_raw,
    get_statcast_pitcher_features,
    get_recent_pitcher_k9,
    get_recent_pitcher_era,
    pitcher_throws,
    name_to_mlbam_id,
)

try:
    from fetch import fetch_monthly_park_factors
except Exception:  # pragma: no cover
    fetch_monthly_park_factors = None  # type: ignore

from precompute_everything import (  # type: ignore
    predict_k9,
    predict_k9_interval,
    IP_REG,
    sample_starter_ip,
    features_base,
    league_feature_means,
    BATTER_CAREER,
    NB_K_theta,
    NB_K_p,
)

# Optional upgraded decomposition / matchup helpers.
try:  # pragma: no cover
    from precompute_everything import (  # type: ignore
        predict_ip_start,
        predict_bf_per_ip,
        predict_k_per_bf,
        predict_k9_decomp,
        predict_k_start_decomp,
        lineup_pitchtype_k_boost,
    )
except Exception:  # pragma: no cover
    predict_ip_start = None  # type: ignore
    predict_bf_per_ip = None  # type: ignore
    predict_k_per_bf = None  # type: ignore
    predict_k9_decomp = None  # type: ignore
    predict_k_start_decomp = None  # type: ignore
    lineup_pitchtype_k_boost = None  # type: ignore


YEAR = pd.Timestamp.today().year
BASE_COLS = [
    'Team', 'Name', 'Pitcher_ID',
    'Proj_IP', 'Proj_BF',
    'BF_per_IP', 'K_per_BF',
    'Pred_K9', 'Mean_K_start', 'Median_K_start',
]
LEAGUE_K_RATE = 0.22
LEAGUE_OBP = 0.315
LEAGUE_BF_PER_IP = 4.15


def _safe_float(v: object, default: float = np.nan) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    return float(x)


def _safe_pct(v: object, default: float) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    return float(x / 100.0 if x > 1.0 else x)


def _weighted_mean(vals: list[float], weights: list[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(arr) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float(np.nan)
    return float(np.average(arr[mask], weights=w[mask]))


@lru_cache(maxsize=8)
def _statcast_window(days: int) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=int(days))
    raw = fetch_statcast_raw(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    return raw if isinstance(raw, pd.DataFrame) else pd.DataFrame()


@lru_cache(maxsize=1)
def _batting_table() -> pd.DataFrame:
    try:
        df = batting_stats(YEAR, qual=0)
    except Exception:
        df = pd.DataFrame()
    if isinstance(df, pd.DataFrame) and not df.empty:
        out = df.copy()
        if 'Name' in out.columns:
            out['Name_norm'] = out['Name'].map(lambda x: clean_name(strip_pos(x)))
        return out
    return pd.DataFrame()


def _recent_pitcher_form(pid: int | None) -> dict[str, float]:
    if not pid:
        return {'k9_7': np.nan, 'k9_14': np.nan, 'k9_30': np.nan, 'era_30': np.nan}

    def _window_k9(days: int) -> float:
        direct = get_recent_pitcher_k9(int(pid), days)
        if direct is not None and np.isfinite(direct):
            return float(direct)
        raw = _statcast_window(days)
        if raw.empty or 'pitcher' not in raw.columns or 'events' not in raw.columns:
            return np.nan
        df = raw[raw['pitcher'] == int(pid)].copy()
        if df.empty:
            return np.nan
        events = df['events'].astype(str)
        pa = max(int(events.notna().sum()), len(df))
        if pa <= 0:
            return np.nan
        ks = int(events.isin(['strikeout', 'strikeout_double_play']).sum())
        return float(np.clip((ks / pa) * 27.0, 2.5, 16.0))

    era30 = get_recent_pitcher_era(int(pid), 30)
    try:
        era30 = float(era30) if era30 is not None else np.nan
    except Exception:
        era30 = np.nan
    return {
        'k9_7': _window_k9(7),
        'k9_14': _window_k9(14),
        'k9_30': _window_k9(30),
        'era_30': era30,
    }


def _career_lineup_row_by_name(name: str) -> pd.Series | None:
    if isinstance(BATTER_CAREER, pd.DataFrame) and not BATTER_CAREER.empty and 'Name_norm' in BATTER_CAREER.columns:
        key = clean_name(strip_pos(name))
        hit = BATTER_CAREER[BATTER_CAREER['Name_norm'] == key]
        if not hit.empty:
            return hit.iloc[0]
    return None


def _batter_k_obp_from_name(name: str) -> tuple[float, float]:
    bat = _batting_table()
    key = clean_name(strip_pos(name))
    if not bat.empty and 'Name_norm' in bat.columns:
        hit = bat[bat['Name_norm'] == key]
        if not hit.empty:
            row = hit.iloc[0]
            k_rate = _safe_pct(row.get('K%', row.get('SO', 0.0) / max(float(row.get('PA', 1.0) or 1.0), 1.0)), LEAGUE_K_RATE)
            obp = float(row.get('OBP', LEAGUE_OBP) or LEAGUE_OBP)
            return float(np.clip(k_rate, 0.08, 0.38)), float(np.clip(obp, 0.240, 0.430))
    career = _career_lineup_row_by_name(name)
    if career is not None:
        pa = float(career.get('PA', 0.0) or 0.0)
        h_rate = float(career.get('H_rate', 0.225) or 0.225)
        k_rate = float(career.get('K_rate', LEAGUE_K_RATE) or LEAGUE_K_RATE) if 'K_rate' in career else LEAGUE_K_RATE
        obp = min(max(h_rate + 0.07, 0.255), 0.410)
        if pa < 50:
            k_rate = 0.65 * LEAGUE_K_RATE + 0.35 * k_rate
            obp = 0.7 * LEAGUE_OBP + 0.3 * obp
        return float(np.clip(k_rate, 0.10, 0.36)), float(np.clip(obp, 0.240, 0.430))
    return LEAGUE_K_RATE, LEAGUE_OBP


def _lineup_k_tendency(lineup: list[str], pitcher_hand: str, batting_team: str | None = None) -> dict[str, float]:
    raw = _statcast_window(365)
    rows = []
    batter_ids: list[int] = []
    resolved_n = 0

    for name in list(lineup or [])[:9]:
        nm = strip_pos(name or '').strip()
        if not nm or nm.upper() == 'TBD':
            continue

        pid = None
        if batting_team:
            try:
                pid, resolved_name = _strict_resolve_batter(batting_team, nm)
                nm = resolved_name or nm
            except Exception:
                pid = None
        if not pid:
            try:
                pid = name_to_mlbam_id(nm)
            except Exception:
                pid = None

        if pid:
            batter_ids.append(int(pid))
            resolved_n += 1

        base_k, base_obp = _batter_k_obp_from_name(nm)

        if raw.empty or 'batter' not in raw.columns or 'events' not in raw.columns or not pid:
            rows.append({'pa': 30.0, 'k_rate_raw': base_k, 'k_rate': base_k, 'obp_proxy': base_obp, 'source': 'fallback'})
            continue

        df = raw[raw['batter'] == int(pid)].copy()
        if df.empty:
            rows.append({'pa': 30.0, 'k_rate_raw': base_k, 'k_rate': base_k, 'obp_proxy': base_obp, 'source': 'fallback'})
            continue

        if pitcher_hand and 'p_throws' in df.columns:
            sub = df[df['p_throws'].astype(str).str.upper().str.startswith(pitcher_hand.upper()[:1])]
            if len(sub) >= 12:
                df = sub

        pa_df = df[df['events'].notna()].copy()
        if pa_df.empty:
            rows.append({'pa': 30.0, 'k_rate_raw': base_k, 'k_rate': base_k, 'obp_proxy': base_obp, 'source': 'fallback'})
            continue

        events = pa_df['events'].astype(str)
        pa = max(int(len(pa_df)), 1)
        ks = int(events.isin(['strikeout', 'strikeout_double_play']).sum())
        hits = int(events.isin(['single', 'double', 'triple', 'home_run']).sum())
        bb = int(events.isin(['walk', 'hit_by_pitch']).sum())

        raw_k = ks / pa
        raw_obp = (hits + bb) / pa

        # Let real split data move the estimate more than before.
        k_rate = ((raw_k * pa) + (base_k * 35.0)) / (pa + 35.0)
        obp = ((raw_obp * pa) + (base_obp * 28.0)) / (pa + 28.0)

        rows.append({
            'pa': float(pa),
            'k_rate_raw': float(np.clip(raw_k, 0.08, 0.42)),
            'k_rate': float(np.clip(k_rate, 0.08, 0.40)),
            'obp_proxy': float(np.clip(obp, 0.240, 0.430)),
            'source': 'statcast'
        })

    if not rows:
        return {
            'k_rate_raw': LEAGUE_K_RATE,
            'k_rate_shrunk': LEAGUE_K_RATE,
            'k_rate': LEAGUE_K_RATE,
            'obp_proxy': LEAGUE_OBP,
            'batter_ids': [],
            'pa_seen': 0.0,
            'resolved_n': 0,
        }

    df = pd.DataFrame(rows)
    w = pd.to_numeric(df['pa'], errors='coerce').fillna(1.0).clip(lower=1.0)
    k_rate_raw = float(np.average(pd.to_numeric(df['k_rate_raw'], errors='coerce'), weights=w))
    k_rate_shrunk = float(np.average(pd.to_numeric(df['k_rate'], errors='coerce'), weights=w))
    obp_proxy = float(np.average(pd.to_numeric(df['obp_proxy'], errors='coerce'), weights=w))

    # Mild final shrink only when too few hitters resolve.
    if resolved_n < 5:
        alpha = 0.55
        k_rate_shrunk = alpha * LEAGUE_K_RATE + (1.0 - alpha) * k_rate_shrunk
        obp_proxy = alpha * LEAGUE_OBP + (1.0 - alpha) * obp_proxy

    return {
        'k_rate_raw': float(np.clip(k_rate_raw, 0.10, 0.34)),
        'k_rate_shrunk': float(np.clip(k_rate_shrunk, 0.10, 0.34)),
        'k_rate': float(np.clip(k_rate_shrunk, 0.10, 0.34)),
        'obp_proxy': float(np.clip(obp_proxy, 0.255, 0.395)),
        'batter_ids': batter_ids,
        'pa_seen': float(w.sum()),
        'resolved_n': int(resolved_n),
    }


def _estimate_bf(proj_ip: float, whip: float, opp_obp_proxy: float) -> float:
    whip = float(whip if np.isfinite(whip) else 1.28)
    opp_obp_proxy = float(opp_obp_proxy if np.isfinite(opp_obp_proxy) else LEAGUE_OBP)
    bf = 3.0 * float(proj_ip) * (1.0 + 0.06 * (whip - 1.28) + 0.55 * (opp_obp_proxy - LEAGUE_OBP))
    return float(np.clip(bf, 12.0, 34.0))


def _season_pitcher_anchor(rec: pd.Series, model_guess: float) -> tuple[float, float]:
    season_k9 = float(rec.get('K/9', rec.get('K9', np.nan))) if pd.notna(rec.get('K/9', rec.get('K9', np.nan))) else np.nan
    k_pct = _safe_pct(rec.get('K%', np.nan), np.nan)
    swstr = _safe_pct(rec.get('SwStr%', np.nan), np.nan)
    anchors = []
    weights = []
    if np.isfinite(season_k9):
        anchors.append(float(season_k9))
        weights.append(0.45)
    if np.isfinite(k_pct):
        anchors.append(float(np.clip(k_pct * 27.0, 4.0, 15.0)))
        weights.append(0.22)
    if np.isfinite(swstr):
        anchors.append(float(np.clip(4.2 + 55.0 * swstr, 4.0, 15.0)))
        weights.append(0.13)
    if np.isfinite(model_guess):
        anchors.append(float(model_guess))
        weights.append(0.20)
    if not anchors:
        return model_guess, float(rec.get('WHIP', 1.28) if pd.notna(rec.get('WHIP', 1.28)) else 1.28)
    base = float(np.average(anchors, weights=weights))
    whip = float(rec.get('WHIP', 1.28) if pd.notna(rec.get('WHIP', 1.28)) else 1.28)
    return base, whip


def _safe_model_predict(model_func, row: pd.Series, default: float = np.nan) -> float:
    if callable(model_func):
        try:
            return float(model_func(row))
        except Exception:
            return float(default)
    return float(default)


def _park_k_mult(team_abbr: str) -> float:
    if not callable(fetch_monthly_park_factors):
        return 1.0
    try:
        mpf = fetch_monthly_park_factors(YEAR)
        m = datetime.today().month
        return float(mpf.get(team_abbr, {}).get(m, {}).get("SO", 100) / 100.0)
    except Exception:
        return 1.0


def _simulate_k_distribution(mu_k: float, row: dict) -> dict:
    mu = float(np.clip(mu_k, 0.0, 15.0))
    if np.isfinite(_safe_float(row.get('park_k_mult', np.nan), np.nan)):
        mu *= float(np.clip(row.get('park_k_mult', 1.0), 0.90, 1.10))

    if np.isfinite(_safe_float(row.get('proj_k_start_model', np.nan), np.nan)):
        mu = float(np.clip(0.60 * mu + 0.40 * row.get('proj_k_start_model', mu), 0.0, 15.0))

    sims = 6000
    if nbinom is not None and np.isfinite(NB_K_theta) and _safe_float(NB_K_theta, np.nan) > 0:
        theta = float(max(NB_K_theta, 0.1))
        p_i = theta / (theta + mu) if (theta + mu) > 0 else 1.0
        samp = nbinom(n=theta, p=p_i).rvs(size=sims)
    else:
        samp = np.random.poisson(mu, size=sims)

    mean_k = float(np.mean(samp))
    med = int(np.percentile(samp, 50))
    probs = {f"P(K≥{k})": float((samp >= k).mean()) for k in range(2, 11)}
    lo_cnt = int(np.percentile(samp, 5))
    hi_cnt = int(np.percentile(samp, 95))
    return {
        "Mean_K_start": mean_k,
        "Median_K_start": med,
        **probs,
        "90% CI": f"{lo_cnt}–{hi_cnt}",
    }



def _resolve_pitcher_id_hard(team: str | None, name: str | None, pid) -> int | None:
    """
    Robust pitcher ID resolver for KO/zone engine.
    Fixes cases where schedule probable pitcher name exists but Pitcher_ID is None/NaN.
    """
    try:
        if pid is not None and str(pid).strip().lower() not in {"", "nan", "none"}:
            return int(float(pid))
    except Exception:
        pass

    nm = strip_pos(str(name or "")).strip()
    if not nm or nm.upper() == "TBD":
        return None

    key = clean_name(nm)

    # 1) Active pitching staff map from fetch.py
    try:
        staff = fetch.full_pitching_staff(str(team or "").strip().upper()) or {}
        if key in staff and staff[key]:
            return int(staff[key])
    except Exception:
        pass

    # 2) Active/full roster map fallback
    try:
        rm = fetch.roster_map(str(team or "").strip().upper()) or {}
        if key in rm and rm[key]:
            return int(rm[key])
    except Exception:
        pass

    # 3) StatsAPI people/search fallback
    try:
        resp = fetch.safe_get(
            "https://statsapi.mlb.com/api/v1/people/search",
            {"sportId": 1, "names": nm}
        )
        js = resp.json() if resp else {}
        people = js.get("people", []) or []

        for person in people:
            full = person.get("fullName") or ""
            mid = person.get("id")
            if mid and clean_name(full) == key:
                return int(mid)

        for person in people:
            mid = person.get("id")
            if mid:
                return int(mid)
    except Exception:
        pass

    # 4) pybaseball fallback
    try:
        mid = name_to_mlbam_id(nm)
        if mid:
            return int(mid)
    except Exception:
        pass

    return None


def build_starter_dict(team, name, pid, opp_lineup=None, opp_team=None, ump_feats=None, framing_feats=None, as_of_date: str | None = None):
    pid = _resolve_pitcher_id_hard(team, name, pid)
    r = {f: league_feature_means.get(f, 0.0) for f in features_base}
    try:
        df = pitching_stats(YEAR, qual=0)
    except Exception as exc:
        LOGGER.warning("pitching_stats fallback for %s (%s): %s", name, team, exc)
        df = pd.DataFrame()

    rec = pd.DataFrame()
    if isinstance(df, pd.DataFrame) and not df.empty and pid is not None:
        id_col = next((c for c in df.columns if str(c).lower() in ('pid', 'playerid', 'player_id', 'mlbam_id')), None)
        if id_col:
            try:
                df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
                rec = df[df[id_col] == int(pid)]
            except Exception:
                rec = pd.DataFrame()
    if rec.empty and isinstance(name, str) and isinstance(df, pd.DataFrame) and not df.empty and 'Name' in df.columns:
        df = df.copy()
        df['Name_norm'] = df.Name.apply(clean_name)
        rec = df[df.Name_norm == clean_name(name)]

    model_guess = float(np.clip(_safe_model_predict(predict_k9, pd.Series(r), 8.2), 4.0, 15.0))
    if not rec.empty:
        rec = rec.iloc[0]
        ipps = float(rec.IP / max(rec.GS, 1)) if rec.GS and rec.GS > 0 else 5.5
        base_k9, whip = _season_pitcher_anchor(rec, model_guess)
        for c in ('FIP', 'WHIP', 'K%', 'BB%', 'SwStr%'):
            if c in rec and pd.notna(rec[c]):
                r[c] = float(rec[c])
    else:
        ipps = 5.5
        whip = 1.28
        sc = get_statcast_pitcher_features(int(pid), days=180) if pid else {}
        whiff2s = float((sc or {}).get('whiff_2S', 0.11) or 0.11)
        swstr_fb = float((sc or {}).get('FF_whiff_rate', 0.10) or 0.10)
        swstr_os = float((sc or {}).get('SL_whiff_rate', (sc or {}).get('CH_whiff_rate', 0.12)) or 0.12)
        sc_guess = float(np.clip(7.4 + 18.0 * (whiff2s - 0.11) + 6.0 * (0.5 * swstr_fb + 0.5 * swstr_os - 0.11), 5.2, 12.8))
        base_k9 = float(np.clip(0.58 * model_guess + 0.42 * sc_guess, 4.5, 13.0))

    form = _recent_pitcher_form(int(pid) if pid else 0)
    recent_candidates = [x for x in (form['k9_7'], form['k9_14'], form['k9_30']) if np.isfinite(x)]
    recent_blend = _weighted_mean(recent_candidates, [3.0, 2.0, 1.0][:len(recent_candidates)]) if recent_candidates else base_k9

    opp_hand = pitcher_throws(int(pid)) if pid else 'R'
    lineup_ctx = _lineup_k_tendency(list(opp_lineup or []), opp_hand, batting_team=opp_team)
    lineup_k_mult = float(np.clip(lineup_ctx['k_rate_shrunk'] / LEAGUE_K_RATE, 0.78, 1.24))
    zone_pitch_ctx = summarize_zone_pitch_ko_context(
        int(pid) if pid else None,
        list(opp_lineup or []),
        opp_team=opp_team,
        as_of_date=as_of_date,
        pitcher_hand=opp_hand,
        days=365,
    )
    pitchtype_k_mult = 1.0
    if callable(lineup_pitchtype_k_boost):
        try:
            pitchtype_k_mult = float(np.clip(lineup_pitchtype_k_boost(int(pid or 0), lineup_ctx.get('batter_ids', [])), 0.88, 1.12))
        except Exception:
            pitchtype_k_mult = 1.0

    row = pd.Series(r).copy()
    row['IP_per_start'] = ipps
    row['Pred_K9'] = base_k9
    row['Recent_K9_7d'] = form['k9_7']
    row['Recent_K9_14d'] = form['k9_14']
    row['Recent_K9_30d'] = form['k9_30']
    row['Recent_ERA_30d'] = form['era_30']
    row['opp_lineup_k_rate_raw'] = lineup_ctx.get('k_rate_raw', lineup_ctx['k_rate'])
    row['opp_lineup_k_rate_shrunk'] = lineup_ctx.get('k_rate_shrunk', lineup_ctx['k_rate'])
    row['opp_lineup_k_rate'] = lineup_ctx.get('k_rate_shrunk', lineup_ctx['k_rate'])
    row['opp_lineup_obp_proxy'] = lineup_ctx['obp_proxy']
    row['opp_lineup_pa_seen'] = lineup_ctx.get('pa_seen', np.nan)
    row['opp_lineup_resolved_n'] = lineup_ctx.get('resolved_n', np.nan)

    decomp_ip = _safe_model_predict(predict_ip_start, row, np.nan)
    decomp_bfip = _safe_model_predict(predict_bf_per_ip, row, np.nan)
    decomp_kbf = _safe_model_predict(predict_k_per_bf, row, np.nan)
    decomp_k9 = _safe_model_predict(predict_k9_decomp, row, np.nan)
    decomp_k_start = _safe_model_predict(predict_k_start_decomp, row, np.nan)

    k9_parts = [base_k9, recent_blend, model_guess]
    k9_weights = [0.40, 0.28, 0.18]
    if np.isfinite(decomp_k9):
        k9_parts.append(decomp_k9)
        k9_weights.append(0.14)

    base_pred_k9 = float(np.clip(_weighted_mean(k9_parts, k9_weights), 4.0, 15.0))

    # IP model
    ip_reg_recent = recent_blend if np.isfinite(recent_blend) else base_pred_k9
    try:
        ip_reg_pred = float(IP_REG.predict([[ipps, base_pred_k9, ip_reg_recent]])[0])
    except Exception:
        ip_reg_pred = ipps

    pred_mean_ip = _weighted_mean(
        [ipps, ip_reg_pred, decomp_ip],
        [0.20, 0.55, 0.25] if np.isfinite(decomp_ip) else [0.30, 0.70]
    )
    if not np.isfinite(pred_mean_ip):
        pred_mean_ip = ip_reg_pred if np.isfinite(ip_reg_pred) else ipps
    if np.isfinite(form['era_30']):
        pred_mean_ip *= float(np.clip(1.0 - 0.015 * (form['era_30'] - 4.10), 0.90, 1.07))
    pred_mean_ip *= float(np.clip(1.0 - 0.42 * (lineup_ctx['obp_proxy'] - LEAGUE_OBP), 0.92, 1.06))
    proj_ip = float(min(sample_starter_ip(pred_mean_ip, base_pred_k9), 8.0))

    # BF/IP and K/BF
    heuristic_bfip = _estimate_bf(1.0, whip, lineup_ctx['obp_proxy'])
    if not np.isfinite(heuristic_bfip):
        heuristic_bfip = LEAGUE_BF_PER_IP
    bf_per_ip = _weighted_mean(
        [heuristic_bfip, decomp_bfip],
        [0.65, 0.35] if np.isfinite(decomp_bfip) else [1.0]
    )
    if not np.isfinite(bf_per_ip):
        bf_per_ip = heuristic_bfip
    bf_per_ip = float(np.clip(bf_per_ip, 3.0, 5.7))

    heuristic_kbf = base_pred_k9 / max(9.0 * bf_per_ip, 1e-6)
    k_per_bf = _weighted_mean(
        [heuristic_kbf, decomp_kbf],
        [0.58, 0.42] if np.isfinite(decomp_kbf) else [1.0]
    )
    if not np.isfinite(k_per_bf):
        k_per_bf = heuristic_kbf

    if ump_feats and 'ump_k9' in ump_feats and pd.notna(ump_feats['ump_k9']):
        k_per_bf *= float(np.clip(ump_feats['ump_k9'], 0.90, 1.10))
    if framing_feats:
        frame_boost = 0.0
        for k in ('away_frame', 'home_frame'):
            if k in framing_feats and pd.notna(framing_feats[k]):
                frame_boost += float(framing_feats[k]) / 100.0
        k_per_bf *= float(np.clip(1.0 + frame_boost, 0.95, 1.08))

    k_per_bf *= lineup_k_mult
    k_per_bf *= pitchtype_k_mult
    k_per_bf *= float(zone_pitch_ctx.get('zone_pitch_k_mult', 1.0) or 1.0)
    k_per_bf = float(np.clip(k_per_bf, 0.08, 0.42))

    pred_k9 = float(np.clip(9.0 * bf_per_ip * k_per_bf, 4.0, 15.5))
    proj_bf = float(np.clip(proj_ip * bf_per_ip, 12.0, 35.0))
    proj_k_mu = float(np.clip(proj_bf * k_per_bf, 0.0, 15.0))

    park_k_mult = _park_k_mult(team)

    r.update({
        'IP_per_start': ipps,
        'Pred_K9': pred_k9,
        'Proj_IP': proj_ip,
        'Proj_BF': proj_bf,
        'BF_per_IP': bf_per_ip,
        'K_per_BF': k_per_bf,
        'WHIP': whip,
        'recent_k9_7': form['k9_7'],
        'recent_k9_14': form['k9_14'],
        'recent_k9_30': form['k9_30'],
        'recent_era_30': form['era_30'],
        'opp_lineup_k_rate_raw': lineup_ctx.get('k_rate_raw', lineup_ctx['k_rate']),
        'opp_lineup_k_rate_shrunk': lineup_ctx.get('k_rate_shrunk', lineup_ctx['k_rate']),
        'opp_lineup_k_rate': lineup_ctx.get('k_rate_shrunk', lineup_ctx['k_rate']),
        'opp_lineup_obp_proxy': lineup_ctx['obp_proxy'],
        'zone_pitch_k_mult': zone_pitch_ctx.get('zone_pitch_k_mult'),
        'zone_pitch_whiff_mult': zone_pitch_ctx.get('zone_pitch_whiff_mult'),
        'zone_pitch_called_mult': zone_pitch_ctx.get('zone_pitch_called_mult'),
        'zone_pitch_putaway_mult': zone_pitch_ctx.get('zone_pitch_putaway_mult'),
        'zone_pitch_foul_suppress_mult': zone_pitch_ctx.get('zone_pitch_foul_suppress_mult'),
        'zone_pitch_confidence': zone_pitch_ctx.get('zone_pitch_confidence'),
        'zone_pitch_resolved_n': zone_pitch_ctx.get('zone_pitch_resolved_n'),
        'zone_pitch_top_family': zone_pitch_ctx.get('zone_pitch_top_family'),
        'zone_pitch_attack_zone': zone_pitch_ctx.get('zone_pitch_attack_zone'),
        'pitchtype_k_mult': pitchtype_k_mult,
        'lineup_k_mult': lineup_k_mult,
        'proj_k_start_model': decomp_k_start,
        'proj_k_start_mu': proj_k_mu,
        'park_k_mult': park_k_mult,
        'Team': team,
        'Name': name or '',
        'Pitcher_ID': pid,
        'zone_pitch_k_mult': zone_pitch_ctx.get('zone_pitch_k_mult'),
        'zone_pitch_whiff_mult': zone_pitch_ctx.get('zone_pitch_whiff_mult'),
        'zone_pitch_called_mult': zone_pitch_ctx.get('zone_pitch_called_mult'),
        'zone_pitch_putaway_mult': zone_pitch_ctx.get('zone_pitch_putaway_mult'),
        'zone_pitch_foul_suppress_mult': zone_pitch_ctx.get('zone_pitch_foul_suppress_mult'),
        'zone_pitch_confidence': zone_pitch_ctx.get('zone_pitch_confidence'),
        'zone_pitch_resolved_n': zone_pitch_ctx.get('zone_pitch_resolved_n'),
        'zone_pitch_top_family': zone_pitch_ctx.get('zone_pitch_top_family'),
        'zone_pitch_attack_zone': zone_pitch_ctx.get('zone_pitch_attack_zone'),
        'zone_pitch_second_family': zone_pitch_ctx.get('zone_pitch_second_family'),
        'zone_pitch_second_zone': zone_pitch_ctx.get('zone_pitch_second_zone'),
        'zone_pitch_mix_entropy': zone_pitch_ctx.get('zone_pitch_mix_entropy'),
        'zone_pitch_player_weight': zone_pitch_ctx.get('zone_pitch_player_weight'),
        'zone_pitch_family_weight': zone_pitch_ctx.get('zone_pitch_family_weight'),
        'zone_pitch_archetype_weight': zone_pitch_ctx.get('zone_pitch_archetype_weight'),
        'zone_pitch_hand_weight': zone_pitch_ctx.get('zone_pitch_hand_weight'),
        'zone_pitch_league_weight': zone_pitch_ctx.get('zone_pitch_league_weight'),
        'zone_pitch_pitcher_arch': zone_pitch_ctx.get('zone_pitch_pitcher_arch'),
        'zone_pitch_exact_pitcher_rows': zone_pitch_ctx.get('zone_pitch_exact_pitcher_rows'),
        'zone_pitch_pitcher_family_rows': zone_pitch_ctx.get('zone_pitch_pitcher_family_rows'),
        'zone_pitch_pitcher_archetype_rows': zone_pitch_ctx.get('zone_pitch_pitcher_archetype_rows'),
        'zone_pitch_hand_rows': zone_pitch_ctx.get('zone_pitch_hand_rows'),
        'zone_pitch_league_rows': zone_pitch_ctx.get('zone_pitch_league_rows'),
        'zone_pitch_candidate_source': zone_pitch_ctx.get('zone_pitch_candidate_source'),
        'zone_pitch_sample_regime': zone_pitch_ctx.get('zone_pitch_sample_regime'),
        'zone_pitch_effect_strength': zone_pitch_ctx.get('zone_pitch_effect_strength'),
        'zone_pitch_early_season_shrink': zone_pitch_ctx.get('zone_pitch_early_season_shrink'),
    })
    return r


def predict_matchup_ko(date_str: str, matchup: str, override_path: str | None = None) -> pd.DataFrame:
    matchup = matchup.replace(' @ ', '@').upper().strip()
    away, home = matchup.split('@')
    meta = _resolve_lineups(date_str, away, home, override_path)

    away_r = build_starter_dict(
        away,
        meta.get('away_pitcher_name'),
        meta.get('away_pitcher_id'),
        opp_lineup=meta.get('home_lineup', []),
        opp_team=home,
        ump_feats=meta.get('ump_feats'),
        framing_feats={'away_frame': meta.get('framing_feats', {}).get('away_frame', 0.0)},
        as_of_date=date_str,
    )
    home_r = build_starter_dict(
        home,
        meta.get('home_pitcher_name'),
        meta.get('home_pitcher_id'),
        opp_lineup=meta.get('away_lineup', []),
        opp_team=away,
        ump_feats=meta.get('ump_feats'),
        framing_feats={'home_frame': meta.get('framing_feats', {}).get('home_frame', 0.0)},
        as_of_date=date_str,
    )

    starter_rows = []
    for team, st, opp_lineup in [
        (away, away_r, meta.get('home_lineup', [])),
        (home, home_r, meta.get('away_lineup', [])),
    ]:
        sr = _simulate_k_distribution(float(st.get('proj_k_start_mu', np.nan)), st)
        sr.update(
            {
                'date': date_str,
                'matchup': matchup,
                'Team': team,
                'Name': st.get('Name'),
                'Pitcher_ID': st.get('Pitcher_ID'),
                'Proj_IP': st.get('Proj_IP'),
                'Proj_BF': st.get('Proj_BF'),
                'BF_per_IP': st.get('BF_per_IP'),
                'K_per_BF': st.get('K_per_BF'),
                'IP_per_start': st.get('IP_per_start'),
                'Pred_K9': st.get('Pred_K9'),
                'pitchtype_k_mult': st.get('pitchtype_k_mult'),
                'lineup_k_mult': st.get('lineup_k_mult'),
                'zone_pitch_k_mult': st.get('zone_pitch_k_mult'),
                'zone_pitch_whiff_mult': st.get('zone_pitch_whiff_mult'),
                'zone_pitch_called_mult': st.get('zone_pitch_called_mult'),
                'zone_pitch_putaway_mult': st.get('zone_pitch_putaway_mult'),
                'zone_pitch_foul_suppress_mult': st.get('zone_pitch_foul_suppress_mult'),
                'zone_pitch_confidence': st.get('zone_pitch_confidence'),
                'zone_pitch_resolved_n': st.get('zone_pitch_resolved_n'),
                'zone_pitch_top_family': st.get('zone_pitch_top_family'),
                'zone_pitch_attack_zone': st.get('zone_pitch_attack_zone'),
                'opp_lineup_source': meta.get('lineup_source', 'projected'),
                'lineup_weight': meta.get('lineup_weight', 0.84),
                'lineup_certainty': meta.get('lineup_certainty', meta.get('lineup_weight', 0.84)),
                'recent_k9_7': st.get('recent_k9_7'),
                'recent_k9_14': st.get('recent_k9_14'),
                'recent_k9_30': st.get('recent_k9_30'),
                'opp_lineup_k_rate_raw': st.get('opp_lineup_k_rate_raw'),
                'opp_lineup_k_rate_shrunk': st.get('opp_lineup_k_rate_shrunk'),
                'opp_lineup_k_rate': st.get('opp_lineup_k_rate'),
                'opp_lineup_pa_seen': st.get('opp_lineup_pa_seen'),
                'opp_lineup_resolved_n': st.get('opp_lineup_resolved_n'),
                'zone_pitch_second_family': st.get('zone_pitch_second_family'),
                'zone_pitch_second_zone': st.get('zone_pitch_second_zone'),
                'zone_pitch_mix_entropy': st.get('zone_pitch_mix_entropy'),
                'zone_pitch_player_weight': st.get('zone_pitch_player_weight'),
                'zone_pitch_family_weight': st.get('zone_pitch_family_weight'),
                'zone_pitch_archetype_weight': st.get('zone_pitch_archetype_weight'),
                'zone_pitch_hand_weight': st.get('zone_pitch_hand_weight'),
                'zone_pitch_league_weight': st.get('zone_pitch_league_weight'),
                'zone_pitch_pitcher_arch': st.get('zone_pitch_pitcher_arch'),
                'zone_pitch_exact_pitcher_rows': st.get('zone_pitch_exact_pitcher_rows'),
                'zone_pitch_pitcher_family_rows': st.get('zone_pitch_pitcher_family_rows'),
                'zone_pitch_pitcher_archetype_rows': st.get('zone_pitch_pitcher_archetype_rows'),
                'zone_pitch_hand_rows': st.get('zone_pitch_hand_rows'),
                'zone_pitch_league_rows': st.get('zone_pitch_league_rows'),
                'zone_pitch_candidate_source': st.get('zone_pitch_candidate_source'),
                'zone_pitch_sample_regime': st.get('zone_pitch_sample_regime'),
                'zone_pitch_effect_strength': st.get('zone_pitch_effect_strength'),
                'zone_pitch_early_season_shrink': st.get('zone_pitch_early_season_shrink'),
            }
        )
        starter_rows.append(sr)

    df = pd.DataFrame(starter_rows)
    prob_cols = sorted([c for c in df.columns if str(c).startswith('P(K≥')], key=lambda x: int(str(x).split('≥', 1)[1].split(')', 1)[0]))

    keep = [
    c for c in [
        'date', 'matchup', *BASE_COLS,
        'hazard_adj_proj_bf', 'hazard_adj_proj_k',
        'recent_k9_7', 'recent_k9_14', 'recent_k9_30',
        'opp_lineup_k_rate_raw', 'opp_lineup_k_rate_shrunk', 'opp_lineup_k_rate',
        'opp_lineup_pa_seen', 'opp_lineup_resolved_n',
        'opp_lineup_role_k_mult', 'opp_lineup_role_top5_usual_slot_mean', 'opp_lineup_role_dislocation',
        'opp_lineup_role_hist_n', 'opp_lineup_role_promoted_bottom_n', 'opp_lineup_role_demoted_top_n',

        'zone_pitch_k_mult', 'zone_pitch_whiff_mult', 'zone_pitch_called_mult',
        'zone_pitch_putaway_mult', 'zone_pitch_foul_suppress_mult',
        'zone_pitch_confidence', 'zone_pitch_resolved_n',
        'zone_pitch_top_family', 'zone_pitch_second_family',
        'zone_pitch_attack_zone', 'zone_pitch_second_zone',
        'zone_pitch_mix_entropy',
        'zone_pitch_player_weight', 'zone_pitch_family_weight',
        'zone_pitch_archetype_weight', 'zone_pitch_hand_weight',
        'zone_pitch_league_weight',
        'zone_pitch_pitcher_arch',
        'zone_pitch_exact_pitcher_rows',
        'zone_pitch_pitcher_family_rows',
        'zone_pitch_pitcher_archetype_rows',
        'zone_pitch_hand_rows',
        'zone_pitch_league_rows',
        'zone_pitch_candidate_source',
        'zone_pitch_sample_regime',
        'zone_pitch_effect_strength',
        'zone_pitch_early_season_shrink',

        'count_path_k_prob_raw', 'count_path_k_prob_shrunk', 'count_path_k_prob',
        'expected_two_strike_reach_rate', 'expected_putaway_efficiency',
        'called_strike_plus_whiff_path_prob', 'foul_extension_penalty',
        'starter_pull_prob_by_18BF', 'starter_pull_prob_by_21BF', 'starter_pull_prob_by_24BF',
        'pitchtype_k_mult', 'lineup_k_mult', 'count_path_k_mult',
        'survival_mult', 'stuff_day_state',
        'regime_prob_whiff_up', 'regime_prob_short_leash', 'regime_prob_laboring', 'regime_prob_neutral',
        *prob_cols, '90% CI', 'opp_lineup_source', 'lineup_weight', 'lineup_certainty',
        ] if c in df.columns
    ]
    if keep:
        df = df[keep]
    return df


def predict_slate_ko(date_str: str, override_path: str | None = None) -> pd.DataFrame:
    frames = []
    for gm in fetch_matchups_for_date(date_str):
        try:
            df = predict_matchup_ko(date_str, gm.replace(' @ ', '@'), override_path=override_path)
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)
            else:
                LOGGER.warning("Empty KO frame for %s on %s", gm, date_str)
                frames.append(pd.DataFrame([{
                    "date": date_str,
                    "matchup": gm.replace(' @ ', '@'),
                    "Team": "",
                    "Name": "",
                    "status": "matchup_empty_frame",
                }]))
        except Exception as exc:
            LOGGER.exception("Slate KO failure for %s on %s", gm, date_str)
            frames.append(pd.DataFrame([{
                "date": date_str,
                "matchup": gm.replace(' @ ', '@'),
                "Team": "",
                "Name": "",
                "status": f"matchup_error:{type(exc).__name__}:{str(exc)[:140]}",
            }]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



def recommended_k_angles(ko_df: pd.DataFrame, prob_floor: float = 0.55) -> pd.DataFrame:
    """Summarize conservative recommended K-over angles from a KO output table."""
    if ko_df is None or ko_df.empty:
        return pd.DataFrame()
    prob_cols = sorted(
    [c for c in ko_df.columns if str(c).startswith('P(K≥')],
    key=lambda x: int(str(x).split('≥', 1)[1].split(')', 1)[0])
)
    rows = []
    for _, row in ko_df.iterrows():
        best_line = None
        best_prob = None
        conservative_line = None
        conservative_prob = None
        for c in prob_cols:
            try:
                k = int(str(c).split('≥', 1)[1].split(')', 1)[0])
                p = float(row[c])
            except Exception:
                continue
            if best_prob is None or p > best_prob:
                best_line, best_prob = k, p
            if p >= prob_floor:
                conservative_line, conservative_prob = k, p
        rows.append({
            'date': row.get('date'),
            'matchup': row.get('matchup'),
            'Team': row.get('Team'),
            'Pitcher': row.get('Name'),
            'Mean_K_start': row.get('Mean_K_start'),
            'Median_K_start': row.get('Median_K_start'),
            'best_prob_line': best_line,
            'best_prob': best_prob,
            'recommended_over_line': conservative_line,
            'recommended_over_prob': conservative_prob,
        })
    return pd.DataFrame(rows)
