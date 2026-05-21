from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
import logging
import re

import numpy as np
import pandas as pd

from .compat import ROOT, LEGACY  # noqa: F401
from .lineups import (
    get_override_for_game,
    lineup_source_weight,
    batting_order_pa,
    lineup_certainty_score,
    fill_tbd_lineup,
    lineup_needs_projection,
)
from .context_features import apply_context_stack, load_team_defense_proxies
from .lineup_role_features import lookup_lineup_role_features

import fetch  # type: ignore
from fetch import (
    safe_get,
    ID2ABBR,
    fetch_unofficial_lineup,
    fetch_lineup_and_starters,
    name_to_mlbam_id,
    strip_pos,
    fetch_statcast_raw,
    get_statcast_pitcher_features,
    get_statcast_batter_features,
    pitcher_throws,
    batter_bats,
    fetch_monthly_park_factors,
    team_bullpen_hrpa,
    YEAR,
    clean_name,
    roster_map,
    ABBR2ID,
    active_hitter_names,
)
from precompute_everything import batter_game_hr_prob, BATTER_CAREER  # type: ignore
try:
    from precompute_everything import iso_pa_calibrators  # type: ignore
except Exception:
    iso_pa_calibrators = {}

try:
    from .zone_hr_features import summarize_zone_hr_context
except Exception:
    summarize_zone_hr_context = None

try:
    from .markov_game_context import apply_markov_context_to_batter_df  # type: ignore
except Exception:
    apply_markov_context_to_batter_df = None

LEAGUE_HIT_PA = 0.225
LEAGUE_2B_SHARE = 0.21
LEAGUE_3B_SHARE = 0.02
LEAGUE_HR_SHARE = 0.14
LEAGUE_1B_SHARE = 1.0 - LEAGUE_2B_SHARE - LEAGUE_3B_SHARE - LEAGUE_HR_SHARE

LEAGUE_FLYBALL_PCT = 0.36
LEAGUE_PULL_AIR_PCT = 0.12
LEAGUE_XWOBACON = 0.37
LEAGUE_BARREL_PCT = 0.08
LEAGUE_HARD_HIT_PCT = 0.36

LOGGER = logging.getLogger(__name__)


def _safe_mean(series) -> float:
    try:
        arr = pd.to_numeric(series, errors='coerce')
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return 0.0
        val = float(arr.mean())
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


def _safe_frac(mask) -> float:
    try:
        ser = pd.Series(mask)
        if len(ser) == 0:
            return 0.0
        return float(pd.to_numeric(ser, errors='coerce').fillna(0).mean())
    except Exception:
        return 0.0


def _normalize_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    if df.empty:
        return df.copy()
    out = df.copy()
    for c in list(out.columns):
        if str(c).lower() in {'date', 'game_date', 'as_of_date'}:
            dt = pd.to_datetime(out[c], errors='coerce')
            out[c] = dt.dt.strftime('%Y-%m-%d').where(dt.notna(), out[c].astype(str))
    return out




def _nanmean_safe(values, default: float = 0.0) -> float:
    try:
        arr = pd.to_numeric(pd.Series(values), errors='coerce')
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return float(default)
        val = float(arr.mean())
        return val if np.isfinite(val) else float(default)
    except Exception:
        return float(default)


def _err_status(exc: Exception, limit: int = 140) -> str:
    msg = str(exc).strip().replace('\n', ' ').replace('\r', ' ')
    if not msg:
        msg = 'no_message'
    if len(msg) > limit:
        msg = msg[:limit]
    return f'row_error:{type(exc).__name__}:{msg}'


def _team_roster_names(team_abbr: str) -> dict[str, int]:
    return _team_roster_lookup(_canonical_team_abbr(team_abbr))


def _active_roster_keyset(team_abbr: str) -> set[str]:
    try:
        return {clean_name(strip_pos(nm)) for nm in active_hitter_names(_canonical_team_abbr(team_abbr)) if str(nm).strip()}
    except Exception:
        return set()


def _row_eligibility_flag(base_row: dict, active_flag: int, batter_pid) -> tuple[int, str]:
    if not batter_pid:
        return 0, 'unresolved_team_roster_player'
    if int(active_flag) != 1:
        return 0, 'off_active_roster'
    nm = str(base_row.get('player_name', '') or '').strip().upper()
    if nm in {'', 'TBD', 'NONE'}:
        return 0, 'missing_player_name'
    opp = str(base_row.get('opp_pitcher', '') or '').strip().upper()
    if opp in {'', 'TBD', 'NONE'}:
        return 0, 'missing_opp_pitcher'
    if not base_row.get('opp_pitcher_id'):
        return 0, 'missing_opp_pitcher_id'
    lineup_source = str(base_row.get('lineup_source', '') or '').strip().lower()
    if lineup_source != 'confirmed':
        return 0, 'non_confirmed_lineup'
    try:
        certainty = float(base_row.get('lineup_certainty', 0.0) or 0.0)
    except Exception:
        certainty = 0.0
    if certainty < 0.95:
        return 0, 'low_lineup_certainty'
    try:
        slot = int(base_row.get('batting_order', 0) or 0)
    except Exception:
        slot = 0
    if slot < 1 or slot > 9:
        return 0, 'invalid_batting_order'
    return 1, ''


def _strict_resolve_batter(team_abbr: str, hitter_name: str | None):
    hitter_name = strip_pos(hitter_name or '').strip()
    if not hitter_name or hitter_name.upper() == 'TBD':
        return None, None

    roster = _team_roster_names(team_abbr)
    key = clean_name(hitter_name)
    if key in roster:
        return int(roster[key]), hitter_name

    alias = re.sub(r"\b(JR|SR|II|III|IV)\b", '', key).replace('.', ' ').strip()
    alias = re.sub(r'\s+', ' ', alias)
    for nm, pid in roster.items():
        nm2 = re.sub(r"\b(JR|SR|II|III|IV)\b", '', str(nm)).replace('.', ' ').strip()
        nm2 = re.sub(r'\s+', ' ', nm2)
        if nm2 == alias:
            return int(pid), hitter_name

    pid = _resolve_batter_id(team_abbr, hitter_name)
    if pid and int(pid) in set(roster.values()):
        return int(pid), hitter_name
    return None, None


def _sanitize_lineup_names(team_abbr: str, lineup: list[str]) -> list[str]:
    cleaned = []
    for nm in list(lineup or [])[:9]:
        raw = strip_pos(str(nm or '')).strip()
        if not raw or raw.upper() == 'TBD':
            cleaned.append('TBD')
        else:
            cleaned.append(raw)
    return cleaned


def _career_hr_game_prob(player_name: str, exp_pa: float) -> float:
    base = _career_baseline_rates(player_name)
    hit_pa = float(base.get('hit_pa', LEAGUE_HIT_PA))
    hr_share = float(base.get('HR_share', LEAGUE_HR_SHARE))
    hr_pa = max(hit_pa * hr_share, 0.0)
    return _binom_at_least_one(hr_pa, exp_pa)


def _calibrate_hr_probability(
    raw_game_prob: float,
    player_name: str,
    recent: dict[str, float],
    exp_pa: float,
    lineup_certainty: float,
    power: dict[str, float],
    pitcher_danger: dict[str, float],
) -> float:
    raw = float(np.clip(raw_game_prob, 0.0, 0.95))
    baseline = _career_hr_game_prob(player_name, exp_pa)

    sample_n = float(recent.get('pa_30d', 0)) + 0.35 * float(recent.get('pa_365d', 0))
    w = float(np.clip(sample_n / (sample_n + 180.0), 0.15, 0.78))
    shrunk = w * raw + (1.0 - w) * baseline

    barrel = float(power.get('barrel_pct', LEAGUE_BARREL_PCT))
    hh = float(power.get('hard_hit_pct', LEAGUE_HARD_HIT_PCT))
    p_hr = float(pitcher_danger.get('hr_pa_allowed', LEAGUE_HIT_PA * LEAGUE_HR_SHARE))

    ceiling = 0.10 + 0.90 * barrel + 0.10 * max(hh - LEAGUE_HARD_HIT_PCT, 0.0) + 2.0 * p_hr
    ceiling = float(np.clip(ceiling, 0.10, 0.24))

    certainty_penalty = 0.92 + 0.08 * float(np.clip(lineup_certainty, 0.0, 1.0))
    calibrated = min(shrunk * certainty_penalty, ceiling)
    return float(np.clip(calibrated, 0.01, 0.30))


FDOdds_PATH = ROOT / 'data' / 'FDOdds.csv'


def _load_fd_odds(path: Path = FDOdds_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    rename = {}
    for c in out.columns:
        lc = c.lower()
        if lc in {'odds', 'fd_line', 'fd_price', 'fanduel_odds'}:
            rename[c] = 'fd_odds'
        elif lc in {'name', 'player'}:
            rename[c] = 'player_name'
    if rename:
        out = out.rename(columns=rename)
    need = {'date', 'matchup', 'player_name', 'fd_odds'}
    if not need.issubset(out.columns):
        return pd.DataFrame()
    out['date'] = out['date'].astype(str).str.strip()
    out['matchup'] = out['matchup'].astype(str).str.replace(' @ ', '@', regex=False).str.strip().str.upper()
    out['player_name'] = out['player_name'].astype(str).str.strip()
    out['fd_odds'] = pd.to_numeric(out['fd_odds'], errors='coerce')
    return out.dropna(subset=['fd_odds'])


def _american_to_implied_prob(odds) -> float:
    try:
        o = float(odds)
    except Exception:
        return float('nan')
    if not np.isfinite(o) or o == 0:
        return float('nan')
    if o > 0:
        return float(100.0 / (o + 100.0))
    return float(abs(o) / (abs(o) + 100.0))


def _prob_to_american(p) -> float:
    try:
        p = float(p)
    except Exception:
        return float('nan')
    if not np.isfinite(p) or p <= 0 or p >= 1:
        return float('nan')
    if p < 0.5:
        return float(100.0 * (1.0 - p) / p)
    return float(-100.0 * p / (1.0 - p))


def _partial_pool(raw: float, sample_n: float, baseline: float, prior_n: float = 80.0) -> float:
    try:
        raw = float(raw); sample_n = float(sample_n); baseline = float(baseline); prior_n = float(prior_n)
    except Exception:
        return float(baseline)
    if not np.isfinite(raw):
        raw = baseline
    if not np.isfinite(sample_n) or sample_n < 0:
        sample_n = 0.0
    if not np.isfinite(baseline):
        baseline = raw
    return float((raw * sample_n + baseline * prior_n) / max(sample_n + prior_n, 1e-9))


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _score_from_delta(value: float, baseline: float, scale: float, center: float = 50.0) -> float:
    try:
        z = (float(value) - float(baseline)) / max(float(scale), 1e-9)
    except Exception:
        return 50.0
    return float(np.clip(center + 12.0 * z, 0.0, 100.0))


def _hr_prob_interval(model_prob_hr: float, exp_pa: float, sample_n: float, lineup_certainty: float) -> tuple[float, float]:
    p = _clip01(model_prob_hr)
    n_eff = max(float(sample_n), 5.0) * max(0.65, float(lineup_certainty))
    se = np.sqrt(max(p * (1.0 - p), 1e-8) / n_eff)
    half = float(np.clip(1.96 * se, 0.02, 0.20))
    return (_clip01(p - half), _clip01(p + half))


def _confidence_pct(model_prob_hr: float, low: float, high: float, lineup_certainty: float, pa_recent: float, status) -> float:
    width = max(float(high) - float(low), 0.0)
    sample_component = np.clip(np.log1p(max(pa_recent, 0.0)) / np.log(121.0), 0.0, 1.0)
    interval_component = np.clip(1.0 - width / 0.35, 0.0, 1.0)
    lineup_component = np.clip(float(lineup_certainty), 0.0, 1.0)
    error_penalty = 0.65 if pd.notna(status) and str(status).strip() else 1.0
    conf = 100.0 * (0.45 * interval_component + 0.35 * sample_component + 0.20 * lineup_component) * error_penalty
    return float(np.clip(conf, 1.0, 99.0))


def _power_score(power: dict[str, float]) -> float:
    vals = [
        _score_from_delta(power.get('barrel_pct', LEAGUE_BARREL_PCT), LEAGUE_BARREL_PCT, 0.03),
        _score_from_delta(power.get('hard_hit_pct', LEAGUE_HARD_HIT_PCT), LEAGUE_HARD_HIT_PCT, 0.08),
        _score_from_delta(power.get('xiso_est', 0.17), 0.17, 0.05),
        _score_from_delta(power.get('flyball_pct', LEAGUE_FLYBALL_PCT), LEAGUE_FLYBALL_PCT, 0.08),
        _score_from_delta(power.get('pulled_air_pct', LEAGUE_PULL_AIR_PCT), LEAGUE_PULL_AIR_PCT, 0.04),
        _score_from_delta(power.get('ev_mean', 89.0), 89.0, 3.0),
        _score_from_delta(power.get('form_hr_mult', 1.0), 1.0, 0.10),
    ]
    return float(np.clip(np.average(vals, weights=[1.4, 1.2, 1.2, 0.8, 0.9, 0.7, 0.8]), 0.0, 100.0))


def _pitcher_vuln_score(d: dict[str, float]) -> float:
    vals = [
        _score_from_delta(d.get('barrel_allowed_pct', LEAGUE_BARREL_PCT), LEAGUE_BARREL_PCT, 0.03),
        _score_from_delta(d.get('hard_hit_allowed_pct', LEAGUE_HARD_HIT_PCT), LEAGUE_HARD_HIT_PCT, 0.08),
        _score_from_delta(d.get('flyball_allowed_pct', LEAGUE_FLYBALL_PCT), LEAGUE_FLYBALL_PCT, 0.08),
        _score_from_delta(d.get('hr_pa_allowed', LEAGUE_HIT_PA * LEAGUE_HR_SHARE), LEAGUE_HIT_PA * LEAGUE_HR_SHARE, 0.02),
        _score_from_delta(d.get('xwobacon_allowed', LEAGUE_XWOBACON), LEAGUE_XWOBACON, 0.04),
        _score_from_delta(d.get('danger_mult', 1.0), 1.0, 0.12),
        _score_from_delta(d.get('recent_danger_mult', 1.0), 1.0, 0.10),
    ]
    return float(np.clip(np.average(vals, weights=[1.4, 1.1, 0.8, 1.2, 1.0, 1.0, 0.9]), 0.0, 100.0))


def _context_score(ctx: dict[str, float], lineup_certainty: float, exp_pa: float, bp: dict[str, float], park_hr_mult: float) -> float:
    vals = [
        _score_from_delta(park_hr_mult, 1.0, 0.08),
        _score_from_delta(float(ctx.get('weather_hr_mult', 1.0)), 1.0, 0.08),
        _score_from_delta(float(ctx.get('pitch_type_mult', 1.0)), 1.0, 0.08),
        _score_from_delta(float(ctx.get('zone_mult', 1.0)), 1.0, 0.04),
        _score_from_delta(float(bp.get('bullpen_hr_mult', 1.0)), 1.0, 0.08),
        _score_from_delta(float(exp_pa), 4.25, 0.35),
        _score_from_delta(float(lineup_certainty), 0.85, 0.10),
    ]
    return float(np.clip(np.average(vals, weights=[1.2, 1.0, 0.9, 0.5, 0.8, 1.0, 0.8]), 0.0, 100.0))


def _composite_hr_score(power_score: float, vuln_score: float, context_score: float, model_prob_hr: float) -> float:
    comp = 0.42 * power_score + 0.33 * vuln_score + 0.17 * context_score + 8.0 * float(model_prob_hr)
    return float(np.clip(comp, 0.0, 100.0))


def _breakout_tag(composite_hr_score: float, edge_prob) -> str:
    try:
        edge_prob = float(edge_prob)
    except Exception:
        edge_prob = float('nan')
    if np.isfinite(edge_prob):
        if composite_hr_score >= 72 and edge_prob >= 0.03:
            return 'Elite Edge'
        if composite_hr_score >= 62 and edge_prob >= 0.015:
            return 'Strong Edge'
        if composite_hr_score >= 54 and edge_prob >= 0.005:
            return 'Lean'
        return 'Pass'
    if composite_hr_score >= 72:
        return 'A'
    if composite_hr_score >= 62:
        return 'B'
    if composite_hr_score >= 54:
        return 'C'
    return 'Pass'


def _merge_fd_odds(out: pd.DataFrame) -> pd.DataFrame:
    odds = _load_fd_odds()
    if odds.empty or out.empty:
        for col in ['fd_odds', 'fd_implied_prob', 'fair_odds_american', 'edge_prob', 'edge_pct_pts']:
            if col not in out.columns:
                out[col] = np.nan
        return out
    m = out.copy()
    m['date'] = m['date'].astype(str)
    m['matchup'] = m['matchup'].astype(str).str.upper()
    m['player_name'] = m['player_name'].astype(str).str.strip()
    merged = m.merge(odds[['date', 'matchup', 'player_name', 'fd_odds']], on=['date', 'matchup', 'player_name'], how='left')
    merged['fd_implied_prob'] = merged['fd_odds'].map(_american_to_implied_prob)
    merged['fair_odds_american'] = pd.to_numeric(merged.get('model_prob_hr'), errors='coerce').map(_prob_to_american)
    merged['edge_prob'] = pd.to_numeric(merged.get('model_prob_hr'), errors='coerce') - pd.to_numeric(merged['fd_implied_prob'], errors='coerce')
    merged['edge_pct_pts'] = 100.0 * merged['edge_prob']
    return merged


def _add_breakout_columns(out: pd.DataFrame) -> pd.DataFrame:
    if out.empty:
        return out
    rows = []
    for _, row in out.iterrows():
        if pd.notna(row.get('status')) and str(row.get('status')).strip():
            rows.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        pscore = _power_score({
            'barrel_pct': row.get('barrel_pct', np.nan),
            'hard_hit_pct': row.get('hard_hit_pct', np.nan),
            'xiso_est': row.get('xiso_est', np.nan),
            'flyball_pct': row.get('flyball_pct', np.nan),
            'pulled_air_pct': row.get('pulled_air_pct', np.nan),
            'ev_mean': row.get('ev_mean', np.nan),
            'form_hr_mult': row.get('batter_form_hr_mult', np.nan),
        })
        vscore = _pitcher_vuln_score({
            'barrel_allowed_pct': row.get('pitcher_barrel_allowed_pct', np.nan),
            'hard_hit_allowed_pct': row.get('pitcher_hard_hit_allowed_pct', np.nan),
            'flyball_allowed_pct': row.get('pitcher_flyball_allowed_pct', np.nan),
            'hr_pa_allowed': row.get('pitcher_hr_pa_allowed', np.nan),
            'xwobacon_allowed': row.get('pitcher_xwobacon_allowed', np.nan),
            'danger_mult': row.get('pitcher_hr_danger_mult', np.nan),
            'recent_danger_mult': row.get('pitcher_recent_danger_mult', np.nan),
        })
        cscore = _context_score({
            'weather_hr_mult': row.get('weather_hr_mult', np.nan),
            'pitch_type_mult': row.get('pitch_type_mult', np.nan),
            'zone_mult': row.get('zone_mult', np.nan),
        }, row.get('lineup_certainty', 1.0), row.get('exp_pa', 4.2), {'bullpen_hr_mult': row.get('bullpen_hr_mult', 1.0)}, row.get('park_hr_mult', 1.0))
        model_prob = pd.to_numeric(pd.Series([row.get('P(HR>=1)', np.nan)]), errors='coerce').iloc[0]
        low, high = _hr_prob_interval(model_prob if np.isfinite(model_prob) else 0.0, row.get('exp_pa', 4.2), row.get('pa_30d', 0) + 0.35 * row.get('pa_365d', 0), row.get('lineup_certainty', 1.0))
        conf = _confidence_pct(model_prob if np.isfinite(model_prob) else 0.0, low, high, row.get('lineup_certainty', 1.0), row.get('pa_30d', 0) + 0.35 * row.get('pa_365d', 0), row.get('status'))
        comp = _composite_hr_score(pscore, vscore, cscore, model_prob if np.isfinite(model_prob) else 0.0)
        rows.append((pscore, vscore, cscore, comp, conf, low, high))
    arr = pd.DataFrame(rows, columns=['power_score', 'pitcher_vulnerability_score', 'context_score', 'composite_hr_score', 'confidence_pct', 'hr_prob_low', 'hr_prob_high'])
    out = pd.concat([out.reset_index(drop=True), arr], axis=1)
    out['model_prob_hr'] = pd.to_numeric(out.get('P(HR>=1)'), errors='coerce')
    out = _merge_fd_odds(out)
    out['breakout_tag'] = [
        _breakout_tag(cs if np.isfinite(cs) else 0.0, ep)
        for cs, ep in zip(pd.to_numeric(out.get('composite_hr_score'), errors='coerce'), pd.to_numeric(out.get('edge_prob'), errors='coerce'))
    ]
    out['breakout_rank'] = pd.to_numeric(out['composite_hr_score'], errors='coerce').fillna(-1).rank(method='first', ascending=False)
    return out

def _estimate_xwobacon(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return LEAGUE_XWOBACON
    for col in ('estimated_woba_using_speedangle', 'woba_value', 'woba_denom'):
        if col in df.columns:
            if col == 'estimated_woba_using_speedangle':
                vals = pd.to_numeric(df[col], errors='coerce')
                val = _nanmean_safe(vals, default=LEAGUE_XWOBACON)
                return val if np.isfinite(val) and val > 0 else LEAGUE_XWOBACON
    events = df.get('events', pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
    weights = np.select(
        [events.eq('single'), events.eq('double'), events.eq('triple'), events.eq('home_run')],
        [0.90, 1.25, 1.60, 2.00],
        default=0.0,
    )
    val = _nanmean_safe(weights, default=LEAGUE_XWOBACON)
    return val if np.isfinite(val) and val > 0 else LEAGUE_XWOBACON


def _event_iso_proxy(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.17
    events = df.get('events', pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
    pa = max(int(len(df)), 1)
    doubles = int(events.eq('double').sum())
    triples = int(events.eq('triple').sum())
    hrs = int(events.eq('home_run').sum())
    return float((doubles + 2 * triples + 3 * hrs) / pa)


@lru_cache(maxsize=16)
def _statcast_window(days: int) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=int(days))
    raw = fetch_statcast_raw(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    return raw if isinstance(raw, pd.DataFrame) else pd.DataFrame()


def fetch_matchups_for_date(date_str: str) -> list[str]:
    resp = safe_get('https://statsapi.mlb.com/api/v1/schedule', {'sportId': 1, 'date': date_str})
    js = resp.json() if resp else {}
    out = []
    for d in js.get('dates', []):
        for g in d.get('games', []):
            try:
                away = ID2ABBR[g['teams']['away']['team']['id']]
                home = ID2ABBR[g['teams']['home']['team']['id']]
                out.append(f'{away} @ {home}')
            except Exception:
                continue
    return out



def _get_boxscore_starting_pitchers(game_pk):
    """
    Historical-game fallback for starter IDs.

    For completed/past games, schedule probablePitchers can be empty.
    The boxscore pitcher list usually has the starter as the first pitcher used.
    """
    if not game_pk:
        return None, None, None, None

    try:
        resp = safe_get(f"https://statsapi.mlb.com/api/v1/game/{int(game_pk)}/boxscore")
        js = resp.json() if resp else {}
    except Exception:
        return None, None, None, None

    def _side(side: str):
        try:
            team = js.get("teams", {}).get(side, {}) or {}
            pitchers = team.get("pitchers", []) or []
            players = team.get("players", {}) or {}

            if not pitchers:
                return None, None

            pid = int(pitchers[0])
            rec = players.get(f"ID{pid}", {}) or {}
            name = (
                rec.get("person", {}).get("fullName")
                or rec.get("person", {}).get("boxscoreName")
                or None
            )
            return name, pid
        except Exception:
            return None, None

    away_name, away_pid = _side("away")
    home_name, home_pid = _side("home")
    return away_name, home_name, away_pid, home_pid


def _get_probables_for_date(date_str: str, away: str, home: str):
    resp = safe_get('https://statsapi.mlb.com/api/v1/schedule', {'sportId': 1, 'date': date_str})
    js = resp.json() if resp else {}
    game_pk = None
    away_name = home_name = None
    away_pid = home_pid = None
    for d in js.get('dates', []):
        for g in d.get('games', []):
            try:
                a = ID2ABBR[g['teams']['away']['team']['id']]
                h = ID2ABBR[g['teams']['home']['team']['id']]
            except Exception:
                continue
            if (a, h) != (away, home):
                continue
            game_pk = g.get('gamePk')
            pp = g.get('probablePitchers', {})
            away_name = strip_pos(pp.get('away', {}).get('fullName', '') or '') or None
            home_name = strip_pos(pp.get('home', {}).get('fullName', '') or '') or None
            away_pid = pp.get('away', {}).get('id')
            home_pid = pp.get('home', {}).get('id')

            # Historical fallback: completed games often no longer expose probablePitchers.
            if game_pk and (not away_pid or not home_pid):
                bx_away_name, bx_home_name, bx_away_pid, bx_home_pid = _get_boxscore_starting_pitchers(game_pk)
                away_name = away_name or bx_away_name
                home_name = home_name or bx_home_name
                away_pid = away_pid or bx_away_pid
                home_pid = home_pid or bx_home_pid

            return away_name, home_name, away_pid, home_pid, game_pk
    return away_name, home_name, away_pid, home_pid, game_pk




TEAM_ABBR_ALIASES = {
    'KC': 'KCR',
    'AZ': 'ARI',
    'CWS': 'CHW',
    'WSH': 'WSN',
    'TB': 'TBR',
    'SF': 'SFG',
    'SD': 'SDP',
}


def _canonical_team_abbr(team_abbr: str) -> str:
    ab = str(team_abbr or '').strip().upper()
    if ab in ABBR2ID:
        return ab
    return TEAM_ABBR_ALIASES.get(ab, ab)


@lru_cache(maxsize=64)
def _team_roster_lookup(team_abbr: str) -> dict[str, int]:
    ab = _canonical_team_abbr(team_abbr)
    tid = ABBR2ID.get(ab)
    out: dict[str, int] = {}
    if not tid:
        return out

    # Try the fetch.py cached roster map first.
    try:
        rm = roster_map(ab) or {}
        for nm, pid in rm.items():
            if pid:
                out[clean_name(nm)] = int(pid)
    except Exception:
        pass

    # Then hit StatsAPI active / 40-man roster endpoints directly for same-day reliability.
    endpoints = [
        f'https://statsapi.mlb.com/api/v1/teams/{tid}/roster/active',
        f'https://statsapi.mlb.com/api/v1/teams/{tid}/roster/40Man',
    ]
    for url in endpoints:
        try:
            resp = safe_get(url)
            js = resp.json() if resp else {}
            for p in js.get('roster', []):
                nm = p.get('person', {}).get('fullName')
                pid = p.get('person', {}).get('id')
                if nm and pid:
                    out[clean_name(nm)] = int(pid)
        except Exception:
            continue

    return out


def _search_people_by_name(hitter_name: str | None) -> int | None:
    nm = strip_pos(hitter_name or '').strip()
    if not nm or nm.upper() == 'TBD':
        return None
    try:
        resp = safe_get('https://statsapi.mlb.com/api/v1/people/search', {'sportId': 1, 'names': nm})
        js = resp.json() if resp else {}
        people = js.get('people', []) or []
        key = clean_name(nm)
        # exact clean-name first
        for p in people:
            full = p.get('fullName') or ''
            pid = p.get('id')
            if pid and clean_name(full) == key:
                return int(pid)
        # otherwise first returned person
        for p in people:
            pid = p.get('id')
            if pid:
                return int(pid)
    except Exception:
        return None
    return None

def _resolve_batter_id(team_abbr: str, hitter_name: str | None):
    hitter_name = strip_pos(hitter_name or '').strip()
    if not hitter_name or hitter_name.upper() == 'TBD':
        return None

    key = clean_name(hitter_name)
    ab = _canonical_team_abbr(team_abbr)

    # 1) Robust team roster lookup from multiple roster endpoints.
    try:
        rm = _team_roster_lookup(ab)
        if key in rm:
            return int(rm[key])
    except Exception:
        pass

    # 2) Normalize suffixes / punctuation and retry against team roster.
    alias = re.sub(r"\b(JR|SR|II|III|IV)\b", '', key).replace('.', ' ').strip()
    alias = re.sub(r'\s+', ' ', alias)
    try:
        rm = _team_roster_lookup(ab)
        if alias in rm:
            return int(rm[alias])
        for nm, pid in rm.items():
            nm2 = re.sub(r"\b(JR|SR|II|III|IV)\b", '', str(nm)).replace('.', ' ').strip()
            nm2 = re.sub(r'\s+', ' ', nm2)
            if nm2 == alias:
                return int(pid)
    except Exception:
        pass

    # 3) StatsAPI people/search fallback.
    pid = _search_people_by_name(hitter_name)
    if pid:
        return int(pid)

    # 4) Legacy pybaseball lookup fallback.
    try:
        pid = name_to_mlbam_id(hitter_name)
        if pid:
            return int(pid)
    except Exception:
        pass

    return None


def _resolve_pitcher_id_hard(team_abbr: str | None, pitcher_name: str | None, pid=None):
    """
    Robust pitcher resolver for lineup metadata.

    This prevents downstream KO/HR context from losing opponent pitcher IDs when
    schedule probables or pybaseball lookup fail.
    """
    try:
        if pid is not None and str(pid).strip().lower() not in {"", "nan", "none"}:
            return int(float(pid))
    except Exception:
        pass

    nm = strip_pos(str(pitcher_name or "")).strip()
    if not nm or nm.upper() == "TBD":
        return None

    key = clean_name(nm)
    team = _canonical_team_abbr(str(team_abbr or "").strip().upper())

    # 1) Pitching staff map from fetch.py
    try:
        staff = fetch.full_pitching_staff(team) or {}
        if key in staff and staff[key]:
            return int(staff[key])
    except Exception:
        pass

    # 2) Team roster map fallback
    try:
        rm = roster_map(team) or {}
        if key in rm and rm[key]:
            return int(rm[key])
    except Exception:
        pass

    # 3) StatsAPI people/search fallback
    try:
        resp = safe_get(
            "https://statsapi.mlb.com/api/v1/people/search",
            {"sportId": 1, "names": nm},
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

    # 4) Legacy pybaseball/name lookup fallback
    try:
        mid = name_to_mlbam_id(nm)
        if mid:
            return int(mid)
    except Exception:
        pass

    return None


def _resolve_lineups(date_str: str, away: str, home: str, override_path: str | None = None):
    matchup_key = f'{away}@{home}'
    override = get_override_for_game(date_str, matchup_key, override_path)

    lineup_source = 'projected'
    ump_feats = {'ump_k9': 1.0, 'ump_bb9': 1.0}
    framing_feats = {'away_frame': 0.0, 'home_frame': 0.0}
    away_cid = home_cid = None

    if date_str == datetime.today().strftime('%Y-%m-%d'):
        try:
            a_name, h_name, a_line, h_line, a_pid, h_pid, away_cid, home_cid, ump_feats, framing_feats = fetch_lineup_and_starters(away, home)
            _, _, _, _, game_pk = _get_probables_for_date(date_str, away, home)
            lineup_source = 'confirmed' if len(a_line) >= 9 and len(h_line) >= 9 else 'hybrid'
        except Exception:
            a_name, h_name, a_pid, h_pid, game_pk = _get_probables_for_date(date_str, away, home)
            a_line = fetch_unofficial_lineup(away)
            h_line = fetch_unofficial_lineup(home)
            lineup_source = 'projected'
    else:
        a_name, h_name, a_pid, h_pid, game_pk = _get_probables_for_date(date_str, away, home)
        a_line = fetch_unofficial_lineup(away)
        h_line = fetch_unofficial_lineup(home)
        lineup_source = 'projected'

    if override:
        a_line = override.get('away_lineup', a_line)
        h_line = override.get('home_lineup', h_line)
        a_name = override.get('away_pitcher_name', a_name)
        h_name = override.get('home_pitcher_name', h_name)
        a_pid = override.get('away_pitcher_id', a_pid)
        h_pid = override.get('home_pitcher_id', h_pid)
        game_pk = override.get('game_pk', game_pk)
        lineup_source = str(override.get('lineup_source', 'hybrid' if lineup_source == 'confirmed' else lineup_source))

    a_pid = _resolve_pitcher_id_hard(away, a_name, a_pid)
    h_pid = _resolve_pitcher_id_hard(home, h_name, h_pid)

    away_opp_hand = pitcher_throws(int(h_pid)) if h_pid else 'R'
    home_opp_hand = pitcher_throws(int(a_pid)) if a_pid else 'R'

    a_line = _sanitize_lineup_names(away, a_line)
    h_line = _sanitize_lineup_names(home, h_line)

    away_before = list(a_line or [])
    home_before = list(h_line or [])
    if lineup_needs_projection(a_line):
        a_line = fill_tbd_lineup(a_line, away, pitcher_hand=away_opp_hand, n=9)
    if lineup_needs_projection(h_line):
        h_line = fill_tbd_lineup(h_line, home, pitcher_hand=home_opp_hand, n=9)

    if lineup_source == 'confirmed' and (lineup_needs_projection(away_before) or lineup_needs_projection(home_before)):
        lineup_source = 'hybrid_projected'
    elif lineup_source in {'projected', 'hybrid'}:
        away_changed = [strip_pos(str(x or '')).strip() for x in away_before[:9]] != [strip_pos(str(x or '')).strip() for x in a_line[:9]]
        home_changed = [strip_pos(str(x or '')).strip() for x in home_before[:9]] != [strip_pos(str(x or '')).strip() for x in h_line[:9]]
        if away_changed or home_changed:
            lineup_source = 'projected_roster' if lineup_source == 'projected' else 'hybrid_projected'

    lw = lineup_source_weight(lineup_source)
    return {
        'away_pitcher_name': a_name,
        'home_pitcher_name': h_name,
        'away_lineup': a_line,
        'home_lineup': h_line,
        'away_pitcher_id': a_pid,
        'home_pitcher_id': h_pid,
        'game_pk': game_pk,
        'lineup_source': lineup_source,
        'lineup_weight': lw,
        'lineup_certainty': lineup_certainty_score(lineup_source, len(a_line) >= 9 and len(h_line) >= 9),
        'ump_feats': ump_feats,
        'framing_feats': framing_feats,
        'away_catcher_id': away_cid,
        'home_catcher_id': home_cid,
    }


def _hand_filtered_batter_events(batter_pid: int, pitcher_hand: str, days: int) -> pd.DataFrame:
    raw = _statcast_window(days)
    if raw is None or raw.empty or 'batter' not in raw.columns:
        return pd.DataFrame()
    df = raw[raw['batter'] == batter_pid].copy()
    if df.empty:
        return df
    if pitcher_hand and 'p_throws' in df.columns:
        sub = df[df['p_throws'].astype(str).str.upper().str.startswith(pitcher_hand.upper()[:1])]
        if len(sub) >= 12:
            df = sub
    return df


def _single_window_rates(df: pd.DataFrame) -> dict[str, float]:
    if df.empty or 'events' not in df.columns:
        return {'pa': 0.0, 'hit_pa': LEAGUE_HIT_PA, '1B_share': LEAGUE_1B_SHARE, '2B_share': LEAGUE_2B_SHARE, '3B_share': LEAGUE_3B_SHARE, 'HR_share': LEAGUE_HR_SHARE}
    events = df['events'].astype(str)
    pa = max(int(events.notna().sum()), len(df))
    n1 = int((events == 'single').sum())
    n2 = int((events == 'double').sum())
    n3 = int((events == 'triple').sum())
    nhr = int((events == 'home_run').sum())
    hits = n1 + n2 + n3 + nhr
    prior_pa = 60
    hit_pa = (hits + LEAGUE_HIT_PA * prior_pa) / (pa + prior_pa)
    denom = max(hits, 1)
    return {
        'pa': float(pa),
        'hit_pa': float(hit_pa),
        '1B_share': float((n1 + LEAGUE_1B_SHARE * 12) / (denom + 12)),
        '2B_share': float((n2 + LEAGUE_2B_SHARE * 12) / (denom + 12)),
        '3B_share': float((n3 + LEAGUE_3B_SHARE * 12) / (denom + 12)),
        'HR_share': float((nhr + LEAGUE_HR_SHARE * 12) / (denom + 12)),
    }


def _career_baseline_rates(player_name: str) -> dict[str, float]:
    nm = clean_name(player_name)
    if isinstance(BATTER_CAREER, pd.DataFrame) and not BATTER_CAREER.empty and 'Name_norm' in BATTER_CAREER.columns:
        rec = BATTER_CAREER[BATTER_CAREER['Name_norm'] == nm]
        if not rec.empty:
            row = rec.iloc[0]
            return {
                'PA': float(row.get('PA', 0.0) or 0.0),
                'hit_pa': float(row.get('H_rate', LEAGUE_HIT_PA)),
                '1B_share': float(row.get('1B_rate', LEAGUE_1B_SHARE)) / max(float(row.get('H_rate', LEAGUE_HIT_PA)), 1e-6),
                '2B_share': float(row.get('2B_rate', LEAGUE_2B_SHARE)) / max(float(row.get('H_rate', LEAGUE_HIT_PA)), 1e-6),
                '3B_share': float(row.get('3B_rate', LEAGUE_3B_SHARE)) / max(float(row.get('H_rate', LEAGUE_HIT_PA)), 1e-6),
                'HR_share': float(row.get('HR_rate', LEAGUE_HR_SHARE)) / max(float(row.get('H_rate', LEAGUE_HIT_PA)), 1e-6),
            }
    return {'PA': 0.0, 'hit_pa': LEAGUE_HIT_PA, '1B_share': LEAGUE_1B_SHARE, '2B_share': LEAGUE_2B_SHARE, '3B_share': LEAGUE_3B_SHARE, 'HR_share': LEAGUE_HR_SHARE}


def _weighted_recent_rates(batter_pid: int, player_name: str, pitcher_hand: str) -> dict[str, float]:
    windows = [(7, 0.25), (14, 0.25), (30, 0.25), (365, 0.25)]
    agg = {'hit_pa': 0.0, '1B_share': 0.0, '2B_share': 0.0, '3B_share': 0.0, 'HR_share': 0.0}
    total_w = 0.0
    details = {}
    for days, w in windows:
        df = _hand_filtered_batter_events(int(batter_pid), pitcher_hand, days)
        rates = _single_window_rates(df)
        sample_adj = min(1.0, rates['pa'] / (20.0 if days <= 14 else 40.0 if days <= 30 else 80.0))
        ew = w * sample_adj
        for k in agg:
            agg[k] += ew * rates[k]
        total_w += ew
        details[f'pa_{days}d'] = int(rates['pa'])
        details[f'hit_pa_{days}d'] = float(rates['hit_pa'])
    career = _career_baseline_rates(player_name)
    career_w = max(0.15, 1.0 - total_w)
    for k in agg:
        agg[k] += career_w * career[k]
    total_w += career_w
    if total_w <= 0:
        out = {'hit_pa': LEAGUE_HIT_PA, '1B_share': LEAGUE_1B_SHARE, '2B_share': LEAGUE_2B_SHARE, '3B_share': LEAGUE_3B_SHARE, 'HR_share': LEAGUE_HR_SHARE}
    else:
        out = {k: float(v / total_w) for k, v in agg.items()}

    recent_pa = float(details.get('pa_30d', 0) + 0.35 * details.get('pa_365d', 0))
    career_pa = float(career.get('PA', 0.0) or 0.0)
    base_n = recent_pa + 0.15 * career_pa
    out['hit_pa'] = _partial_pool(out['hit_pa'], base_n, career['hit_pa'], prior_n=120.0)
    out['1B_share'] = _partial_pool(out['1B_share'], base_n, career['1B_share'], prior_n=90.0)
    out['2B_share'] = _partial_pool(out['2B_share'], base_n, career['2B_share'], prior_n=90.0)
    out['3B_share'] = _partial_pool(out['3B_share'], base_n, career['3B_share'], prior_n=120.0)
    out['HR_share'] = _partial_pool(out['HR_share'], base_n, career['HR_share'], prior_n=110.0)
    shares = _normalize_shares({'1B': out['1B_share'], '2B': out['2B_share'], '3B': out['3B_share'], 'HR': out['HR_share']})
    out['1B_share'], out['2B_share'], out['3B_share'], out['HR_share'] = shares['1B'], shares['2B'], shares['3B'], shares['HR']
    out.update(details)
    return out




def _batter_power_profile(batter_pid: int, pitcher_hand: str) -> dict[str, float]:
    feats = get_statcast_batter_features(int(batter_pid)) or {}
    raw365 = _hand_filtered_batter_events(int(batter_pid), pitcher_hand, 365)
    raw30 = _hand_filtered_batter_events(int(batter_pid), pitcher_hand, 30)
    raw14 = _hand_filtered_batter_events(int(batter_pid), pitcher_hand, 14)

    def _fb_pct(df: pd.DataFrame) -> float:
        if df is None or df.empty or 'bb_type' not in df.columns:
            return LEAGUE_FLYBALL_PCT
        return _safe_frac(df['bb_type'].astype(str).str.lower().eq('fly_ball'))

    def _pulled_air(df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return LEAGUE_PULL_AIR_PCT
        la = pd.to_numeric(df.get('launch_angle', pd.Series(index=df.index, dtype=float)), errors='coerce')
        fb = df.get('bb_type', pd.Series(index=df.index, dtype=object)).astype(str).str.lower().eq('fly_ball')
        return _safe_frac(fb & la.between(-20, 20))

    ev = float(feats.get('ev_mean', 89.0) or 89.0)
    hh = float(feats.get('hard_hit_pct', LEAGUE_HARD_HIT_PCT) or LEAGUE_HARD_HIT_PCT)
    hh14 = float(feats.get('hardhit_14d_pct', hh) or hh)
    br = float(feats.get('barrel_pct', feats.get('barrel_14d_pct', LEAGUE_BARREL_PCT)) or LEAGUE_BARREL_PCT)
    br14 = float(feats.get('barrel_14d_pct', br) or br)
    la = float(feats.get('la_mean', 12.0) or 12.0)
    sweet = float(feats.get('sweet_spot_frac', 0.06) or 0.06)
    hr_fb = float(feats.get('HR_FB_rate', LEAGUE_HR_SHARE / max(LEAGUE_FLYBALL_PCT, 1e-6)) or 0.0)
    fb365 = _fb_pct(raw365)
    fb30 = _fb_pct(raw30)
    pa365 = float(len(raw365))
    xw365 = _estimate_xwobacon(raw365)
    xw14 = _estimate_xwobacon(raw14)
    iso365 = _event_iso_proxy(raw365)
    iso30 = _event_iso_proxy(raw30)
    pulled_air = _pulled_air(raw365)

    contact_mult = float(np.clip(
        1.0 + 0.30 * (hh - LEAGUE_HARD_HIT_PCT) + 0.18 * (xw365 - LEAGUE_XWOBACON),
        0.90, 1.12,
    ))
    xbh_mult = float(np.clip(
        1.0 + 0.55 * (iso365 - 0.17) + 0.22 * (fb365 - LEAGUE_FLYBALL_PCT) + 0.25 * (sweet - 0.06),
        0.86, 1.22,
    ))
    raw_hr_contact_mult = (
        1.0 + 0.90 * (br - LEAGUE_BARREL_PCT) + 0.30 * (hh - LEAGUE_HARD_HIT_PCT) + 0.35 * (fb365 - LEAGUE_FLYBALL_PCT)
        + 0.35 * (pulled_air - LEAGUE_PULL_AIR_PCT) + 0.20 * (sweet - 0.06) + 0.02 * (ev - 89.0)
    )
    hr_contact_mult = float(np.clip(1.0 + 1.45 * (raw_hr_contact_mult - 1.0), 0.45, 1.70))
    form_hr_mult = float(np.clip(
        1.0 + 1.18 * (br14 - br) + 0.48 * (hh14 - hh) + 0.68 * (xw14 - xw365) + 0.58 * (iso30 - iso365)
        + 0.32 * (fb30 - fb365),
        0.65, 1.45,
    ))

    return {
        'ev_mean': ev,
        'la_mean': la,
        'hard_hit_pct': hh,
        'hard_hit_14d_pct': hh14,
        'barrel_pct': br,
        'barrel_14d_pct': br14,
        'sweet_spot_frac': sweet,
        'flyball_pct': fb365,
        'flyball_30d_pct': fb30,
        'pulled_air_pct': pulled_air,
        'xwobacon': xw365,
        'xwobacon_14d': xw14,
        'xiso_est': iso365,
        'xiso_30d_est': iso30,
        'hr_fb_rate': hr_fb,
        'contact_mult': contact_mult,
        'xbh_mult': xbh_mult,
        'raw_hr_contact_mult': float(raw_hr_contact_mult),
        'hr_contact_mult': hr_contact_mult,
        'form_hr_mult': form_hr_mult,
        'pa_365': pa365,
    }


def _pitcher_batted_ball_window(pitcher_pid: int, batter_stand: str, days: int) -> pd.DataFrame:
    raw = _statcast_window(days)
    if raw is None or raw.empty or 'pitcher' not in raw.columns:
        return pd.DataFrame()
    df = raw[raw['pitcher'] == pitcher_pid].copy()
    if df.empty:
        return df
    if batter_stand and 'stand' in df.columns:
        sub = df[df['stand'].astype(str).str.upper().str.startswith(str(batter_stand).upper()[:1])]
        if len(sub) >= 15:
            df = sub
    return df


def _pitcher_hr_danger_profile(pitcher_pid: int | None, batter_stand: str) -> dict[str, float]:
    if not pitcher_pid:
        return {
            'hard_hit_allowed_pct': LEAGUE_HARD_HIT_PCT,
            'barrel_allowed_pct': LEAGUE_BARREL_PCT,
            'flyball_allowed_pct': LEAGUE_FLYBALL_PCT,
            'hr_pa_allowed': LEAGUE_HIT_PA * LEAGUE_HR_SHARE,
            'xwobacon_allowed': LEAGUE_XWOBACON,
            'ev_allowed_mean': 89.0,
            'raw_danger_mult': 1.0,
            'danger_mult': 1.0,
            'hit_suppress_mult': 1.0,
            'raw_recent_danger_mult': 1.0,
            'recent_danger_mult': 1.0,
        }
    pf = get_statcast_pitcher_features(int(pitcher_pid)) or {}
    df365 = _pitcher_batted_ball_window(int(pitcher_pid), batter_stand, 365)
    df30 = _pitcher_batted_ball_window(int(pitcher_pid), batter_stand, 30)

    def _barrel_allowed(df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return LEAGUE_BARREL_PCT
        if 'barrel' in df.columns:
            return _safe_frac(pd.to_numeric(df['barrel'], errors='coerce').fillna(0) == 1)
        ls = pd.to_numeric(df.get('launch_speed', pd.Series(index=df.index, dtype=float)), errors='coerce')
        la = pd.to_numeric(df.get('launch_angle', pd.Series(index=df.index, dtype=float)), errors='coerce')
        return _safe_frac((ls >= 98) & la.between(26, 30))

    def _hard_hit(df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return LEAGUE_HARD_HIT_PCT
        ls = pd.to_numeric(df.get('launch_speed', pd.Series(index=df.index, dtype=float)), errors='coerce')
        return _safe_frac(ls >= 95)

    def _fb(df: pd.DataFrame) -> float:
        if df is None or df.empty or 'bb_type' not in df.columns:
            return LEAGUE_FLYBALL_PCT
        return _safe_frac(df['bb_type'].astype(str).str.lower().eq('fly_ball'))

    def _hr_pa(df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return LEAGUE_HIT_PA * LEAGUE_HR_SHARE
        events = df.get('events', pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
        return float(events.eq('home_run').mean())

    hh = _hard_hit(df365)
    br = _barrel_allowed(df365)
    fb = _fb(df365)
    hrpa = _hr_pa(df365)
    xw = _estimate_xwobacon(df365)
    ev_allowed = _safe_mean(df365.get('launch_speed', pd.Series(index=df365.index, dtype=float))) if not df365.empty else 89.0
    hh30 = _hard_hit(df30)
    br30 = _barrel_allowed(df30)
    hr30 = _hr_pa(df30)
    xw30 = _estimate_xwobacon(df30)
    whiff2s = float(pf.get('whiff_2S', 0.11) or 0.11)
    hit_rate_vs = float(pf.get(f"vs_{'L' if str(batter_stand).upper().startswith('L') else 'R'}_BA", LEAGUE_HIT_PA) or LEAGUE_HIT_PA)

    raw_danger_mult = (
        1.0 + 0.95 * (br - LEAGUE_BARREL_PCT) + 0.35 * (hh - LEAGUE_HARD_HIT_PCT) + 0.28 * (fb - LEAGUE_FLYBALL_PCT)
        + 0.55 * (xw - LEAGUE_XWOBACON) + 2.20 * (hrpa - LEAGUE_HIT_PA * LEAGUE_HR_SHARE) - 0.18 * (whiff2s - 0.11)
    )
    danger_mult = float(np.clip(1.0 + 1.35 * (raw_danger_mult - 1.0), 0.45, 1.55))
    hit_suppress_mult = float(np.clip(
        1.0 + 0.30 * (hit_rate_vs - LEAGUE_HIT_PA) + 0.20 * (hh - LEAGUE_HARD_HIT_PCT) + 0.18 * (xw - LEAGUE_XWOBACON)
        - 0.12 * (whiff2s - 0.11),
        0.90, 1.12,
    ))
    raw_recent_danger_mult = (
        1.0 + 0.75 * (br30 - br) + 0.30 * (hh30 - hh) + 0.40 * (xw30 - xw) + 1.80 * (hr30 - hrpa)
    )
    recent_danger_mult = float(np.clip(1.0 + 1.22 * (raw_recent_danger_mult - 1.0), 0.55, 1.38))

    return {
        'hard_hit_allowed_pct': hh,
        'barrel_allowed_pct': br,
        'flyball_allowed_pct': fb,
        'hr_pa_allowed': hrpa,
        'xwobacon_allowed': xw,
        'ev_allowed_mean': ev_allowed,
        'raw_danger_mult': float(raw_danger_mult),
        'danger_mult': danger_mult,
        'hit_suppress_mult': hit_suppress_mult,
        'raw_recent_danger_mult': float(raw_recent_danger_mult),
        'recent_danger_mult': recent_danger_mult,
    }


def _park_hr_factor(home_team_abbr: str, month: int, batter_stand: str) -> float:
    mpf = fetch_monthly_park_factors(YEAR)
    rec = mpf.get(home_team_abbr, {}).get(month, {}) if isinstance(mpf, dict) else {}
    raw = rec.get('HR', rec.get('hr', 100.0))
    try:
        mult = float(raw) / 100.0
    except Exception:
        mult = 1.0
    # mild handedness shaping so extreme parks matter a bit more to same-side power archetypes
    if str(batter_stand).upper().startswith('L'):
        mult = 1.0 + 1.05 * (mult - 1.0)
    else:
        mult = 1.0 + 0.95 * (mult - 1.0)
    return float(np.clip(mult, 0.88, 1.18))

def _pitcher_allow_rates(pitcher_pid: int | None, batter_stand: str) -> dict[str, float]:
    if not pitcher_pid:
        return {'hit_pa': LEAGUE_HIT_PA, 'hr_pa': LEAGUE_HIT_PA * LEAGUE_HR_SHARE, 'whiff_2s': 0.11}
    pf = get_statcast_pitcher_features(int(pitcher_pid)) or {}
    side = 'L' if str(batter_stand).upper().startswith('L') else 'R'
    raw_hit_pa = float(pf.get(f'vs_{side}_BA', LEAGUE_HIT_PA))
    raw_hr_pa = float(pf.get(f'vs_{side}_HR_rate', LEAGUE_HIT_PA * LEAGUE_HR_SHARE))
    whiff_2s = float(pf.get('whiff_2S', 0.11) or 0.11)
    df365 = _pitcher_batted_ball_window(int(pitcher_pid), batter_stand, 365)
    pa_n = float(len(df365))
    hit_pa = _partial_pool(raw_hit_pa, pa_n, LEAGUE_HIT_PA, prior_n=140.0)
    hr_pa = _partial_pool(raw_hr_pa, pa_n, LEAGUE_HIT_PA * LEAGUE_HR_SHARE, prior_n=180.0)
    return {'hit_pa': hit_pa, 'hr_pa': hr_pa, 'whiff_2s': whiff_2s}


def _safe_game_hr_pa(game_hr_prob: float, exp_pa: float) -> float:
    exp_pa = max(float(exp_pa), 1.0)
    game_hr_prob = float(np.clip(game_hr_prob, 0.0, 0.95))
    return float(1.0 - (1.0 - game_hr_prob) ** (1.0 / exp_pa))


def _park_ba_factor(home_team_abbr: str, month: int) -> float:
    mpf = fetch_monthly_park_factors(YEAR)
    return float(mpf.get(home_team_abbr, {}).get(month, {}).get('BA', 100.0)) / 100.0


def _normalize_shares(shares: dict[str, float]) -> dict[str, float]:
    s = sum(max(v, 0.0) for v in shares.values())
    if s <= 0:
        return {'1B': LEAGUE_1B_SHARE, '2B': LEAGUE_2B_SHARE, '3B': LEAGUE_3B_SHARE, 'HR': LEAGUE_HR_SHARE}
    return {k: max(v, 0.0) / s for k, v in shares.items()}


def _binom_at_least_one(p: float, n: float) -> float:
    n_int = max(int(round(n)), 1)
    p = float(np.clip(p, 0.0, 0.999))
    return float(1.0 - (1.0 - p) ** n_int)


def _contact_quality_multiplier(batter_pid: int) -> dict[str, float]:
    feats = get_statcast_batter_features(int(batter_pid)) or {}
    ev = float(feats.get('ev_mean', 89.0) or 89.0)
    hh = float(feats.get('hard_hit_pct', 0.36) or 0.36)
    br = float(feats.get('barrel_pct', feats.get('barrel_14d_pct', 0.08)) or 0.08)
    xb_mult = float(np.clip(1.0 + 0.012 * (ev - 89.0) + 0.45 * (hh - 0.36) + 0.65 * (br - 0.08), 0.88, 1.18))
    hit_mult = float(np.clip(1.0 + 0.25 * (hh - 0.36), 0.95, 1.06))
    hr_mult = float(np.clip(1.0 + 0.70 * (br - 0.08) + 0.02 * (ev - 89.0), 0.84, 1.24))
    return {'hit_mult': hit_mult, 'xb_mult': xb_mult, 'hr_mult': hr_mult}


def _bullpen_followthrough(fielding_team: str, starter_pid: int | None, starter_hr_pa: float, starter_share: float) -> dict[str, float]:
    try:
        bp_hr_pa, bp_sample = team_bullpen_hrpa(fielding_team, exclude_pid=int(starter_pid) if starter_pid else None, days=180)
    except Exception:
        bp_hr_pa, bp_sample = (starter_hr_pa, 0)
    if not np.isfinite(bp_hr_pa) or bp_hr_pa <= 0:
        bp_hr_pa = starter_hr_pa
    starter_share = float(np.clip(starter_share, 0.45, 0.85))
    blended_hr_pa = starter_share * float(starter_hr_pa) + (1.0 - starter_share) * float(bp_hr_pa)
    hr_mult = float(np.clip(blended_hr_pa / max(float(starter_hr_pa), 1e-6), 0.88, 1.16))
    hit_mult = float(np.clip(1.0 + 0.30 * (hr_mult - 1.0), 0.96, 1.05))
    return {'starter_share': starter_share, 'bullpen_hr_mult': hr_mult, 'bullpen_hit_mult': hit_mult, 'bullpen_sample': int(bp_sample)}


def predict_matchup_batters(date_str: str, matchup: str, override_path: str | None = None) -> pd.DataFrame:
    away, home = [x.strip().upper() for x in matchup.replace(' @ ', '@').split('@')]
    meta = _resolve_lineups(date_str, away, home, override_path)
    month = pd.to_datetime(date_str).month
    defense_df = _normalize_date_cols(load_team_defense_proxies())

    rows: list[dict] = []
    contexts = [
        (away, home, meta['away_lineup'], meta['home_pitcher_id'], meta['home_pitcher_name'], False),
        (home, away, meta['home_lineup'], meta['away_pitcher_id'], meta['away_pitcher_name'], True),
    ]

    lineup_weight = float(meta['lineup_weight'])
    lineup_certainty = float(meta.get('lineup_certainty', lineup_weight))
    away_frame = float(meta.get('framing_feats', {}).get('away_frame', 0.0) or 0.0)
    home_frame = float(meta.get('framing_feats', {}).get('home_frame', 0.0) or 0.0)
    ump_k_mult = float(meta.get('ump_feats', {}).get('ump_k9', 1.0) or 1.0)

    for batting_team, fielding_team, lineup, opp_pid, opp_name, is_home in contexts:
        active_keys = _active_roster_keyset(batting_team)
        opp_hand = pitcher_throws(int(opp_pid)) if opp_pid else 'R'
        park_mult = _park_ba_factor(home, month)
        framing_runs = home_frame if fielding_team == home else away_frame
        framing_hit_mult = float(np.clip(1.0 - 0.0012 * framing_runs, 0.97, 1.03))
        ump_hit_mult = float(np.clip(1.0 - 0.012 * (ump_k_mult - 1.0), 0.97, 1.03))

        for slot, hitter_name in enumerate(lineup[:9], start=1):
            hitter_name = strip_pos(hitter_name or '')
            role_hint = lookup_lineup_role_features(
                team=batting_team,
                player_name=hitter_name,
                player_id=None,
                current_slot=slot,
                as_of_date=date_str,
                lineup_source=meta.get('lineup_source', 'projected'),
                lineup_certainty=lineup_certainty,
            )
            base_row = {
                'date': date_str,
                'matchup': f'{away}@{home}',
                'team': batting_team,
                'batting_order': slot,
                'player_name': hitter_name,
                'opp_pitcher': opp_name or 'TBD',
                'opp_pitcher_id': int(opp_pid) if opp_pid else None,
                'opp_pitcher_hand': opp_hand,
                'lineup_source': meta['lineup_source'],
                'lineup_weight': round(lineup_weight, 3),
                'lineup_certainty': round(lineup_certainty, 3),
                'usual_batting_slot': role_hint.get('usual_batting_slot'),
                'usual_role_bucket': role_hint.get('usual_role_bucket'),
                'current_role_bucket': role_hint.get('current_role_bucket'),
                'slot_delta': role_hint.get('slot_delta'),
                'slot_abs_delta': role_hint.get('slot_abs_delta'),
                'role_changed_flag': role_hint.get('role_changed_flag'),
                'confirmed_lineup_role_change_flag': role_hint.get('confirmed_lineup_role_change_flag'),
                'slot_volatility': role_hint.get('slot_volatility'),
                'lineup_role_hist_n': role_hint.get('lineup_role_hist_n'),
                'lineup_role_hist_weight': role_hint.get('lineup_role_hist_weight'),
                'lineup_role_confidence': role_hint.get('lineup_role_confidence'),
                'lineup_role_pa_mult': role_hint.get('lineup_role_pa_mult'),
                'lineup_role_hit_mult': role_hint.get('lineup_role_hit_mult'),
                'lineup_role_hr_mult': role_hint.get('lineup_role_hr_mult'),
                'exp_pa': round(batting_order_pa(slot, is_home, lineup_weight) * float(role_hint.get('lineup_role_pa_mult', 1.0) or 1.0), 2),
            }

            try:
                batter_pid, resolved_name = _strict_resolve_batter(batting_team, hitter_name)
                if resolved_name:
                    base_row['player_name'] = resolved_name
                hitter_key = clean_name(strip_pos(base_row['player_name']))
                active_flag = int(bool(active_keys) and hitter_key in active_keys) if active_keys else 1
                roster_note = '' if active_flag else 'off_active_roster'
                if not batter_pid:
                    rows.append({
                        **base_row,
                        'player_id': None,
                        'bats': None,
                        'active_roster_flag': 0,
                        'strict_confirmed_lineup_flag': 0,
                        'eligible_for_top_boards': 0,
                        'display_terminal_flag': 0,
                        'top_board_block_reason': 'unresolved_team_roster_player',
                        'roster_integrity_note': 'unresolved_team_roster_player',
                        'status': 'unresolved_team_roster_player',
                    })
                    continue
                if not active_flag:
                    rows.append({
                        **base_row,
                        'player_id': int(batter_pid),
                        'bats': None,
                        'active_roster_flag': 0,
                        'strict_confirmed_lineup_flag': 0,
                        'eligible_for_top_boards': 0,
                        'display_terminal_flag': 0,
                        'top_board_block_reason': 'off_active_roster',
                        'roster_integrity_note': roster_note,
                        'status': 'off_active_roster',
                    })
                    continue

                stand = batter_bats(int(batter_pid))
                role_feats = lookup_lineup_role_features(
                    team=batting_team,
                    player_name=base_row['player_name'],
                    player_id=int(batter_pid),
                    current_slot=slot,
                    as_of_date=date_str,
                    lineup_source=meta.get('lineup_source', 'projected'),
                    lineup_certainty=lineup_certainty,
                )
                park_hr_mult = _park_hr_factor(home, month, stand)
                recent = _weighted_recent_rates(int(batter_pid), hitter_name, opp_hand)
                pitch_allow = _pitcher_allow_rates(int(opp_pid) if opp_pid else None, stand)
                cq = _contact_quality_multiplier(int(batter_pid))
                power = _batter_power_profile(int(batter_pid), opp_hand)
                pitcher_danger = _pitcher_hr_danger_profile(int(opp_pid) if opp_pid else None, stand)

                zone_hr_ctx = {}
                zone_hr_overall_mult = 1.0
                if callable(summarize_zone_hr_context) and batter_pid and opp_pid:
                    try:
                        zone_hr_ctx = summarize_zone_hr_context(
                            int(batter_pid),
                            int(opp_pid),
                            batter_stand=stand,
                            pitcher_hand=opp_hand,
                            as_of_date=date_str,
                            days=365,
                        )
                        zone_hr_overall_mult = float(zone_hr_ctx.get('zone_hr_overall_mult', 1.0) or 1.0)
                    except Exception:
                        zone_hr_ctx = {}
                        zone_hr_overall_mult = 1.0
                exp_pa = batting_order_pa(slot, is_home, lineup_weight) * float(role_feats.get('lineup_role_pa_mult', 1.0) or 1.0)
                starter_share = float(np.clip(0.78 - 0.03 * max(slot - 1, 0), 0.50, 0.78))

                base_hit_pa = (
                    0.45 * recent['hit_pa']
                    + 0.25 * pitch_allow['hit_pa']
                    + 0.15 * _career_baseline_rates(hitter_name)['hit_pa']
                    + 0.15 * min(max(power['xwobacon'] * 0.42, 0.10), 0.32)
                )
                hit_pa = float(np.clip(
                    base_hit_pa * park_mult * framing_hit_mult * ump_hit_mult
                    * cq['hit_mult'] * power['contact_mult'] * pitcher_danger['hit_suppress_mult'],
                    0.08, 0.45
                ))

                try:
                    game_hr_prob = batter_game_hr_prob(
                        batter_pid=int(batter_pid),
                        batter_stand=stand,
                        batting_team=batting_team,
                        opp_team=fielding_team,
                        opp_starter_pid=int(opp_pid) if opp_pid else 0,
                        exp_pa=exp_pa,
                        game_pk=meta['game_pk'],
                    )
                except Exception:
                    fallback_hr_pa = 0.55 * recent['HR_share'] * hit_pa + 0.45 * pitch_allow['hr_pa']
                    game_hr_prob = _binom_at_least_one(fallback_hr_pa, exp_pa)

                game_hr_prob = _calibrate_hr_probability(
                    game_hr_prob, hitter_name, recent, exp_pa, lineup_certainty, power, pitcher_danger
                )
                hr_pa = _safe_game_hr_pa(game_hr_prob, exp_pa)
                bp = _bullpen_followthrough(fielding_team, int(opp_pid) if opp_pid else None, hr_pa, starter_share)
                hr_pa = float(np.clip(
                    hr_pa * bp['bullpen_hr_mult'] * cq['hr_mult'] * power['hr_contact_mult']
                    * power['form_hr_mult'] * pitcher_danger['danger_mult']
                    * pitcher_danger['recent_danger_mult'] * park_hr_mult * zone_hr_overall_mult,
                    0.0, 0.26
                ))
                hit_pa = float(np.clip(hit_pa * bp['bullpen_hit_mult'], 0.06, 0.48))

                non_hr_hit_pa = max(hit_pa - hr_pa, 0.0)
                shares = _normalize_shares({
                    '1B': recent['1B_share'],
                    '2B': recent['2B_share'],
                    '3B': recent['3B_share'],
                    'HR': recent['HR_share'],
                })
                non_hr_total_share = max(shares['1B'] + shares['2B'] + shares['3B'], 1e-6)
                p1b = non_hr_hit_pa * (shares['1B'] / non_hr_total_share)
                p2b = non_hr_hit_pa * (shares['2B'] / non_hr_total_share) * cq['xb_mult'] * power['xbh_mult']
                p3b = non_hr_hit_pa * (shares['3B'] / non_hr_total_share) * cq['xb_mult'] * power['xbh_mult']
                phr = hr_pa

                ctx = apply_context_stack(
                    hit_pa, p1b, p2b, p3b, phr,
                    batter_pid=int(batter_pid),
                    pitcher_pid=int(opp_pid) if opp_pid else None,
                    pitcher_hand=opp_hand,
                    game_pk=meta['game_pk'],
                    fielding_team=fielding_team,
                    defense_df=defense_df,
                )
                p1b = float(ctx['1b_pa'])
                p2b = float(ctx['2b_pa'])
                p3b = float(ctx['3b_pa'])
                phr = float(ctx['hr_pa'])
                hit_mult_role = float(role_feats.get('lineup_role_hit_mult', 1.0) or 1.0)
                hr_mult_role = float(role_feats.get('lineup_role_hr_mult', 1.0) or 1.0)
                p1b = float(np.clip(p1b * hit_mult_role, 0.0, 0.35))
                p2b = float(np.clip(p2b * (0.70 * hit_mult_role + 0.30 * hr_mult_role), 0.0, 0.18))
                p3b = float(np.clip(p3b * (0.65 * hit_mult_role + 0.35 * hr_mult_role), 0.0, 0.06))
                phr = float(np.clip(phr * hr_mult_role, 0.0, 0.26))
                hit_pa = float(np.clip(p1b + p2b + p3b + phr, 0.06, 0.48))
                game_hr_prob = _binom_at_least_one(phr, exp_pa)
                game_hr_prob = _calibrate_hr_probability(
                    game_hr_prob, hitter_name, recent, exp_pa, lineup_certainty, power, pitcher_danger
                )
                phr = min(float(phr), _safe_game_hr_pa(game_hr_prob, exp_pa))
                hit_game_prob = _binom_at_least_one(min(hit_pa, 0.95), exp_pa)
                eligible_for_top_boards, eligibility_reason = _row_eligibility_flag(base_row, active_flag, batter_pid)
                strict_confirmed_lineup_flag = int(
                    str(base_row.get('lineup_source', '') or '').strip().lower() == 'confirmed'
                    and float(base_row.get('lineup_certainty', 0.0) or 0.0) >= 0.95
                    and base_row.get('opp_pitcher_id') is not None
                )

                rows.append({
                    **base_row,
                    'player_id': int(batter_pid),
                    'bats': stand,
                    'usual_batting_slot': role_feats.get('usual_batting_slot'),
                    'usual_role_bucket': role_feats.get('usual_role_bucket'),
                    'current_role_bucket': role_feats.get('current_role_bucket'),
                    'slot_delta': role_feats.get('slot_delta'),
                    'slot_abs_delta': role_feats.get('slot_abs_delta'),
                    'role_changed_flag': role_feats.get('role_changed_flag'),
                    'confirmed_lineup_role_change_flag': role_feats.get('confirmed_lineup_role_change_flag'),
                    'slot_volatility': role_feats.get('slot_volatility'),
                    'lineup_role_hist_n': role_feats.get('lineup_role_hist_n'),
                    'lineup_role_hist_weight': role_feats.get('lineup_role_hist_weight'),
                    'lineup_role_confidence': role_feats.get('lineup_role_confidence'),
                    'lineup_role_pa_mult': role_feats.get('lineup_role_pa_mult'),
                    'lineup_role_hit_mult': role_feats.get('lineup_role_hit_mult'),
                    'lineup_role_hr_mult': role_feats.get('lineup_role_hr_mult'),
                    'exp_pa': round(exp_pa, 2),
                    'starter_share': round(bp['starter_share'], 3),
                    'bullpen_hr_mult': round(bp['bullpen_hr_mult'], 3),
                    'park_hr_mult': round(park_hr_mult, 3),
                    'ev_mean': round(power['ev_mean'], 2),
                    'la_mean': round(power['la_mean'], 2),
                    'hard_hit_pct': round(power['hard_hit_pct'], 4),
                    'hard_hit_14d_pct': round(power['hard_hit_14d_pct'], 4),
                    'barrel_pct': round(power['barrel_pct'], 4),
                    'barrel_14d_pct': round(power['barrel_14d_pct'], 4),
                    'sweet_spot_frac': round(power['sweet_spot_frac'], 4),
                    'flyball_pct': round(power['flyball_pct'], 4),
                    'flyball_30d_pct': round(power['flyball_30d_pct'], 4),
                    'pulled_air_pct': round(power['pulled_air_pct'], 4),
                    'xwobacon_est': round(power['xwobacon'], 4),
                    'xwobacon_14d_est': round(power['xwobacon_14d'], 4),
                    'xiso_est': round(power['xiso_est'], 4),
                    'xiso_30d_est': round(power['xiso_30d_est'], 4),
                    'batter_contact_mult': round(power['contact_mult'], 3),
                    'batter_xbh_mult': round(power['xbh_mult'], 3),
                    'raw_batter_hr_mult': round(power['raw_hr_contact_mult'], 3),
                    'batter_hr_mult': round(power['hr_contact_mult'], 3),
                    'batter_form_hr_mult': round(power['form_hr_mult'], 3),
                    'pitcher_hard_hit_allowed_pct': round(pitcher_danger['hard_hit_allowed_pct'], 4),
                    'pitcher_barrel_allowed_pct': round(pitcher_danger['barrel_allowed_pct'], 4),
                    'pitcher_flyball_allowed_pct': round(pitcher_danger['flyball_allowed_pct'], 4),
                    'pitcher_hr_pa_allowed': round(pitcher_danger['hr_pa_allowed'], 4),
                    'pitcher_xwobacon_allowed': round(pitcher_danger['xwobacon_allowed'], 4),
                    'raw_pitcher_hr_danger_mult': round(pitcher_danger['raw_danger_mult'], 3),
                    'pitcher_hr_danger_mult': round(pitcher_danger['danger_mult'], 3),
                    'raw_pitcher_recent_danger_mult': round(pitcher_danger['raw_recent_danger_mult'], 3),
                    'pitcher_recent_danger_mult': round(pitcher_danger['recent_danger_mult'], 3),
                    'hit_pa': round(hit_pa, 4),
                    '1b_pa': round(p1b, 4),
                    '2b_pa': round(p2b, 4),
                    '3b_pa': round(p3b, 4),
                    'hr_pa': round(phr, 4),
                    'pitch_type_mult': round(float(ctx['pitch_type_mult']), 3),
                    'zone_mult': round(float(ctx['zone_mult']), 3),
                    'zone_hr_overall_mult': round(float(zone_hr_ctx.get('zone_hr_overall_mult', 1.0) or 1.0), 3),
                    'zone_hr_contact_mult': round(float(zone_hr_ctx.get('zone_hr_contact_mult', 1.0) or 1.0), 3),
                    'zone_hr_damage_mult': round(float(zone_hr_ctx.get('zone_hr_damage_mult', 1.0) or 1.0), 3),
                    'zone_hr_barrel_mult': round(float(zone_hr_ctx.get('zone_hr_barrel_mult', 1.0) or 1.0), 3),
                    'zone_hr_air_mult': round(float(zone_hr_ctx.get('zone_hr_air_mult', 1.0) or 1.0), 3),
                    'zone_hr_pull_air_mult': round(float(zone_hr_ctx.get('zone_hr_pull_air_mult', 1.0) or 1.0), 3),
                    'zone_hr_k_suppress_mult': round(float(zone_hr_ctx.get('zone_hr_k_suppress_mult', 1.0) or 1.0), 3),
                    'zone_hr_confidence': zone_hr_ctx.get('zone_hr_confidence'),
                    'zone_hr_candidate_source': zone_hr_ctx.get('zone_hr_candidate_source'),
                    'zone_hr_sample_regime': zone_hr_ctx.get('zone_hr_sample_regime'),
                    'zone_hr_batter_archetype': zone_hr_ctx.get('zone_hr_batter_archetype'),
                    'zone_hr_pitcher_arch': zone_hr_ctx.get('zone_hr_pitcher_arch'),
                    'zone_hr_exact_batter_rows': zone_hr_ctx.get('zone_hr_exact_batter_rows'),
                    'zone_hr_batter_archetype_rows': zone_hr_ctx.get('zone_hr_batter_archetype_rows'),
                    'zone_hr_pitcher_archetype_rows': zone_hr_ctx.get('zone_hr_pitcher_archetype_rows'),
                    'zone_hr_hand_rows': zone_hr_ctx.get('zone_hr_hand_rows'),
                    'zone_hr_league_rows': zone_hr_ctx.get('zone_hr_league_rows'),
                    'zone_hr_early_season_shrink': zone_hr_ctx.get('zone_hr_early_season_shrink'),
                    'weather_hr_mult': round(float(ctx['weather_hr_mult']), 3),
                    'defense_factor': round(float(ctx['defense_factor']), 3),
                    'umpire': ctx['umpire'],
                    'exp_hits': round(exp_pa * hit_pa, 3),
                    'exp_1b': round(exp_pa * p1b, 3),
                    'exp_2b': round(exp_pa * p2b, 3),
                    'exp_3b': round(exp_pa * p3b, 3),
                    'exp_hr': round(exp_pa * phr, 3),
                    'P(H>=1)': round(hit_game_prob, 4),
                    'P(1B>=1)': round(_binom_at_least_one(p1b, exp_pa), 4),
                    'P(2B>=1)': round(_binom_at_least_one(p2b, exp_pa), 4),
                    'P(3B>=1)': round(_binom_at_least_one(p3b, exp_pa), 4),
                    'P(HR>=1)': round(game_hr_prob, 4),
                    'hit_pa_7d': round(float(recent.get('hit_pa_7d', recent.get('hit_pa', LEAGUE_HIT_PA))), 4),
                    'hit_pa_14d': round(float(recent.get('hit_pa_14d', recent.get('hit_pa', LEAGUE_HIT_PA))), 4),
                    'pa_7d': int(recent.get('pa_7d', 0)),
                    'pa_14d': int(recent.get('pa_14d', 0)),
                    'pa_30d': int(recent.get('pa_30d', 0)),
                    'pa_365d': int(recent.get('pa_365d', 0)),
                    'active_roster_flag': 1,
                    'strict_confirmed_lineup_flag': int(strict_confirmed_lineup_flag),
                    'eligible_for_top_boards': int(eligible_for_top_boards),
                    'display_terminal_flag': int(eligible_for_top_boards),
                    'top_board_block_reason': '' if int(eligible_for_top_boards) == 1 else eligibility_reason,
                    'roster_integrity_note': eligibility_reason,
                    'status': np.nan if int(eligible_for_top_boards) == 1 else 'ineligible_top_board',
                })
            except Exception as exc:
                rows.append({
                    **base_row,
                    'player_id': None,
                    'bats': None,
                    'active_roster_flag': 0,
                    'strict_confirmed_lineup_flag': 0,
                    'eligible_for_top_boards': 0,
                    'display_terminal_flag': 0,
                    'top_board_block_reason': 'row_exception',
                    'roster_integrity_note': 'row_exception',
                    'status': _err_status(exc),
                })

    out = pd.DataFrame(rows)
    out = _normalize_date_cols(out)
    if not out.empty:
        if apply_markov_context_to_batter_df is not None:
            try:
                out = apply_markov_context_to_batter_df(out)
            except Exception as exc:
                LOGGER.warning("Markov context enrichment failed for %s: %s", matchup, exc)
        out = _add_breakout_columns(out)
        sort_cols = [c for c in ['matchup', 'team', 'batting_order'] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, kind='stable').reset_index(drop=True)
    return out



def predict_slate_batters(date_str: str, override_path: str | None = None) -> pd.DataFrame:
    frames = []
    for gm in fetch_matchups_for_date(date_str):
        try:
            df = predict_matchup_batters(date_str, gm.replace(' @ ', '@'), override_path=override_path)
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)
            else:
                LOGGER.warning('Empty batter frame for %s on %s', gm, date_str)
                frames.append(pd.DataFrame([{
                    'date': date_str,
                    'matchup': gm.replace(' @ ', '@'),
                    'team': '',
                    'batting_order': np.nan,
                    'player_name': '',
                    'status': 'matchup_empty_frame',
                }]))
        except Exception as exc:
            LOGGER.exception('Slate batter failure for %s on %s', gm, date_str)
            frames.append(pd.DataFrame([{
                'date': date_str,
                'matchup': gm.replace(' @ ', '@'),
                'team': '',
                'batting_order': np.nan,
                'player_name': '',
                'status': f'matchup_error:{type(exc).__name__}:{str(exc)[:140]}',
            }]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
