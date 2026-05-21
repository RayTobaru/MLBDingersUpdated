
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .compat import ROOT

import fetch  # type: ignore
from fetch import fetch_statcast_raw, name_to_mlbam_id, strip_pos

try:
    from fetch import batter_bats
except Exception:  # pragma: no cover
    batter_bats = None  # type: ignore

try:
    from .predict_batter_outcomes import _strict_resolve_batter  # type: ignore
except Exception:  # pragma: no cover
    _strict_resolve_batter = None  # type: ignore


PITCH_FAMILY_MAP = {
    'FF': 'FF_CT', 'FA': 'FF_CT', 'FC': 'FF_CT',
    'SI': 'SI_FT', 'FT': 'SI_FT',
    'SL': 'SL_SW', 'ST': 'SL_SW', 'SV': 'SL_SW',
    'KC': 'CB_KC', 'CU': 'CB_KC', 'CS': 'CB_KC',
    'CH': 'CH_SP', 'FS': 'CH_SP', 'FO': 'CH_SP', 'SC': 'CH_SP',
}
DEFAULT_COUNT_BUCKET_WEIGHTS = {0: 0.22, 1: 0.30, 2: 0.48}
SWING_DESCRIPTIONS = {
    'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt',
    'missed_bunt', 'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score',
}
WHIFF_DESCRIPTIONS = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
CONTACT_DESCRIPTIONS = {'foul', 'foul_tip', 'foul_bunt', 'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
INPLAY_DESCRIPTIONS = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}


def _cache_dir() -> Path:
    p = ROOT / 'cache'
    p.mkdir(parents=True, exist_ok=True)
    return p


def _pitch_family(pt: object) -> str:
    s = str(pt or '').upper().strip()
    return PITCH_FAMILY_MAP.get(s, 'OTHER')


def _zone_cell(zone, plate_x, plate_z) -> str:
    try:
        z = int(float(zone))
        if 1 <= z <= 9:
            return f'Z{z}'
    except Exception:
        pass
    try:
        x = float(plate_x)
        z = float(plate_z)
    except Exception:
        return 'UNK'
    x_cuts = (-0.28, 0.28)
    z_cuts = (2.0, 3.0)
    col = 0 if x < x_cuts[0] else (1 if x <= x_cuts[1] else 2)
    row = 0 if z > z_cuts[1] else (1 if z >= z_cuts[0] else 2)
    idx = row * 3 + col + 1
    return f'Z{idx}'


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return float(x)


def _first_non_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _series_like(obj) -> bool:
    return isinstance(obj, (pd.Series, dict))


def _row_get(row, key: str, default=None):
    if row is None:
        return default
    if isinstance(row, pd.Series):
        return row.get(key, default)
    if isinstance(row, dict):
        return row.get(key, default)
    return default


def _shrink(raw: float, n: float, prior: float, prior_n: float) -> float:
    raw = _safe_float(raw, prior)
    n = max(_safe_float(n, 0.0), 0.0)
    prior = _safe_float(prior, 0.0)
    prior_n = max(_safe_float(prior_n, 0.0), 0.0)
    return float((raw * n + prior * prior_n) / max(n + prior_n, 1e-9))


def _normalize_weights(d: dict[str, float]) -> dict[str, float]:
    out = {k: max(0.0, _safe_float(v, 0.0)) for k, v in d.items()}
    s = sum(out.values())
    if s <= 0:
        n = len(out)
        return {k: 1.0 / max(n, 1) for k in out}
    return {k: v / s for k, v in out.items()}


def _entropy_norm(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size <= 1:
        return 0.0
    p = p / p.sum()
    ent = -np.sum(p * np.log(p))
    return float(ent / np.log(max(len(p), 2)))


def _resolve_batter_ids(lineup_names: list[str], team: str | None = None) -> list[tuple[str, int | None, str]]:
    out: list[tuple[str, int | None, str]] = []
    for nm in list(lineup_names or [])[:9]:
        name = strip_pos(str(nm or '')).strip()
        if not name or name.upper() == 'TBD':
            continue
        pid = None
        if callable(_strict_resolve_batter) and team:
            try:
                pid, resolved = _strict_resolve_batter(team, name)
                name = resolved or name
            except Exception:
                pid = None
        if not pid:
            try:
                pid = name_to_mlbam_id(name)
            except Exception:
                pid = None
        side = 'R'
        try:
            if callable(batter_bats) and pid:
                side = str(batter_bats(int(pid)) or 'R').upper()[:1]
        except Exception:
            side = 'R'
        if side not in {'L', 'R', 'S'}:
            side = 'R'
        out.append((name, int(pid) if pid else None, side))
    return out


def _window_dates(as_of_date: str | None, days: int) -> tuple[str, str, str]:
    end = pd.to_datetime(as_of_date, errors='coerce') if as_of_date else pd.Timestamp.today().normalize()
    if pd.isna(end):
        end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(days))
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), end.strftime('%Y%m%d')


def _pitcher_archetype_key(g: pd.DataFrame) -> str:
    hand = str(g['pitcher_hand'].iloc[0]) if not g.empty else 'R'
    fam_usage = g.groupby('pitch_family', dropna=False)['pitch_count'].sum().sort_values(ascending=False)
    fams = fam_usage.index.tolist()
    top1 = fams[0] if fams else 'OTHER'
    top2 = fams[1] if len(fams) > 1 and fam_usage.iloc[1] / max(fam_usage.sum(), 1.0) >= 0.18 else 'NONE'
    return f'{hand}|{top1}|{top2}'


@lru_cache(maxsize=8)
def _prepared_window(as_of_date: str | None, days: int = 120) -> pd.DataFrame:
    start_s, end_s, end_key = _window_dates(as_of_date, days)
    cache_path = _cache_dir() / f'zone_pitch_prepared_v4_{end_key}_{days}.pkl'
    if cache_path.exists():
        try:
            return pd.read_pickle(cache_path)
        except Exception:
            pass
    raw = fetch_statcast_raw(start_s, end_s)
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()

    need = [
        'pitcher', 'batter', 'pitch_type', 'p_throws', 'stand', 'zone', 'plate_x', 'plate_z',
        'description', 'events', 'strikes', 'launch_speed', 'launch_angle'
    ]
    keep = [c for c in need if c in raw.columns]
    df = raw[keep].copy()
    if df.empty:
        return pd.DataFrame()

    df['pitcher'] = pd.to_numeric(df.get('pitcher'), errors='coerce').astype('Int64')
    df['batter'] = pd.to_numeric(df.get('batter'), errors='coerce').astype('Int64')
    df = df[df['pitcher'].notna() & df['batter'].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    desc = df.get('description', pd.Series('', index=df.index)).astype(str).str.lower()
    df['pitch_type'] = df.get('pitch_type', pd.Series('OTHER', index=df.index)).astype(str).str.upper().str.strip().replace({'': 'OTHER'})
    df['pitch_family'] = df['pitch_type'].map(_pitch_family)
    df['zone_cell'] = [_zone_cell(z, x, y) for z, x, y in zip(df.get('zone'), df.get('plate_x'), df.get('plate_z'))]
    df['pitcher_hand'] = df.get('p_throws', pd.Series('R', index=df.index)).astype(str).str.upper().str[:1].replace({'': 'R'})
    df['batter_side'] = df.get('stand', pd.Series('R', index=df.index)).astype(str).str.upper().str[:1].replace({'': 'R'})
    df['hand_matchup'] = df['pitcher_hand'] + df['batter_side'].replace({'S': 'L'})
    strikes = pd.to_numeric(df.get('strikes'), errors='coerce').fillna(0).clip(lower=0, upper=2)
    df['count_bucket'] = strikes.astype(int)
    df['pitch_count'] = 1.0
    df['swing'] = desc.isin(SWING_DESCRIPTIONS).astype(int)
    df['whiff'] = desc.isin(WHIFF_DESCRIPTIONS).astype(int)
    df['contact'] = desc.isin(CONTACT_DESCRIPTIONS).astype(int)
    df['called_strike'] = desc.eq('called_strike').astype(int)
    df['take'] = (1 - df['swing']).clip(lower=0)
    df['foul'] = desc.str.startswith('foul').astype(int)
    df['in_play'] = desc.isin(INPLAY_DESCRIPTIONS).astype(int)
    ls = pd.to_numeric(df.get('launch_speed'), errors='coerce').fillna(0.0)
    la = pd.to_numeric(df.get('launch_angle'), errors='coerce').fillna(0.0)
    df['damage_contact'] = ((ls >= 95.0) & (la >= 18.0) & (la <= 40.0)).astype(int)

    # pitcher archetypes from pitch mix on this rolling window
    arch_map = (
        df.groupby('pitcher', dropna=False)
          .apply(_pitcher_archetype_key)
          .rename('pitcher_archetype')
          .reset_index()
    )
    df = df.merge(arch_map, on='pitcher', how='left')
    df['pitcher_archetype'] = df['pitcher_archetype'].fillna('R|OTHER|NONE')

    out_cols = [
        'pitcher', 'batter', 'pitch_type', 'pitch_family', 'zone_cell',
        'pitcher_hand', 'batter_side', 'hand_matchup', 'count_bucket', 'pitcher_archetype',
        'pitch_count', 'swing', 'whiff', 'contact', 'called_strike', 'take', 'foul', 'in_play', 'damage_contact'
    ]
    df = df[out_cols].copy()
    try:
        df.to_pickle(cache_path)
    except Exception:
        pass
    return df


def _add_rates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    pc = pd.to_numeric(out['pitch_count'], errors='coerce').fillna(0.0)
    sw = pd.to_numeric(out['swing'], errors='coerce').fillna(0.0)
    tk = pd.to_numeric(out['take'], errors='coerce').fillna(0.0)
    ct = pd.to_numeric(out['contact'], errors='coerce').fillna(0.0)
    out['swing_rate'] = sw / pc.clip(lower=1.0)
    out['whiff_rate'] = pd.to_numeric(out['whiff'], errors='coerce').fillna(0.0) / sw.clip(lower=1.0)
    out['called_strike_rate'] = pd.to_numeric(out['called_strike'], errors='coerce').fillna(0.0) / tk.clip(lower=1.0)
    out['foul_rate'] = pd.to_numeric(out['foul'], errors='coerce').fillna(0.0) / sw.clip(lower=1.0)
    out['in_play_rate'] = pd.to_numeric(out['in_play'], errors='coerce').fillna(0.0) / ct.clip(lower=1.0)
    out['damage_contact_rate'] = pd.to_numeric(out['damage_contact'], errors='coerce').fillna(0.0) / ct.clip(lower=1.0)
    return out


@lru_cache(maxsize=8)
def _agg_tables(as_of_date: str | None, days: int = 120) -> dict[str, pd.DataFrame]:
    end_key = _window_dates(as_of_date, days)[2]
    cache_path = _cache_dir() / f'zone_pitch_tables_v4_{end_key}_{days}.pkl'
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass

    df = _prepared_window(as_of_date, days)
    if df.empty:
        empty = pd.DataFrame()
        return {
            'league_type': empty, 'league_family': empty,
            'hand_type': empty, 'hand_family': empty,
            'pitcher_type': empty, 'pitcher_family': empty,
            'batter_type': empty, 'batter_family': empty,
            'pitcher_arch_family': empty,
        }

    metrics = ['pitch_count', 'swing', 'whiff', 'contact', 'called_strike', 'take', 'foul', 'in_play', 'damage_contact']

    def _group(keys):
        return _add_rates(df.groupby(keys, dropna=False)[metrics].sum().reset_index())

    tables = {
        'league_type': _group(['pitch_type', 'zone_cell', 'count_bucket']),
        'league_family': _group(['pitch_family', 'zone_cell', 'count_bucket']),
        'hand_type': _group(['hand_matchup', 'pitch_type', 'zone_cell', 'count_bucket']),
        'hand_family': _group(['hand_matchup', 'pitch_family', 'zone_cell', 'count_bucket']),
        'pitcher_type': _group(['pitcher', 'hand_matchup', 'pitch_type', 'zone_cell', 'count_bucket']),
        'pitcher_family': _group(['pitcher', 'hand_matchup', 'pitch_family', 'zone_cell', 'count_bucket']),
        'batter_type': _group(['batter', 'hand_matchup', 'pitch_type', 'zone_cell', 'count_bucket']),
        'batter_family': _group(['batter', 'hand_matchup', 'pitch_family', 'zone_cell', 'count_bucket']),
        'pitcher_arch_family': _group(['pitcher_archetype', 'hand_matchup', 'pitch_family', 'zone_cell', 'count_bucket']),
    }
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(tables, f)
    except Exception:
        pass
    return tables


def _subset_one(df: pd.DataFrame, filters: dict) -> pd.Series | None:
    if df is None or df.empty:
        return None
    sub = df
    for k, v in filters.items():
        if k not in sub.columns:
            return None
        sub = sub[sub[k] == v]
        if sub.empty:
            return None
    return sub.iloc[0]


def _rate_from_row(row, metric: str, default_rate: float) -> tuple[float, float]:
    if row is None:
        return float(default_rate), 0.0
    return _safe_float(_row_get(row, metric, default_rate), default_rate), _safe_float(_row_get(row, 'pitch_count', 0.0), 0.0)


def _player_layer_weights(exact_n: float, fam_n: float, arch_n: float, hand_n: float, league_n: float) -> dict[str, float]:
    # More aggressive toward player family and archetype, weaker league fallback.
    exact_s = min(exact_n / 22.0, 1.0)
    fam_s = min(fam_n / 34.0, 1.0)
    arch_s = min(arch_n / 42.0, 1.0)
    hand_s = min(hand_n / 60.0, 1.0)

    player = 0.42 * exact_s
    family = 0.34 * (1.0 - exact_s) * fam_s + 0.18 * fam_s
    arch = 0.24 * (1.0 - exact_s) * (1.0 - 0.45 * fam_s) * arch_s + 0.12 * arch_s
    hand = 0.16 * (1.0 - exact_s) * (1.0 - fam_s) * hand_s + 0.08 * hand_s
    weights = _normalize_weights({
        'player': player,
        'family': family,
        'archetype': arch,
        'hand': hand,
        'league': 0.18,  # weaker default league fallback
    })
    # If there is any meaningful upper-layer evidence, cap league dominance.
    upper = weights['player'] + weights['family'] + weights['archetype'] + weights['hand']
    if upper > 0.20 and weights['league'] > 0.45:
        weights['league'] = 0.45
        rem = 0.55
        up = _normalize_weights({k: weights[k] for k in ('player', 'family', 'archetype', 'hand')})
        for k in up:
            weights[k] = rem * up[k]
    return _normalize_weights(weights)


def _batter_layer_weights(exact_n: float, fam_n: float, hand_n: float, league_n: float) -> dict[str, float]:
    exact_s = min(exact_n / 16.0, 1.0)
    fam_s = min(fam_n / 28.0, 1.0)
    hand_s = min(hand_n / 48.0, 1.0)
    weights = _normalize_weights({
        'player': 0.40 * exact_s,
        'family': 0.34 * (1.0 - exact_s) * fam_s + 0.20 * fam_s,
        'hand': 0.18 * (1.0 - exact_s) * (1.0 - fam_s) * hand_s + 0.10 * hand_s,
        'league': 0.20,
    })
    upper = weights['player'] + weights['family'] + weights['hand']
    if upper > 0.18 and weights['league'] > 0.50:
        weights['league'] = 0.50
        rem = 0.50
        up = _normalize_weights({k: weights[k] for k in ('player', 'family', 'hand')})
        for k in up:
            weights[k] = rem * up[k]
    return _normalize_weights(weights)


def _metric_mix_player(metric: str, dflt: float, rows: dict) -> tuple[float, dict[str, float]]:
    lg_v, lg_n = _rate_from_row(_first_non_none(rows.get('league_family'), rows.get('league_type')), metric, dflt)
    hand_v, hand_n = _rate_from_row(_first_non_none(rows.get('hand_type'), rows.get('hand_family')), metric, lg_v)
    arch_v, arch_n = _rate_from_row(rows.get('arch_family'), metric, hand_v)
    fam_v, fam_n = _rate_from_row(rows.get('pitcher_family'), metric, arch_v)
    exact_v, exact_n = _rate_from_row(rows.get('pitcher_type'), metric, fam_v)

    weights = _player_layer_weights(exact_n, fam_n, arch_n, hand_n, lg_n)
    mixed = (
        weights['player'] * exact_v
        + weights['family'] * fam_v
        + weights['archetype'] * arch_v
        + weights['hand'] * hand_v
        + weights['league'] * lg_v
    )
    return float(mixed), weights


def _metric_mix_batter(metric: str, dflt: float, rows: dict) -> tuple[float, dict[str, float]]:
    lg_v, lg_n = _rate_from_row(_first_non_none(rows.get('league_family'), rows.get('league_type')), metric, dflt)
    hand_v, hand_n = _rate_from_row(_first_non_none(rows.get('hand_type'), rows.get('hand_family')), metric, lg_v)
    fam_v, fam_n = _rate_from_row(rows.get('batter_family'), metric, hand_v)
    exact_v, exact_n = _rate_from_row(rows.get('batter_type'), metric, fam_v)

    weights = _batter_layer_weights(exact_n, fam_n, hand_n, lg_n)
    mixed = (
        weights['player'] * exact_v
        + weights['family'] * fam_v
        + weights['hand'] * hand_v
        + weights['league'] * lg_v
    )
    return float(mixed), weights




def _arch_families_from_label(pitcher_arch: str) -> list[str]:
    parts = str(pitcher_arch or "").split("|")
    fams = []
    for part in parts[1:3]:
        fam = str(part or "").strip()
        if fam and fam not in {"NONE", "OTHER", "nan"}:
            fams.append(fam)
    # preserve order, remove duplicates
    out = []
    for fam in fams:
        if fam not in out:
            out.append(fam)
    return out


def _all_pitch_candidates(tabs: dict, pitcher_id: int, hand_matchup: str, pitcher_arch: str) -> pd.DataFrame:
    """
    Candidate pitch/zone/count usage table for the 3x3 matchup engine.

    Evidence tiers:
      1. exact_pitcher
      2. pitcher_family
      3. pitcher_archetype
      4. hand_matchup
      5. league

    If the true pitcher_archetype table has no rows for a derived archetype,
    we synthesize an archetype layer from same-hand/league family rows filtered
    to the pitcher's archetype families. This avoids jumping straight from
    no exact pitcher rows to broad hand/league fallback.
    """
    pt = tabs.get('pitcher_type', pd.DataFrame())
    pf = tabs.get('pitcher_family', pd.DataFrame())
    af = tabs.get('pitcher_arch_family', pd.DataFrame())
    hf = tabs.get('hand_family', pd.DataFrame())
    lf = tabs.get('league_family', pd.DataFrame())

    out = []
    arch_added = False
    arch_fams = _arch_families_from_label(pitcher_arch)

    # 1) Exact pitcher + pitch type
    if pt is not None and not pt.empty:
        ss = pt[
            (pd.to_numeric(pt.get('pitcher'), errors='coerce') == int(pitcher_id)) &
            (pt.get('hand_matchup').astype(str) == str(hand_matchup))
        ].copy()
        if not ss.empty:
            if 'pitch_family' not in ss.columns:
                ss['pitch_family'] = ss['pitch_type'].map(_pitch_family) if 'pitch_type' in ss.columns else 'OTHER'
            ss['candidate_type'] = 'exact_pitcher'
            ss['source_weight'] = 1.00
            out.append(ss)

    # 2) Pitcher + family
    if pf is not None and not pf.empty:
        ss = pf[
            (pd.to_numeric(pf.get('pitcher'), errors='coerce') == int(pitcher_id)) &
            (pf.get('hand_matchup').astype(str) == str(hand_matchup))
        ].copy()
        if not ss.empty:
            ss = ss.assign(
                pitch_type=ss['pitch_family'],
                candidate_type='pitcher_family',
                source_weight=0.78,
            )
            out.append(ss)

    # 3) True pitcher archetype table
    if af is not None and not af.empty:
        ss = af[
            (af.get('pitcher_archetype').astype(str) == str(pitcher_arch)) &
            (af.get('hand_matchup').astype(str) == str(hand_matchup))
        ].copy()
        if not ss.empty:
            ss = ss.assign(
                pitch_type=ss['pitch_family'],
                candidate_type='pitcher_archetype',
                source_weight=0.58,
            )
            out.append(ss)
            arch_added = True

    # 3b) Synthetic archetype from same-hand family rows, filtered to archetype families.
    if not arch_added and arch_fams and hf is not None and not hf.empty:
        ss = hf[
            (hf.get('hand_matchup').astype(str) == str(hand_matchup)) &
            (hf.get('pitch_family').astype(str).isin(arch_fams))
        ].copy()
        if not ss.empty:
            ss = ss.assign(
                pitch_type=ss['pitch_family'],
                pitcher_archetype=str(pitcher_arch),
                candidate_type='pitcher_archetype',
                source_weight=0.50,
            )
            out.append(ss)
            arch_added = True

    # 3c) Synthetic archetype from league family rows if hand-family is unavailable.
    if not arch_added and arch_fams and lf is not None and not lf.empty:
        ss = lf[lf.get('pitch_family').astype(str).isin(arch_fams)].copy()
        if not ss.empty:
            ss = ss.assign(
                pitch_type=ss['pitch_family'],
                hand_matchup=str(hand_matchup),
                pitcher_archetype=str(pitcher_arch),
                candidate_type='pitcher_archetype',
                source_weight=0.42,
            )
            out.append(ss)
            arch_added = True

    # 4) Broad hand-family fallback
    if hf is not None and not hf.empty:
        ss = hf[hf.get('hand_matchup').astype(str) == str(hand_matchup)].copy()
        if not ss.empty:
            ss = ss.assign(
                pitch_type=ss['pitch_family'],
                candidate_type='hand_matchup',
                source_weight=0.36,
            )
            out.append(ss)

    # 5) League fallback
    if lf is not None and not lf.empty:
        ss = lf.copy()
        if not ss.empty:
            ss = ss.assign(
                pitch_type=ss['pitch_family'],
                candidate_type='league',
                source_weight=0.24,
            )
            out.append(ss)

    if not out:
        return pd.DataFrame()

    allc = pd.concat(out, ignore_index=True, sort=False)
    allc['pitch_count'] = pd.to_numeric(allc.get('pitch_count'), errors='coerce').fillna(0.0)
    allc['source_weight'] = pd.to_numeric(allc.get('source_weight'), errors='coerce').fillna(1.0)
    allc = allc[allc['pitch_count'] > 0].copy()
    return allc


def _profile_pitch_mix_candidates(pitcher_id: int) -> dict[str, float]:
    """
    Best-effort pitch family mix from available fetch.py helpers.

    Purpose:
      If a pitcher has no exact rows in the prepared 3x3 window, classify him
      by pitch profile so the engine can use pitcher_archetype rows before
      falling to hand/league.
    """
    out: dict[str, float] = {}

    def _add_family_from_key(key_obj, val_obj, scale: float = 1.0):
        key = str(key_obj or "").upper().strip()
        if not key:
            return

        # Accept direct pitch type keys like FF, SI, SL.
        candidates = [key]

        # Accept feature-style keys like FF_whiff_rate, SL_pct, CH_usage.
        for sep in ("_", "-", " "):
            if sep in key:
                candidates.append(key.split(sep)[0])

        fam = "OTHER"
        for cand in candidates:
            fam = _pitch_family(cand)
            if fam != "OTHER":
                break

        if fam == "OTHER":
            return

        try:
            val = float(val_obj)
        except Exception:
            return
        if not np.isfinite(val) or val <= 0:
            return

        out[fam] = out.get(fam, 0.0) + float(val) * float(scale)

    # 1) Try pitch-mix/profile helpers if available.
    helper_names = [
        "pitcher_mix_last_starts",
        "get_pitch_type_profile",
    ]

    for helper_name in helper_names:
        fn = getattr(fetch, helper_name, None)
        if not callable(fn):
            continue

        result = None
        try:
            result = fn(int(pitcher_id))
        except TypeError:
            try:
                result = fn(int(pitcher_id), days=365)
            except Exception:
                result = None
        except Exception:
            result = None

        if not isinstance(result, dict):
            continue

        for k, v in result.items():
            # Flat profile: {"FF": 0.42, "SL": 0.25}
            if isinstance(v, (int, float, np.number)):
                _add_family_from_key(k, v, scale=1.0)
                continue

            # Nested profile: {"FF": {"usage": .42, "pitches": 300}}
            if isinstance(v, dict):
                for usage_key in ("usage", "pct", "pitch_pct", "share", "mix", "rate", "pitches", "count"):
                    if usage_key in v:
                        _add_family_from_key(k, v.get(usage_key), scale=1.0)
                        break

    # 2) Statcast feature fallback. These are weaker than true usage, but still
    # better than no archetype. They help identify arsenal families.
    try:
        feats = fetch.get_statcast_pitcher_features(int(pitcher_id), days=365) or {}
    except TypeError:
        try:
            feats = fetch.get_statcast_pitcher_features(int(pitcher_id)) or {}
        except Exception:
            feats = {}
    except Exception:
        feats = {}

    if isinstance(feats, dict):
        for k, v in feats.items():
            ks = str(k or "").upper()

            # Examples: FF_whiff_rate, SL_whiff_rate, CH_usage, SI_pct
            if any(token in ks for token in ("WHIFF", "USAGE", "PCT", "RATE", "PITCH")):
                _add_family_from_key(ks, v, scale=0.18)

    return out


def _available_arch_counts(tabs: dict) -> pd.DataFrame:
    af = tabs.get("pitcher_arch_family", pd.DataFrame())
    if af is None or af.empty or "pitcher_archetype" not in af.columns:
        return pd.DataFrame(columns=["pitcher_archetype", "pitch_count"])

    out = af.copy()
    out["pitch_count"] = pd.to_numeric(out.get("pitch_count"), errors="coerce").fillna(0.0)
    return (
        out.groupby("pitcher_archetype", dropna=False)["pitch_count"]
           .sum()
           .reset_index()
           .sort_values("pitch_count", ascending=False)
    )


def _closest_available_pitcher_arch(tabs: dict, desired_arch: str) -> str:
    """
    If desired archetype is not present in pitcher_arch_family, map it to the
    closest existing archetype by same hand/top family/secondary family.
    """
    arch_counts = _available_arch_counts(tabs)
    if arch_counts.empty:
        return desired_arch

    available = set(arch_counts["pitcher_archetype"].astype(str))
    if desired_arch in available:
        return desired_arch

    parts = str(desired_arch or "").split("|")
    hand = parts[0] if len(parts) > 0 and parts[0] in {"L", "R"} else "R"
    top1 = parts[1] if len(parts) > 1 else "OTHER"
    top2 = parts[2] if len(parts) > 2 else "NONE"

    best_arch = desired_arch
    best_score = -1.0

    for _, row in arch_counts.iterrows():
        arch = str(row.get("pitcher_archetype", ""))
        cnt = float(row.get("pitch_count", 0.0) or 0.0)
        ap = arch.split("|")
        ah = ap[0] if len(ap) > 0 else ""
        a1 = ap[1] if len(ap) > 1 else "OTHER"
        a2 = ap[2] if len(ap) > 2 else "NONE"

        score = 0.0
        if ah == hand:
            score += 5.0
        else:
            continue

        if a1 == top1:
            score += 4.0
        if a2 == top2 and top2 != "NONE":
            score += 2.0
        if top2 != "NONE" and a1 == top2:
            score += 1.0
        if a2 == top1:
            score += 1.0

        # Tiny volume tiebreaker only.
        score += min(np.log1p(max(cnt, 0.0)) / 20.0, 0.75)

        if score > best_score:
            best_score = score
            best_arch = arch

    return best_arch


def _fallback_pitcher_arch_from_profile(tabs: dict, pitcher_id: int, pitcher_hand: str | None = None) -> str:
    hand = str(pitcher_hand or "R").upper()[:1]
    if hand not in {"L", "R"}:
        hand = "R"

    fam_usage = _profile_pitch_mix_candidates(int(pitcher_id))
    fam_usage = {
        k: float(v)
        for k, v in fam_usage.items()
        if k and np.isfinite(float(v)) and float(v) > 0
    }

    if fam_usage:
        fam_sorted = sorted(fam_usage.items(), key=lambda kv: kv[1], reverse=True)
        total = max(sum(v for _, v in fam_sorted), 1e-9)

        top1 = fam_sorted[0][0]
        top2 = "NONE"
        if len(fam_sorted) > 1 and fam_sorted[1][1] / total >= 0.14:
            top2 = fam_sorted[1][0]

        desired = f"{hand}|{top1}|{top2}"
        return _closest_available_pitcher_arch(tabs, desired)

    # Final fallback: best same-hand archetype available.
    arch_counts = _available_arch_counts(tabs)
    if not arch_counts.empty:
        same_hand = arch_counts[arch_counts["pitcher_archetype"].astype(str).str.startswith(hand + "|")]
        if not same_hand.empty:
            return str(same_hand.iloc[0]["pitcher_archetype"])

    return f"{hand}|OTHER|NONE"


def _find_pitcher_arch(tabs: dict, pitcher_id: int, pitcher_hand: str | None = None) -> str:
    """
    Find pitcher archetype.

    Priority:
      1. Exact pitcher rows from the prepared 3x3 table.
      2. Profile-derived pitch mix / whiff profile.
      3. Closest same-hand available archetype.
      4. Generic hand|OTHER|NONE.
    """
    pt = tabs.get("pitcher_type", pd.DataFrame())
    s = pd.DataFrame()

    if pt is not None and not pt.empty:
        s = pt[pd.to_numeric(pt.get("pitcher"), errors="coerce") == int(pitcher_id)].copy()

    if not s.empty:
        hand = str(pitcher_hand or "R").upper()[:1]
        if "hand_matchup" in s.columns and s["hand_matchup"].notna().any():
            hm = str(s["hand_matchup"].dropna().iloc[0])
            if hm:
                hand = hm[:1]
        if hand not in {"L", "R"}:
            hand = "R"

        if "pitch_family" not in s.columns:
            if "pitch_type" in s.columns:
                s["pitch_family"] = s["pitch_type"].map(_pitch_family)
            else:
                s["pitch_family"] = "OTHER"

        fam_usage = s.groupby("pitch_family", dropna=False)["pitch_count"].sum().sort_values(ascending=False)
        fams = fam_usage.index.tolist()

        top1 = fams[0] if fams else "OTHER"
        top2 = fams[1] if len(fams) > 1 and fam_usage.iloc[1] / max(fam_usage.sum(), 1.0) >= 0.18 else "NONE"

        desired = f"{hand}|{top1}|{top2}"
        return _closest_available_pitcher_arch(tabs, desired)

    return _fallback_pitcher_arch_from_profile(tabs, int(pitcher_id), pitcher_hand)


def _context_rows(tabs: dict, pitcher_id: int, batter_id: int | None, hand_matchup: str, pitch_type: str, pitch_family: str, zone_cell: str, count_bucket: int, pitcher_arch: str) -> dict:
    return {
        'league_type': _subset_one(tabs['league_type'], {'pitch_type': pitch_type, 'zone_cell': zone_cell, 'count_bucket': count_bucket}),
        'league_family': _subset_one(tabs['league_family'], {'pitch_family': pitch_family, 'zone_cell': zone_cell, 'count_bucket': count_bucket}),
        'hand_type': _subset_one(tabs['hand_type'], {'hand_matchup': hand_matchup, 'pitch_type': pitch_type, 'zone_cell': zone_cell, 'count_bucket': count_bucket}),
        'hand_family': _subset_one(tabs['hand_family'], {'hand_matchup': hand_matchup, 'pitch_family': pitch_family, 'zone_cell': zone_cell, 'count_bucket': count_bucket}),
        'pitcher_type': _subset_one(tabs['pitcher_type'], {'pitcher': int(pitcher_id), 'hand_matchup': hand_matchup, 'pitch_type': pitch_type, 'zone_cell': zone_cell, 'count_bucket': count_bucket}),
        'pitcher_family': _subset_one(tabs['pitcher_family'], {'pitcher': int(pitcher_id), 'hand_matchup': hand_matchup, 'pitch_family': pitch_family, 'zone_cell': zone_cell, 'count_bucket': count_bucket}),
        'arch_family': _subset_one(tabs['pitcher_arch_family'], {'pitcher_archetype': pitcher_arch, 'hand_matchup': hand_matchup, 'pitch_family': pitch_family, 'zone_cell': zone_cell, 'count_bucket': count_bucket}),
        'batter_type': _subset_one(tabs['batter_type'], {'batter': int(batter_id), 'hand_matchup': hand_matchup, 'pitch_type': pitch_type, 'zone_cell': zone_cell, 'count_bucket': count_bucket}) if batter_id is not None else None,
        'batter_family': _subset_one(tabs['batter_family'], {'batter': int(batter_id), 'hand_matchup': hand_matchup, 'pitch_family': pitch_family, 'zone_cell': zone_cell, 'count_bucket': count_bucket}) if batter_id is not None else None,
    }



def _source_count_summary(candidates_cache: dict[str, pd.DataFrame]) -> dict[str, float]:
    out = {
        'exact_pitcher': 0.0,
        'pitcher_family': 0.0,
        'pitcher_archetype': 0.0,
        'hand_matchup': 0.0,
        'league': 0.0,
    }
    for cand in candidates_cache.values():
        if cand is None or cand.empty or 'candidate_type' not in cand.columns:
            continue
        for src, g in cand.groupby('candidate_type', dropna=False):
            src = str(src)
            if src in out:
                out[src] += float(pd.to_numeric(g.get('pitch_count'), errors='coerce').fillna(0.0).sum())
    return out


def _top_candidate_source(src_counts: dict[str, float]) -> str:
    """
    Return best available evidence tier, not the largest row-count tier.
    League will almost always have the most rows, but that does not mean it is
    the most meaningful source.
    """
    priority = [
        "exact_pitcher",
        "pitcher_family",
        "pitcher_archetype",
        "hand_matchup",
        "league",
    ]
    for src in priority:
        if float(src_counts.get(src, 0.0) or 0.0) > 0:
            return src
    return "none"


def _season_elapsed_factor(as_of_date: str | None) -> float:
    """
    0.0 early, 1.0 later.
    Uses a rough MLB regular-season anchor so early April/May stays conservative.
    """
    dt = pd.to_datetime(as_of_date, errors='coerce') if as_of_date else pd.Timestamp.today()
    if pd.isna(dt):
        dt = pd.Timestamp.today()
    season_start = pd.Timestamp(year=int(dt.year), month=3, day=20)
    season_mature = pd.Timestamp(year=int(dt.year), month=7, day=15)
    span = max((season_mature - season_start).days, 1)
    elapsed = max((dt - season_start).days, 0)
    return float(np.clip(elapsed / span, 0.0, 1.0))


def _early_season_shrink(as_of_date: str | None, source_counts: dict[str, float], conf: float) -> float:
    """
    Shrinks the size of the zone multiplier's deviation from 1.0.
    Early season + low exact/pitcher-family evidence = conservative.
    """
    elapsed = _season_elapsed_factor(as_of_date)
    exact_n = float(source_counts.get('exact_pitcher', 0.0) or 0.0)
    fam_n = float(source_counts.get('pitcher_family', 0.0) or 0.0)
    arch_n = float(source_counts.get('pitcher_archetype', 0.0) or 0.0)

    player_evidence = min(1.0, exact_n / 180.0)
    family_evidence = min(1.0, fam_n / 260.0)
    arch_evidence = min(1.0, arch_n / 360.0)
    evidence = 0.50 * player_evidence + 0.32 * family_evidence + 0.18 * arch_evidence

    # Early season shrink can still reach moderate strength if evidence is real.
    shrink = 0.42 + 0.28 * elapsed + 0.22 * evidence + 0.08 * float(np.clip(conf, 0.0, 1.0))
    return float(np.clip(shrink, 0.35, 1.00))


def _sample_regime_label(source_counts: dict[str, float], conf: float, as_of_date: str | None) -> tuple[str, str]:
    elapsed = _season_elapsed_factor(as_of_date)
    exact_n = float(source_counts.get('exact_pitcher', 0.0) or 0.0)
    fam_n = float(source_counts.get('pitcher_family', 0.0) or 0.0)
    arch_n = float(source_counts.get('pitcher_archetype', 0.0) or 0.0)
    hand_n = float(source_counts.get('hand_matchup', 0.0) or 0.0)

    if exact_n >= 220 and conf >= 0.55:
        regime = 'pitcher_specific'
    elif fam_n >= 260 and conf >= 0.42:
        regime = 'pitcher_family'
    elif arch_n >= 360 and conf >= 0.32:
        regime = 'pitcher_archetype'
    elif hand_n > 0:
        regime = 'hand_family_league'
    else:
        regime = 'league_fallback'

    if elapsed < 0.55 and regime != 'pitcher_specific':
        regime = 'early_season_' + regime

    if conf >= 0.58:
        strength = 'normal'
    elif conf >= 0.34:
        strength = 'conservative'
    else:
        strength = 'very_conservative'

    return regime, strength


def summarize_zone_pitch_ko_context(
    pitcher_id: int | None,
    opp_lineup: list[str],
    opp_team: str | None = None,
    as_of_date: str | None = None,
    pitcher_hand: str | None = None,
    days: int = 120,
) -> dict[str, float | int | str]:
    defaults = {
        'zone_pitch_k_mult': 1.0,
        'zone_pitch_whiff_mult': 1.0,
        'zone_pitch_called_mult': 1.0,
        'zone_pitch_putaway_mult': 1.0,
        'zone_pitch_foul_suppress_mult': 1.0,
        'zone_pitch_confidence': 0.0,
        'zone_pitch_resolved_n': 0,
        'zone_pitch_top_family': '',
        'zone_pitch_second_family': '',
        'zone_pitch_attack_zone': '',
        'zone_pitch_second_zone': '',
        'zone_pitch_pitcher_arch': '',
        'zone_pitch_mix_entropy': 0.0,
        'zone_pitch_player_weight': 0.0,
        'zone_pitch_family_weight': 0.0,
        'zone_pitch_archetype_weight': 0.0,
        'zone_pitch_hand_weight': 0.0,
        'zone_pitch_league_weight': 1.0,
        'zone_pitch_exact_pitcher_rows': 0.0,
        'zone_pitch_pitcher_family_rows': 0.0,
        'zone_pitch_pitcher_archetype_rows': 0.0,
        'zone_pitch_hand_rows': 0.0,
        'zone_pitch_league_rows': 0.0,
        'zone_pitch_candidate_source': 'none',
        'zone_pitch_sample_regime': 'default',
        'zone_pitch_effect_strength': 'none',
        'zone_pitch_early_season_shrink': 0.0,
    }
    if not pitcher_id or not opp_lineup:
        return defaults

    tabs = _agg_tables(as_of_date, days)
    if tabs['league_family'].empty and tabs['pitcher_family'].empty:
        return defaults

    p_hand = str(pitcher_hand or 'R').upper()[:1]
    if p_hand not in {'L', 'R'}:
        p_hand = 'R'
    pitcher_arch = _find_pitcher_arch(tabs, int(pitcher_id), pitcher_hand)
    resolved = _resolve_batter_ids(opp_lineup, opp_team)
    if not resolved:
        return defaults

    per_batter = []
    family_scores: dict[str, float] = {}
    zone_scores: dict[str, float] = {}
    mix_weights_accum = {'player': 0.0, 'family': 0.0, 'archetype': 0.0, 'hand': 0.0, 'league': 0.0}
    entropy_list = []

    candidates_cache: dict[str, pd.DataFrame] = {}

    for _, batter_id, side in resolved:
        hm = p_hand + ('L' if side == 'S' else side)
        if hm not in candidates_cache:
            candidates_cache[hm] = _all_pitch_candidates(tabs, int(pitcher_id), hm, pitcher_arch)
        cand = candidates_cache[hm]
        if cand.empty:
            continue

        # Pitch usage: exact counts first; sharpen when evidence exists.
        usage = cand.copy()
        if 'pitch_family' not in usage.columns:
            if 'pitch_type' in usage.columns:
                usage['pitch_family'] = usage['pitch_type'].map(_pitch_family)
            else:
                usage['pitch_family'] = 'OTHER'
        usage['count_w'] = usage['count_bucket'].map(DEFAULT_COUNT_BUCKET_WEIGHTS).fillna(0.30)
        raw_w = (pd.to_numeric(usage['pitch_count'], errors='coerce').fillna(0.0) * pd.to_numeric(usage['count_w'], errors='coerce').fillna(0.30) * pd.to_numeric(usage.get('source_weight', 1.0), errors='coerce').fillna(1.0))
        fam_mass = usage.groupby('pitch_family')['pitch_count'].sum()
        top_share = float(fam_mass.max() / max(fam_mass.sum(), 1.0)) if not fam_mass.empty else 0.0
        evidence = min(np.sqrt(max(float(raw_w.sum()), 0.0)) / 18.0, 1.0)
        alpha = 1.0 + 0.60 * evidence + 0.35 * max(0.0, top_share - 0.35)
        usage['raw_weight'] = np.power(np.clip(raw_w, 1e-9, None), alpha)
        usage = usage.groupby(['pitch_type', 'pitch_family', 'zone_cell', 'count_bucket'], dropna=False, as_index=False)['raw_weight'].sum()
        usage['weight'] = usage['raw_weight'] / max(usage['raw_weight'].sum(), 1e-9)

        if usage.empty:
            continue

        entropy_list.append(_entropy_norm(usage['weight'].to_numpy(dtype=float)))

        whiff_parts = []
        called_parts = []
        putaway_parts = []
        foul_parts = []
        k_parts = []
        conf_parts = []

        for row in usage.itertuples(index=False):
            pitch_type = str(row.pitch_type)
            pitch_family = str(row.pitch_family)
            zone_cell = str(row.zone_cell)
            count_bucket = int(row.count_bucket)
            w = float(row.weight)

            rows = _context_rows(tabs, int(pitcher_id), batter_id, hm, pitch_type, pitch_family, zone_cell, count_bucket, pitcher_arch)

            # baselines
            base_whiff = _safe_float(_row_get(_first_non_none(rows['hand_type'], rows['hand_family'], rows['league_family'], rows['league_type']), 'whiff_rate', 0.24), 0.24)
            base_called = _safe_float(_row_get(_first_non_none(rows['hand_type'], rows['hand_family'], rows['league_family'], rows['league_type']), 'called_strike_rate', 0.17), 0.17)
            base_foul = _safe_float(_row_get(_first_non_none(rows['hand_type'], rows['hand_family'], rows['league_family'], rows['league_type']), 'foul_rate', 0.18), 0.18)

            pit_whiff, pit_w = _metric_mix_player('whiff_rate', base_whiff, rows)
            pit_called, pit_wc = _metric_mix_player('called_strike_rate', base_called, rows)
            pit_foul, pit_wf = _metric_mix_player('foul_rate', base_foul, rows)

            bat_whiff, bat_w = _metric_mix_batter('whiff_rate', base_whiff, rows)
            bat_called, bat_wc = _metric_mix_batter('called_strike_rate', base_called, rows)
            bat_foul, bat_wf = _metric_mix_batter('foul_rate', base_foul, rows)

            whiff_mult = float(np.sqrt(np.clip(pit_whiff / max(base_whiff, 1e-6), 0.70, 1.55) * np.clip(bat_whiff / max(base_whiff, 1e-6), 0.70, 1.55)))
            called_mult = float(np.sqrt(np.clip(pit_called / max(base_called, 1e-6), 0.75, 1.45) * np.clip(bat_called / max(base_called, 1e-6), 0.75, 1.45)))
            foul_suppress = float(np.sqrt(np.clip(base_foul / max(pit_foul, 1e-6), 0.75, 1.35) * np.clip(base_foul / max(bat_foul, 1e-6), 0.75, 1.35)))

            putaway = whiff_mult
            if count_bucket == 2:
                putaway = float(np.clip(0.70 * whiff_mult + 0.30 * called_mult, 0.82, 1.30))
            elif count_bucket == 1:
                putaway = float(np.clip(0.82 * whiff_mult + 0.18 * called_mult, 0.84, 1.24))
            else:
                putaway = float(np.clip(0.88 * whiff_mult + 0.12 * called_mult, 0.86, 1.18))

            overall = float(np.clip(0.45 * whiff_mult + 0.24 * called_mult + 0.21 * putaway + 0.10 * foul_suppress, 0.88, 1.14))

            # combined confidence from layer usage; less league dependence => more confidence
            player_mix = 0.5 * (pit_w['player'] + bat_w['player'])
            fam_mix = 0.5 * (pit_w['family'] + bat_w['family'])
            arch_mix = pit_w['archetype']
            hand_mix = 0.5 * (pit_w['hand'] + bat_w['hand'])
            lg_mix = 0.5 * (pit_w['league'] + bat_w['league'])
            local_conf = float(np.clip(0.20 + 0.55 * (player_mix + fam_mix + arch_mix) + 0.20 * hand_mix - 0.25 * lg_mix, 0.15, 1.0))

            whiff_parts.append(w * whiff_mult)
            called_parts.append(w * called_mult)
            putaway_parts.append(w * putaway)
            foul_parts.append(w * foul_suppress)
            k_parts.append(w * overall)
            conf_parts.append(w * local_conf)

            family_scores[pitch_family] = family_scores.get(pitch_family, 0.0) + w * overall
            zone_scores[zone_cell] = zone_scores.get(zone_cell, 0.0) + w * overall

            mix_weights_accum['player'] += w * player_mix
            mix_weights_accum['family'] += w * fam_mix
            mix_weights_accum['archetype'] += w * arch_mix
            mix_weights_accum['hand'] += w * hand_mix
            mix_weights_accum['league'] += w * lg_mix

        if not k_parts:
            continue

        per_batter.append({
            'overall': float(np.sum(k_parts)),
            'whiff': float(np.sum(whiff_parts)),
            'called': float(np.sum(called_parts)),
            'putaway': float(np.sum(putaway_parts)),
            'foul': float(np.sum(foul_parts)),
            'conf': float(np.sum(conf_parts)),
        })

    if not per_batter:
        return defaults

    arr = pd.DataFrame(per_batter)
    conf_raw = float(arr['conf'].mean())
    family_share = max(family_scores.values()) / max(sum(family_scores.values()), 1e-9) if family_scores else 0.0
    ent_norm = float(np.mean(entropy_list)) if entropy_list else 0.0

    # Final confidence: more family concentration and more non-league weight should help.
    mix_weights_accum = _normalize_weights(mix_weights_accum)
    conf = float(np.clip(0.40 * conf_raw + 0.25 * (1.0 - ent_norm) + 0.20 * (1.0 - mix_weights_accum.get('league', 1.0)) + 0.15 * family_share, 0.0, 1.0))

    raw_overall = float(arr['overall'].mean())
    raw_whiff = float(arr['whiff'].mean())
    raw_called = float(arr['called'].mean())
    raw_putaway = float(arr['putaway'].mean())
    raw_foul = float(arr['foul'].mean())

    source_counts = _source_count_summary(candidates_cache)
    candidate_source = _top_candidate_source(source_counts)
    early_shrink = _early_season_shrink(as_of_date, source_counts, conf)
    sample_regime, effect_strength = _sample_regime_label(source_counts, conf, as_of_date)

    def _blend(raw: float, lo: float = 0.94, hi: float = 1.08) -> float:
        # Two-stage shrink: confidence first, then early-season/sample-regime shrink.
        shifted = 1.0 + conf * early_shrink * (_safe_float(raw, 1.0) - 1.0)
        return float(np.clip(shifted, lo, hi))

    fam_sorted = sorted(family_scores.items(), key=lambda kv: kv[1], reverse=True)
    zone_sorted = sorted(zone_scores.items(), key=lambda kv: kv[1], reverse=True)

    return {
        'zone_pitch_k_mult': _blend(raw_overall, 0.94, 1.08),
        'zone_pitch_whiff_mult': _blend(raw_whiff, 0.94, 1.08),
        'zone_pitch_called_mult': _blend(raw_called, 0.95, 1.07),
        'zone_pitch_putaway_mult': _blend(raw_putaway, 0.94, 1.09),
        'zone_pitch_foul_suppress_mult': _blend(raw_foul, 0.95, 1.06),
        'zone_pitch_confidence': round(conf, 3),
        'zone_pitch_resolved_n': int(len(per_batter)),
        'zone_pitch_top_family': str(fam_sorted[0][0]) if fam_sorted else '',
        'zone_pitch_second_family': str(fam_sorted[1][0]) if len(fam_sorted) > 1 else '',
        'zone_pitch_attack_zone': str(zone_sorted[0][0]) if zone_sorted else '',
        'zone_pitch_second_zone': str(zone_sorted[1][0]) if len(zone_sorted) > 1 else '',
        'zone_pitch_pitcher_arch': pitcher_arch,
        'zone_pitch_mix_entropy': round(ent_norm, 3),
        'zone_pitch_player_weight': round(mix_weights_accum.get('player', 0.0), 3),
        'zone_pitch_family_weight': round(mix_weights_accum.get('family', 0.0), 3),
        'zone_pitch_archetype_weight': round(mix_weights_accum.get('archetype', 0.0), 3),
        'zone_pitch_hand_weight': round(mix_weights_accum.get('hand', 0.0), 3),
        'zone_pitch_league_weight': round(mix_weights_accum.get('league', 1.0), 3),
        'zone_pitch_exact_pitcher_rows': round(float(source_counts.get('exact_pitcher', 0.0)), 1),
        'zone_pitch_pitcher_family_rows': round(float(source_counts.get('pitcher_family', 0.0)), 1),
        'zone_pitch_pitcher_archetype_rows': round(float(source_counts.get('pitcher_archetype', 0.0)), 1),
        'zone_pitch_hand_rows': round(float(source_counts.get('hand_matchup', 0.0)), 1),
        'zone_pitch_league_rows': round(float(source_counts.get('league', 0.0)), 1),
        'zone_pitch_candidate_source': candidate_source,
        'zone_pitch_sample_regime': sample_regime,
        'zone_pitch_effect_strength': effect_strength,
        'zone_pitch_early_season_shrink': round(float(early_shrink), 3),
    }
