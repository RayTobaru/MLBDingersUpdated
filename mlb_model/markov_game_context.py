from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

_BASE_STATES: List[Tuple[int, int, int, int]] = [
    (outs, b1, b2, b3)
    for outs in range(3)
    for b1 in (0, 1)
    for b2 in (0, 1)
    for b3 in (0, 1)
]

SINGLE_2B_SCORE = 0.62
SINGLE_1B_TO_3B = 0.42
DOUBLE_1B_SCORE = 0.58


def _clip_prob(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 0.95))


def _row_event_probs(row: pd.Series) -> Dict[str, float]:
    p1 = _clip_prob(row.get("1b_pa", 0.0))
    p2 = _clip_prob(row.get("2b_pa", 0.0))
    p3 = _clip_prob(row.get("3b_pa", 0.0))
    phr = _clip_prob(row.get("hr_pa", 0.0))
    hit = _clip_prob(row.get("hit_pa", p1 + p2 + p3 + phr))
    non_hit = max(0.0, 1.0 - hit)

    # conservative default walk/HBP share from non-hit space
    pbb = min(0.085, non_hit * 0.45)
    pout = max(0.0, 1.0 - (p1 + p2 + p3 + phr + pbb))

    total = p1 + p2 + p3 + phr + pbb + pout
    if total <= 0:
        return {"OUT": 1.0}
    scale = 1.0 / total
    return {
        "1B": p1 * scale,
        "2B": p2 * scale,
        "3B": p3 * scale,
        "HR": phr * scale,
        "BB": pbb * scale,
        "OUT": pout * scale,
    }


def _advance_walk(state: Tuple[int, int, int, int]) -> Dict[Tuple[Tuple[int, int, int, int], int], float]:
    outs, b1, b2, b3 = state
    run = 1 if (b1 and b2 and b3) else 0
    nb3 = 1 if (b3 or (b1 and b2)) else 0
    nb2 = 1 if b1 else b2
    nb1 = 1
    return {((outs, nb1, nb2, nb3), run): 1.0}


def _advance_single(state: Tuple[int, int, int, int]) -> Dict[Tuple[Tuple[int, int, int, int], int], float]:
    outs, b1, b2, b3 = state
    out = defaultdict(float)
    for second_scores in ([0, 1] if b2 else [0]):
        p_second = SINGLE_2B_SCORE if second_scores else (1.0 - SINGLE_2B_SCORE)
        if not b2:
            p_second = 1.0
        for first_to_third in ([0, 1] if b1 else [0]):
            p_first = SINGLE_1B_TO_3B if first_to_third else (1.0 - SINGLE_1B_TO_3B)
            if not b1:
                p_first = 1.0
            prob = p_second * p_first
            runs = int(b3) + int(second_scores and b2)
            nb3 = 1 if ((b2 and not second_scores) or (b1 and first_to_third)) else 0
            nb2 = 1 if (b1 and not first_to_third) else 0
            nb1 = 1
            out[((outs, nb1, nb2, nb3), runs)] += prob
    return dict(out)


def _advance_double(state: Tuple[int, int, int, int]) -> Dict[Tuple[Tuple[int, int, int, int], int], float]:
    outs, b1, b2, b3 = state
    out = defaultdict(float)
    for first_scores in ([0, 1] if b1 else [0]):
        p_first = DOUBLE_1B_SCORE if first_scores else (1.0 - DOUBLE_1B_SCORE)
        if not b1:
            p_first = 1.0
        runs = int(b3) + int(b2) + int(first_scores and b1)
        nb3 = 1 if (b1 and not first_scores) else 0
        nb2 = 1  # batter
        nb1 = 0
        out[((outs, nb1, nb2, nb3), runs)] += p_first
    return dict(out)


def _advance_triple(state: Tuple[int, int, int, int]) -> Dict[Tuple[Tuple[int, int, int, int], int], float]:
    outs, b1, b2, b3 = state
    runs = int(b1) + int(b2) + int(b3)
    return {((outs, 0, 0, 1), runs): 1.0}


def _advance_hr(state: Tuple[int, int, int, int]) -> Dict[Tuple[Tuple[int, int, int, int], int], float]:
    outs, b1, b2, b3 = state
    runs = 1 + int(b1) + int(b2) + int(b3)
    return {((outs, 0, 0, 0), runs): 1.0}


def _advance_out(state: Tuple[int, int, int, int]) -> Dict[Tuple[Tuple[int, int, int, int], int], float]:
    outs, b1, b2, b3 = state
    return {((min(outs + 1, 3), b1, b2, b3), 0): 1.0}


_ADVANCERS = {
    "BB": _advance_walk,
    "1B": _advance_single,
    "2B": _advance_double,
    "3B": _advance_triple,
    "HR": _advance_hr,
    "OUT": _advance_out,
}


def _half_inning_from_start(lineup_probs: List[Dict[str, float]], start_idx: int, max_steps: int = 40):
    n = len(lineup_probs)
    dist: Dict[Tuple[int, int, int, int, int], float] = {(0, 0, 0, 0, int(start_idx) % n): 1.0}
    next_start = np.zeros(n, dtype=float)
    pa_counts = np.zeros(n, dtype=float)
    exp_runs = 0.0

    for _ in range(max_steps):
        if not dist:
            break
        newdist: Dict[Tuple[int, int, int, int, int], float] = defaultdict(float)
        live_mass = 0.0

        for (outs, b1, b2, b3, idx), mass in list(dist.items()):
            if mass < 1e-12:
                continue
            live_mass += mass
            pa_counts[idx] += mass
            state = (outs, b1, b2, b3)
            probs = lineup_probs[idx]
            next_idx = (idx + 1) % n

            for evt, p_evt in probs.items():
                if p_evt <= 0:
                    continue
                for (ns, runs), p_branch in _ADVANCERS[evt](state).items():
                    outs2, nb1, nb2, nb3 = ns
                    prob = mass * p_evt * p_branch
                    exp_runs += prob * runs
                    if outs2 >= 3:
                        next_start[next_idx] += prob
                    else:
                        newdist[(outs2, nb1, nb2, nb3, next_idx)] += prob

        dist = newdist
        if live_mass < 1e-9:
            break

    total = next_start.sum()
    if total > 0:
        next_start /= total
    else:
        next_start[0] = 1.0
    return exp_runs, pa_counts, next_start


def lineup_markov_game_context(lineup_df: pd.DataFrame, innings: int = 9) -> pd.DataFrame:
    if not isinstance(lineup_df, pd.DataFrame) or lineup_df.empty:
        return pd.DataFrame()
    df = lineup_df.sort_values("batting_order", kind="stable").reset_index(drop=True).copy()
    lineup_probs = [_row_event_probs(r) for _, r in df.iterrows()]
    n = len(lineup_probs)
    if n == 0:
        return df

    start_dist = np.zeros(n, dtype=float)
    start_dist[0] = 1.0
    total_runs = 0.0
    pa_totals = np.zeros(n, dtype=float)

    for _ in range(int(max(innings, 1))):
        new_start = np.zeros(n, dtype=float)
        for idx, p_start in enumerate(start_dist):
            if p_start <= 0:
                continue
            runs_i, pa_i, next_i = _half_inning_from_start(lineup_probs, idx)
            total_runs += p_start * runs_i
            pa_totals += p_start * pa_i
            new_start += p_start * next_i
        s = new_start.sum()
        start_dist = (new_start / s) if s > 0 else np.r_[1.0, np.zeros(n - 1)]

    mean_pa = float(np.mean(pa_totals)) if np.mean(pa_totals) > 0 else 4.2
    team_env_mult = float(np.clip(total_runs / 4.4, 0.75, 1.30))
    slot_pa_mult = np.clip(pa_totals / max(mean_pa, 1e-9), 0.85, 1.18)

    run_slot_bias = np.array([1.06, 1.04, 1.00, 0.97, 0.95, 0.97, 1.00, 1.02, 1.03])[:n]
    rbi_slot_bias = np.array([0.90, 0.96, 1.05, 1.08, 1.07, 1.02, 0.97, 0.93, 0.90])[:n]

    run_ctx = np.clip(team_env_mult * slot_pa_mult * run_slot_bias, 0.82, 1.22)
    rbi_ctx = np.clip(team_env_mult * slot_pa_mult * rbi_slot_bias, 0.82, 1.24)
    hr_ctx = np.clip(1.0 + 0.42 * (slot_pa_mult - 1.0) + 0.22 * (team_env_mult - 1.0), 0.88, 1.14)

    out = df.copy()
    out["markov_team_exp_runs"] = float(total_runs)
    out["markov_slot_exp_pa"] = pa_totals
    out["markov_team_env_mult"] = float(team_env_mult)
    out["markov_run_context_mult"] = run_ctx
    out["markov_rbi_context_mult"] = rbi_ctx
    out["markov_hr_context_mult"] = hr_ctx
    return out


def apply_markov_context_to_batter_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if not isinstance(df, pd.DataFrame) else df
    out = df.copy()
    needed = {"matchup", "team", "batting_order"}
    if not needed.issubset(out.columns):
        return out

    parts = []
    for _, sub in out.groupby(["matchup", "team"], sort=False):
        try:
            ctx = lineup_markov_game_context(sub)
            parts.append(ctx)
        except Exception:
            sub = sub.copy()
            sub["markov_team_exp_runs"] = np.nan
            sub["markov_slot_exp_pa"] = np.nan
            sub["markov_team_env_mult"] = np.nan
            sub["markov_run_context_mult"] = np.nan
            sub["markov_rbi_context_mult"] = np.nan
            sub["markov_hr_context_mult"] = np.nan
            parts.append(sub)

    merged = pd.concat(parts, ignore_index=True)
    if "exp_hr" in merged.columns and "markov_hr_context_mult" in merged.columns:
        merged["exp_hr_ctx"] = pd.to_numeric(merged["exp_hr"], errors="coerce") * pd.to_numeric(merged["markov_hr_context_mult"], errors="coerce").fillna(1.0)
    return merged
