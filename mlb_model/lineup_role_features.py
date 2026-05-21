
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .compat import ROOT

try:
    import fetch  # type: ignore
    from fetch import clean_name, strip_pos
except Exception:  # pragma: no cover
    def clean_name(x: str) -> str:
        return str(x or "").strip().lower()
    def strip_pos(x: str) -> str:
        return str(x or "").strip()


ROLE_BUCKET_ORDER = {"top": 0, "heart": 1, "bridge": 2, "bottom": 3}


def role_bucket(slot) -> str:
    try:
        s = int(round(float(slot)))
    except Exception:
        return "unknown"
    if s <= 3:
        return "top"
    if s <= 5:
        return "heart"
    if s == 6:
        return "bridge"
    return "bottom"


def _history_outputs_dir(outputs_dir: str | Path | None = None) -> Path:
    return Path(outputs_dir) if outputs_dir else (ROOT / "outputs")


def _is_full_batter_output(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith(".csv"):
        return False
    if not name.startswith("batters"):
        return False
    banned = [
        "_compact", "_top_hr", "_top_hits", "_top_xbh", "_team_totals",
        "actuals_", "market_template", "calibration_test", "breakout",
    ]
    return not any(b in name for b in banned)


def _safe_read_batter_history(path: Path) -> pd.DataFrame:
    want = [
        "date", "team", "player_name", "player_id", "batting_order",
        "lineup_source", "lineup_weight", "lineup_certainty",
        "P(H>=1)", "P(HR>=1)", "exp_hits", "exp_hr",
    ]
    try:
        df = pd.read_csv(path, usecols=lambda c: c in want)
    except Exception:
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
        keep = [c for c in want if c in df.columns]
        df = df[keep].copy()
    return df


@lru_cache(maxsize=32)
def lineup_role_history(as_of_date: str, lookback_days: int = 120, outputs_dir: str | None = None) -> pd.DataFrame:
    outdir = _history_outputs_dir(outputs_dir)
    if not outdir.exists():
        return pd.DataFrame()

    asof = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(asof):
        asof = pd.Timestamp.today().normalize()
    start = asof - pd.Timedelta(days=int(lookback_days))

    frames: list[pd.DataFrame] = []
    for p in sorted(outdir.glob("batters*.csv")):
        if not _is_full_batter_output(p):
            continue
        df = _safe_read_batter_history(p)
        if df.empty:
            continue
        if "date" not in df.columns or "team" not in df.columns or "player_name" not in df.columns or "batting_order" not in df.columns:
            continue
        dt = pd.to_datetime(df["date"], errors="coerce")
        mask = dt.notna() & (dt < asof) & (dt >= start)
        if not mask.any():
            continue
        use = df.loc[mask].copy()
        use["date"] = dt.loc[mask].dt.strftime("%Y-%m-%d")
        use["team"] = use["team"].astype(str).str.upper().str.strip()
        use["player_name"] = use["player_name"].astype(str).map(strip_pos)
        use["player_key"] = use["player_name"].map(clean_name)
        use["player_id"] = pd.to_numeric(use.get("player_id"), errors="coerce").astype("Int64")
        use["batting_order"] = pd.to_numeric(use["batting_order"], errors="coerce")
        use = use[use["batting_order"].between(1, 9, inclusive="both")].copy()
        if use.empty:
            continue
        ls = use.get("lineup_source", pd.Series("", index=use.index)).fillna("").astype(str).str.lower()
        lw = pd.to_numeric(use.get("lineup_weight"), errors="coerce").fillna(0.84)
        lc = pd.to_numeric(use.get("lineup_certainty"), errors="coerce").fillna(lw)
        confirmed_bonus = np.where(ls.eq("confirmed"), 1.0, np.where(ls.str.contains("hybrid"), 0.65, 0.45))
        use["hist_weight"] = np.clip(lc * confirmed_bonus, 0.15, 1.0)
        use["role_bucket_hist"] = use["batting_order"].apply(role_bucket)
        use["hr_proxy"] = pd.to_numeric(use.get("P(HR>=1)"), errors="coerce").fillna(pd.to_numeric(use.get("exp_hr"), errors="coerce").fillna(0.0))
        use["hit_proxy"] = pd.to_numeric(use.get("P(H>=1)"), errors="coerce").fillna(pd.to_numeric(use.get("exp_hits"), errors="coerce").fillna(0.0))
        frames.append(use)

    if not frames:
        return pd.DataFrame()

    hist = pd.concat(frames, ignore_index=True)
    grp = hist.groupby(["team", "player_key"], dropna=False)

    rows: list[dict] = []
    for (team, player_key), g in grp:
        w = pd.to_numeric(g["hist_weight"], errors="coerce").fillna(0.25).to_numpy(dtype=float)
        slots = pd.to_numeric(g["batting_order"], errors="coerce").to_numpy(dtype=float)
        if len(slots) == 0 or np.nansum(w) <= 0:
            continue
        mean_slot = float(np.average(slots, weights=w))
        var_slot = float(np.average((slots - mean_slot) ** 2, weights=w))
        std_slot = float(np.sqrt(max(var_slot, 0.0)))
        top_rate = float(np.average((slots <= 3).astype(float), weights=w))
        heart_rate = float(np.average(((slots >= 4) & (slots <= 5)).astype(float), weights=w))
        bottom_rate = float(np.average((slots >= 7).astype(float), weights=w))
        bridge_rate = float(np.average((slots == 6).astype(float), weights=w))
        role_scores = {
            "top": top_rate,
            "heart": heart_rate,
            "bridge": bridge_rate,
            "bottom": bottom_rate,
        }
        role_usual = max(role_scores.items(), key=lambda kv: (kv[1], -ROLE_BUCKET_ORDER[kv[0]]))[0]
        last_slot = float(pd.to_numeric(g["batting_order"], errors="coerce").dropna().iloc[-1]) if len(g) else mean_slot
        pid_vals = pd.to_numeric(g.get("player_id"), errors="coerce").dropna()
        pid = int(pid_vals.mode().iloc[0]) if len(pid_vals) else None
        rows.append({
            "team": team,
            "player_key": player_key,
            "player_id": pid,
            "hist_rows": int(len(g)),
            "hist_weight_sum": float(np.nansum(w)),
            "usual_batting_slot": mean_slot,
            "slot_volatility": std_slot,
            "usual_role_bucket": role_usual,
            "top_rate": top_rate,
            "heart_rate": heart_rate,
            "bridge_rate": bridge_rate,
            "bottom_rate": bottom_rate,
            "last_slot": last_slot,
            "hr_proxy_hist": float(np.average(pd.to_numeric(g["hr_proxy"], errors="coerce").fillna(0.0), weights=w)),
            "hit_proxy_hist": float(np.average(pd.to_numeric(g["hit_proxy"], errors="coerce").fillna(0.0), weights=w)),
        })
    return pd.DataFrame(rows)


def lookup_lineup_role_features(
    team: str,
    player_name: str,
    player_id,
    current_slot,
    as_of_date: str,
    lineup_source: str = "projected",
    lineup_certainty: float = 0.84,
    lookback_days: int = 120,
    outputs_dir: str | None = None,
) -> dict[str, float | int | str]:
    hist = lineup_role_history(as_of_date, lookback_days=lookback_days, outputs_dir=outputs_dir)
    team = str(team or "").upper().strip()
    player_key = clean_name(strip_pos(player_name))
    current_slot = float(pd.to_numeric(pd.Series([current_slot]), errors="coerce").iloc[0]) if pd.notna(pd.to_numeric(pd.Series([current_slot]), errors="coerce").iloc[0]) else 9.0
    cur_bucket = role_bucket(current_slot)

    row = None
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        pid_num = pd.to_numeric(pd.Series([player_id]), errors="coerce").iloc[0]
        if pd.notna(pid_num) and "player_id" in hist.columns:
            hit = hist[(hist["team"] == team) & (pd.to_numeric(hist["player_id"], errors="coerce") == int(pid_num))]
            if not hit.empty:
                row = hit.iloc[0]
        if row is None:
            hit = hist[(hist["team"] == team) & (hist["player_key"] == player_key)]
            if not hit.empty:
                row = hit.iloc[0]

    hist_n = int(row["hist_rows"]) if row is not None else 0
    hist_weight = float(row["hist_weight_sum"]) if row is not None else 0.0
    usual_slot = float(row["usual_batting_slot"]) if row is not None else current_slot
    usual_bucket = str(row["usual_role_bucket"]) if row is not None else cur_bucket
    slot_vol = float(row["slot_volatility"]) if row is not None else 0.0

    slot_delta = float(current_slot - usual_slot)
    role_changed = int(cur_bucket != usual_bucket or abs(slot_delta) >= 0.75)
    confirmed_change = int(str(lineup_source or "").lower() == "confirmed" and role_changed == 1)

    conf = float(np.clip(lineup_certainty, 0.0, 1.0))
    hist_conf = float(np.clip(hist_weight / 6.0, 0.0, 1.0))
    role_conf = float(np.clip(conf * hist_conf, 0.0, 1.0))

    pa_bucket = {"top": 1.030, "heart": 1.000, "bridge": 0.985, "bottom": 0.955}
    hit_bucket = {"top": 1.018, "heart": 1.000, "bridge": 0.992, "bottom": 0.968}
    hr_bucket = {"top": 0.990, "heart": 1.030, "bridge": 1.000, "bottom": 0.955}

    pa_raw = pa_bucket.get(cur_bucket, 1.0) * float(np.clip(1.0 - 0.012 * slot_delta, 0.94, 1.06))
    hit_raw = hit_bucket.get(cur_bucket, 1.0) * float(np.clip(1.0 - 0.006 * slot_delta, 0.96, 1.04))
    hr_raw = hr_bucket.get(cur_bucket, 1.0)

    if cur_bucket == "heart" and usual_bucket != "heart":
        hr_raw *= 1.018
    if usual_bucket == "heart" and cur_bucket != "heart":
        hr_raw *= 0.980
    if int(round(current_slot)) == 5 and int(round(usual_slot)) == 4:
        hr_raw *= 1.010
    if int(round(current_slot)) == 4 and int(round(usual_slot)) == 5:
        hr_raw *= 1.010
    if int(round(current_slot)) <= 3 and usual_slot >= 5.0:
        hr_raw *= 0.985
    if int(round(current_slot)) >= 6 and usual_slot <= 4.0:
        hr_raw *= 0.975
    hr_raw = float(np.clip(hr_raw, 0.95, 1.06))

    pa_mult = float(np.clip(1.0 + role_conf * (pa_raw - 1.0), 0.95, 1.05))
    hit_mult = float(np.clip(1.0 + role_conf * (hit_raw - 1.0), 0.96, 1.04))
    hr_mult = float(np.clip(1.0 + role_conf * (hr_raw - 1.0), 0.95, 1.06))

    return {
        "usual_batting_slot": round(usual_slot, 2),
        "usual_role_bucket": usual_bucket,
        "current_role_bucket": cur_bucket,
        "slot_delta": round(slot_delta, 2),
        "slot_abs_delta": round(abs(slot_delta), 2),
        "role_changed_flag": int(role_changed),
        "confirmed_lineup_role_change_flag": int(confirmed_change),
        "slot_volatility": round(slot_vol, 3),
        "lineup_role_hist_n": int(hist_n),
        "lineup_role_hist_weight": round(hist_weight, 3),
        "lineup_role_confidence": round(role_conf, 3),
        "lineup_role_pa_mult": round(pa_mult, 3),
        "lineup_role_hit_mult": round(hit_mult, 3),
        "lineup_role_hr_mult": round(hr_mult, 3),
    }


def summarize_lineup_role_for_pitcher(
    opp_team: str,
    lineup_names: list[str],
    as_of_date: str,
    lineup_source: str = "projected",
    lineup_certainty: float = 0.84,
    lookback_days: int = 120,
    outputs_dir: str | None = None,
) -> dict[str, float | int]:
    rows = []
    for slot, nm in enumerate((lineup_names or [])[:9], start=1):
        feats = lookup_lineup_role_features(
            team=opp_team,
            player_name=nm,
            player_id=None,
            current_slot=slot,
            as_of_date=as_of_date,
            lineup_source=lineup_source,
            lineup_certainty=lineup_certainty,
            lookback_days=lookback_days,
            outputs_dir=outputs_dir,
        )
        rows.append({"slot": slot, **feats})

    if not rows:
        return {
            "lineup_role_k_mult": 1.0,
            "lineup_role_top5_usual_slot_mean": 4.0,
            "lineup_role_dislocation": 0.0,
            "lineup_role_hist_n": 0,
            "lineup_role_promoted_bottom_n": 0,
            "lineup_role_demoted_top_n": 0,
        }

    df = pd.DataFrame(rows)
    top5 = df[df["slot"] <= 5].copy()
    if top5.empty:
        top5 = df.copy()

    top5_usual = float(pd.to_numeric(top5["usual_batting_slot"], errors="coerce").fillna(top5["slot"]).mean())
    dislocation = float(pd.to_numeric(df["slot_abs_delta"], errors="coerce").fillna(0.0).mean())
    hist_n = int(pd.to_numeric(df["lineup_role_hist_n"], errors="coerce").fillna(0).sum())
    promoted_bottom_n = int(((pd.to_numeric(df["usual_batting_slot"], errors="coerce").fillna(df["slot"]) >= 7.0) & (df["slot"] <= 5)).sum())
    demoted_top_n = int(((pd.to_numeric(df["usual_batting_slot"], errors="coerce").fillna(df["slot"]) <= 4.0) & (df["slot"] >= 6)).sum())

    conf = float(np.clip(lineup_certainty, 0.0, 1.0))
    hist_conf = float(np.clip(hist_n / 30.0, 0.0, 1.0))
    role_conf = conf * hist_conf

    # Higher top5_usual mean implies weaker top-5 than normal -> slightly easier K environment.
    raw_k_mult = 1.0 + 0.015 * (top5_usual - 4.0) + 0.010 * promoted_bottom_n - 0.008 * demoted_top_n
    raw_k_mult = float(np.clip(raw_k_mult, 0.96, 1.05))
    k_mult = float(np.clip(1.0 + role_conf * (raw_k_mult - 1.0), 0.97, 1.04))

    return {
        "lineup_role_k_mult": round(k_mult, 3),
        "lineup_role_top5_usual_slot_mean": round(top5_usual, 3),
        "lineup_role_dislocation": round(dislocation, 3),
        "lineup_role_hist_n": int(hist_n),
        "lineup_role_promoted_bottom_n": int(promoted_bottom_n),
        "lineup_role_demoted_top_n": int(demoted_top_n),
    }
