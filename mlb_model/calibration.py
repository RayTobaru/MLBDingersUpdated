
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd


def _load_any_table(pathlike) -> pd.DataFrame:
    p = Path(pathlike)
    if not p.exists():
        raise FileNotFoundError(p)
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf in {".csv", ".txt"}:
        return pd.read_csv(p)
    if suf in {".pkl", ".pickle"}:
        return pd.read_pickle(p)
    return pd.read_csv(p)


def _normalize_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        name = str(c).lower()
        if name in {"date", "game_date", "as_of_date", "event_date"}:
            dt = pd.to_datetime(out[c], errors="coerce")
            out[c] = dt.dt.strftime("%Y-%m-%d").where(dt.notna(), out[c].astype(str))
    return out


def _coerce_prob(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.clip(1e-6, 1 - 1e-6)


def _coerce_binary(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.clip(0, 1)


def _implied_prob_from_american(odds) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if not np.isfinite(o) or o == 0:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def _brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def reliability_table(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    bins: int = 10,
    by: str | None = None,
) -> pd.DataFrame:
    use = df[[pred_col, actual_col] + ([by] if by and by in df.columns else [])].copy()
    use[pred_col] = _coerce_prob(use[pred_col])
    use[actual_col] = _coerce_binary(use[actual_col])
    use = use.dropna(subset=[pred_col, actual_col])

    if use.empty:
        return pd.DataFrame()

    use["bucket"] = pd.cut(use[pred_col], bins=np.linspace(0, 1, bins + 1), include_lowest=True)

    group_cols = ["bucket"] + ([by] if by and by in use.columns else [])
    rel = (
        use.groupby(group_cols, dropna=False)
        .agg(
            n=(actual_col, "size"),
            pred_mean=(pred_col, "mean"),
            actual_mean=(actual_col, "mean"),
            pred_var=(pred_col, "var"),
        )
        .reset_index()
    )
    rel["calibration_gap"] = rel["pred_mean"] - rel["actual_mean"]
    return rel


def expected_calibration_error(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    bins: int = 10,
) -> float:
    rel = reliability_table(df, pred_col, actual_col, bins=bins)
    if rel.empty:
        return np.nan
    w = rel["n"] / rel["n"].sum()
    return float((w * (rel["pred_mean"] - rel["actual_mean"]).abs()).sum())


def brier_decomposition(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    bins: int = 10,
) -> dict:
    use = df[[pred_col, actual_col]].copy()
    use[pred_col] = _coerce_prob(use[pred_col])
    use[actual_col] = _coerce_binary(use[actual_col])
    use = use.dropna(subset=[pred_col, actual_col])
    if use.empty:
        return {"uncertainty": np.nan, "resolution": np.nan, "reliability": np.nan, "brier": np.nan}

    rel = reliability_table(use, pred_col, actual_col, bins=bins)
    base_rate = float(use[actual_col].mean())
    total_n = float(len(use))

    uncertainty = base_rate * (1.0 - base_rate)
    resolution = float((((rel["actual_mean"] - base_rate) ** 2) * rel["n"] / total_n).sum())
    reliability = float((((rel["pred_mean"] - rel["actual_mean"]) ** 2) * rel["n"] / total_n).sum())
    brier = _brier_score(use[actual_col].values.astype(float), use[pred_col].values.astype(float))

    return {
        "uncertainty": float(uncertainty),
        "resolution": float(resolution),
        "reliability": float(reliability),
        "brier": float(brier),
    }


def sharpness_diagnostics(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    bins: int = 10,
) -> pd.DataFrame:
    use = df[[pred_col, actual_col]].copy()
    use[pred_col] = _coerce_prob(use[pred_col])
    use[actual_col] = _coerce_binary(use[actual_col])
    use = use.dropna(subset=[pred_col, actual_col])
    if use.empty:
        return pd.DataFrame([{
            "rows": 0,
            "avg_pred": np.nan,
            "pred_var": np.nan,
            "pred_std": np.nan,
            "brier": np.nan,
            "logloss": np.nan,
            "ece": np.nan,
            "uncertainty": np.nan,
            "resolution": np.nan,
            "reliability": np.nan,
        }])

    decomp = brier_decomposition(use, pred_col, actual_col, bins=bins)
    return pd.DataFrame([{
        "rows": int(len(use)),
        "avg_pred": float(use[pred_col].mean()),
        "pred_var": float(use[pred_col].var(ddof=0)),
        "pred_std": float(use[pred_col].std(ddof=0)),
        "brier": _brier_score(use[actual_col].values.astype(float), use[pred_col].values.astype(float)),
        "logloss": _log_loss(use[actual_col].values.astype(float), use[pred_col].values.astype(float)),
        "ece": expected_calibration_error(use, pred_col, actual_col, bins=bins),
        **decomp,
    }])


def purged_embargo_time_series_splits(
    df: pd.DataFrame,
    date_col: str,
    n_splits: int = 5,
    purge_days: int = 1,
    embargo_days: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if date_col not in df.columns:
        raise KeyError(f"{date_col} not in DataFrame")

    dates = pd.to_datetime(df[date_col], errors="coerce")
    use = df.copy()
    use["_cv_date"] = dates
    use = use.dropna(subset=["_cv_date"]).sort_values("_cv_date", kind="stable").reset_index()

    uniq_dates = np.array(sorted(use["_cv_date"].dt.normalize().unique()))
    if len(uniq_dates) < n_splits + 1:
        return iter(())

    fold_edges = np.linspace(0, len(uniq_dates), n_splits + 1, dtype=int)

    for i in range(1, len(fold_edges)):
        test_dates = uniq_dates[fold_edges[i - 1]: fold_edges[i]]
        if len(test_dates) == 0:
            continue

        test_start = pd.Timestamp(test_dates[0])
        train_mask = use["_cv_date"] < (test_start - pd.Timedelta(days=purge_days))

        train_idx = use.loc[train_mask, "index"].to_numpy()
        test_idx = use.loc[use["_cv_date"].dt.normalize().isin(test_dates), "index"].to_numpy()

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def walkforward_segment_report(
    df: pd.DataFrame,
    date_col: str,
    pred_col: str,
    actual_col: str,
    n_splits: int = 5,
    purge_days: int = 1,
    embargo_days: int = 1,
) -> pd.DataFrame:
    if date_col not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])

    if tmp.empty or tmp[date_col].dt.normalize().nunique() < 2:
        return pd.DataFrame()

    rows = []
    for fold_num, (_, te) in enumerate(
        purged_embargo_time_series_splits(tmp, date_col, n_splits=n_splits, purge_days=purge_days, embargo_days=embargo_days),
        start=1,
    ):
        fold = tmp.loc[te].copy()
        if fold.empty:
            continue
        sharp = sharpness_diagnostics(fold, pred_col, actual_col).iloc[0].to_dict()
        sharp["fold"] = int(fold_num)
        sharp["test_start"] = str(pd.to_datetime(fold[date_col]).min().date())
        sharp["test_end"] = str(pd.to_datetime(fold[date_col]).max().date())
        rows.append(sharp)
    return pd.DataFrame(rows)


def resegment_name(col: str) -> str:
    return (
        str(col)
        .replace("(", "")
        .replace(")", "")
        .replace(">=", "ge")
        .replace("≥", "ge")
        .replace(" ", "_")
        .replace("/", "_")
    )


def add_default_segments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "lineup_source" in out.columns:
        out["seg_lineup_source"] = out["lineup_source"].fillna("unknown").astype(str)
    if "lineup_certainty" in out.columns:
        lc = pd.to_numeric(out["lineup_certainty"], errors="coerce")
        out["seg_lineup_certainty"] = pd.cut(
            lc,
            bins=[-np.inf, 0.80, 0.95, np.inf],
            labels=["low", "medium", "high"]
        ).astype(str)
    if "batter_hand" in out.columns:
        out["seg_handedness"] = out["batter_hand"].fillna("unknown").astype(str)
    elif "p_throws" in out.columns:
        out["seg_handedness"] = out["p_throws"].fillna("unknown").astype(str)

    for col in ["P(HR>=1)", "P(K>=6)", "P(K≥6)", "P(K≥5)", "P(H>=1)"]:
        if col in out.columns:
            nm = resegment_name(col)
            rank_s = pd.to_numeric(out[col], errors="coerce").rank(method="average")
            q = min(4, max(2, out[col].notna().sum() // 50 or 2))
            out[f"seg_{nm}_tier"] = pd.qcut(rank_s, q=q, duplicates="drop", labels=False)

    if "tail_wind_pct" in out.columns:
        tw = pd.to_numeric(out["tail_wind_pct"], errors="coerce").fillna(0.0)
        out["seg_wind"] = np.where(tw >= 0.10, "high_wind", "normal_wind")
    return out


def calibration_by_segments(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    segment_cols: Sequence[str],
    bins: int = 10,
) -> pd.DataFrame:
    use = df.copy()
    rows = []
    for seg in segment_cols:
        if seg not in use.columns:
            continue
        for level, sub in use.groupby(seg, dropna=False):
            sharp = sharpness_diagnostics(sub, pred_col, actual_col, bins=bins).iloc[0].to_dict()
            sharp["segment"] = seg
            sharp["level"] = str(level)
            rows.append(sharp)
    return pd.DataFrame(rows)


def edge_bucket_roi(
    df: pd.DataFrame,
    model_prob_col: str,
    actual_col: str,
    american_odds_col: str | None = None,
    book_prob_col: str | None = None,
    edge_bins: Sequence[float] = (-1.0, -0.05, -0.02, 0.0, 0.02, 0.05, 1.0),
) -> pd.DataFrame:
    use = df.copy()
    use[model_prob_col] = _coerce_prob(use[model_prob_col])
    use[actual_col] = _coerce_binary(use[actual_col])

    if book_prob_col and book_prob_col in use.columns:
        use["book_prob"] = _coerce_prob(use[book_prob_col])
    elif american_odds_col and american_odds_col in use.columns:
        use["book_prob"] = use[american_odds_col].apply(_implied_prob_from_american)
    else:
        raise ValueError("Need either book_prob_col or american_odds_col")

    use = use.dropna(subset=[model_prob_col, actual_col, "book_prob"])
    if use.empty:
        return pd.DataFrame()

    use["edge"] = use[model_prob_col] - use["book_prob"]
    use["edge_bucket"] = pd.cut(use["edge"], bins=edge_bins, include_lowest=True)

    if american_odds_col and american_odds_col in use.columns:
        odds = pd.to_numeric(use[american_odds_col], errors="coerce")
        payout = np.where(odds > 0, odds / 100.0, 100.0 / np.abs(odds))
    else:
        payout = (1.0 / use["book_prob"]) - 1.0

    use["profit"] = np.where(use[actual_col] > 0.5, payout, -1.0)

    out = (
        use.groupby("edge_bucket", dropna=False)
        .agg(
            n=(actual_col, "size"),
            avg_edge=("edge", "mean"),
            hit_rate=(actual_col, "mean"),
            avg_model_prob=(model_prob_col, "mean"),
            avg_book_prob=("book_prob", "mean"),
            total_profit=("profit", "sum"),
        )
        .reset_index()
    )
    out["roi"] = out["total_profit"] / out["n"].clip(lower=1)
    return out


def clv_summary(
    df: pd.DataFrame,
    model_prob_col: str,
    open_prob_col: str | None = None,
    close_prob_col: str | None = None,
    open_odds_col: str | None = None,
    close_odds_col: str | None = None,
) -> pd.DataFrame:
    use = df.copy()
    use[model_prob_col] = _coerce_prob(use[model_prob_col])

    if open_prob_col and open_prob_col in use.columns:
        use["open_prob"] = _coerce_prob(use[open_prob_col])
    elif open_odds_col and open_odds_col in use.columns:
        use["open_prob"] = use[open_odds_col].apply(_implied_prob_from_american)

    if close_prob_col and close_prob_col in use.columns:
        use["close_prob"] = _coerce_prob(use[close_prob_col])
    elif close_odds_col and close_odds_col in use.columns:
        use["close_prob"] = use[close_odds_col].apply(_implied_prob_from_american)

    req = ["open_prob", "close_prob", model_prob_col]
    use = use.dropna(subset=[c for c in req if c in use.columns])
    if use.empty:
        return pd.DataFrame()

    use["model_vs_open"] = use[model_prob_col] - use["open_prob"]
    use["model_vs_close"] = use[model_prob_col] - use["close_prob"]
    use["close_minus_open"] = use["close_prob"] - use["open_prob"]

    return pd.DataFrame([{
        "rows": int(len(use)),
        "avg_model_vs_open": float(use["model_vs_open"].mean()),
        "avg_model_vs_close": float(use["model_vs_close"].mean()),
        "avg_close_minus_open": float(use["close_minus_open"].mean()),
        "clv_hit_rate": float((use["model_vs_open"] > 0).mean()),
        "close_better_than_open_rate": float((use["close_minus_open"] > 0).mean()),
    }])


def drift_monitor(
    df: pd.DataFrame,
    date_col: str,
    pred_col: str,
    actual_col: str,
    book_prob_col: str | None = None,
    windows: Sequence[int] = (7, 30),
) -> pd.DataFrame:
    use = df.copy()
    if date_col not in use.columns:
        return pd.DataFrame()
    use[date_col] = pd.to_datetime(use[date_col], errors="coerce")
    use = use.dropna(subset=[date_col]).sort_values(date_col, kind="stable")
    if use.empty or use[date_col].dt.normalize().nunique() < 2:
        return pd.DataFrame()

    latest = use[date_col].max()
    rows = []

    def _one_block(label: str, sub: pd.DataFrame):
        sharp = sharpness_diagnostics(sub, pred_col, actual_col).iloc[0].to_dict()
        sharp["window"] = label
        if book_prob_col and book_prob_col in sub.columns:
            bp = _coerce_prob(sub[book_prob_col])
            mp = _coerce_prob(sub[pred_col])
            y = _coerce_binary(sub[actual_col])
            edge = mp - bp
            sharp["mean_edge"] = float(edge.mean())
            sharp["positive_edge_rate"] = float((edge > 0).mean())
            sharp["hit_rate_pos_edge"] = float(y[edge > 0].mean()) if (edge > 0).any() else np.nan
            sharp["calibration_gap"] = float(mp.mean() - y.mean())
        else:
            sharp["mean_edge"] = np.nan
            sharp["positive_edge_rate"] = np.nan
            sharp["hit_rate_pos_edge"] = np.nan
            sharp["calibration_gap"] = float(_coerce_prob(sub[pred_col]).mean() - _coerce_binary(sub[actual_col]).mean())
        return sharp

    rows.append(_one_block("season", use.copy()))
    for w in windows:
        start = latest - pd.Timedelta(days=int(w) - 1)
        sub = use.loc[use[date_col] >= start].copy()
        if not sub.empty:
            rows.append(_one_block(f"last_{int(w)}d", sub))

    return pd.DataFrame(rows)


def _prepare_eval_merge(preds: pd.DataFrame, acts: pd.DataFrame, join_keys: Sequence[str], pred_col: str, actual_col: str) -> pd.DataFrame:
    keys = [k for k in join_keys if k in preds.columns and k in acts.columns]
    if not keys:
        raise ValueError("No overlapping join keys found between predictions and actuals.")

    pred_use = preds[keys + ([pred_col] if pred_col in preds.columns else [])].copy()
    act_use = acts[keys + ([actual_col] if actual_col in acts.columns else [])].copy()

    merged = pred_use.merge(act_use, on=keys, how="inner")
    if pred_col not in merged.columns:
        raise KeyError(f"{pred_col} not found after merge. Check the prediction file and pred_col.")
    if actual_col not in merged.columns:
        raise KeyError(f"{actual_col} not found after merge. Check the actuals file and actual_col.")
    return merged.dropna(subset=[pred_col, actual_col]).copy()


def evaluate_prediction_file(
    preds_path,
    actuals_path,
    join_keys=("date", "matchup", "player_name"),
    pred_col="P(HR>=1)",
    actual_col="actual_hr",
    date_col="date",
    bins: int = 10,
) -> dict:
    preds = _normalize_date_cols(_load_any_table(preds_path))
    acts = _normalize_date_cols(_load_any_table(actuals_path))

    merged = _prepare_eval_merge(preds, acts, join_keys, pred_col, actual_col)

    seg_df = add_default_segments(merged)
    out = {
        "merged": merged,
        "sharpness": sharpness_diagnostics(merged, pred_col, actual_col, bins=bins),
        "reliability": reliability_table(merged, pred_col, actual_col, bins=bins),
        "walkforward": walkforward_segment_report(merged, date_col, pred_col, actual_col) if date_col in merged.columns else pd.DataFrame(),
        "segments": calibration_by_segments(seg_df, pred_col, actual_col, [
            "seg_lineup_source",
            "seg_lineup_certainty",
            "seg_handedness",
            "seg_PHRge1_tier",
            "seg_wind",
        ], bins=bins),
        "drift": drift_monitor(seg_df, date_col, pred_col, actual_col) if date_col in seg_df.columns else pd.DataFrame(),
    }
    return out


def evaluate_market_file(
    preds_path,
    actuals_path,
    market_path,
    join_keys=("date", "matchup", "player_name"),
    pred_col="P(HR>=1)",
    actual_col="actual_hr",
    market_prob_col="book_prob",
    open_prob_col="open_prob",
    close_prob_col="close_prob",
    date_col="date",
) -> dict:
    preds = _normalize_date_cols(_load_any_table(preds_path))
    acts = _normalize_date_cols(_load_any_table(actuals_path))
    mkt = _normalize_date_cols(_load_any_table(market_path))

    keys = [k for k in join_keys if k in preds.columns and k in acts.columns and k in mkt.columns]
    if not keys:
        raise ValueError("No overlapping join keys found across predictions, actuals, and market.")

    pred_use = preds[keys + ([pred_col] if pred_col in preds.columns else [])].copy()
    act_use = acts[keys + ([actual_col] if actual_col in acts.columns else [])].copy()

    market_keep = keys[:]
    for c in [market_prob_col, open_prob_col, close_prob_col]:
        if c in mkt.columns:
            market_keep.append(c)
    if "american_odds" in mkt.columns:
        market_keep.append("american_odds")
    mkt_use = mkt[market_keep].copy()

    merged = pred_use.merge(act_use, on=keys, how="inner").merge(mkt_use, on=keys, how="inner")
    merged = merged.dropna(subset=[pred_col, actual_col]).copy()

    seg_df = add_default_segments(merged)
    out = {
        "merged": merged,
        "sharpness": sharpness_diagnostics(merged, pred_col, actual_col),
        "segments": calibration_by_segments(seg_df, pred_col, actual_col, [
            "seg_lineup_source",
            "seg_lineup_certainty",
            "seg_handedness",
            "seg_PHRge1_tier",
            "seg_wind",
        ]),
        "edge_roi": edge_bucket_roi(merged, pred_col, actual_col, book_prob_col=market_prob_col if market_prob_col in merged.columns else None, american_odds_col="american_odds" if "american_odds" in merged.columns else None),
        "clv": clv_summary(merged, pred_col, open_prob_col=open_prob_col if open_prob_col in merged.columns else None, close_prob_col=close_prob_col if close_prob_col in merged.columns else None),
        "drift": drift_monitor(seg_df, date_col, pred_col, actual_col, book_prob_col=market_prob_col if market_prob_col in merged.columns else None) if date_col in seg_df.columns else pd.DataFrame(),
    }
    return out


def write_calibration_bundle(bundle: dict, outdir, stem: str) -> dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    for k, v in bundle.items():
        if isinstance(v, pd.DataFrame):
            p = out / f"{stem}_{k}.csv"
            v.to_csv(p, index=False)
            written[k] = str(p)

    if "sharpness" in bundle and isinstance(bundle["sharpness"], pd.DataFrame):
        summary_path = out / f"{stem}_summary.txt"
        summary_path.write_text(bundle["sharpness"].to_string(index=False), encoding="utf-8")
        written["summary"] = str(summary_path)

    return written


__all__ = [
    "reliability_table",
    "expected_calibration_error",
    "brier_decomposition",
    "sharpness_diagnostics",
    "purged_embargo_time_series_splits",
    "walkforward_segment_report",
    "add_default_segments",
    "calibration_by_segments",
    "edge_bucket_roi",
    "clv_summary",
    "drift_monitor",
    "evaluate_prediction_file",
    "evaluate_market_file",
    "write_calibration_bundle",
]
