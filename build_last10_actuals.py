
from __future__ import annotations

import argparse
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _load_fetch_module():
    candidates = ["fetch", "legacy.fetch"]
    last_exc = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:
            last_exc = exc
    raise ModuleNotFoundError(
        "Could not import fetch helpers from either 'fetch' or 'legacy.fetch'."
    ) from last_exc


_fetch = _load_fetch_module()
fetch_statcast_raw = getattr(_fetch, "fetch_statcast_raw")
clean_name = getattr(_fetch, "clean_name")
strip_pos = getattr(_fetch, "strip_pos")
name_to_mlbam_id = getattr(_fetch, "name_to_mlbam_id")

HIT_EVENTS = {"single", "double", "triple", "home_run"}
HR_EVENT = "home_run"


@dataclass
class PredFile:
    path: Path
    date: str
    matchup: str


def _parse_full_batter_filename(path: Path) -> PredFile | None:
    # full files only: batters_YYYY-MM-DD_AWAY_HOME.csv
    m = re.match(r"^batters_(\d{4}-\d{2}-\d{2})_([A-Z0-9]+)_([A-Z0-9]+)\.csv$", path.name)
    if not m:
        return None
    date_s, away, home = m.groups()
    return PredFile(path=path, date=date_s, matchup=f"{away}@{home}")


def _iter_full_prediction_files(outputs_dir: Path, days: int) -> list[PredFile]:
    files: list[PredFile] = []
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days - 1)
    for p in sorted(outputs_dir.glob("batters_*.csv")):
        info = _parse_full_batter_filename(p)
        if info is None:
            continue
        dt = pd.to_datetime(info.date, errors="coerce")
        if pd.isna(dt):
            continue
        if dt.normalize() >= cutoff:
            files.append(info)
    return files


def _pick_first_existing_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def _load_full_prediction_union(pred_files: list[PredFile]) -> pd.DataFrame:
    frames = []
    for pf in pred_files:
        try:
            df = pd.read_csv(pf.path)
        except Exception:
            continue
        if df.empty:
            continue

        df = df.copy()
        df["date"] = pf.date
        df["matchup"] = pf.matchup

        name_col = _pick_first_existing_col(df, ["player_name", "Name", "name"])
        if name_col is None:
            continue
        df["player_name"] = df[name_col].astype(str).map(strip_pos)
        df["name_norm"] = df["player_name"].map(clean_name)

        id_col = _pick_first_existing_col(
            df,
            ["player_id", "Player_ID", "batter_id", "Batter_ID", "mlbam_id", "batter_mlbam_id", "player_mlbam_id"],
        )
        if id_col is not None:
            df["player_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
        else:
            df["player_id"] = df["name_norm"].map(name_to_mlbam_id)
            df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")

        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _normalize_events_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    g = df.copy()
    if "game_date" in g.columns:
        g["date"] = pd.to_datetime(g["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        g["date"] = pd.NaT

    g["events"] = g.get("events", pd.Series("", index=g.index)).astype(str).str.lower()
    g["batter"] = pd.to_numeric(g.get("batter", pd.Series(np.nan, index=g.index)), errors="coerce").astype("Int64")
    g["is_pa"] = g["events"].notna().astype(int)
    g["actual_1b"] = g["events"].eq("single").astype(int)
    g["actual_2b"] = g["events"].eq("double").astype(int)
    g["actual_3b"] = g["events"].eq("triple").astype(int)
    g["actual_hr_count"] = g["events"].eq(HR_EVENT).astype(int)
    g["actual_hit_event"] = g["events"].isin(HIT_EVENTS).astype(int)
    g["actual_tb"] = g["actual_1b"] + 2 * g["actual_2b"] + 3 * g["actual_3b"] + 4 * g["actual_hr_count"]
    return g


def _fetch_statcast_for_dates(dates: Iterable[str]) -> pd.DataFrame:
    frames = []
    for d in sorted(set(dates)):
        try:
            sc = fetch_statcast_raw(d, d)
        except Exception:
            sc = None
        if isinstance(sc, pd.DataFrame) and not sc.empty:
            frames.append(sc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_actuals_from_full(pred_df: pd.DataFrame, statcast_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()

    out = pred_df.copy()
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")

    sc = _normalize_events_frame(statcast_df)
    if sc.empty:
        for c in [
            "actual_pa", "actual_hits_total", "actual_hit", "actual_1b", "actual_2b",
            "actual_3b", "actual_hr_count", "actual_hr", "actual_tb"
        ]:
            out[c] = np.nan
        return out

    by_pid = (
        sc.groupby(["date", "batter"], dropna=False)
        .agg(
            actual_pa=("is_pa", "sum"),
            actual_hits_total=("actual_hit_event", "sum"),
            actual_1b=("actual_1b", "sum"),
            actual_2b=("actual_2b", "sum"),
            actual_3b=("actual_3b", "sum"),
            actual_hr_count=("actual_hr_count", "sum"),
            actual_tb=("actual_tb", "sum"),
        )
        .reset_index()
        .rename(columns={"batter": "player_id"})
    )
    by_pid["player_id"] = pd.to_numeric(by_pid["player_id"], errors="coerce").astype("Int64")

    merged = out.merge(by_pid, on=["date", "player_id"], how="left")
    merged["actual_hit"] = (pd.to_numeric(merged["actual_hits_total"], errors="coerce").fillna(0) > 0).astype(int)
    merged["actual_hr"] = (pd.to_numeric(merged["actual_hr_count"], errors="coerce").fillna(0) > 0).astype(int)
    return merged


def _make_hr_eval_file(full_actuals: pd.DataFrame) -> pd.DataFrame:
    want = [
        "date", "matchup", "player_name", "player_id",
        "P(HR>=1)", "exp_hr",
        "actual_pa", "actual_hits_total", "actual_hit",
        "actual_1b", "actual_2b", "actual_3b", "actual_hr_count", "actual_hr", "actual_tb",
    ]
    keep = [c for c in want if c in full_actuals.columns]
    return full_actuals[keep].copy()


def _make_hits_eval_file(full_actuals: pd.DataFrame) -> pd.DataFrame:
    want = [
        "date", "matchup", "player_name", "player_id",
        "P(H>=1)", "exp_hits",
        "actual_pa", "actual_hits_total", "actual_hit",
        "actual_1b", "actual_2b", "actual_3b", "actual_hr_count", "actual_tb",
    ]
    keep = [c for c in want if c in full_actuals.columns]
    return full_actuals[keep].copy()


def _make_market_template(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c for c in ["date", "matchup", "player_name", "player_id", prob_col] if c in df.columns]
    out = df[cols].copy()
    out["book_prob"] = np.nan
    out["open_prob"] = np.nan
    out["close_prob"] = np.nan
    out["american_odds"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description="Build HR/hits actuals from FULL saved batter outputs, not top board files.")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    pred_files = _iter_full_prediction_files(outputs_dir, args.days)
    if not pred_files:
        raise SystemExit(f"No full batter prediction files found in {outputs_dir} for the last {args.days} days.")

    pred_full = _load_full_prediction_union(pred_files)
    if pred_full.empty:
        raise SystemExit("Could not load any full batter prediction rows.")

    dates = set(pred_full["date"].dropna().astype(str).tolist())
    sc = _fetch_statcast_for_dates(dates)
    full_actuals = _build_actuals_from_full(pred_full, sc)

    hr_df = _make_hr_eval_file(full_actuals)
    hit_df = _make_hits_eval_file(full_actuals)

    hr_actuals_path = outputs_dir / f"actuals_last{args.days}d_hr.csv"
    hit_actuals_path = outputs_dir / f"actuals_last{args.days}d_hits.csv"
    hr_market_template_path = outputs_dir / f"market_template_last{args.days}d_hr.csv"
    hit_market_template_path = outputs_dir / f"market_template_last{args.days}d_hits.csv"

    if not hr_df.empty:
        hr_df.to_csv(hr_actuals_path, index=False)
        _make_market_template(hr_df, "P(HR>=1)").to_csv(hr_market_template_path, index=False)
        print(f"[wrote] {hr_actuals_path.resolve()}")
        print(f"[wrote] {hr_market_template_path.resolve()}")

    if not hit_df.empty:
        hit_df.to_csv(hit_actuals_path, index=False)
        _make_market_template(hit_df, "P(H>=1)").to_csv(hit_market_template_path, index=False)
        print(f"[wrote] {hit_actuals_path.resolve()}")
        print(f"[wrote] {hit_market_template_path.resolve()}")

    summary = pd.DataFrame([{
        "days": int(args.days),
        "prediction_files_found": int(len(pred_files)),
        "rows_full": int(len(full_actuals)),
        "rows_hr": int(len(hr_df)),
        "rows_hits": int(len(hit_df)),
        "matched_nonnull_pa": int(pd.to_numeric(full_actuals.get("actual_pa"), errors="coerce").notna().sum()) if "actual_pa" in full_actuals.columns else 0,
        "sum_actual_hr": float(pd.to_numeric(full_actuals.get("actual_hr"), errors="coerce").fillna(0).sum()) if "actual_hr" in full_actuals.columns else 0.0,
        "sum_actual_hit": float(pd.to_numeric(full_actuals.get("actual_hit"), errors="coerce").fillna(0).sum()) if "actual_hit" in full_actuals.columns else 0.0,
        "nonnull_player_id": int(pd.to_numeric(full_actuals.get("player_id"), errors="coerce").notna().sum()) if "player_id" in full_actuals.columns else 0,
        "dates_covered": ", ".join(sorted(dates)[:10]) + (" ..." if len(dates) > 10 else ""),
    }])
    summary_path = outputs_dir / f"actuals_build_summary_last{args.days}d.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[wrote] {summary_path.resolve()}")


if __name__ == "__main__":
    main()
