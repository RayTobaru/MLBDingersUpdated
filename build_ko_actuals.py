
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

K_EVENTS = {"strikeout", "strikeout_double_play"}


@dataclass
class KOFile:
    path: Path
    date: str


def _parse_ko_filename(path: Path) -> KOFile | None:
    name = path.name
    if name.endswith("_compact.csv"):
        return None
    m1 = re.match(r"^ko_(\d{4}-\d{2}-\d{2})_[A-Z0-9]+_[A-Z0-9]+\.csv$", name)
    if m1:
        return KOFile(path=path, date=m1.group(1))
    m2 = re.match(r"^ko_slate_(\d{4}-\d{2}-\d{2})\.csv$", name)
    if m2:
        return KOFile(path=path, date=m2.group(1))
    return None


def _iter_ko_prediction_files(outputs_dir: Path, days: int) -> list[KOFile]:
    files: list[KOFile] = []
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days - 1)
    for p in sorted(outputs_dir.glob("ko*.csv")):
        info = _parse_ko_filename(p)
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


def _load_ko_prediction_union(ko_files: list[KOFile]) -> pd.DataFrame:
    frames = []
    for kf in ko_files:
        try:
            df = pd.read_csv(kf.path)
        except Exception:
            continue
        if df.empty:
            continue

        df = df.copy()
        if "date" not in df.columns:
            df["date"] = kf.date
        else:
            dt = pd.to_datetime(df["date"], errors="coerce")
            df["date"] = dt.dt.strftime("%Y-%m-%d").where(dt.notna(), df["date"].astype(str))

        name_col = _pick_first_existing_col(df, ["Name", "player_name", "Pitcher", "pitcher_name"])
        if name_col is None:
            continue
        df["Name"] = df[name_col].astype(str).str.strip()

        id_col = _pick_first_existing_col(df, ["Pitcher_ID", "pitcher_id", "player_id", "Player_ID"])
        if id_col is None:
            continue
        df["Pitcher_ID"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")

        matchup_col = _pick_first_existing_col(df, ["matchup", "Matchup"])
        if matchup_col is not None:
            df["matchup"] = df[matchup_col].astype(str).str.strip()
        else:
            df["matchup"] = ""

        team_col = _pick_first_existing_col(df, ["Team", "team"])
        if team_col is not None:
            df["Team"] = df[team_col].astype(str).str.strip()
        else:
            df["Team"] = ""

        df["_source_file"] = kf.path.name
        df["_source_cols"] = len(df.columns)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "matchup", "Pitcher_ID", "_source_cols"], ascending=[True, True, True, False], kind="stable")
    out = out.drop_duplicates(subset=["date", "matchup", "Pitcher_ID"], keep="first").reset_index(drop=True)
    return out


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


def _normalize_statcast_for_ko(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    g = df.copy()
    if "game_date" in g.columns:
        g["date"] = pd.to_datetime(g["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        g["date"] = pd.NaT
    g["pitcher"] = pd.to_numeric(g.get("pitcher", pd.Series(np.nan, index=g.index)), errors="coerce").astype("Int64")
    g["events"] = g.get("events", pd.Series("", index=g.index)).astype(str).str.lower()
    g["actual_k_event"] = g["events"].isin(K_EVENTS).astype(int)
    return g


def _build_ko_actuals(pred_df: pd.DataFrame, statcast_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()

    out = pred_df.copy()
    out["Pitcher_ID"] = pd.to_numeric(out["Pitcher_ID"], errors="coerce").astype("Int64")

    sc = _normalize_statcast_for_ko(statcast_df)
    if sc.empty:
        out["actual_k"] = np.nan
    else:
        by_pid = (
            sc.groupby(["date", "pitcher"], dropna=False)
            .agg(actual_k=("actual_k_event", "sum"))
            .reset_index()
            .rename(columns={"pitcher": "Pitcher_ID"})
        )
        by_pid["Pitcher_ID"] = pd.to_numeric(by_pid["Pitcher_ID"], errors="coerce").astype("Int64")
        out = out.merge(by_pid, on=["date", "Pitcher_ID"], how="left")

    out["actual_k"] = pd.to_numeric(out.get("actual_k"), errors="coerce").fillna(0).astype(int)
    for k in [4, 5, 6, 7, 8]:
        out[f"actual_k_ge{k}"] = (out["actual_k"] >= k).astype(int)
    return out


def _make_market_template(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    keep = [c for c in [
        "date", "matchup", "Team", "Name", "Pitcher_ID",
        "P(K≥4)", "P(K≥5)", "P(K≥6)", "P(K≥7)", "P(K≥8)"
    ] if c in df.columns]
    out = df[keep].copy()
    for k in [4, 5, 6, 7, 8]:
        if f"P(K≥{k})" in out.columns:
            out[f"book_prob_k_ge{k}"] = np.nan
            out[f"open_prob_k_ge{k}"] = np.nan
            out[f"close_prob_k_ge{k}"] = np.nan
            out[f"american_odds_k_ge{k}"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description="Build strikeout actuals from saved KO outputs.")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ko_files = _iter_ko_prediction_files(outputs_dir, args.days)
    if not ko_files:
        raise SystemExit(f"No KO prediction files found in {outputs_dir} for the last {args.days} days.")

    preds = _load_ko_prediction_union(ko_files)
    if preds.empty:
        raise SystemExit("Could not load any KO prediction rows.")

    dates = set(preds["date"].dropna().astype(str).tolist())
    sc = _fetch_statcast_for_dates(dates)
    ko_actuals = _build_ko_actuals(preds, sc)

    actuals_path = outputs_dir / f"actuals_last{args.days}d_ko.csv"
    market_path = outputs_dir / f"market_template_last{args.days}d_ko.csv"
    summary_path = outputs_dir / f"actuals_build_summary_last{args.days}d_ko.csv"

    ko_actuals.to_csv(actuals_path, index=False)
    _make_market_template(ko_actuals).to_csv(market_path, index=False)

    summary = pd.DataFrame([{
        "days": int(args.days),
        "prediction_files_found": int(len(ko_files)),
        "rows_ko": int(len(ko_actuals)),
        "nonnull_pitcher_id": int(pd.to_numeric(ko_actuals.get("Pitcher_ID"), errors="coerce").notna().sum()) if "Pitcher_ID" in ko_actuals.columns else 0,
        "sum_actual_k": float(pd.to_numeric(ko_actuals.get("actual_k"), errors="coerce").fillna(0).sum()) if "actual_k" in ko_actuals.columns else 0.0,
        "rows_k_ge4": int(pd.to_numeric(ko_actuals.get("actual_k_ge4"), errors="coerce").fillna(0).sum()) if "actual_k_ge4" in ko_actuals.columns else 0,
        "dates_covered": ", ".join(sorted(dates)[:10]) + (" ..." if len(dates) > 10 else ""),
    }])
    summary.to_csv(summary_path, index=False)

    print(f"[wrote] {actuals_path.resolve()}")
    print(f"[wrote] {market_path.resolve()}")
    print(f"[wrote] {summary_path.resolve()}")


if __name__ == "__main__":
    main()
