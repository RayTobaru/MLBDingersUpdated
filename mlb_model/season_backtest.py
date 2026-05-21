from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from .predict_pitcher_ko import predict_slate_ko
from .predict_batter_outcomes import predict_slate_batters
from .shared_game_utils import fetch_matchups_for_date
from .reporting import save_report_bundle, slate_prediction_report

__all__ = ["run_season_replay"]


def _daterange(start_date: str, end_date: str):
    cur = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    while cur <= end:
        yield cur.isoformat()
        cur += timedelta(days=1)


def run_season_replay(
    start_date: str,
    end_date: str,
    outdir: str | Path,
    override_path: str | None = None,
    *,
    save_reports: bool = True,
) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for d in _daterange(start_date, end_date):
        matchups = fetch_matchups_for_date(d)
        if not matchups:
            continue

        ko = predict_slate_ko(d, override_path=override_path)
        bat = predict_slate_batters(d, override_path=override_path)

        ko_path = outdir / f"ko_slate_{d}.csv"
        bat_path = outdir / f"batters_slate_{d}.csv"

        if isinstance(ko, pd.DataFrame) and not ko.empty:
            ko.to_csv(ko_path, index=False)
        if isinstance(bat, pd.DataFrame) and not bat.empty:
            bat.to_csv(bat_path, index=False)

        report_paths = {}
        if save_reports:
            try:
                bundle = slate_prediction_report(d, override_path=override_path)
                report_paths = save_report_bundle(bundle, outdir, f"slate_{d}")
            except Exception:
                report_paths = {}

        rows.append(
            {
                "date": d,
                "games": len(matchups),
                "ko_rows": int(len(ko)) if isinstance(ko, pd.DataFrame) else 0,
                "batter_rows": int(len(bat)) if isinstance(bat, pd.DataFrame) else 0,
                "ko_file": str(ko_path) if ko_path.exists() else "",
                "batters_file": str(bat_path) if bat_path.exists() else "",
                "report_file": str(report_paths.get("report", "")) if report_paths else "",
                "ko_compact_file": str(report_paths.get("ko_compact", "")) if report_paths else "",
                "batters_compact_file": str(report_paths.get("batters_compact", "")) if report_paths else "",
                "top_hr_file": str(report_paths.get("top_hr", "")) if report_paths else "",
                "top_hits_file": str(report_paths.get("top_hits", "")) if report_paths else "",
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("date", kind="stable")
        out.to_csv(outdir / "season_replay_summary.csv", index=False)
    return out
