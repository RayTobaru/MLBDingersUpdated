
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlb_model.ko_ladder_calibration import (
    THRESHOLDS,
    apply_ko_ladder_calibrators,
    fit_all_ko_ladder_calibrators,
    load_calibrators,
    save_calibrators,
)


def main():
    ap = argparse.ArgumentParser(description="Fit and apply per-threshold KO ladder calibrators.")
    ap.add_argument("--history", default=r"outputs\actuals_last180d_ko_filtered.csv", help="Historical KO file with raw ladder probs and actual_k_geX columns")
    ap.add_argument("--apply-to", default=None, help="Target KO file to calibrate; defaults to the same history file")
    ap.add_argument("--outdir", default=r"outputs\calibration_test", help="Output folder")
    ap.add_argument("--calibrator-path", default=r"outputs\calibration_test\ko_ladder_calibrators.pkl", help="Where to save the calibrators")
    args = ap.parse_args()

    hist_path = Path(args.history)
    apply_path = Path(args.apply_to) if args.apply_to else hist_path
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hist = pd.read_csv(hist_path)
    calibrators, summary = fit_all_ko_ladder_calibrators(hist, THRESHOLDS)
    save_calibrators(calibrators, args.calibrator_path)

    # write fitted history with calibrated columns
    hist_cal = apply_ko_ladder_calibrators(hist, calibrators)
    hist_cal_path = outdir / f"{hist_path.stem}_with_calibrated_ladders.csv"
    hist_cal.to_csv(hist_cal_path, index=False)

    # apply to target
    target = pd.read_csv(apply_path)
    target_cal = apply_ko_ladder_calibrators(target, calibrators)
    target_cal_path = outdir / f"{apply_path.stem}_calibrated.csv"
    target_cal.to_csv(target_cal_path, index=False)

    summary_path = outdir / "ko_ladder_calibration_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"[wrote] {Path(args.calibrator_path).resolve()}")
    print(f"[wrote] {summary_path.resolve()}")
    print(f"[wrote] {hist_cal_path.resolve()}")
    print(f"[wrote] {target_cal_path.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
