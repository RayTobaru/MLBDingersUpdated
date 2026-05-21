
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlb_model.ko_ladder_calibration import apply_ko_ladder_calibrators, load_calibrators


def main():
    ap = argparse.ArgumentParser(description="Apply previously fitted KO ladder calibrators to a KO CSV.")
    ap.add_argument("--input", required=True, help="KO csv file to calibrate")
    ap.add_argument("--calibrators", default=r"outputs\calibration_test\ko_ladder_calibrators.pkl", help="Path to saved calibrators")
    ap.add_argument("--outdir", default=r"outputs\calibration_test", help="Output folder")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    calibrators = load_calibrators(args.calibrators)
    out = apply_ko_ladder_calibrators(df, calibrators)

    # useful summary helpers
    if "P_cal(K≥6)" in out.columns:
        out["anchor_k6"] = out["P_cal(K≥6)"]
    if "P_cal(K≥5)" in out.columns and "P_cal(K≥6)" in out.columns:
        out["k5_k6_blend"] = 0.4 * out["P_cal(K≥5)"] + 0.6 * out["P_cal(K≥6)"]
    if "P_cal(K≥7)" in out.columns and "P_cal(K≥8)" in out.columns:
        out["tail_k7_k8_blend"] = 0.65 * out["P_cal(K≥7)"] + 0.35 * out["P_cal(K≥8)"]

    out_path = outdir / f"{in_path.stem}_with_saved_ko_calibration.csv"
    out.to_csv(out_path, index=False)

    compact_cols = [
        "date", "matchup", "Team", "Name", "Pitcher_ID",
        "Proj_IP", "Pred_K9", "Mean_K_start",
        "P(K≥4)", "P_cal(K≥4)",
        "P(K≥5)", "P_cal(K≥5)",
        "P(K≥6)", "P_cal(K≥6)",
        "P(K≥7)", "P_cal(K≥7)",
        "P(K≥8)", "P_cal(K≥8)",
        "anchor_k6", "k5_k6_blend", "tail_k7_k8_blend",
    ]
    compact_cols = [c for c in compact_cols if c in out.columns]
    compact = out[compact_cols].copy()
    compact_path = outdir / f"{in_path.stem}_with_saved_ko_calibration_compact.csv"
    compact.to_csv(compact_path, index=False)

    print(f"[wrote] {out_path.resolve()}")
    print(f"[wrote] {compact_path.resolve()}")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
