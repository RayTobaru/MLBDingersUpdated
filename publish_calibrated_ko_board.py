
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlb_model.ko_ladder_calibration import apply_ko_ladder_calibrators, load_calibrators


def _find_latest_ko_csv(outputs_dir: Path) -> Path:
    files = [
        p for p in outputs_dir.glob("ko*.csv")
        if p.suffix.lower() == ".csv" and not p.name.endswith("_compact.csv")
    ]
    if not files:
        raise FileNotFoundError(f"No non-compact KO csv files found in {outputs_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def _add_ko_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "P_cal(K≥6)" in out.columns:
        out["anchor_k6"] = pd.to_numeric(out["P_cal(K≥6)"], errors="coerce")
    elif "P(K≥6)" in out.columns:
        out["anchor_k6"] = pd.to_numeric(out["P(K≥6)"], errors="coerce")

    if "P_cal(K≥5)" in out.columns and "P_cal(K≥6)" in out.columns:
        out["k5_k6_blend"] = (
            0.40 * pd.to_numeric(out["P_cal(K≥5)"], errors="coerce").fillna(0)
            + 0.60 * pd.to_numeric(out["P_cal(K≥6)"], errors="coerce").fillna(0)
        )
    elif "P(K≥5)" in out.columns and "P(K≥6)" in out.columns:
        out["k5_k6_blend"] = (
            0.40 * pd.to_numeric(out["P(K≥5)"], errors="coerce").fillna(0)
            + 0.60 * pd.to_numeric(out["P(K≥6)"], errors="coerce").fillna(0)
        )

    if "P_cal(K≥7)" in out.columns and "P_cal(K≥8)" in out.columns:
        out["tail_k7_k8_blend"] = (
            0.65 * pd.to_numeric(out["P_cal(K≥7)"], errors="coerce").fillna(0)
            + 0.35 * pd.to_numeric(out["P_cal(K≥8)"], errors="coerce").fillna(0)
        )
    elif "P(K≥7)" in out.columns and "P(K≥8)" in out.columns:
        out["tail_k7_k8_blend"] = (
            0.65 * pd.to_numeric(out["P(K≥7)"], errors="coerce").fillna(0)
            + 0.35 * pd.to_numeric(out["P(K≥8)"], errors="coerce").fillna(0)
        )

    if "anchor_k6" in out.columns:
        out["rank_anchor_k6"] = out["anchor_k6"].rank(method="dense", ascending=False).astype("Int64")
    if "k5_k6_blend" in out.columns:
        out["rank_balanced"] = out["k5_k6_blend"].rank(method="dense", ascending=False).astype("Int64")
    if "tail_k7_k8_blend" in out.columns:
        out["rank_tail"] = out["tail_k7_k8_blend"].rank(method="dense", ascending=False).astype("Int64")

    # simple operational label
    def _focus(row):
        try:
            mean_k = float(row.get("Mean_K_start", 0) or 0)
        except Exception:
            mean_k = 0.0
        if mean_k >= 6.0:
            return "tail"
        if mean_k >= 5.0:
            return "balanced"
        return "anchor"

    out["recommended_focus"] = out.apply(_focus, axis=1)
    return out


def _compact_board(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date", "matchup", "Team", "Name", "Pitcher_ID",
        "Proj_IP", "Pred_K9", "Mean_K_start", "Median_K_start",
        "P(K≥4)", "P_cal(K≥4)",
        "P(K≥5)", "P_cal(K≥5)",
        "P(K≥6)", "P_cal(K≥6)",
        "P(K≥7)", "P_cal(K≥7)",
        "P(K≥8)", "P_cal(K≥8)",
        "anchor_k6", "k5_k6_blend", "tail_k7_k8_blend",
        "rank_anchor_k6", "rank_balanced", "rank_tail",
        "recommended_focus",
    ]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()

    sort_col = "anchor_k6" if "anchor_k6" in out.columns else (
        "P_cal(K≥6)" if "P_cal(K≥6)" in out.columns else (
            "P(K≥6)" if "P(K≥6)" in out.columns else None
        )
    )
    if sort_col:
        out = out.sort_values(sort_col, ascending=False, kind="stable").reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Produce a report-ready calibrated KO board from a KO csv.")
    ap.add_argument("--input", default=None, help="KO csv file to calibrate; if omitted, uses latest non-compact KO file in outputs/")
    ap.add_argument("--outputs-dir", default="outputs", help="Folder containing KO outputs")
    ap.add_argument("--calibrators", default=r"outputs\calibration_test\ko_ladder_calibrators.pkl", help="Path to saved KO calibrators")
    ap.add_argument("--outdir", default=r"outputs\final_ko", help="Folder for final calibrated KO board files")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    in_path = Path(args.input) if args.input else _find_latest_ko_csv(outputs_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    calibrators = load_calibrators(args.calibrators)
    cal = apply_ko_ladder_calibrators(df, calibrators)
    cal = _add_ko_summary_columns(cal)
    compact = _compact_board(cal)

    full_path = outdir / f"{in_path.stem}_final_ko_board.csv"
    compact_path = outdir / f"{in_path.stem}_final_ko_board_compact.csv"

    cal.to_csv(full_path, index=False)
    compact.to_csv(compact_path, index=False)

    print(f"[input]  {in_path.resolve()}")
    print(f"[wrote]  {full_path.resolve()}")
    print(f"[wrote]  {compact_path.resolve()}")
    print()
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
