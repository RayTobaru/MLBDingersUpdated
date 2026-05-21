from __future__ import annotations

import pandas as pd


def _safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _round_display(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].round(digits)
    return out


def _to_terminal_string(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "[no rows]"
    return df.head(max_rows).to_string(index=False, max_colwidth=22)




def _ko_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "date", "matchup", "Team", "Name",
        "Proj_IP", "Pred_K9", "Mean_K_start", "Median_K_start",
        "P(K≥4)", "P_cal(K≥4)", "P(K≥5)", "P_cal(K≥5)",
        "P(K≥6)", "P_cal(K≥6)", "P(K≥7)", "P_cal(K≥7)",
        "P(K≥8)", "P_cal(K≥8)",
        "anchor_k6", "k5_k6_blend", "tail_k7_k8_blend", "recommended_focus",
    ]
    out = df[_safe_cols(df, keep)].copy()
    sort_col = "anchor_k6" if "anchor_k6" in out.columns else ("P_cal(K≥6)" if "P_cal(K≥6)" in out.columns else ("P(K≥6)" if "P(K≥6)" in out.columns else None))
    if sort_col:
        out[sort_col] = pd.to_numeric(out[sort_col], errors="coerce")
        out = out.sort_values(sort_col, ascending=False, kind="stable")
    return out


def compact_ko_terminal(df: pd.DataFrame, max_rows: int = 20) -> str:
    out = _ko_display_frame(df)
    out = _round_display(out, 3)
    return _to_terminal_string(out, max_rows=max_rows)


def compact_batters_terminal(df: pd.DataFrame, max_rows: int = 20, mode: str = "hits") -> str:
    if mode == "hr":
        keep = [
            "team", "batting_order", "player_name", "opp_pitcher",
            "exp_hr", "P(HR>=1)", "hr_pa",
            "batter_hr_mult", "batter_form_hr_mult",
            "pitcher_hr_danger_mult", "status",
        ]
        sort_primary = "P(HR>=1)"
        sort_secondary = "exp_hr"
    else:
        keep = [
            "team", "batting_order", "player_name", "opp_pitcher",
            "exp_hits", "P(H>=1)", "exp_1b", "exp_2b", "exp_3b", "exp_hr",
            "hit_pa", "hr_pa", "status",
        ]
        sort_primary = "P(H>=1)"
        sort_secondary = "exp_hits"

    out = df[_safe_cols(df, keep)].copy()
    if "status" in out.columns:
        out["status"] = out["status"].fillna("")
    if sort_primary in out.columns and sort_secondary in out.columns:
        out = out.sort_values([sort_primary, sort_secondary], ascending=[False, False])
    out = _round_display(out, 3)
    return _to_terminal_string(out, max_rows=max_rows)
