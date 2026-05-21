from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .predict_batter_outcomes import predict_matchup_batters, predict_slate_batters
from .predict_pitcher_ko import predict_matchup_ko, predict_slate_ko, recommended_k_angles



ROOT = Path(__file__).resolve().parent.parent


def _ko_calibrator_path() -> Path:
    return ROOT / "outputs" / "calibration_test" / "ko_ladder_calibrators.pkl"


def _apply_ko_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if not isinstance(df, pd.DataFrame) else df.copy()
    out = df.copy()
    if "P_cal(K≥6)" in out.columns:
        out["anchor_k6"] = pd.to_numeric(out["P_cal(K≥6)"], errors="coerce")
    elif "P(K≥6)" in out.columns:
        out["anchor_k6"] = pd.to_numeric(out["P(K≥6)"], errors="coerce")
    if "P_cal(K≥5)" in out.columns and "P_cal(K≥6)" in out.columns:
        out["k5_k6_blend"] = 0.40 * pd.to_numeric(out["P_cal(K≥5)"], errors="coerce").fillna(0) + 0.60 * pd.to_numeric(out["P_cal(K≥6)"], errors="coerce").fillna(0)
    elif "P(K≥5)" in out.columns and "P(K≥6)" in out.columns:
        out["k5_k6_blend"] = 0.40 * pd.to_numeric(out["P(K≥5)"], errors="coerce").fillna(0) + 0.60 * pd.to_numeric(out["P(K≥6)"], errors="coerce").fillna(0)
    if "P_cal(K≥7)" in out.columns and "P_cal(K≥8)" in out.columns:
        out["tail_k7_k8_blend"] = 0.65 * pd.to_numeric(out["P_cal(K≥7)"], errors="coerce").fillna(0) + 0.35 * pd.to_numeric(out["P_cal(K≥8)"], errors="coerce").fillna(0)
    elif "P(K≥7)" in out.columns and "P(K≥8)" in out.columns:
        out["tail_k7_k8_blend"] = 0.65 * pd.to_numeric(out["P(K≥7)"], errors="coerce").fillna(0) + 0.35 * pd.to_numeric(out["P(K≥8)"], errors="coerce").fillna(0)
    if "anchor_k6" in out.columns:
        out["rank_anchor_k6"] = out["anchor_k6"].rank(method="dense", ascending=False).astype("Int64")
    if "k5_k6_blend" in out.columns:
        out["rank_balanced"] = out["k5_k6_blend"].rank(method="dense", ascending=False).astype("Int64")
    if "tail_k7_k8_blend" in out.columns:
        out["rank_tail"] = out["tail_k7_k8_blend"].rank(method="dense", ascending=False).astype("Int64")

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


def _try_apply_ko_calibration(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if not isinstance(df, pd.DataFrame) else df.copy()
    out = df.copy()
    if "P(K≥6)" not in out.columns:
        return out
    try:
        cal_path = _ko_calibrator_path()
        if cal_path.exists():
            from .ko_ladder_calibration import apply_ko_ladder_calibrators, load_calibrators
            out = apply_ko_ladder_calibrators(out, load_calibrators(cal_path))
    except Exception:
        pass
    return _apply_ko_summary_columns(out)


def _normalize_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if not isinstance(df, pd.DataFrame) else df.copy()
    out = df.copy()
    for c in list(out.columns):
        name = str(c).lower()
        if name in {"date", "game_date", "as_of_date"}:
            dt = pd.to_datetime(out[c], errors="coerce")
            out[c] = dt.dt.strftime("%Y-%m-%d").where(dt.notna(), out[c].astype(str))
    return out


def _fmt_pct(x: Any) -> str:
    try:
        return f"{100 * float(x):.1f}%"
    except Exception:
        return ""


def _fmt_num(x: Any, digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return ""


def _eligible_topboard_mask(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=bool)
    mask = pd.Series(True, index=df.index)
    if 'status' in df.columns:
        st = df['status'].fillna('').astype(str).str.strip()
        mask &= st.eq('')
    if 'eligible_for_top_boards' in df.columns:
        etb = pd.to_numeric(df['eligible_for_top_boards'], errors='coerce').fillna(0)
        mask &= etb.eq(1)
    if 'strict_confirmed_lineup_flag' in df.columns:
        sflag = pd.to_numeric(df['strict_confirmed_lineup_flag'], errors='coerce').fillna(0)
        mask &= sflag.eq(1)
    if 'active_roster_flag' in df.columns:
        arf = pd.to_numeric(df['active_roster_flag'], errors='coerce').fillna(0)
        mask &= arf.eq(1)
    if 'roster_integrity_note' in df.columns:
        rin = df['roster_integrity_note'].fillna('').astype(str).str.strip()
        mask &= rin.eq('')
    if 'top_board_block_reason' in df.columns:
        tbr = df['top_board_block_reason'].fillna('').astype(str).str.strip()
        mask &= tbr.eq('')
    if 'opp_pitcher' in df.columns:
        opp = df['opp_pitcher'].fillna('').astype(str).str.strip().str.upper()
        mask &= ~opp.isin(['', 'TBD', 'NONE'])
    if 'opp_pitcher_id' in df.columns:
        oppid = pd.to_numeric(df['opp_pitcher_id'], errors='coerce')
        mask &= oppid.notna()
    if 'player_id' in df.columns:
        pid = pd.to_numeric(df['player_id'], errors='coerce')
        mask &= pid.notna()
    if 'lineup_source' in df.columns:
        ls = df['lineup_source'].fillna('').astype(str).str.lower()
        mask &= ls.eq('confirmed')
    return mask


def _safe_top_table(df: pd.DataFrame, sort_col: str, cols: list[str], n: int = 5) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty or sort_col not in df.columns:
        return pd.DataFrame(columns=cols)
    tmp = df.copy()
    mask = _eligible_topboard_mask(tmp)
    if len(mask):
        tmp = tmp[mask]
    tmp[sort_col] = pd.to_numeric(tmp[sort_col], errors="coerce")
    tmp = tmp.sort_values(sort_col, ascending=False, kind="stable").head(n).copy()
    keep = [c for c in cols if c in tmp.columns]
    return tmp[keep]


def _breakout_board(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = _normalize_date_cols(df)
    keep = [
        'matchup','team','batting_order','player_name','opp_pitcher',
        'power_score','pitcher_vulnerability_score','context_score',
        'composite_hr_score','confidence_pct','model_prob_hr','exp_hr',
        'fd_odds','fd_implied_prob','fair_odds_american','edge_prob','edge_pct_pts',
        'breakout_tag','breakout_rank','status'
    ]
    keep = [c for c in keep if c in out.columns]
    board = out.copy()
    mask = _eligible_topboard_mask(board)
    if len(mask):
        board = board[mask]
    keep = [c for c in keep if c in board.columns]
    board = board[keep].copy()
    sort_col = 'composite_hr_score' if 'composite_hr_score' in board.columns else 'model_prob_hr'
    if sort_col in board.columns:
        board[sort_col] = pd.to_numeric(board[sort_col], errors='coerce')
        board = board.sort_values(sort_col, ascending=False, kind='stable').reset_index(drop=True)
    return board


def _compact_ko_view(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "date", "matchup", "Team", "Name", "Pitcher_ID",
        "Proj_IP", "Proj_BF", "Pred_K9", "Mean_K_start", "Median_K_start",
        "recent_k9_7", "recent_k9_14", "recent_k9_30", "opp_lineup_k_rate_raw", "opp_lineup_k_rate_shrunk", "opp_lineup_k_rate", "opp_lineup_pa_seen", "opp_lineup_resolved_n",
        "P(K≥4)", "P_cal(K≥4)", "P(K≥5)", "P_cal(K≥5)", "P(K≥6)", "P_cal(K≥6)", "P(K≥7)", "P_cal(K≥7)", "P(K≥8)", "P_cal(K≥8)",
        "anchor_k6", "k5_k6_blend", "tail_k7_k8_blend", "rank_anchor_k6", "rank_balanced", "rank_tail", "recommended_focus",
        "90% CI", "opp_lineup_source", "lineup_weight", "lineup_certainty",
    ]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=keep)
    out = _try_apply_ko_calibration(_normalize_date_cols(df))
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()
    sort_col = "anchor_k6" if "anchor_k6" in out.columns else ("P_cal(K≥6)" if "P_cal(K≥6)" in out.columns else ("Mean_K_start" if "Mean_K_start" in out.columns else None))
    if sort_col:
        out = out.sort_values(sort_col, ascending=False, kind="stable")
    return out


def _batter_views(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "compact": pd.DataFrame(),
            "top_hr": pd.DataFrame(),
            "top_hit": pd.DataFrame(),
            "top_xbh": pd.DataFrame(),
            "team_totals": pd.DataFrame(),
            "breakout_board": pd.DataFrame(),
        }

    out = _normalize_date_cols(df)

    numeric_cols = [
        "exp_hits", "exp_1b", "exp_2b", "exp_3b", "exp_hr",
        "P(H>=1)", "P(1B>=1)", "P(2B>=1)", "P(3B>=1)", "P(HR>=1)",
        "hit_pa", "1b_pa", "2b_pa", "3b_pa", "hr_pa",
    ]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if {"exp_2b", "exp_3b", "exp_hr"}.issubset(out.columns):
        out["exp_xbh"] = out["exp_2b"].fillna(0) + out["exp_3b"].fillna(0) + out["exp_hr"].fillna(0)
    else:
        out["exp_xbh"] = 0.0

    compact_cols = [
        "date", "matchup", "team", "batting_order", "player_name", "player_id",
        "bats", "opp_pitcher", "opp_pitcher_id", "opp_pitcher_hand",
        "lineup_source", "lineup_weight", "lineup_certainty", "exp_pa",
        "exp_hits", "exp_1b", "exp_2b", "exp_3b", "exp_hr",
        "P(H>=1)", "P(1B>=1)", "P(2B>=1)", "P(3B>=1)", "P(HR>=1)",
        "hit_pa", "1b_pa", "2b_pa", "3b_pa", "hr_pa", "exp_hr_ctx", "markov_hr_context_mult", "raw_batter_hr_mult", "raw_pitcher_hr_danger_mult", "raw_pitcher_recent_danger_mult", "batter_hr_mult", "pitcher_hr_danger_mult", "active_roster_flag", "strict_confirmed_lineup_flag", "eligible_for_top_boards", "top_board_block_reason", "roster_integrity_note", "status",
    ]
    compact = out[[c for c in compact_cols if c in out.columns]].copy()
    sort_cols = [c for c in ["date", "matchup", "team", "batting_order"] if c in compact.columns]
    if sort_cols:
        compact = compact.sort_values(sort_cols, kind="stable")

    hr_sort_col = "exp_hr_ctx" if "exp_hr_ctx" in out.columns else "P(HR>=1)"
    top_hr = _safe_top_table(
        out,
        hr_sort_col,
        ["matchup", "team", "batting_order", "player_name", "opp_pitcher", "opp_pitcher_id", "P(HR>=1)", "exp_hr", "exp_hr_ctx", "markov_hr_context_mult", "raw_batter_hr_mult", "batter_hr_mult", "raw_pitcher_hr_danger_mult", "pitcher_hr_danger_mult", "zone_hr_overall_mult", "zone_hr_damage_mult", "zone_hr_barrel_mult", "zone_hr_air_mult", "zone_hr_confidence", "zone_hr_candidate_source", "zone_hr_sample_regime", "zone_hr_batter_archetype", "zone_hr_pitcher_arch", "zone_hr_exact_batter_rows", "zone_hr_batter_archetype_rows", "zone_hr_pitcher_archetype_rows", "zone_hr_hand_rows", "zone_hr_league_rows", "zone_hr_early_season_shrink", "weather_hr_mult", "pitch_type_mult", "eligible_for_top_boards", "display_terminal_flag", "top_board_block_reason", "roster_integrity_note"],
        n=20,
    )
    top_hit = _safe_top_table(
        out,
        "P(H>=1)",
        ["matchup", "team", "batting_order", "player_name", "opp_pitcher", "opp_pitcher_id", "P(H>=1)", "exp_hits", "markov_team_exp_runs", "markov_run_context_mult", "lineup_source", "strict_confirmed_lineup_flag", "eligible_for_top_boards", "display_terminal_flag", "top_board_block_reason", "roster_integrity_note"],
        n=20,
    )
    top_xbh = _safe_top_table(
        out,
        "exp_xbh",
        ["matchup", "team", "batting_order", "player_name", "opp_pitcher", "exp_xbh", "P(2B>=1)", "P(3B>=1)", "P(HR>=1)"],
        n=8,
    )

    group_cols = [c for c in ["matchup", "team"] if c in out.columns]
    if group_cols:
        agg_map = {}
        for c in ["exp_hits", "exp_hr", "exp_2b", "exp_3b"]:
            if c in out.columns:
                agg_map[c] = "sum"
        if "markov_team_exp_runs" in out.columns:
            agg_map["markov_team_exp_runs"] = "max"
        team_totals = out.groupby(group_cols, as_index=False).agg(agg_map) if agg_map else pd.DataFrame()
        if not team_totals.empty:
            rename_map = {
                "exp_hits": "lineup_exp_hits",
                "exp_hr": "lineup_exp_hr",
                "exp_2b": "lineup_exp_2b",
                "exp_3b": "lineup_exp_3b",
                "markov_team_exp_runs": "markov_team_exp_runs",
            }
            team_totals = team_totals.rename(columns=rename_map).sort_values(group_cols, kind="stable")
    else:
        team_totals = pd.DataFrame()

    return {
        "compact": compact,
        "top_hr": top_hr,
        "top_hit": top_hit,
        "top_xbh": top_xbh,
        "team_totals": team_totals,
        "breakout_board": _breakout_board(out),
    }


def matchup_prediction_report(date_str: str, matchup: str, override_path: str | None = None) -> dict[str, Any]:
    ko_df = predict_matchup_ko(date_str, matchup, override_path=override_path)
    bat_df = predict_matchup_batters(date_str, matchup, override_path=override_path)
    return _assemble_report(date_str, matchup, ko_df, bat_df)


def slate_prediction_report(date_str: str, override_path: str | None = None) -> dict[str, Any]:
    ko_df = predict_slate_ko(date_str, override_path=override_path)
    bat_df = predict_slate_batters(date_str, override_path=override_path)
    return _assemble_report(date_str, None, ko_df, bat_df)


def _assemble_report(date_str: str, matchup: str | None, ko_df: pd.DataFrame, bat_df: pd.DataFrame) -> dict[str, Any]:
    ko_df = _normalize_date_cols(ko_df) if isinstance(ko_df, pd.DataFrame) else pd.DataFrame()
    bat_df = _normalize_date_cols(bat_df) if isinstance(bat_df, pd.DataFrame) else pd.DataFrame()

    lines: list[str] = []
    title = f"MLB Model Report — {date_str}"
    if matchup:
        title += f" — {matchup.replace('@', ' @ ')}"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    ko_df = _try_apply_ko_calibration(ko_df)
    ko_compact = _compact_ko_view(ko_df)
    batter_views = _batter_views(bat_df)
    top_hr = batter_views["top_hr"]
    top_hit = batter_views["top_hit"]
    top_xbh = batter_views["top_xbh"]
    team_totals = batter_views["team_totals"]
    breakout_board = batter_views["breakout_board"]

    if not ko_compact.empty:
        lines.append("Pitcher KO outlook")
        lines.append("------------------")
        for _, row in ko_compact.iterrows():
            lines.append(
                f"{row.get('Name', '')} ({row.get('Team', '')}): mean K {_fmt_num(row.get('Mean_K_start'))}, "
                f"median K {_fmt_num(row.get('Median_K_start'))}, "
                f"cal K≥6 {_fmt_pct(row.get('P_cal(K≥6)', row.get('P(K≥6)')))}, "
                f"balanced {_fmt_pct(row.get('k5_k6_blend'))}, "
                f"tail {_fmt_pct(row.get('tail_k7_k8_blend'))}, "
                f"focus {row.get('recommended_focus', '')}"
            )
        lines.append("")

    if not bat_df.empty:
        lines.append("Batter outlook")
        lines.append("--------------")
        if "lineup_source" in bat_df.columns and "matchup" in bat_df.columns:
            by_game = (
                bat_df.groupby("matchup", dropna=False)["lineup_source"]
                .agg(lambda s: pd.Series(s).mode().iloc[0] if len(pd.Series(s).mode()) else s.iloc[0])
            )
            for mk, src in by_game.items():
                lines.append(f"{str(mk).replace('@', ' @ ')} lineup source: {src}")
        if "umpire" in bat_df.columns and "matchup" in bat_df.columns:
            by_game_ump = bat_df.groupby("matchup", dropna=False)["umpire"].agg(
                lambda s: next((x for x in s if str(x).strip()), "")
            )
            for mk, ump in by_game_ump.items():
                if str(ump).strip():
                    lines.append(f"{str(mk).replace('@', ' @ ')} umpire context: {ump}")
        lines.append("")

        lines.append("Top HR threats")
        lines.append("~~~~~~~~~~~~~~")
        for _, row in top_hr.iterrows():
            lines.append(
                f"{row.get('matchup', '')} | {row.get('player_name', '')} ({row.get('team', '')}) vs {row.get('opp_pitcher', '')} — "
                f"HR {_fmt_pct(row.get('P(HR>=1)'))}, exp HR {_fmt_num(row.get('exp_hr'), 3)}"
            )
        lines.append("")

        lines.append("Top hit threats")
        lines.append("~~~~~~~~~~~~~~~")
        for _, row in top_hit.iterrows():
            lines.append(
                f"{row.get('matchup', '')} | {row.get('player_name', '')} ({row.get('team', '')}) vs {row.get('opp_pitcher', '')} — "
                f"Hit {_fmt_pct(row.get('P(H>=1)'))}, exp hits {_fmt_num(row.get('exp_hits'), 3)}"
            )
        lines.append("")

        lines.append("Top extra-base-hit threats")
        lines.append("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for _, row in top_xbh.iterrows():
            lines.append(
                f"{row.get('matchup', '')} | {row.get('player_name', '')} ({row.get('team', '')}) vs {row.get('opp_pitcher', '')} — "
                f"exp XBH {_fmt_num(row.get('exp_xbh'), 3)}, HR {_fmt_pct(row.get('P(HR>=1)'))}"
            )
        lines.append("")

        if not team_totals.empty:
            lines.append("Team hitting outlook")
            lines.append("~~~~~~~~~~~~~~~~~~~~")
            for _, row in team_totals.iterrows():
                lines.append(
                    f"{row.get('matchup', '')} | {row.get('team', '')} — exp hits {_fmt_num(row.get('lineup_exp_hits'), 2)}, "
                    f"exp 2B {_fmt_num(row.get('lineup_exp_2b'), 2)}, exp 3B {_fmt_num(row.get('lineup_exp_3b'), 2)}, exp HR {_fmt_num(row.get('lineup_exp_hr'), 2)}"
                )
            lines.append("")

        if not breakout_board.empty:
            lines.append("HR breakout board")
            lines.append("~~~~~~~~~~~~~~~~")
            for _, row in breakout_board.head(12).iterrows():
                edge = row.get('edge_pct_pts', np.nan)
                edge_txt = f", edge {_fmt_num(edge, 1)} pts" if pd.notna(edge) else ""
                lines.append(
                    f"{row.get('matchup', '')} | {row.get('player_name', '')} ({row.get('team', '')}) vs {row.get('opp_pitcher', '')} — "
                    f"comp {_fmt_num(row.get('composite_hr_score'), 1)}, conf {_fmt_num(row.get('confidence_pct'), 0)}%, "
                    f"HR {_fmt_pct(row.get('model_prob_hr'))}{edge_txt}, tag {row.get('breakout_tag', '')}"
                )
            lines.append("")

    report_text = "\n".join(lines).strip() + "\n"
    return {
        "text": report_text,
        "ko_df": ko_df,
        "bat_df": bat_df,
        "ko_compact": ko_compact,
        "bat_compact": batter_views["compact"],
        "top_hr": top_hr,
        "top_hit": top_hit,
        "top_xbh": top_xbh,
        "team_totals": team_totals,
        "breakout_board": breakout_board,
    }


def save_report_bundle(bundle: dict[str, Any], outdir: str | Path, stem: str) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    txt_path = outdir / f"{stem}_report.txt"
    txt_path.write_text(str(bundle.get("text", "")), encoding="utf-8")
    paths["report"] = txt_path

    def _save_df(key: str, suffix: str):
        df = bundle.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            p = outdir / f"{stem}_{suffix}.csv"
            _normalize_date_cols(df).to_csv(p, index=False)
            paths[suffix] = p

    _save_df("ko_df", "ko")
    _save_df("ko_compact", "ko_compact")
    _save_df("bat_df", "batters")
    _save_df("bat_compact", "batters_compact")
    _save_df("top_hr", "top_hr")
    _save_df("top_hit", "top_hits")
    _save_df("top_xbh", "top_xbh")
    _save_df("team_totals", "team_totals")
    _save_df("breakout_board", "breakout_hr_board")

    return paths


__all__ = [
    "matchup_prediction_report",
    "slate_prediction_report",
    "save_report_bundle",
]
