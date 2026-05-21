#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
try:
    from mlb_model.fd_value import (
        build_and_save_hr_values,
        build_hr_value_board,
        find_fd_odds_file,
        write_fd_odds_template,
    )
except Exception:
    build_and_save_hr_values = None
    build_hr_value_board = None
    find_fd_odds_file = None
    write_fd_odds_template = None

import numpy as np

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / 'legacy'
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RUNTIME = None


def _calibration_attr(mod, primary: str, fallback: str | None = None):
    fn = getattr(mod, primary, None)
    if fn is None and fallback:
        fn = getattr(mod, fallback, None)
    return fn


def _runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    try:
        bat_mod = importlib.import_module('mlb_model.predict_batter_outcomes')
        ko_mod = importlib.import_module('mlb_model.predict_pitcher_ko')
        rep_mod = importlib.import_module('mlb_model.reporting')
        cal_mod = importlib.import_module('mlb_model.calibration')
        ctx_mod = importlib.import_module('mlb_model.context_features')
        bkt_mod = importlib.import_module('mlb_model.season_backtest')
    except ModuleNotFoundError as exc:
        missing = getattr(exc, 'name', str(exc))
        raise SystemExit(
            f"Missing Python dependency or module: {missing}\n"
            "From the Dingers_hotfix folder, install requirements first:\n"
            "  pip install -r requirements.txt\n"
            "Then run:\n"
            "  python cli.py"
        )
    _RUNTIME = {
        'predict_matchup_batters': bat_mod.predict_matchup_batters,
        'predict_slate_batters': bat_mod.predict_slate_batters,
        'fetch_matchups_for_date': bat_mod.fetch_matchups_for_date,
        'predict_matchup_ko': ko_mod.predict_matchup_ko,
        'predict_slate_ko': ko_mod.predict_slate_ko,
        'matchup_prediction_report': rep_mod.matchup_prediction_report,
        'slate_prediction_report': rep_mod.slate_prediction_report,
        'save_report_bundle': rep_mod.save_report_bundle,
        'load_any_table': _calibration_attr(cal_mod, 'load_any_table', '_load_any_table'),
        'benchmark_closing_lines': _calibration_attr(cal_mod, 'benchmark_closing_lines', 'evaluate_market_file'),
        'fit_walk_forward_summary': _calibration_attr(cal_mod, 'fit_walk_forward_summary', 'evaluate_prediction_file'),
        'auto_build_team_defense_proxies': ctx_mod.auto_build_team_defense_proxies,
        'run_season_replay': bkt_mod.run_season_replay,
    }
    return _RUNTIME


def _run_python(script: str, extra: list[str]) -> int:
    path = LEGACY / script
    cmd = [sys.executable, str(path), *extra]
    return subprocess.call(cmd, cwd=str(ROOT))


def _ensure_outputs() -> Path:
    out = ROOT / 'outputs'
    out.mkdir(parents=True, exist_ok=True)
    return out


def _normalize_date_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    if df.empty:
        return df.copy()
    out = df.copy()
    for c in list(out.columns):
        if str(c).lower() in {'date', 'game_date', 'as_of_date'}:
            dt = pd.to_datetime(out[c], errors='coerce')
            out[c] = dt.dt.strftime('%Y-%m-%d').where(dt.notna(), out[c].astype(str))
    return out


def _print_df(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print('[no rows]')
        return
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 240):
        print(df.to_string(index=False))


def _safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _round_display(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include='number').columns
    out[num_cols] = out[num_cols].round(digits)
    return out


def _to_terminal_string(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return '[no rows]'
    return df.head(max_rows).to_string(index=False, max_colwidth=22)



def _ko_calibrator_path() -> Path:
    return ROOT / 'outputs' / 'calibration_test' / 'ko_ladder_calibrators.pkl'


def _apply_ko_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if not isinstance(df, pd.DataFrame) else df.copy()
    out = df.copy()

    if 'P_cal(K≥6)' in out.columns:
        out['anchor_k6'] = pd.to_numeric(out['P_cal(K≥6)'], errors='coerce')
    elif 'P(K≥6)' in out.columns:
        out['anchor_k6'] = pd.to_numeric(out['P(K≥6)'], errors='coerce')

    if 'P_cal(K≥5)' in out.columns and 'P_cal(K≥6)' in out.columns:
        out['k5_k6_blend'] = (
            0.40 * pd.to_numeric(out['P_cal(K≥5)'], errors='coerce').fillna(0)
            + 0.60 * pd.to_numeric(out['P_cal(K≥6)'], errors='coerce').fillna(0)
        )
    elif 'P(K≥5)' in out.columns and 'P(K≥6)' in out.columns:
        out['k5_k6_blend'] = (
            0.40 * pd.to_numeric(out['P(K≥5)'], errors='coerce').fillna(0)
            + 0.60 * pd.to_numeric(out['P(K≥6)'], errors='coerce').fillna(0)
        )

    if 'P_cal(K≥7)' in out.columns and 'P_cal(K≥8)' in out.columns:
        out['tail_k7_k8_blend'] = (
            0.65 * pd.to_numeric(out['P_cal(K≥7)'], errors='coerce').fillna(0)
            + 0.35 * pd.to_numeric(out['P_cal(K≥8)'], errors='coerce').fillna(0)
        )
    elif 'P(K≥7)' in out.columns and 'P(K≥8)' in out.columns:
        out['tail_k7_k8_blend'] = (
            0.65 * pd.to_numeric(out['P(K≥7)'], errors='coerce').fillna(0)
            + 0.35 * pd.to_numeric(out['P(K≥8)'], errors='coerce').fillna(0)
        )

    if 'anchor_k6' in out.columns:
        out['rank_anchor_k6'] = out['anchor_k6'].rank(method='dense', ascending=False).astype('Int64')
    if 'k5_k6_blend' in out.columns:
        out['rank_balanced'] = out['k5_k6_blend'].rank(method='dense', ascending=False).astype('Int64')
    if 'tail_k7_k8_blend' in out.columns:
        out['rank_tail'] = out['tail_k7_k8_blend'].rank(method='dense', ascending=False).astype('Int64')

    def _focus(row):
        try:
            mean_k = float(row.get('Mean_K_start', 0) or 0)
        except Exception:
            mean_k = 0.0
        if mean_k >= 6.0:
            return 'tail'
        if mean_k >= 5.0:
            return 'balanced'
        return 'anchor'

    out['recommended_focus'] = out.apply(_focus, axis=1)
    return out


def _try_apply_ko_calibration(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if not isinstance(df, pd.DataFrame) else df.copy()
    out = df.copy()
    if 'P(K≥6)' not in out.columns:
        return out
    try:
        cal_path = _ko_calibrator_path()
        if cal_path.exists():
            ko_cal_mod = importlib.import_module('mlb_model.ko_ladder_calibration')
            calibrators = ko_cal_mod.load_calibrators(cal_path)
            out = ko_cal_mod.apply_ko_ladder_calibrators(out, calibrators)
    except Exception:
        pass
    return _apply_ko_summary_columns(out)


def compact_ko_terminal(df: pd.DataFrame, max_rows: int = 20) -> str:
    df = _try_apply_ko_calibration(df)
    keep = [
        'date', 'matchup', 'Team', 'Name',
        'Proj_IP', 'Pred_K9', 'Mean_K_start', 'Median_K_start',
        'opp_lineup_k_rate',
        'P(K≥4)', 'P_cal(K≥4)', 'P(K≥5)', 'P_cal(K≥5)',
        'P(K≥6)', 'P_cal(K≥6)', 'P(K≥7)', 'P_cal(K≥7)',
        'P(K≥8)', 'P_cal(K≥8)',
        'anchor_k6', 'k5_k6_blend', 'tail_k7_k8_blend',
        'recommended_focus', '90% CI', 'opp_lineup_source',
    ]
    out = df[_safe_cols(df, keep)].copy()
    sort_col = 'anchor_k6' if 'anchor_k6' in out.columns else ('P_cal(K≥6)' if 'P_cal(K≥6)' in out.columns else ('P(K≥6)' if 'P(K≥6)' in out.columns else None))
    if sort_col:
        out[sort_col] = pd.to_numeric(out[sort_col], errors='coerce')
        out = out.sort_values(sort_col, ascending=False, kind='stable')
    out = _round_display(out, 3)
    return _to_terminal_string(out, max_rows=max_rows)



def _eligible_cli_batter_mask(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=bool)

    mask = pd.Series(True, index=df.index)

    if 'display_terminal_flag' in df.columns:
        flag = pd.to_numeric(df['display_terminal_flag'], errors='coerce').fillna(0)
        mask &= flag.eq(1)

    if 'eligible_for_top_boards' in df.columns:
        etb = pd.to_numeric(df['eligible_for_top_boards'], errors='coerce').fillna(0)
        mask &= etb.eq(1)

    if 'strict_confirmed_lineup_flag' in df.columns:
        sflag = pd.to_numeric(df['strict_confirmed_lineup_flag'], errors='coerce').fillna(0)
        mask &= sflag.eq(1)

    if 'active_roster_flag' in df.columns:
        arf = pd.to_numeric(df['active_roster_flag'], errors='coerce').fillna(0)
        mask &= arf.eq(1)

    if 'status' in df.columns:
        st = df['status'].fillna('').astype(str).str.strip()
        mask &= st.eq('')

    if 'top_board_block_reason' in df.columns:
        tbr = df['top_board_block_reason'].fillna('').astype(str).str.strip()
        mask &= tbr.eq('')

    if 'roster_integrity_note' in df.columns:
        rin = df['roster_integrity_note'].fillna('').astype(str).str.strip()
        mask &= rin.eq('')

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


def _filtered_cli_batter_board(df: pd.DataFrame, mode: str = 'hits') -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()
    mask = _eligible_cli_batter_mask(out)
    if len(mask) == len(out):
        out = out[mask].copy()

    if mode == 'hr':
        sort_primary = 'exp_hr_ctx' if 'exp_hr_ctx' in out.columns else 'P(HR>=1)'
        sort_secondary = 'exp_hr'
    else:
        sort_primary = 'P(H>=1)'
        sort_secondary = 'exp_hits'

    if sort_primary in out.columns:
        out[sort_primary] = pd.to_numeric(out[sort_primary], errors='coerce')
    if sort_secondary in out.columns:
        out[sort_secondary] = pd.to_numeric(out[sort_secondary], errors='coerce')

    if sort_primary in out.columns and sort_secondary in out.columns:
        out = out.sort_values([sort_primary, sort_secondary], ascending=[False, False], kind='stable')
    return out

def compact_batters_terminal(df: pd.DataFrame, max_rows: int = 20, mode: str = 'hits') -> str:
    if mode == 'hr':
        keep = [
            'team', 'batting_order', 'player_name', 'opp_pitcher',
            'exp_hr', 'P(HR>=1)', 'hr_pa',
            'batter_hr_mult', 'batter_form_hr_mult',
            'pitcher_hr_danger_mult',
        'zone_hr_overall_mult',
        'zone_hr_confidence',
        'zone_hr_candidate_source',
        'zone_hr_sample_regime',
 'status',
        ]
        sort_primary = 'P(HR>=1)'
        sort_secondary = 'exp_hr'
    else:
        keep = [
            'team', 'batting_order', 'player_name', 'opp_pitcher',
            'exp_hits', 'P(H>=1)', 'exp_1b', 'exp_2b', 'exp_3b', 'exp_hr',
            'hit_pa', 'hr_pa', 'status',
        ]
        sort_primary = 'P(H>=1)'
        sort_secondary = 'exp_hits'

    out = df[_safe_cols(df, keep)].copy()
    if 'status' in out.columns:
        out['status'] = out['status'].fillna('')
    if sort_primary in out.columns and sort_secondary in out.columns:
        out = out.sort_values([sort_primary, sort_secondary], ascending=[False, False], kind='stable')
    out = _round_display(out, 3)
    return _to_terminal_string(out, max_rows=max_rows)


def _compact_subset_csvs(df: pd.DataFrame, out: Path) -> list[Path]:
    written: list[Path] = []
    stem = out.stem

    if stem.startswith('ko_'):
        df = _try_apply_ko_calibration(df)
        keep = [
            'date', 'matchup', 'Team', 'Name', 'Pitcher_ID',
            'Proj_IP', 'Proj_BF', 'Pred_K9', 'Mean_K_start', 'Median_K_start',
            'recent_k9_7', 'recent_k9_14', 'recent_k9_30', 'opp_lineup_k_rate',
            'P(K≥4)', 'P_cal(K≥4)', 'P(K≥5)', 'P_cal(K≥5)',
            'P(K≥6)', 'P_cal(K≥6)', 'P(K≥7)', 'P_cal(K≥7)',
            'P(K≥8)', 'P_cal(K≥8)', '90% CI',
            'opp_lineup_source', 'lineup_weight', 'lineup_certainty',
            'anchor_k6', 'k5_k6_blend', 'tail_k7_k8_blend',
            'rank_anchor_k6', 'rank_balanced', 'rank_tail', 'recommended_focus',
        ]
        compact_df = df[_safe_cols(df, keep)].copy()
        compact_out = out.with_name(f'{stem}_compact.csv')
        compact_df.to_csv(compact_out, index=False)
        written.append(compact_out)

    elif stem.startswith('batters_'):
        compact_keep = [
            'date', 'matchup', 'team', 'batting_order', 'player_name', 'player_id',
            'bats', 'opp_pitcher', 'opp_pitcher_id', 'opp_pitcher_hand',
            'lineup_source', 'lineup_weight', 'lineup_certainty', 'exp_pa',
            'exp_hits', 'exp_1b', 'exp_2b', 'exp_3b', 'exp_hr',
            'P(H>=1)', 'P(1B>=1)', 'P(2B>=1)', 'P(3B>=1)', 'P(HR>=1)',
            'hit_pa', '1b_pa', '2b_pa', '3b_pa', 'hr_pa', 'status',
        ]
        compact_df = df[_safe_cols(df, compact_keep)].copy()
        compact_out = out.with_name(f'{stem}_compact.csv')
        compact_df.to_csv(compact_out, index=False)
        written.append(compact_out)


        top_hr_keep = [
            'team', 'batting_order', 'player_name', 'opp_pitcher', 'opp_pitcher_id', 'opp_pitcher_hand',
            'exp_hr', 'exp_hr_ctx', 'P(HR>=1)', 'hr_pa',
            'raw_batter_hr_mult', 'batter_hr_mult', 'batter_form_hr_mult',
            'raw_pitcher_hr_danger_mult', 'pitcher_hr_danger_mult', 'pitcher_recent_danger_mult',
            'zone_hr_overall_mult',
            'zone_hr_damage_mult',
            'zone_hr_barrel_mult',
            'zone_hr_air_mult',
            'zone_hr_confidence',
            'zone_hr_candidate_source',
            'zone_hr_sample_regime',
            'zone_hr_batter_archetype',
            'zone_hr_pitcher_arch',
            'zone_hr_exact_batter_rows',
            'zone_hr_batter_archetype_rows',
            'zone_hr_pitcher_archetype_rows',
            'zone_hr_hand_rows',
            'zone_hr_league_rows',
            'zone_hr_early_season_shrink',

            'display_terminal_flag', 'strict_confirmed_lineup_flag', 'eligible_for_top_boards',
            'top_board_block_reason', 'roster_integrity_note', 'status',
        ]
        top_hr = _filtered_cli_batter_board(df, mode='hr')
        top_hr = top_hr[_safe_cols(top_hr, top_hr_keep)].copy()
        top_hr_out = out.with_name(f'{stem}_top_hr.csv')
        top_hr.to_csv(top_hr_out, index=False)
        written.append(top_hr_out)

        # Optional FanDuel HR values board.
        try:
            if callable(find_fd_odds_file) and callable(build_and_save_hr_values):
                odds_path = find_fd_odds_file()
                if odds_path is not None:
                    values_out = out.with_name(f'{stem}_hr_values.csv')
                    values_df = build_and_save_hr_values(
                        df,
                        values_out,
                        odds_path=odds_path,
                        include_no_odds=True,
                    )
                    if isinstance(values_df, pd.DataFrame):
                        written.append(values_out)
        except Exception as exc:
            print(f'[fd odds][warn] HR values board not written: {exc}')

        top_hits_keep = [
            'team', 'batting_order', 'player_name', 'opp_pitcher', 'opp_pitcher_id', 'opp_pitcher_hand',
            'exp_hits', 'P(H>=1)', 'exp_1b', 'exp_2b', 'exp_3b',
            'hit_pa', '1b_pa', '2b_pa', '3b_pa',
            'display_terminal_flag', 'strict_confirmed_lineup_flag', 'eligible_for_top_boards',
            'top_board_block_reason', 'roster_integrity_note', 'status',
        ]
        top_hits = _filtered_cli_batter_board(df, mode='hits')
        top_hits = top_hits[_safe_cols(top_hits, top_hits_keep)].copy()
        top_hits_out = out.with_name(f'{stem}_top_hits.csv')
        top_hits.to_csv(top_hits_out, index=False)
        written.append(top_hits_out)

    return written




def _run_fd_value_board_cli() -> None:
    print()
    print("[FanDuel HR value board]")
    print("------------------------")

    if not callable(build_and_save_hr_values) or not callable(find_fd_odds_file):
        print("FD value module is not available.")
        return

    odds_path = find_fd_odds_file()

    if odds_path is None:
        if callable(write_fd_odds_template):
            created = write_fd_odds_template("FDodds.csv")
            print(f"No FDodds.csv found. Created template: {created}")
            print("Update FDodds.csv with player_name,fd_odds, then run option 13 again.")
        else:
            print("No FDodds.csv found.")
        return

    print(f"Using odds file: {odds_path}")

    files = [
        p for p in sorted(Path("outputs").glob("batters_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not p.name.endswith("_compact.csv")
        and not p.name.endswith("_top_hr.csv")
        and not p.name.endswith("_top_hits.csv")
        and not p.name.endswith("_hr_values.csv")
        and not p.name.endswith("_top_xbh.csv")
        and not p.name.endswith("_team_totals.csv")
    ]

    if not files:
        print("No full batter output files found in outputs/.")
        return

    default = files[0]
    raw = input(f"Full batter CSV [{default}]: ").strip()
    in_path = Path(raw) if raw else default

    if not in_path.exists():
        print(f"File not found: {in_path}")
        return

    try:
        df = pd.read_csv(in_path)
    except Exception as exc:
        print(f"Could not read {in_path}: {exc}")
        return

    out_path = in_path.with_name(in_path.stem + "_hr_values.csv")

    try:
        values = build_and_save_hr_values(
            df,
            out_path,
            odds_path=odds_path,
            include_no_odds=True,
        )
    except Exception as exc:
        print(f"Could not build value board: {exc}")
        return

    print(f"[wrote] {out_path}")

    show_cols = [
        "team", "player_name", "opp_pitcher", "P(HR>=1)",
        "fd_odds", "fd_implied_prob", "fair_odds_american",
        "edge_pct_pts", "ev_per_100", "value_tag",
        "zone_hr_overall_mult", "zone_hr_confidence",
        "zone_hr_candidate_source",
    ]
    show_cols = [c for c in show_cols if c in values.columns]

    if show_cols:
        print()
        print(values[show_cols].head(25).to_string(index=False))
    else:
        print(values.head(25).to_string(index=False))


def _audit_zone_hr_outputs(outputs_dir: str | Path = "outputs") -> pd.DataFrame:
    """
    Audit saved batter prediction files to see which ones include the HR 3x3 layer.

    This is intentionally lightweight:
      - scans outputs/batters_*.csv
      - excludes compact/top-board files
      - reports whether zone_hr_overall_mult exists
      - reports average zone multiplier/confidence where available
    """
    outputs = Path(outputs_dir)

    rows = []
    files = sorted(outputs.glob("batters_*.csv"))

    for path in files:
        name = path.name

        if (
            name.endswith("_compact.csv")
            or name.endswith("_top_hr.csv")
            or name.endswith("_top_hits.csv")
            or name.endswith("_top_xbh.csv")
            or name.endswith("_team_totals.csv")
            or name.endswith("_breakout_hr_board.csv")
        ):
            continue

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            rows.append({
                "File": name,
                "Rows": 0,
                "HasZoneHR": False,
                "ZoneRows": 0,
                "AvgZoneMult": np.nan,
                "AvgZoneConf": np.nan,
                "Sources": "",
                "Status": f"read_error: {exc}",
            })
            continue

        if df is None or df.empty:
            rows.append({
                "File": name,
                "Rows": 0,
                "HasZoneHR": False,
                "ZoneRows": 0,
                "AvgZoneMult": np.nan,
                "AvgZoneConf": np.nan,
                "Sources": "",
                "Status": "empty",
            })
            continue

        has_zone = "zone_hr_overall_mult" in df.columns

        if has_zone:
            zm = pd.to_numeric(df.get("zone_hr_overall_mult"), errors="coerce")
            zc = pd.to_numeric(df.get("zone_hr_confidence"), errors="coerce") if "zone_hr_confidence" in df.columns else pd.Series(np.nan, index=df.index)
            zone_rows = int(zm.notna().sum())
            avg_mult = float(zm.mean(skipna=True)) if zm.notna().any() else np.nan
            avg_conf = float(zc.mean(skipna=True)) if zc.notna().any() else np.nan

            if "zone_hr_candidate_source" in df.columns:
                srcs = (
                    df["zone_hr_candidate_source"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .replace("", np.nan)
                    .dropna()
                    .value_counts()
                    .head(3)
                )
                src_txt = ", ".join([f"{idx}:{int(val)}" for idx, val in srcs.items()])
            else:
                src_txt = ""
        else:
            zone_rows = 0
            avg_mult = np.nan
            avg_conf = np.nan
            src_txt = ""

        rows.append({
            "File": name,
            "Rows": int(len(df)),
            "HasZoneHR": bool(has_zone),
            "ZoneRows": zone_rows,
            "AvgZoneMult": round(avg_mult, 4) if np.isfinite(avg_mult) else np.nan,
            "AvgZoneConf": round(avg_conf, 4) if np.isfinite(avg_conf) else np.nan,
            "Sources": src_txt,
            "Status": "ok",
        })

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(["HasZoneHR", "File"], ascending=[False, True], kind="stable").reset_index(drop=True)

    return out


def _print_zone_hr_output_audit() -> None:
    audit = _audit_zone_hr_outputs("outputs")

    print()
    print("[Zone HR 3x3 output audit]")
    print("---------------------------")

    if audit.empty:
        print("No batter output files found in outputs/.")
        return

    print(audit.to_string(index=False))

    total_files = int(len(audit))
    zone_files = int(audit["HasZoneHR"].fillna(False).sum())
    total_rows = int(pd.to_numeric(audit["Rows"], errors="coerce").fillna(0).sum())
    zone_rows = int(pd.to_numeric(audit["ZoneRows"], errors="coerce").fillna(0).sum())

    print()
    print(f"Files with ZoneHR: {zone_files}/{total_files}")
    print(f"Rows with ZoneHR:  {zone_rows}/{total_rows}")

    if zone_files < 7:
        print()
        print("Calibration note: not enough new-zone files yet for reliable HR 3x3 calibration.")
        print("Target before tuning alpha: ~7-10 completed slates, 1,000+ batter rows, and 40-50+ actual HR events.")
    else:
        print()
        print("Calibration note: sample size may be getting large enough to run the zone HR backtest.")


def _save_and_print(df: pd.DataFrame, out: Path | None = None) -> None:
    if not isinstance(df, pd.DataFrame):
        _print_df(pd.DataFrame())
        return

    df = _normalize_date_df(df)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.stem.startswith('ko_'):
            df = _try_apply_ko_calibration(df)
        df.to_csv(out, index=False)
        print(f'\n[wrote] {out}')
        for p in _compact_subset_csvs(df, out):
            print(f'[wrote] {p}')

        stem = out.stem
        if stem.startswith('ko_'):
            print('\n[KO compact view]')
            print(compact_ko_terminal(df, max_rows=20))
            return
        if stem.startswith('batters_'):
            print('\n[Batters compact view - top hit angles]')
            print(compact_batters_terminal(df, max_rows=20, mode='hits'))
            print('\n[Batters compact view - top HR angles]')
            print(compact_batters_terminal(df, max_rows=20, mode='hr'))
            return

    _print_df(df)


def _run_stage_and_save(stage_name: str, fn, out: Path) -> pd.DataFrame:
    df = fn()
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'{stage_name} did not return a DataFrame')
    df = _normalize_date_df(df)
    _save_and_print(df, out)
    return df


def _pick_date(default: str | None = None) -> str:
    default = default or date.today().isoformat()
    raw = input(f'Date [YYYY-MM-DD] [{default}]: ').strip()
    return raw or default


def _pick_matchup(date_str: str) -> str | None:
    games = _runtime()['fetch_matchups_for_date'](date_str)
    if not games:
        print(f'No games found for {date_str}.')
        return None
    print(f'\nMatchups for {date_str}:')
    for i, gm in enumerate(games, 1):
        print(f'  {i}) {gm}')
    raw = input('Choose matchup # (or press Enter for full slate): ').strip()
    if not raw:
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(games):
        return games[int(raw) - 1].replace(' @ ', '@')
    return raw.replace(' @ ', '@')


def _menu_build_refresh() -> None:
    rt = _runtime()
    print('\nRefreshing auto defense proxies...')
    df = rt['auto_build_team_defense_proxies'](force=True)
    _print_df(df)
    print('\nRunning precompute / training build...')
    rc = _run_python('precompute_everything.py', [])
    if rc != 0:
        print(f'[warn] precompute exited with code {rc}')


def _menu_predict_ko() -> None:
    rt = _runtime()
    d = _pick_date()
    matchup = _pick_matchup(d)
    if not matchup:
        print('A single matchup is required for KO output.')
        return
    df = rt['predict_matchup_ko'](d, matchup)
    out = _ensure_outputs() / f'ko_{d}_{matchup.replace("@", "_")}.csv'
    _save_and_print(df, out)


def _menu_predict_batters(matchup_required: bool) -> None:
    rt = _runtime()
    d = _pick_date()
    matchup = _pick_matchup(d)
    if matchup_required and not matchup:
        print('A single matchup is required.')
        return
    if matchup:
        df = rt['predict_matchup_batters'](d, matchup)
        out = _ensure_outputs() / f'batters_{d}_{matchup.replace("@", "_")}.csv'
    else:
        df = rt['predict_slate_batters'](d)
        out = _ensure_outputs() / f'batters_slate_{d}.csv'
    _save_and_print(df, out)


def _menu_daily_pipeline(matchup_required: bool) -> None:
    rt = _runtime()
    d = _pick_date()
    matchup = _pick_matchup(d)
    if matchup_required and not matchup:
        print('A single matchup is required.')
        return
    pre = input('Run precompute first? [y/N]: ').strip().lower()
    if pre in {'y', 'yes'}:
        _menu_build_refresh()
    outdir = _ensure_outputs()
    if matchup:
        ko_df = rt['predict_matchup_ko'](d, matchup)
        bat_df = rt['predict_matchup_batters'](d, matchup)
        _save_and_print(ko_df, outdir / f'ko_{d}_{matchup.replace("@", "_")}.csv')
        _save_and_print(bat_df, outdir / f'batters_{d}_{matchup.replace("@", "_")}.csv')
    else:
        ko_df = rt['predict_slate_ko'](d)
        bat_df = rt['predict_slate_batters'](d)
        _save_and_print(ko_df, outdir / f'ko_slate_{d}.csv')
        _save_and_print(bat_df, outdir / f'batters_slate_{d}.csv')


def _menu_walk_forward() -> None:
    rt = _runtime()
    if rt.get('load_any_table') is None or rt.get('fit_walk_forward_summary') is None:
        print('[error] Calibration summary helpers are not available in the current mlb_model.calibration module.')
        return
    preds_raw = input('Predictions file [csv/parquet]: ').strip()
    acts_raw = input('Actuals file [csv/parquet]: ').strip()
    if not preds_raw or not acts_raw:
        print('[error] Predictions and actuals file paths are required.')
        return
    preds = Path(preds_raw)
    acts = Path(acts_raw)
    pred_col = input('Prediction probability column [P(HR>=1)]: ').strip() or 'P(HR>=1)'
    actual_col = input('Actual result column [hit]: ').strip() or 'hit'
    key_cols = [k.strip() for k in (input('Join keys comma-separated [date,matchup,player_name]: ').strip() or 'date,matchup,player_name').split(',') if k.strip()]
    out = rt['fit_walk_forward_summary'](
        rt['load_any_table'](preds),
        rt['load_any_table'](acts),
        key_cols=key_cols,
        pred_col=pred_col,
        actual_col=actual_col,
    )
    _print_df(out)


def _menu_benchmark_lines() -> None:
    rt = _runtime()
    if rt.get('load_any_table') is None or rt.get('benchmark_closing_lines') is None:
        print('[error] Closing-line benchmark helpers are not available in the current mlb_model.calibration module.')
        return
    preds_raw = input('Predictions file [csv/parquet]: ').strip()
    lines_raw = input('Closing-lines file [csv/parquet]: ').strip()
    if not preds_raw or not lines_raw:
        print('[error] Predictions and closing-lines file paths are required.')
        return
    preds = Path(preds_raw)
    lines = Path(lines_raw)
    pred_col = input('Model probability column [P(HR>=1)]: ').strip() or 'P(HR>=1)'
    book_col = input('Book implied probability column [book_prob]: ').strip() or 'book_prob'
    key_cols = [k.strip() for k in (input('Join keys comma-separated [date,matchup,player_name]: ').strip() or 'date,matchup,player_name').split(',') if k.strip()]
    out = rt['benchmark_closing_lines'](
        rt['load_any_table'](preds),
        rt['load_any_table'](lines),
        key_cols=key_cols,
        pred_prob_col=pred_col,
        book_prob_col=book_col,
    )
    _print_df(out)


def _menu_report(full_slate: bool) -> None:
    rt = _runtime()
    d = _pick_date()
    if full_slate:
        bundle = rt['slate_prediction_report'](d)
        stem = f'slate_{d}'
    else:
        matchup = _pick_matchup(d)
        if not matchup:
            print('A single matchup is required for a matchup report.')
            return
        bundle = rt['matchup_prediction_report'](d, matchup)
        stem = f'{d}_{matchup.replace("@", "_")}'
    print()
    print(bundle['text'])
    save = input('Save report + CSV bundle to outputs/? [Y/n]: ').strip().lower()
    if save not in {'n', 'no'}:
        paths = rt['save_report_bundle'](bundle, _ensure_outputs(), stem)
        for k, v in paths.items():
            print(f'[wrote:{k}] {v}')


def _menu_replay() -> None:
    rt = _runtime()
    start = input('Start date [YYYY-MM-DD]: ').strip()
    end = input('End date   [YYYY-MM-DD]: ').strip()
    if not start or not end:
        print('[error] Start and end dates are required for replay.')
        return
    outdir = _ensure_outputs() / f'replay_{start}_to_{end}'
    out = rt['run_season_replay'](start, end, outdir)
    _print_df(out)


def _interactive_menu() -> int:
    while True:
        print('\n=== Dingers Monster Menu ===')
        print('  1) Build / refresh precompute caches')
        print('  2) Refresh auto defense proxies only')
        print('  3) Pitcher KO output for one matchup')
        print('  4) Batter outcomes for one matchup')
        print('  5) Full daily pipeline for one matchup')
        print('  6) Full daily pipeline for full slate')
        print('  7) Prediction report for one matchup')
        print('  8) Prediction report for full slate')
        print('  9) Full-season replay / backtest')
        print(' 10) Walk-forward calibration summary from saved files')
        print(' 11) Closing-line benchmark from saved files')
        print(' 12) Zone HR 3x3 output audit')
        print(' 13) FanDuel HR value board')
        print('  0) Exit')
        choice = input('Select option: ').strip()
        try:
            if choice == '13':

                _run_fd_value_board_cli()

                continue


            if choice == '12':

                _print_zone_hr_output_audit()

                continue


            if choice == '0':
                return 0
            if choice == '1':
                _menu_build_refresh()
            elif choice == '2':
                _print_df(_runtime()['auto_build_team_defense_proxies'](force=True))
            elif choice == '3':
                _menu_predict_ko()
            elif choice == '4':
                _menu_predict_batters(matchup_required=True)
            elif choice == '5':
                _menu_daily_pipeline(matchup_required=True)
            elif choice == '6':
                _menu_daily_pipeline(matchup_required=False)
            elif choice == '7':
                _menu_report(full_slate=False)
            elif choice == '8':
                _menu_report(full_slate=True)
            elif choice == '9':
                _menu_replay()
            elif choice == '10':
                _menu_walk_forward()
            elif choice == '11':
                _menu_benchmark_lines()
            else:
                print('Invalid selection. Try again.')
        except KeyboardInterrupt:
            print('\nCancelled. Returning to menu.')
        except Exception as exc:
            print(f'[error] {exc}')


def main() -> int:
    if len(sys.argv) == 1:
        return _interactive_menu()
    ap = argparse.ArgumentParser(description='Dingers Monster MLB toolkit')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('cache-all')
    sub.add_parser('refresh-defense')
    p_ko = sub.add_parser('predict-ko')
    p_ko.add_argument('--date', required=True)
    p_ko.add_argument('--matchup')
    p_ko.add_argument('--csv')
    p_b = sub.add_parser('predict-batters')
    p_b.add_argument('--date', required=True)
    p_b.add_argument('--matchup')
    p_b.add_argument('--csv')
    p_d = sub.add_parser('daily-pipeline')
    p_d.add_argument('--date', required=True)
    p_d.add_argument('--matchup')
    p_d.add_argument('--precompute', action='store_true')
    p_r = sub.add_parser('prediction-report')
    p_r.add_argument('--date', required=True)
    p_r.add_argument('--matchup')
    p_r.add_argument('--outdir')
    p_s = sub.add_parser('season-replay')
    p_s.add_argument('--start', required=True)
    p_s.add_argument('--end', required=True)
    p_s.add_argument('--outdir')
    args = ap.parse_args()
    rt = _runtime()

    if args.cmd == 'cache-all':
        rt['auto_build_team_defense_proxies'](force=True)
        return _run_python('precompute_everything.py', [])
    if args.cmd == 'refresh-defense':
        _print_df(rt['auto_build_team_defense_proxies'](force=True))
        return 0
    if args.cmd == 'predict-ko':
        df = rt['predict_matchup_ko'](args.date, args.matchup) if args.matchup else rt['predict_slate_ko'](args.date)
        _save_and_print(df, Path(args.csv) if args.csv else None)
        return 0
    if args.cmd == 'predict-batters':
        df = rt['predict_matchup_batters'](args.date, args.matchup) if args.matchup else rt['predict_slate_batters'](args.date)
        _save_and_print(df, Path(args.csv) if args.csv else None)
        return 0
    if args.cmd == 'daily-pipeline':
        if args.precompute:
            _menu_build_refresh()
        outdir = _ensure_outputs()
        if args.matchup:
            stem = f"{args.date}_{args.matchup.replace('@', '_')}"
            print('[stage] Building KO output...')
            _run_stage_and_save('ko', lambda: rt['predict_matchup_ko'](args.date, args.matchup), outdir / f'ko_{stem}.csv')
            print('[stage] Building batter output...')
            _run_stage_and_save('batters', lambda: rt['predict_matchup_batters'](args.date, args.matchup), outdir / f'batters_{stem}.csv')
        else:
            print('[stage] Building KO slate output...')
            _run_stage_and_save('ko_slate', lambda: rt['predict_slate_ko'](args.date), outdir / f'ko_slate_{args.date}.csv')
            print('[stage] Building batter slate output...')
            _run_stage_and_save('batters_slate', lambda: rt['predict_slate_batters'](args.date), outdir / f'batters_slate_{args.date}.csv')
        return 0
    if args.cmd == 'prediction-report':
        bundle = rt['matchup_prediction_report'](args.date, args.matchup) if args.matchup else rt['slate_prediction_report'](args.date)
        outdir = Path(args.outdir) if args.outdir else _ensure_outputs()
        stem = f"{args.date}_{args.matchup.replace('@', '_')}" if args.matchup else f'slate_{args.date}'
        print(bundle['text'])
        rt['save_report_bundle'](bundle, outdir, stem)
        return 0
    if args.cmd == 'season-replay':
        outdir = Path(args.outdir) if args.outdir else (_ensure_outputs() / f'replay_{args.start}_to_{args.end}')
        _print_df(rt['run_season_replay'](args.start, args.end, outdir))
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
