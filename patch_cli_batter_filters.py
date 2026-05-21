from pathlib import Path
import re
import sys

CLI_PATH = Path("cli.py")

NEW_HELPERS = """
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
"""

NEW_COMPACT = """
def compact_batters_terminal(df: pd.DataFrame, max_rows: int = 20, mode: str = 'hits') -> str:
    board = _filtered_cli_batter_board(df, mode=mode)

    if mode == 'hr':
        keep = [
            'team', 'batting_order', 'player_name', 'opp_pitcher',
            'exp_hr', 'P(HR>=1)', 'exp_hr_ctx', 'hr_pa',
            'batter_hr_mult', 'batter_form_hr_mult',
            'pitcher_hr_danger_mult', 'status',
        ]
    else:
        keep = [
            'team', 'batting_order', 'player_name', 'opp_pitcher',
            'exp_hits', 'P(H>=1)', 'exp_1b', 'exp_2b', 'exp_3b', 'exp_hr',
            'hit_pa', 'hr_pa', 'status',
        ]

    out = board[_safe_cols(board, keep)].copy()
    if 'status' in out.columns:
        out['status'] = out['status'].fillna('')
    out = _round_display(out, 3)
    return _to_terminal_string(out, max_rows=max_rows)
"""

NEW_COMPANION = """
        top_hr_keep = [
            'team', 'batting_order', 'player_name', 'opp_pitcher', 'opp_pitcher_id', 'opp_pitcher_hand',
            'exp_hr', 'exp_hr_ctx', 'P(HR>=1)', 'hr_pa',
            'raw_batter_hr_mult', 'batter_hr_mult', 'batter_form_hr_mult',
            'raw_pitcher_hr_danger_mult', 'pitcher_hr_danger_mult', 'pitcher_recent_danger_mult',
            'display_terminal_flag', 'strict_confirmed_lineup_flag', 'eligible_for_top_boards',
            'top_board_block_reason', 'roster_integrity_note', 'status',
        ]
        top_hr = _filtered_cli_batter_board(df, mode='hr')
        top_hr = top_hr[_safe_cols(top_hr, top_hr_keep)].copy()
        top_hr_out = out.with_name(f'{stem}_top_hr.csv')
        top_hr.to_csv(top_hr_out, index=False)
        written.append(top_hr_out)

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
"""

def main():
    if not CLI_PATH.exists():
        print("Could not find cli.py in the current folder.")
        sys.exit(1)

    src = CLI_PATH.read_text(encoding="utf-8")

    if "_eligible_cli_batter_mask" not in src:
        marker = "def compact_batters_terminal(df: pd.DataFrame, max_rows: int = 20, mode: str = 'hits') -> str:\n"
        if marker not in src:
            print("Could not find compact_batters_terminal in cli.py")
            sys.exit(1)
        src = src.replace(marker, NEW_HELPERS + "\n" + marker, 1)

    src, n1 = re.subn(
        r"def compact_batters_terminal\(df: pd\.DataFrame, max_rows: int = 20, mode: str = 'hits'\) -> str:\n(?:    .*\n)+?(?=def _save_and_print)",
        NEW_COMPACT + "\n",
        src,
        count=1,
    )

    src, n2 = re.subn(
        r"        top_hr_keep = \[\n(?:.*\n)+?        written\.append\(top_hits_out\)\n",
        NEW_COMPANION,
        src,
        count=1,
    )

    CLI_PATH.write_text(src, encoding="utf-8")
    print("Patched cli.py successfully.")
    print(f"compact_batters_terminal replaced: {n1}")
    print(f"companion top_hr/top_hits block replaced: {n2}")

if __name__ == '__main__':
    main()
