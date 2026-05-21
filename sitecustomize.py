
"""
sitecustomize.py

Put this file in the Dingers_hotfix project root as:
    sitecustomize.py

Goal:
- Option 6 / 7 batter output prints a compact all-player table in terminal
- KO table stays unchanged
- Full CSVs are still written untouched
- Compact companion CSVs are still written
"""

from __future__ import annotations

import pandas as pd
import os
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


# ----------------------------
# Helpers
# ----------------------------
def _safe_cols(df: "pd.DataFrame", keep: Iterable[str]) -> list[str]: # type: ignore
    return [c for c in keep if c in df.columns]


def _round_df(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame: # type: ignore
    out = df.copy()
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].round(digits)
    return out


def _clean_status(df: pd.DataFrame) -> pd.DataFrame: # type: ignore
    out = df.copy()
    if "status" in out.columns:
        out["status"] = out["status"].fillna("")
    return out


# ----------------------------
# Compact batter views
# ----------------------------
BATTER_TERMINAL_KEEP = [
    "matchup", "team", "batting_order", "player_name", "bats",
    "opp_pitcher", "exp_pa",
    "exp_hits", "exp_1b", "exp_2b", "exp_3b", "exp_hr",
    "P(H>=1)", "P(HR>=1)",
    "hit_pa", "hr_pa",
    "status",
]

BATTER_COMPACT_KEEP = [
    "date", "matchup", "team", "batting_order", "player_name", "player_id",
    "bats", "opp_pitcher", "opp_pitcher_id", "opp_pitcher_hand",
    "lineup_source", "lineup_weight", "lineup_certainty", "exp_pa",
    "exp_hits", "exp_1b", "exp_2b", "exp_3b", "exp_hr",
    "P(H>=1)", "P(1B>=1)", "P(2B>=1)", "P(3B>=1)", "P(HR>=1)",
    "hit_pa", "1b_pa", "2b_pa", "3b_pa", "hr_pa",
    "starter_share", "bullpen_hr_mult", "park_hr_mult",
    "batter_contact_mult", "batter_xbh_mult", "batter_hr_mult", "batter_form_hr_mult",
    "pitcher_hr_danger_mult", "pitcher_recent_danger_mult",
    "zone_hr_overall_mult",
    "zone_hr_damage_mult",
    "zone_hr_barrel_mult",
    "zone_hr_air_mult",
    "zone_hr_confidence",
    "zone_hr_candidate_source",
    "zone_hr_sample_regime",
    "zone_hr_batter_archetype",
    "zone_hr_pitcher_arch",
    "zone_hr_exact_batter_rows",
    "zone_hr_batter_archetype_rows",
    "zone_hr_pitcher_archetype_rows",
    "zone_hr_hand_rows",
    "zone_hr_league_rows",
    "zone_hr_early_season_shrink",
 "status",
]

BATTER_TOP_HR_KEEP = [
    "team", "matchup", "batting_order", "player_name", "opp_pitcher", "opp_pitcher_hand",
    "exp_hr", "P(HR>=1)", "hr_pa", "starter_share", "bullpen_hr_mult",
    "park_hr_mult", "batter_hr_mult", "batter_form_hr_mult",
    "pitcher_hr_danger_mult", "pitcher_recent_danger_mult",
    "zone_hr_overall_mult",
    "zone_hr_damage_mult",
    "zone_hr_barrel_mult",
    "zone_hr_air_mult",
    "zone_hr_confidence",
    "zone_hr_candidate_source",
    "zone_hr_sample_regime",
    "zone_hr_batter_archetype",
    "zone_hr_pitcher_arch",
    "zone_hr_exact_batter_rows",
    "zone_hr_batter_archetype_rows",
    "zone_hr_pitcher_archetype_rows",
    "zone_hr_hand_rows",
    "zone_hr_league_rows",
    "zone_hr_early_season_shrink",
 "status",
]

BATTER_TOP_HITS_KEEP = [
    "team", "matchup", "batting_order", "player_name", "opp_pitcher", "opp_pitcher_hand",
    "exp_hits", "P(H>=1)", "exp_1b", "exp_2b", "exp_3b",
    "hit_pa", "1b_pa", "2b_pa", "3b_pa", "status",
]


def _looks_like_batter_df(df: "pd.DataFrame") -> bool: # type: ignore
    cols = set(df.columns)
    return {"player_name", "exp_hits", "exp_hr"}.issubset(cols)


def _make_batter_terminal(df: "pd.DataFrame") -> "pd.DataFrame": # type: ignore
    out = df[_safe_cols(df, BATTER_TERMINAL_KEEP)].copy()
    out = _clean_status(out)
    sort_cols = [c for c in ["matchup", "team", "batting_order"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    return _round_df(out, 3)


def _make_batter_compact(df: "pd.DataFrame") -> "pd.DataFrame": # type: ignore
    out = df[_safe_cols(df, BATTER_COMPACT_KEEP)].copy()
    out = _clean_status(out)
    sort_cols = [c for c in ["matchup", "team", "batting_order"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    return _round_df(out, 3)


def _make_batter_top_hr(df: "pd.DataFrame") -> "pd.DataFrame": # type: ignore
    out = df[_safe_cols(df, BATTER_TOP_HR_KEEP)].copy()
    out = _clean_status(out)
    sort_cols = [c for c in ["P(HR>=1)", "exp_hr"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return _round_df(out, 3)


def _make_batter_top_hits(df: "pd.DataFrame") -> "pd.DataFrame": # type: ignore
    out = df[_safe_cols(df, BATTER_TOP_HITS_KEEP)].copy()
    out = _clean_status(out)
    sort_cols = [c for c in ["P(H>=1)", "exp_hits"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return _round_df(out, 3)


# ----------------------------
# Auto-create companion CSVs
# ----------------------------
def _write_pretty_variants(csv_path: os.PathLike | str, df: "pd.DataFrame") -> None: # type: ignore
    path = Path(csv_path)
    name = path.name.lower()

    if any(tag in name for tag in ["_compact.csv", "_top_hr.csv", "_top_hits.csv"]):
        return

    try:
        if name.startswith("batters_") and path.suffix.lower() == ".csv":
            compact = _make_batter_compact(df)
            top_hr = _make_batter_top_hr(df)
            top_hits = _make_batter_top_hits(df)

            compact_path = path.with_name(path.stem + "_compact.csv")
            top_hr_path = path.with_name(path.stem + "_top_hr.csv")
            top_hits_path = path.with_name(path.stem + "_top_hits.csv")

            compact.to_csv(compact_path, index=False)
            top_hr.to_csv(top_hr_path, index=False)
            top_hits.to_csv(top_hits_path, index=False)

            print(f"[pretty] wrote {compact_path}")
            print(f"[pretty] wrote {top_hr_path}")
            print(f"[pretty] wrote {top_hits_path}")
    except Exception as e:
        print(f"[pretty][warn] failed to build readable batter output views for {path.name}: {e}")


# ----------------------------
# Patch terminal printing
# ----------------------------
def _install_terminal_pretty_print() -> None:
    if pd is None:
        return
    if getattr(pd.DataFrame, "_dingers_batter_print_patched", False):
        return

    # Display options only; KO can still print as normal
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 18)
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.max_colwidth", 28)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.show_dimensions", False)
    pd.options.display.float_format = lambda v: f"{v:0.3f}"

    original_repr = pd.DataFrame.__repr__
    original_to_string = pd.DataFrame.to_string

    def patched_repr(self):
        try:
            if _looks_like_batter_df(self):
                return _make_batter_terminal(self).to_string(index=False)
            return original_repr(self)
        except Exception:
            return original_repr(self)

    def patched_to_string(self, *args, **kwargs):
        try:
            if _looks_like_batter_df(self):
                use_index = kwargs.get("index", False)
                return original_to_string(_make_batter_terminal(self), index=use_index)
            return original_to_string(self, *args, **kwargs)
        except Exception:
            return original_to_string(self, *args, **kwargs)

    pd.DataFrame.__repr__ = patched_repr
    pd.DataFrame.to_string = patched_to_string
    pd.DataFrame._dingers_batter_print_patched = True


# ----------------------------
# Hook CSV writes
# ----------------------------
def _install_to_csv_monkeypatch() -> None:
    if pd is None:
        return
    if getattr(pd.DataFrame.to_csv, "_dingers_batter_csv_patched", False):
        return

    original_to_csv = pd.DataFrame.to_csv

    def patched_to_csv(self, path_or_buf=None, *args, **kwargs):
        result = original_to_csv(self, path_or_buf, *args, **kwargs)
        if isinstance(path_or_buf, (str, os.PathLike)):
            _write_pretty_variants(path_or_buf, self.copy())
        return result

    patched_to_csv._dingers_batter_csv_patched = True
    pd.DataFrame.to_csv = patched_to_csv


if pd is not None:
    _install_terminal_pretty_print()
    _install_to_csv_monkeypatch()
