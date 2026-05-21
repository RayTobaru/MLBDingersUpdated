#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pitchers-only game day tool:
- Lists slate
- Lets you pick a matchup
- Prints Starter KO probabilities for both starters

Dependencies expected from your repo:
  - fetch.py: ID2ABBR, fetch_lineup_and_starters, fetch_unofficial_lineup, safe_get,
              pitching_stats, clean_name (optional)
  - precompute_everything.py: kt_montecarlo, predict_k9, IP_REG, sample_starter_ip,
                              features_base, league_feature_means
"""

import os, io, sys, argparse, logging, warnings
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, date as _date

import numpy as np
import pandas as pd

# silence pybaseball banner (optional)
import builtins
_orig_print = builtins.print
def _quiet_print(*args, **kwargs):
    if args and isinstance(args[0], str) and args[0].startswith("Gathering Player Data"):
        return
    _orig_print(*args, **kwargs)
builtins.print = _quiet_print

import fetch
from fetch import (
    ID2ABBR, fetch_lineup_and_starters, fetch_unofficial_lineup,
    safe_get, pitching_stats, clean_name
)

from precompute_everything import (
    kt_montecarlo, predict_k9, IP_REG, sample_starter_ip,
    features_base, league_feature_means
)

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.3f}".format)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pybaseball").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.INFO)

YEAR = datetime.today().year


# ---------- helpers ----------
def _resolve_date(args) -> str:
    base = _date.today()
    if getattr(args, "tomorrow", False):
        base = base + timedelta(days=1)
    elif getattr(args, "yesterday", False):
        base = base - timedelta(days=1)
    elif getattr(args, "date", None):
        return args.date
    return base.strftime("%Y-%m-%d")


def fetch_matchups_for_date(date_str: str):
    resp = safe_get("https://statsapi.mlb.com/api/v1/schedule",
                    {"sportId": 1, "date": date_str})
    js = resp.json() if resp else {}
    games = []
    for d in js.get("dates", []):
        for g in d.get("games", []):
            try:
                away_id = g["teams"]["away"]["team"]["id"]
                home_id = g["teams"]["home"]["team"]["id"]
                games.append(f"{ID2ABBR[away_id]} @ {ID2ABBR[home_id]}")
            except Exception:
                continue
    return games


def build_starter_dict(team, name, pid, ump_feats=None, framing_feats=None):
    """
    Minimal pitcher feature row needed by kt_montecarlo & downstream K projections.
    Uses season stats + (optional) Statcast pitcher features if available.
    """
    r = {f: league_feature_means.get(f, 0.0) for f in features_base}
    df = pitching_stats(YEAR, qual=0)

    # find pitcher by id OR by cleaned name
    rec = pd.DataFrame()
    if pid is not None:
        id_col = next((c for c in df.columns if c.lower() in ("pid","playerid","player_id","mlbam_id")), None)
        if id_col:
            try:
                df[id_col] = df[id_col].astype(int)
                rec = df[df[id_col] == int(pid)]
            except Exception:
                rec = df[df[id_col] == pid]
    if rec.empty and isinstance(name, str):
        df["Name_norm"] = df.Name.apply(clean_name)
        rec = df[df.Name_norm == clean_name(name)]

    # baseline IP per start & a few rate stats if we found him
    if not rec.empty:
        rec = rec.iloc[0]
        ipps = float(rec.IP / max(rec.GS, 1)) if rec.GS and rec.GS > 0 else 5.5
        for c in ("FIP","WHIP","K%","BB%","SwStr%"):
            if c in rec and pd.notna(rec[c]):
                r[c] = float(rec[c])
    else:
        ipps = 5.5  # league-ish fallback

    # optional: pull statcast pitcher feature vector (if your fetch has it)
    if pid:
        sc = fetch.get_statcast_pitcher_features(pid) or {}
        for k, v in sc.items():
            if k in r and pd.notna(v):
                r[k] = float(v)

    r["IP_per_start"] = ipps
    r["Pred_K9"] = float(predict_k9(pd.Series(r)))

    # project IP for the start (uses your trained regression)
    pred_mean_ip = float(IP_REG.predict([[ipps, r["Pred_K9"], r["Pred_K9"]]])[0])
    r["Proj_IP"] = float(min(sample_starter_ip(pred_mean_ip, r["Pred_K9"]), 8.0))

    # light contextual tweaks (ump/framing) if available
    if ump_feats and "ump_k9" in ump_feats:
        r["Pred_K9"] *= float(ump_feats["ump_k9"])
    if framing_feats:
        # keys are usually 'away_frame' / 'home_frame' as % impact
        for k in ("away_frame","home_frame"):
            if k in framing_feats and pd.notna(framing_feats[k]):
                r["Pred_K9"] *= (1.0 + float(framing_feats[k])/100.0)

    r.update({"Team": team, "Name": name or "", "Pitcher_ID": pid})
    return r


# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="Pitchers-only KO model")
    p.add_argument("-g","--game", type=int, help="Pick by index from slate")
    p.add_argument("-m","--matchup", help="Pick by exact string 'AWY @ HOM'")
    p.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--today", action="store_true", help="Force today")
    p.add_argument("--tomorrow", action="store_true", help="Force tomorrow")
    p.add_argument("--yesterday", action="store_true", help="Force yesterday")
    p.add_argument("--csv", type=str, default=None, help="Write KO table to CSV")
    args = p.parse_args()

    use_date = _resolve_date(args)
    games = fetch_matchups_for_date(use_date)
    if not games:
        sys.exit("No MLB games on that date.")
    for i, gm in enumerate(games, 1):
        print(f"{i}. {gm}")

    if args.game and 1 <= args.game <= len(games):
        sel = str(args.game)
    elif args.matchup:
        sel = args.matchup
    else:
        sel = input("Pick # or 'AWY @ HOM': ").strip()

    away, home = (games[int(sel)-1].split(" @ ") if sel.isdigit()
                  else tuple(map(str.strip, sel.split("@"))))

    # lineups + starters (use unofficial if official not hydrated yet)
    try:
        (a_name,h_name,a_line,h_line,a_pid,h_pid, away_cid,home_cid, ump_feats,framing_feats) = \
            fetch_lineup_and_starters(away, home)
    except RuntimeError:
        logging.warning(f"Official lineup not ready for {away}@{home}, using unofficial.")
        a_line = fetch_unofficial_lineup(away)
        h_line = fetch_unofficial_lineup(home)
        a_name = h_name = None
        a_pid = h_pid = None
        ump_feats = {"ump_k9":1.0}
        framing_feats = {"away_frame":0.0,"home_frame":0.0}

    # build starter feature rows
    away_r = build_starter_dict(
        away, a_name, a_pid,
        ump_feats=ump_feats,
        framing_feats={"away_frame":framing_feats.get("away_frame",0.0)}
    )
    home_r = build_starter_dict(
        home, h_name, h_pid,
        ump_feats=ump_feats,
        framing_feats={"home_frame":framing_feats.get("home_frame",0.0)}
    )

    # run KO model for each starter, conditioned on opposing lineup
    starter_rows = []
    for team, st, lineup in [(away, away_r, a_line), (home, home_r, h_line)]:
        sr = kt_montecarlo(st, lineup)  # expects dict with probabilities & summary stats
        sr.update({
            "Team": team,
            "Name": st["Name"],
            "Pitcher_ID": st["Pitcher_ID"],
            "Proj_IP": st["Proj_IP"],
            "IP_per_start": st["IP_per_start"],
            "Pred_K9": st["Pred_K9"],
        })
        starter_rows.append(sr)

    # pretty table
    df = pd.DataFrame(starter_rows)
    # columns provided by your kt_montecarlo: adapt if your keys differ
    prob_cols = [c for c in df.columns if c.startswith("P(K≥")]
    base_cols = ["Team","Name","Pitcher_ID","Proj_IP","IP_per_start","Pred_K9",
                 "Mean_K_start","Median_K_start"]
    if "90% CI" in df.columns:
        base_cols = base_cols + prob_cols + ["90% CI"]
    else:
        base_cols = base_cols + prob_cols

    print("\n=== Starter KO Probabilities ===")
    print(df[base_cols].to_string(index=False))

    if args.csv:
        df[base_cols].to_csv(args.csv, index=False)
        print(f"\n[wrote] {args.csv}")


if __name__ == "__main__":
    main()
