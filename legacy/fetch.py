#!/usr/bin/env python3
"""
fetch.py

All raw-data fetching and low-level helpers:
  • HTTP retry logic
  • name cleanup & ID lookup
  • MLB roster & lineup fetchers
  • Park factors (yearly & monthly)
  • Player PA maps & recent-form lookups (7d/14d/30d)
  • Bulk Statcast download + in-memory filtering
  • Extended Statcast feature stubs (whiff/pitch-type, reverse splits)
  • Batter-vs-Pitcher H2H rates
  • Relief-pitcher selection helpers
  • Bullpen profiling

Refinements:
  • Parquet DataFrame caching (faster, schema-safe) with pickle fallback
  • Pitch-mix & batter-by-pitch xISO helpers (for matchup-aware HR/TB bumps)
"""
import sys
import os
import time
import logging
import re
import functools
import pickle
import requests
import numpy as np
import pandas as pd
import io
import random
from collections import defaultdict

from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta, date
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from multiprocessing.pool import ThreadPool
from bs4 import BeautifulSoup
from pandas.errors import ParserError
from functools import lru_cache
from dateutil import parser

from pybaseball import (
    batting_stats,
    batting_stats_range,
    pitching_stats,
    playerid_lookup,
    statcast
)

from requests.exceptions import ConnectionError as ReqConnErr
from urllib3.exceptions import ProtocolError

def _safe_mean(series, default=0.0):
    try:
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=False)
        if arr.size == 0:
            return float(default)
        mask = np.isfinite(arr)
        if not mask.any():
            return float(default)
        return float(arr[mask].mean())
    except Exception:
        return float(default)

def _safe_std(series, default=0.0):
    try:
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=False)
        if arr.size == 0:
            return float(default)
        mask = np.isfinite(arr)
        if mask.sum() < 2:
            return float(default)
        return float(arr[mask].std(ddof=0))
    except Exception:
        return float(default)


def _safe_frac(mask_like, default=0.0):
    try:
        # Works for bool/0-1 Series/arrays, robust to NA
        arr = pd.Series(mask_like, copy=False).astype("float64").to_numpy(copy=False)
        if arr.size == 0:
            return float(default)
        m = np.nanmean(arr)
        return float(default) if np.isnan(m) else float(m)
    except Exception:
        return float(default)


# ───────────────────────────────────────────────────────────────────────────────
# GLOBALS & CONFIG
# ───────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
YEAR = datetime.today().year


TEAM_ABBR_ALIASES = {
    "KC": "KCR",
    "AZ": "ARI",
    "CWS": "CHW",
    "WSH": "WSN",
    "TB": "TBR",
    "SF": "SFG",
    "SD": "SDP",
    "ATH": "OAK",
}

def _canonical_team_abbr(team_abbr: str) -> str:
    ab = str(team_abbr or "").strip().upper()
    if 'ABBR2ID' in globals() and ab in ABBR2ID:
        return ab
    return TEAM_ABBR_ALIASES.get(ab, ab)

# ───────────────────────────────────────────────────────────────────────────────
# CACHE DECORATORS WITH TTL
# ───────────────────────────────────────────────────────────────────────────────

def _cache_name_from_pattern(pattern: str, args: tuple, kwargs: dict) -> str:
    """
    Safely build a cache filename from a format pattern.
    If pattern.format(*args, **kwargs) fails because of bad placeholders,
    fall back to a deterministic sanitized name instead of crashing.
    """
    if "{" not in pattern:
        return pattern
    try:
        return pattern.format(*args, **kwargs)
    except Exception:
        base, ext = os.path.splitext(pattern)
        # strip format fields like {0}, {name}
        base = re.sub(r"\{[^{}]*\}", "", base).strip("._- ") or "cache"
        parts = []
        for a in args:
            s = str(a)
            s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
            if s:
                parts.append(s[:60])
        for k in sorted(kwargs):
            s = f"{k}_{kwargs[k]}"
            s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
            if s:
                parts.append(s[:60])
        suffix = "_".join(parts) if parts else "auto"
        ext = ext or ".pkl"
        return f"{base}_{suffix}{ext}"
def disk_cache(key: str, ttl_days: int = 30):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            path = CACHE_DIR / _cache_name_from_pattern(key, args, kwargs)
            # evict if stale
            if path.exists():
                age = (time.time() - path.stat().st_mtime) / 86400.0
                if age < ttl_days:
                    try:
                        return pickle.loads(path.read_bytes())
                    except:
                        pass
                else:
                    try: path.unlink()
                    except: pass
            res = func(*args, **kwargs)
            try:
                path.write_bytes(pickle.dumps(res))
            except:
                pass
            return res
        return wrapper
    return decorator

def disk_cache_pid(prefix: str, ttl_days: Optional[int] = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(pid, *args, **kwargs):
            fname = f"{prefix}_{pid}.pkl"
            path = CACHE_DIR / fname
            if path.exists() and ttl_days is not None:
                age = (time.time() - path.stat().st_mtime) / 86400.0
                if age > ttl_days:
                    try: path.unlink()
                    except: pass
            if path.exists():
                try:
                    return pickle.loads(path.read_bytes())
                except:
                    pass
            res = func(pid, *args, **kwargs)
            try:
                path.write_bytes(pickle.dumps(res))
            except:
                pass
            return res
        return wrapper
    return decorator

# NEW: Parquet cache for DataFrames (fast; safe pickle fallback)
def parquet_cache_df(key_fmt: str, ttl_days: int = 7):
    """
    Cache DataFrames as Parquet; if Parquet engine unavailable, fall back to pickle.
    Keeps TTL behavior consistent with disk_cache.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            fname = _cache_name_from_pattern(key_fmt, args, kwargs)
            parq = CACHE_DIR / fname
            pkl  = CACHE_DIR / (fname.replace(".parquet", "") + ".pkl")

            # try parquet
            if parq.exists():
                age = (time.time() - parq.stat().st_mtime) / 86400.0
                if age < ttl_days:
                    try:
                        return pd.read_parquet(parq)
                    except Exception:
                        try: parq.unlink()
                        except: pass

            # fallback pickle
            if pkl.exists():
                age = (time.time() - pkl.stat().st_mtime) / 86400.0
                if age < ttl_days:
                    try:
                        return pickle.loads(pkl.read_bytes())
                    except Exception:
                        try: pkl.unlink()
                        except: pass

            # compute & save
            df = fn(*args, **kwargs)
            try:
                if isinstance(df, pd.DataFrame):
                    try:
                        df.to_parquet(parq, index=False)
                    except Exception:
                        pkl.write_bytes(pickle.dumps(df))
                else:
                    pkl.write_bytes(pickle.dumps(df))
            except Exception:
                pass
            return df
        return wrapped
    return deco

# these will hold everything in RAM once, so downstream code just does dict lookups
statcast_bat_cache: Dict[int, dict] = {}
h2h_cache: Dict[Tuple[int,int], dict] = {}

def disk_cache_h2h(ttl_days: int = 30):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(bat_pid, pit_pid):
            key  = f"h2h_{bat_pid}_{pit_pid}.pkl"
            path = CACHE_DIR / key
            if path.exists():
                age = (time.time() - path.stat().st_mtime) / 86400.0
                if age < ttl_days:
                    try:
                        return pickle.loads(path.read_bytes())
                    except:
                        pass
                else:
                    try: path.unlink()
                    except: pass
            res = func(bat_pid, pit_pid) or {}
            try:
                path.write_bytes(pickle.dumps(res))
            except:
                pass
            return res
        return wrapper
    return decorator

@disk_cache("league_hit_rate.pkl", ttl_days=30)
def _league_hit_rate() -> float:
    """
    League-wide per-PA hit rate (1B+2B+3B+HR)/PA for the current YEAR.
    Cached to avoid a circular import from precompute_everything.
    """
    try:
        df = batting_stats(YEAR, qual=0)
        df = df.copy()
        df["1B"] = df["H"] - df["2B"] - df["3B"] - df["HR"]
        totPA = float(df["PA"].sum()) or 1.0
        rate  = (df["1B"].sum() + df["2B"].sum() + df["3B"].sum() + df["HR"].sum()) / totPA
        return float(rate)
    except Exception:
        # Reasonable fallback if stats fetch fails
        return 0.23


# ───────────────────────────────────────────────────────────────────────────────
# UTILS: HTTP, NAME CLEANUP, ID LOOKUP
# ───────────────────────────────────────────────────────────────────────────────
def safe_get(url, params=None, max_retries=5, backoff=1.0):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 429:
                wait = 60 + random.uniform(0,5)
                logging.warning(f"429 Rate-limit on {url}, sleeping {wait:.1f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = backoff*(2**attempt) + random.uniform(0,backoff)
                logging.warning(f"Retry {attempt+1} for {url} in {wait:.1f}s: {e}")
                time.sleep(wait)
            else:
                logging.error(f"SAFE_GET failed for {url}: {e}")
    return None

import unicodedata
def strip_pos(name: str) -> str:
    """
    Remove parenthetical and trailing position codes.
    """
    name = unicodedata.normalize("NFKD", name).encode("ascii","ignore").decode()
    name = re.sub(r"\s*\([A-Z]{1,3}\)\s*$","",name)
    name = re.sub(r"\s+(?:[A-Z0-9]{1,3})\s*$","",name)
    return name.strip()

def clean_name(name: str) -> str:
    s = re.sub(r"\s*\(.*?\)","", name or "")
    s = re.sub(r"[^\w\s]","", s)
    return s.strip().upper()

def name_to_mlbam_id(full_name: str) -> Optional[int]:
    parts = full_name.strip().split()
    if not parts:
        return None
    first, last = parts[0], parts[-1]
    try:
        df = playerid_lookup(last, first[0])
    except:
        return None
    if df.empty or "key_mlbam" not in df.columns:
        return None
    df["UP"] = df.name_first_last.str.upper()
    m = df[df.UP == full_name.upper()]
    return int(m.key_mlbam.iloc[0]) if not m.empty else None

# ───────────────────────────────────────────────────────────────────────────────
# TEAMS, ROSTERS & VENUES
# ───────────────────────────────────────────────────────────────────────────────
@disk_cache("teams_info.pkl", ttl_days=30)
def load_teams_info():
    r = safe_get("https://statsapi.mlb.com/api/v1/teams",{"sportIds":1})
    teams = r.json().get("teams",[]) if r else []
    ABBR2ID = {t["abbreviation"]:t["id"] for t in teams}
    ID2ABBR = {v:k for k,v in ABBR2ID.items()}
    FULL2ABBR = {t["teamName"]:t["abbreviation"] for t in teams}
    return ABBR2ID, ID2ABBR, FULL2ABBR

ABBR2ID, ID2ABBR, FULL2ABBR = load_teams_info()

@functools.lru_cache(None)
def full_pitching_staff(team_abbr: str) -> dict:
    tid = ABBR2ID.get(team_abbr)
    if not tid: return {}
    r = safe_get(f"https://statsapi.mlb.com/api/v1/teams/{tid}/roster/active")
    if not r: return {}
    staff=[]
    for p in r.json().get("roster",[]):
        if p.get("position",{}).get("type","").lower()=="pitcher":
            nm = p["person"]["fullName"]
            staff.append((clean_name(nm),p["person"]["id"]))
    return dict(staff)

@functools.lru_cache(None)
def roster_map(team_abbr: str) -> dict:
    tid = ABBR2ID.get(team_abbr)
    if not tid: return {}
    r = safe_get(f"https://statsapi.mlb.com/api/v1/teams/{tid}",{"expand":"team.roster"})
    if not r: return {}
    js = r.json().get("teams",[])
    rost = js[0].get("roster",{}).get("roster",[]) if js else []
    return {clean_name(p["person"]["fullName"]):p["person"]["id"] for p in rost}


@functools.lru_cache(None)
def active_hitter_names(team_abbr: str) -> List[str]:
    tid = ABBR2ID.get(team_abbr)
    if not tid:
        return []
    r = safe_get(f"https://statsapi.mlb.com/api/v1/teams/{tid}/roster/active")
    if not r:
        return []
    out: List[str] = []
    for p in r.json().get("roster", []):
        pos = p.get("position", {}) or {}
        pos_type = str(pos.get("type", "")).lower()
        pos_code = str(pos.get("code", ""))
        if pos_type == "pitcher" or pos_code == "1":
            continue
        nm = strip_pos(p.get("person", {}).get("fullName", "") or "")
        if nm:
            out.append(nm)
    seen = set()
    uniq = []
    for nm in out:
        key = clean_name(nm)
        if key and key not in seen:
            seen.add(key)
            uniq.append(nm)
    return uniq


# ───────────────────────────────────────────────────────────────────────────────
# PARK FACTORS
# ───────────────────────────────────────────────────────────────────────────────
LOCAL_PF = {}
# Canonical -> list of aliases that might appear in schedule/output
PF_CANON_ALIASES = {
    "OAK": ["ATH"],
    "ARI": ["AZ"],
    "CHW": ["CWS"],
    "WSN": ["WSH"],
    "KCR": ["KC"],
    "TBR": ["TB"],
    "SFG": ["SF"],
    "SDP": ["SD"],
}
if Path("park_factors.csv").exists():
    pf_df = pd.read_csv("park_factors.csv")
    for _,row in pf_df.iterrows():
        ab,m = row.Team.strip().upper(), int(row.Month)
        LOCAL_PF.setdefault(ab,{})[m]={"R":row.R,"HR":row.HR,"SO":row.SO,"BA":row.BA}

@disk_cache("yearly_pf.pkl")
def fetch_yearly_park_factors(year:int)->dict:
    url = f"https://www.fangraphs.com/park-factors?season={year}&teamId=0&position=all"
    r = safe_get(url)
    if not r: return {}
    tbl=pd.read_html(io.StringIO(r.text))[0]
    tbl.columns=[str(c).strip() for c in tbl.columns]
    out={}
    team_col=tbl.columns[0]
    for _,row in tbl.iterrows():
        ab=FULL2ABBR.get(row[team_col])
        if not ab: continue
        out[ab]={"R":float(row.get("R",100)),"HR":float(row.get("HR",100)),
                 "SO":float(row.get("SO",row.get("K%",100))),
                 "BA":float(row.get("BA",100))}
    return out

@disk_cache("monthly_pf.pkl")
def fetch_monthly_park_factors(year:int)->dict:
    if LOCAL_PF:
        # Also mirror aliases into LOCAL_PF so lookups never miss
        out = {ab: months.copy() for ab, months in LOCAL_PF.items()}
        for canon, aliases in PF_CANON_ALIASES.items():
            if canon in out:
                for al in aliases:
                    out[al] = out[canon]
        return out

    url=f"https://www.fangraphs.com/park-factors?season={year}&teamId=0&position=all&split=monthly"
    r=safe_get(url)
    if not r:
        yearly=fetch_yearly_park_factors(year)
        base = {ab:{m:v.copy() for m in range(1,13)} for ab,v in yearly.items()}
        # Mirror canonical -> aliases
        for canon, aliases in PF_CANON_ALIASES.items():
            if canon in base:
                for al in aliases:
                    base[al] = base[canon]
        return base

    tbl=pd.read_html(io.StringIO(r.text))[0]
    tbl.columns=[str(c).strip() for c in tbl.columns]
    mm={datetime(year,i,1).strftime("%b"):i for i in range(1,13)}
    out={}
    for _,row in tbl.iterrows():
        ab=FULL2ABBR.get(row[tbl.columns[0]])
        if not ab: continue
        out.setdefault(ab,{})
        for col in tbl.columns[1:]:
            parts = col.split()
            if len(parts) != 2:
                continue
            mon,met = parts
            m=mm.get(mon)
            if not m: continue
            val=float(row[col]) if pd.notna(row[col]) else 100.0
            slot=out[ab].setdefault(m,{"R":100,"HR":100,"SO":100,"BA":100})
            if   met=="R":  slot["R"]=val
            elif met=="HR": slot["HR"]=val
            elif met in ("SO","K%"): slot["SO"]=val
            elif met=="BA": slot["BA"]=val

    # Add alias keys so mpf.get(home, {}) works for TB/ATH/AZ etc.
    for canon, aliases in PF_CANON_ALIASES.items():
        if canon in out:
            for al in aliases:
                out[al] = out[canon]
    return out

# ───────────────────────────────────────────────────────────────────────────────
# UMPIRE CONTEXT
# ───────────────────────────────────────────────────────────────────────────────
@disk_cache("boxscore_officials_{0}.pkl", ttl_days=1)
def fetch_boxscore_officials(game_pk:int)->List[str]:
    r=safe_get(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore")
    if not r: return []
    offs=r.json().get("gameData",{}).get("officials",[])
    return [o["official"]["fullName"] for o in offs if o.get("official",{}).get("fullName")]

@disk_cache("umpire_network_stats.pkl", ttl_days=7)
def load_umpire_network_stats()->pd.DataFrame:
    url="https://data.scorenetwork.org/baseball/mlb_umpires_2008-2023.html"
    r=safe_get(url)
    if not r: return pd.DataFrame()
    df=pd.read_html(io.StringIO(r.text), header=0)[0]
    df.columns=[c.strip() for c in df.columns]
    name_col=None
    for cand in ("Umpire","Name","Umpire Name"):
        if cand in df.columns:
            name_col=cand; break
    if name_col and name_col!="Umpire":
        df=df.rename(columns={name_col:"Umpire"})
    return df

def get_home_plate_umpire(game_pk:int)->str:
    r=safe_get(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore")
    if not r: return ""
    offs=r.json().get("gameData",{}).get("officials",[])
    for o in offs:
        if o.get("officialType","").lower()=="home plate":
            return o.get("official",{}).get("fullName","")
    return ""

def umpire_adjustments(game_pk:int)->Tuple[float,float]:
    home = get_home_plate_umpire(game_pk) or ""
    df = load_umpire_network_stats()
    if df.empty: return 1.0,1.0
    name_col = "Umpire" if "Umpire" in df.columns else df.columns[0]
    rec = df[df[name_col]==home]
    if rec.empty: return 1.0,1.0
    row=rec.iloc[0]
    return float(row.get("k_rate",1.0)), float(row.get("bb_rate",1.0))

# ───────────────────────────────────────────────────────────────────────────────
# CATCHER FRAMING
# ───────────────────────────────────────────────────────────────────────────────
# in fetch.py
def get_catcher_framing_leaderboard(year: int) -> pd.DataFrame:
    """
    Pulls Baseball Savant's Catcher Framing leaderboard as CSV.
    Note: Savant renders tables client-side; use the CSV export.
    """
    base = "https://baseballsavant.mlb.com/leaderboard/catcher-framing"
    headers = {"User-Agent": "Mozilla/5.0"}
    # Try explicit season; if the site ignores it, you'll still get current season
    for params in ({"season": str(year), "csv": "true"}, {"csv": "true"}):
        r = requests.get(base, params=params, headers=headers, timeout=30)
        if r.ok and "," in r.text:
            return pd.read_csv(io.StringIO(r.text))
    raise RuntimeError("Unable to fetch Savant catcher framing CSV")

def framing_runs_for(catcher_name: str) -> float:
    df = get_catcher_framing_leaderboard(YEAR)
    # normalize column names and pick the framing-runs column
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("player") or cols.get("player_name") or cols.get("catcher") or "player"
    # Savant labels can change; handle common cases
    frm_col = (cols.get("framing runs") or cols.get("runs (framing)") or
               cols.get("runs_framing") or cols.get("frm"))
    if frm_col is None:
        return 0.0
    df["Name_norm"] = df[name_col].astype(str).str.upper().str.strip()
    key = catcher_name.upper().strip()
    row = df[df["Name_norm"] == key]
    return float(row.iloc[0][frm_col]) if not row.empty else 0.0


# ───────────────────────────────────────────────────────────────────────────────
# DAYS REST & TRAVEL
# ───────────────────────────────────────────────────────────────────────────────
@lru_cache(None)
def get_days_rest(pid:int)->Optional[int]:
    if not pid: return None
    today=datetime.today().strftime("%Y-%m-%d")
    url=f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
    params={"stats":"pitching","group":"pitching","season":YEAR}
    r=safe_get(url,params=params)
    if not r: return None
    try:
        last_date=r.json()["stats"][0]["splits"][-1]["date"]
        dt_last=parser.isoparse(last_date).date()
        return (date.today()-dt_last).days
    except:
        return None

@lru_cache(None)
def get_travel_days(team_abbr:str)->Optional[int]:
    today=datetime.today().strftime("%Y-%m-%d")
    r=safe_get("https://statsapi.mlb.com/api/v1/schedule",
               {"sportId":1,"teamId":ABBR2ID[team_abbr],"date":today,"hydrate":"teams"})
    if not r: return None
    dates=[]
    for day in r.json().get("dates",[]):
        for g in day.get("games",[]):
            for side in ("away","home"):
                if ID2ABBR[g["teams"][side]["team"]["id"]]==team_abbr:
                    dt=parser.isoparse(g["gameDate"]).date()
                    dates.append((dt,side))
    if len(dates)<2: return None
    dates.sort()
    (d0,loc0),(d1,loc1)=dates[-2],dates[-1]
    travel=1 if loc0!=loc1 else 0
    return (d1-d0).days + travel

# ───────────────────────────────────────────────────────────────────────────────
# RETROSHEET PBP
# ───────────────────────────────────────────────────────────────────────────────
@parquet_cache_df("retrosheet_pbp_{0}.parquet", ttl_days=30)  # switched to parquet cache
def fetch_retrosheet_pbp(year: int) -> pd.DataFrame:
    # 1) get the full season schedule
    sched = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={
            "sportId": 1,
            "startDate": f"{year}-03-01",
            "endDate":   f"{year}-11-01",
            "hydrate":   "gamePk"
        }
    ).json()
    game_pks = [
        g["gamePk"]
        for d in sched.get("dates", [])
        for g in d["games"]
    ]

    rows = []
    for pk in game_pks:
        feed = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live").json()
        for play in feed["liveData"]["plays"]["allPlays"]:
            ob = play.get("count", {}).get("outs", 0)
            bb = (
                (1 if play["matchup"].get("onFirst")  else 0) +
                (2 if play["matchup"].get("onSecond") else 0) +
                (4 if play["matchup"].get("onThird")  else 0)
            )
            # sum all the RBIs in the playEvents
            rs = sum(evt.get("rbi", 0) for evt in play.get("playEvents", []))
            rows.append({
                "game_pk":     pk,
                "inning":      play["about"]["inning"],
                "outs_before": ob,
                "bases_before":bb,
                "runs_scored": rs
            })

    return pd.DataFrame(rows, columns=[
        "game_pk",
        "inning",
        "outs_before",
        "bases_before",
        "runs_scored"
    ])

# ───────────────────────────────────────────────────────────────────────────────
# TODAY’S GAMES & LINEUPS
# ───────────────────────────────────────────────────────────────────────────────
@disk_cache("todays_matchups.pkl", ttl_days=1)
def list_todays_matchups()->list:
    today=datetime.today().strftime("%Y-%m-%d")
    r=safe_get("https://statsapi.mlb.com/api/v1/schedule",{"sportId":1,"date":today})
    if not r: return []
    out=[]
    for d in r.json().get("dates",[]):
        for g in d["games"]:
            a=ID2ABBR[g["teams"]["away"]["team"]["id"]]
            h=ID2ABBR[g["teams"]["home"]["team"]["id"]]
            out.append(f"{a} @ {h}")
    return out

@disk_cache("active_pitchers.pkl")
def fetch_all_active_pitchers():
    return {ab: full_pitching_staff(ab) for ab in ABBR2ID}

ACTIVE_PITCHERS = fetch_all_active_pitchers()

@functools.lru_cache(None)
def fetch_lineup_and_starters(away:str, home:str):
    today=datetime.today().strftime("%Y-%m-%d")
    sched=safe_get("https://statsapi.mlb.com/api/v1/schedule",{"sportId":1,"date":today})
    if not sched:
        raise RuntimeError("Schedule fetch failed")

    gamePk=None; a_name=h_name=""; a_pid=h_pid=None
    for day in sched.json().get("dates",[]):
        for g in day["games"]:
            A=ID2ABBR[g["teams"]["away"]["team"]["id"]]; H=ID2ABBR[g["teams"]["home"]["team"]["id"]]
            if (A,H)==(away,home):
                gamePk=g["gamePk"]
                pp=g.get("probablePitchers",{})
                a_name=strip_pos(pp.get("away",{}).get("fullName","") or a_name)
                h_name=strip_pos(pp.get("home",{}).get("fullName","") or h_name)
                a_pid=pp.get("away",{}).get("id",a_pid)
                h_pid=pp.get("home",{}).get("id",h_pid)
                break
        if gamePk:
            break
    if not gamePk:
        raise RuntimeError(f"No game for {away}@{home}")

    away_lineup=[]; home_lineup=[]
    bx=safe_get(f"https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore")
    if bx:
        js=bx.json()
        gd=js.get("gameData",{})
        pp=gd.get("probablePitchers",{})
        a_name=strip_pos(pp.get("away",{}).get("fullName",a_name))
        h_name=strip_pos(pp.get("home",{}).get("fullName",h_name))
        a_pid=pp.get("away",{}).get("id",a_pid)
        h_pid=pp.get("home",{}).get("id",h_pid)

        ld=js.get("liveData",{})
        teams=ld.get("boxscore",{}).get("teams",{})
        idsA=teams.get("away",{}).get("batters",[])[:9]
        idsH=teams.get("home",{}).get("batters",[])[:9]
        players=gd.get("players",{})
        away_lineup=[strip_pos(players[str(i)]["person"]["fullName"]) for i in idsA if str(i) in players]
        home_lineup=[strip_pos(players[str(i)]["person"]["fullName"]) for i in idsH if str(i) in players]

    try:
        page=safe_get("https://www.mlb.com/starting-lineups")
        if page:
            soup=BeautifulSoup(page.text,"lxml")
            for blk in soup.select("div.starting-lineups__matchup"):
                codes=[a["data-tri-code"].upper() for a in blk.select(".starting-lineups__team-name--link")]
                if codes!=[away,home]:
                    continue
                ps=blk.select("div.starting-lineups__pitcher-name a")
                if len(ps)>=2:
                    a_name=a_name or strip_pos(ps[0].get_text(strip=True))
                    h_name=h_name or strip_pos(ps[1].get_text(strip=True))
                    a_pid=a_pid or name_to_mlbam_id(clean_name(a_name))
                    h_pid=h_pid or name_to_mlbam_id(clean_name(h_name))
                lists=blk.select("ol.starting-lineups__team")
                if len(lists)>=2:
                    if len(away_lineup) < 9:
                        away_lineup=[strip_pos(li.get_text(strip=True)) for li in lists[0].select("li")][:9]
                    if len(home_lineup) < 9:
                        home_lineup=[strip_pos(li.get_text(strip=True)) for li in lists[1].select("li")][:9]
                break
    except Exception as exc:
        logging.warning(f"MLB starting-lineups fallback failed for {away}@{home}: {exc}")

    if len(away_lineup)<9:
        alt = fetch_unofficial_lineup(away, lookback_days=5)
        if alt:
            logging.info(f"Backfilled projected roster lineup for {away}")
            away_lineup=alt
    if len(home_lineup)<9:
        alt = fetch_unofficial_lineup(home, lookback_days=5)
        if alt:
            logging.info(f"Backfilled projected roster lineup for {home}")
            home_lineup=alt

    def _find_catcher(side:str)->Optional[int]:
        if not bx:
            return None
        tm=bx.json().get("liveData",{}).get("boxscore",{}).get("teams",{})
        for pid_str in tm.get(side,{}).get("batters",[]):
            pl=tm[side]["players"].get(str(pid_str),{})
            if pl.get("position",{}).get("code")=="2":
                return pl["person"]["id"]
        return None

    away_cid=_find_catcher("away"); home_cid=_find_catcher("home")

    ump_feats=dict(zip(("ump_k9","ump_bb9"), umpire_adjustments(gamePk)))
    roster=roster_map(away)
    nm_map={v:k for k,v in roster.items()}
    away_cname=nm_map.get(away_cid,""); home_cname=nm_map.get(home_cid,"")
    try:
        framing_feats = {
            "away_frame": framing_runs_for(away_cname),
            "home_frame": framing_runs_for(home_cname),
        }
    except Exception as e:
        logging.warning(f"Framing disabled (using 0s): {e}")
        framing_feats = {"away_frame": 0.0, "home_frame": 0.0}

    if not a_name or not h_name:
        logging.warning(f"No starter names for {away}@{home}")
    if len(away_lineup)!=9 or len(home_lineup)!=9:
        logging.warning(f"Incomplete lineup from MLB sources for {away}@{home}; away={len(away_lineup)} home={len(home_lineup)}")

    if a_pid is None and a_name:
        a_pid=full_pitching_staff(away).get(clean_name(a_name))
    if h_pid is None and h_name:
        h_pid=full_pitching_staff(home).get(clean_name(h_name))

    away_lineup = [strip_pos(x) for x in (away_lineup or [])][:9]
    home_lineup = [strip_pos(x) for x in (home_lineup or [])][:9]

    away_bat_ids=[name_to_mlbam_id(clean_name(strip_pos(n))) for n in away_lineup]
    home_bat_ids=[name_to_mlbam_id(clean_name(strip_pos(n))) for n in home_lineup]
    for bid in away_bat_ids+home_bat_ids:
        if bid and bid not in statcast_bat_cache:
            statcast_bat_cache[bid]=get_statcast_batter_features(bid)

    for bid in away_bat_ids+home_bat_ids:
        for pid in (a_pid,h_pid):
            if bid and pid and (bid,pid) not in h2h_cache:
                h2h_cache[(bid,pid)]=get_batter_vs_pitcher_rates(bid,pid) or {}

    return (
        a_name,h_name,
        away_lineup,home_lineup,
        a_pid,h_pid,
        away_cid,home_cid,
        ump_feats,framing_feats
    )

def _boxscore_unofficial(team_abbr:str, lookback_days:int=5)->List[str]:
    today=datetime.today()
    for d in range(1,lookback_days+1):
        date=(today-timedelta(days=d)).strftime("%Y-%m-%d")
        r=safe_get("https://statsapi.mlb.com/api/v1/schedule",{"sportId":1,"date":date})
        if not r: continue
        for day in r.json().get("dates",[]):
            for game in day.get("games",[]):
                away_id=game.get("teams",{}).get("away",{}).get("team",{}).get("id")
                home_id=game.get("teams",{}).get("home",{}).get("team",{}).get("id")
                if away_id is None or home_id is None: continue
                away=ID2ABBR.get(away_id); home=ID2ABBR.get(home_id)
                side="away" if away==team_abbr else ("home" if home==team_abbr else None)
                if not side: continue
                pk=game.get("gamePk")
                if not pk: continue
                bx=safe_get(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore")
                if not bx: continue
                teams=bx.json().get("liveData",{}).get("boxscore",{}).get("teams",{})
                batters=teams.get(side,{}).get("batters",[])[:9]
                players=teams.get(side,{}).get("players",{})
                lineup=[players.get(str(pid),{}).get("person",{}).get("fullName")
                        for pid in batters]
                if len(lineup)==9:
                    return lineup
    return []

def fetch_unofficial_lineup(team_abbr:str, lookback_days:int=5)->List[str]:
    lineup=_boxscore_unofficial(team_abbr,lookback_days)
    if len(lineup)==9:
        return [strip_pos(x) for x in lineup]

    hitters = active_hitter_names(team_abbr)

    recent_seen = []
    recent_keys = set()
    for d in range(1, max(lookback_days, 10) + 1):
        date=(datetime.today()-timedelta(days=d)).strftime("%Y-%m-%d")
        r=safe_get("https://statsapi.mlb.com/api/v1/schedule",{"sportId":1,"date":date})
        if not r:
            continue
        for day in r.json().get("dates",[]):
            for game in day.get("games",[]):
                away_id=game.get("teams",{}).get("away",{}).get("team",{}).get("id")
                home_id=game.get("teams",{}).get("home",{}).get("team",{}).get("id")
                if away_id is None or home_id is None:
                    continue
                away=ID2ABBR.get(away_id); home=ID2ABBR.get(home_id)
                side="away" if away==team_abbr else ("home" if home==team_abbr else None)
                if not side:
                    continue
                pk=game.get("gamePk")
                if not pk:
                    continue
                bx=safe_get(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore")
                if not bx:
                    continue
                teams=bx.json().get("liveData",{}).get("boxscore",{}).get("teams",{})
                batters=teams.get(side,{}).get("batters",[])[:9]
                players=teams.get(side,{}).get("players",{})
                lineup_day=[strip_pos(players.get(str(pid),{}).get("person",{}).get("fullName","")) for pid in batters]
                if len(lineup_day)==9:
                    for nm in lineup_day:
                        key = clean_name(nm)
                        if key and key not in recent_keys:
                            recent_keys.add(key)
                            recent_seen.append(nm)

    ordered = []
    seen = set()
    for nm in recent_seen + hitters:
        key = clean_name(nm)
        if key and key not in seen:
            seen.add(key)
            ordered.append(nm)

    top9 = ordered[:9]
    if len(top9) < 9:
        top9 += ["TBD"] * (9 - len(top9))
    return top9

@functools.lru_cache(None)
def build_player_pa_maps(year:int)->Tuple[Dict[str,float],Dict[str,float]]:
    try:
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            df=batting_stats(year,qual=0)
    except Exception as exc:
        logging.warning(f"build_player_pa_maps fallback for {year}: {exc}")
        return {}, {ab:(27/9) for ab in ABBR2ID.keys()}

    def normalize_name(nm:str)->str:
        txt=strip_pos(nm)
        if ',' in txt:
            last,first=[p.strip() for p in txt.split(',',1)]
            txt=f"{first} {last}"
        return clean_name(txt)

    if df is None or df.empty:
        return {}, {ab:(27/9) for ab in ABBR2ID.keys()}

    df["Name_norm"]=df.Name.apply(normalize_name)
    SEASON_PA_G=dict(zip(df.Name_norm,(df.PA/df.G).fillna(0)))
    tm=df.groupby("Team").agg({"PA":"sum","G":"sum"})
    TEAM_PA_GAME={team:(row.PA/row.G if row.G>0 else 27/9) for team,row in tm.iterrows()}
    return SEASON_PA_G,TEAM_PA_GAME

# ───────────────────────────────────────────────────────────────────────────────
# BULK STATCAST DOWNLOAD + FILTER HELPERS
# ───────────────────────────────────────────────────────────────────────────────
STATCAST_RAW_TTL_DAYS = 7
STATCAST_PITCH_COLS = [
    "game_pk","game_date","pitcher","pitch_type","description",
    "balls","strikes","zone","plate_x","plate_z",
    "release_speed","release_extension","release_spin_rate","spin_axis",
    "pfx_x","pfx_z","effective_speed","events","bb_type"
]
STATCAST_BATTER_COLS = [
    "game_pk","game_date","batter","pitcher","events","bb_type",
    "launch_speed","launch_angle","launch_speed_angle",
    "hc_x","hc_y","estimated_ba_using_speedangle",
    "estimated_slg_using_speedangle","estimated_woba_using_speedangle",
    "barrel","stand"
]


@parquet_cache_df("statcast_raw_{0}_{1}.parquet", ttl_days=STATCAST_RAW_TTL_DAYS)  # switched to parquet cache
def fetch_statcast_raw(start_date:str, end_date:str)->pd.DataFrame:
    df=statcast(start_date,end_date)
    return df if df is not None else pd.DataFrame()

def _statcast_slice(df:pd.DataFrame, cols:List[str])->pd.DataFrame:
    avail=[c for c in cols if c in df.columns]
    return df.loc[:,avail].copy()

def get_statcast_pitch_data(start_date:str, end_date:str)->pd.DataFrame:
    raw=fetch_statcast_raw(start_date,end_date)
    return _statcast_slice(raw, STATCAST_PITCH_COLS)

def get_statcast_batter_data(pid:int, start_date:str, end_date:str)->pd.DataFrame:
    raw=fetch_statcast_raw(start_date,end_date)
    df=raw[raw["batter"]==pid]
    return _statcast_slice(df, STATCAST_BATTER_COLS)

# ───────────────────────────────────────────────────────────────────────────────
# STATCAST PITCHER FEATURES (bulk-sliced)
# ───────────────────────────────────────────────────────────────────────────────
@disk_cache_pid("statcast_pitcher_feats", ttl_days=7)
def get_statcast_pitcher_features(pid:int, days:int=180)->dict:
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today()-timedelta(days=days)).strftime("%Y-%m-%d")
    df = get_statcast_pitch_data(start, end)
    df = df[df["pitcher"] == pid]
    if df.empty:
        return {}
    feats = {}
    total = len(df)

    # Pitch-type mix + velo + spin + whiff_rate
    for ptype, sub in df.groupby("pitch_type"):
        n = len(sub)
        feats[f"pct_{ptype}"]   = n / total
        feats[f"{ptype}_vel"]   = _safe_mean(sub["release_speed"])
        feats[f"{ptype}_spin"]  = _safe_mean(sub["release_spin_rate"])
        swings = sub["description"].astype(str).str.contains("swing", na=False)
        swung  = sub[swings]
        whiffs = swung["description"].astype(str).str.contains("swinging_strike", na=False).sum()
        feats[f"{ptype}_whiff_rate"] = float(whiffs / max(1, len(swung)))

    # Reverse splits
    if {"stand","events"}.issubset(df.columns):
        for side in ("L","R"):
            vs = df[df["stand"] == side]
            pa = len(vs)
            if pa:
                hits = vs["events"].isin(["single","double","triple","home_run"]).mean()
                hr   = (vs["events"] == "home_run").mean()
            else:
                hits = hr = 0.0
            feats[f"vs_{side}_BA"]      = float(hits)
            feats[f"vs_{side}_HR_rate"] = float(hr)

    # Movement & release
    for col in ("pfx_x","pfx_z","release_extension","effective_speed"):
        if col in df.columns:
            feats[f"avg_{col}"] = _safe_mean(df[col])

    # Two-strike whiff
    if {"balls","strikes","description"}.issubset(df.columns):
        ts = df[df["strikes"] == 2]
        if len(ts):
            w2 = ts["description"].astype(str).str.contains("swinging_strike", na=False).mean()
            feats["whiff_2S"] = float(np.nan_to_num(w2, nan=0.0))
        else:
            feats["whiff_2S"] = 0.0

    return feats

# ───────────────────────────────────────────────────────────────────────────────
# Rolling-window recent K9 & ERA (7/14/30 days)
# ───────────────────────────────────────────────────────────────────────────────
@functools.lru_cache(None)
def get_recent_pitcher_k9(pid:int, days:int=7)->Optional[float]:
    if not pid: return None
    since=(datetime.today()-timedelta(days=days)).strftime("%Y-%m-%d")
    today=datetime.today().strftime("%Y-%m-%d")
    df=fetch_statcast_raw(since,today)
    if df is None or df.empty or "pitcher" not in df:
        return None
    df=df[df["pitcher"]==pid]
    if df.empty: return None
    # compute innings
    ip = df.get("innings_pitched", None)
    if ip is not None:
        ip_sum=ip.sum()
    elif "outs" in df.columns:
        ip_sum=df["outs"].sum()/3.0
    else:
        return None
    if ip_sum<=0: return None
    ks = (df.events=="strikeout").sum()
    return 9*ks/ip_sum

@functools.lru_cache(None)
def get_recent_pitcher_era(pid:int, days:int=30)->Optional[float]:
    if not pid: return None
    since=(datetime.today()-timedelta(days=days)).strftime("%Y-%m-%d")
    today=datetime.today().strftime("%Y-%m-%d")
    df=fetch_statcast_raw(since,today)
    if df is None or df.empty or "pitcher" not in df:
        return None
    df=df[df["pitcher"]==pid]
    if df.empty: return None
    # compute innings
    if "innings_pitched" in df.columns:
        ip_sum=df["innings_pitched"].sum()
    elif "outs" in df.columns:
        ip_sum=df["outs"].sum()/3.0
    else:
        return None
    if ip_sum<=0: return None
    er = df.get("earned_runs", df.get("earnedRuns", None))
    if er is None: return None
    return 9*er.sum()/ip_sum

# ───────────────────────────────────────────────────────────────────────────────
# STATCAST BATTER FEATURES (bulk-sliced)
# ───────────────────────────────────────────────────────────────────────────────
@disk_cache_pid("statcast_batter_feats", ttl_days=7)
def get_statcast_batter_features(pid:int, days:int=180)->dict:
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = get_statcast_batter_data(pid, start, end)
    if df.empty or "game_date" not in df.columns:
        return {}

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    now     = datetime.today()
    since7  = now - timedelta(days=7)
    since14 = now - timedelta(days=14)
    df7  = df[df["game_date"] >= since7]
    df14 = df[df["game_date"] >= since14]

    feats = {}

    # Exit velocity + hard-hit (>=95)
    if "launch_speed" in df.columns:
        feats["ev_mean"]        = _safe_mean(df["launch_speed"])
        feats["hard_hit_pct"]   = _safe_frac(df["launch_speed"] > 95)

        feats["ev_mean_14d"]    = _safe_mean(df14["launch_speed"])   # <- renamed
        feats["hardhit_14d_pct"]= _safe_frac(df14["launch_speed"] > 95)  # <- renamed
    else:
        feats.update({
            "ev_mean":0.0, "hard_hit_pct":0.0,
            "ev_mean_14d":0.0, "hardhit_14d_pct":0.0
        })

    # Launch angle
    if "launch_angle" in df.columns:
        feats["la_mean"] = _safe_mean(df["launch_angle"])
    else:
        feats["la_mean"] = 0.0

    # Barrel rate
    if "barrel" in df.columns:
        feats["barrel_pct"]     = _safe_frac(df["barrel"])
        feats["barrel_14d_pct"] = _safe_frac(df14["barrel"])
    else:
        # proxy barrel% if needed (EV >= 98 and 26–30°)
        if {"launch_speed","launch_angle"}.issubset(df.columns):
            prox_all  = ((pd.to_numeric(df["launch_speed"], errors="coerce") >= 98) &
                         pd.to_numeric(df["launch_angle"], errors="coerce").between(26, 30))
            prox_14   = ((pd.to_numeric(df14["launch_speed"], errors="coerce") >= 98) &
                         pd.to_numeric(df14["launch_angle"], errors="coerce").between(26, 30))
            feats["barrel_pct"]     = _safe_frac(prox_all)
            feats["barrel_14d_pct"] = _safe_frac(prox_14)
        else:
            feats["barrel_pct"] = 0.0
            feats["barrel_14d_pct"] = 0.0

    # “Sweet spot” proxy (105+ EV & 25–35°)
    if {"launch_speed","launch_angle"}.issubset(df.columns) and len(df) > 0:
        feats["sweet_spot_frac"] = _safe_frac(
            (pd.to_numeric(df["launch_speed"], errors="coerce") > 105) &
            pd.to_numeric(df["launch_angle"], errors="coerce").between(25, 35)
        )
    else:
        feats["sweet_spot_frac"] = 0.0

    # Pull% proxy (LA in [-20, 20])
    if "launch_angle" in df.columns:
        feats["pull_pct"] = _safe_frac(pd.to_numeric(df["launch_angle"], errors="coerce").between(-20, 20))
    else:
        feats["pull_pct"] = 0.40

    # Stance (bats L/R) – prefer Statcast column; fallback to API
    if "stand" in df.columns:
        s = df["stand"].dropna().astype(str)
        feats["stand"] = s.mode().iloc[0].upper()[0] if not s.empty else batter_bats(pid)
    else:
        feats["stand"] = batter_bats(pid)

    # HR/FB rate over the whole window
    if {"events","bb_type"}.issubset(df.columns):
        evs = df["events"].astype(str).str.lower()
        bbt = df["bb_type"].astype(str).str.lower()
        hr = (evs == "home_run").sum()
        fb = (bbt == "fly_ball").sum()
        feats["HR_FB_rate"] = float(hr / fb) if fb > 0 else 0.0
    else:
        feats["HR_FB_rate"] = 0.0

    return feats

# ───────────────────────────────────────────────────────────────────────────────
# 7d/14d PITCHER-FACED BATTED-BALL SUMMARIES
# ───────────────────────────────────────────────────────────────────────────────
@functools.lru_cache(None)
def get_recent_bb_stats(pid:int, days:int=7)->dict:
    end=datetime.today().strftime("%Y-%m-%d")
    start=(datetime.today()-timedelta(days=days)).strftime("%Y-%m-%d")
    df=fetch_statcast_raw(start,end)
    if df is None or df.empty or "pitcher" not in df.columns:
        return {}
    df=df[df["pitcher"]==pid]
    if df.empty or "launch_speed" not in df.columns:
        return {}
    bb=df[df.events.isin(["single","double","triple","home_run"])]
    if bb.empty: 
        return {}
    return {
        f"ev_mean_{days}d":  _safe_mean(bb["launch_speed"]),
        f"ev_std_{days}d":   _safe_std(bb["launch_speed"]),
        f"la_mean_{days}d":  _safe_mean(bb["launch_angle"]),
        f"barrel_pct_{days}d": _safe_mean(pd.to_numeric(bb.get("barrel", 0), errors="coerce").fillna(0))
    }

def get_pitch_type_profile(pid:int, start:str=None, end:str=None)->dict:
    if start is None:
        start=f"{datetime.today().year}-01-01"
    if end is None:
        end=datetime.today().strftime("%Y-%m-%d")
    df=fetch_statcast_raw(start,end)
    if df is None or df.empty or "pitcher" not in df.columns:
        return {"FB_pct":0.5,"OS_pct":0.5,"SwStr_FB":0.1,"SwStr_OS":0.1}
    df=df[df["pitcher"]==pid]
    if df.empty:
        return {"FB_pct":0.5,"OS_pct":0.5,"SwStr_FB":0.1,"SwStr_OS":0.1}
    fb_types={"FF","FT","FC","FS"}
    df["is_FB"]=df.pitch_type.isin(fb_types)
    total=len(df)
    fb_count=df.is_FB.sum()
    os_count=total-fb_count
    fb_pct=fb_count/total
    os_pct=1-fb_pct
    swstr_fb = df[df["is_FB"]]["description"].astype(str).str.contains("swinging_strike", na=False).sum() / max(1, fb_count)
    swstr_os = df[~df["is_FB"]]["description"].astype(str).str.contains("swinging_strike", na=False).sum() / max(1, os_count)

    return {"FB_pct":fb_pct,"OS_pct":os_pct,"SwStr_FB":swstr_fb,"SwStr_OS":swstr_os}

@functools.lru_cache(None)
def get_batter_vs_pitcher_rates(batter_id:int, pitcher_id:int)->Dict[str,float]:
    if not batter_id or not pitcher_id:
        return {}
    end=datetime.today().strftime("%Y-%m-%d")
    start=(datetime.today()-timedelta(days=365*3)).strftime("%Y-%m-%d")
    df=fetch_statcast_raw(start,end)
    if df is None or df.empty or "batter" not in df.columns:
        return {}
    df=df[df["batter"]==batter_id]
    df=df[df["pitcher"]==pitcher_id]
    df=df[df.events.notnull()]
    pa=len(df)
    if pa==0: return {}
    singles=(df.events=="single")
    doubles=(df.events=="double")
    triples=(df.events=="triple")
    hrs=(df.events=="home_run")
    tb=singles.astype(int)+2*doubles.astype(int)+3*triples.astype(int)+4*hrs.astype(int)
    rbi_col="rbi" if "rbi" in df.columns else None
    runs_col="runs_scored" if "runs_scored" in df.columns else None
    rbi=df[rbi_col].fillna(0).astype(int) if rbi_col else pd.Series(0,index=df.index)
    runs=df[runs_col].fillna(0).astype(int) if runs_col else pd.Series(0,index=df.index)
    return {
        "H":((singles|doubles|triples|hrs).sum()/pa),
        "1B":singles.sum()/pa,
        "2B":doubles.sum()/pa,
        "3B":triples.sum()/pa,
        "HR":hrs.sum()/pa,
        "TB":tb.sum()/pa,
        "RBI":(rbi>0).sum()/pa,
        "R":(runs>0).sum()/pa
    }

from scipy.stats import beta
def get_filtered_batter_vs_pitcher_rates(bat_id:int, pit_id:int,
                                         min_pa:int=10,
                                         launch_speed_thresh:float=85,
                                         launch_angle_range:tuple=(0,50),
                                         prior_pa:int=20) -> dict:
    # Pull last 3 seasons of Statcast and filter to this batter vs this pitcher
    df = fetch_statcast_raw(
        (datetime.today() - timedelta(days=365*3)).strftime("%Y-%m-%d"),
        datetime.today().strftime("%Y-%m-%d")
    )
    if df is None or df.empty or "batter" not in df.columns or "pitcher" not in df.columns:
        return {}

    df = df[(df["batter"] == bat_id) & (df["pitcher"] == pit_id)]
    if df.empty:
        return {}

    # Robust numeric columns (handles missing cols & NA cleanly)
    ls = pd.to_numeric(df.get("launch_speed", pd.Series(index=df.index, dtype=float)), errors="coerce")
    la = pd.to_numeric(df.get("launch_angle", pd.Series(index=df.index, dtype=float)), errors="coerce")

    # Keep only PAs that produced a batted-ball result (hit types)
    if "events" not in df.columns:
        return {}
    hit_ev = df["events"].isin(["single", "double", "triple", "home_run"])

    # Quality-contact filter
    q = df[(ls >= launch_speed_thresh) & (la.between(*launch_angle_range)) & hit_ev]
    # If nothing qualifies, fall back to empty but still shrink to league later
    # (let pa stay 0 so shrink fully to league)
    # Pitch-type mix for weighting (based on all pitches seen vs this pitcher)
    pt_all = df.get("pitch_type", pd.Series(index=df.index, dtype=object)).astype(str).fillna("")
    fb_pct = float((pt_all.str.startswith("FF")).mean())  # share of FF among all pitch_type rows
    off_pct = 1.0 - fb_pct

    # FB vs Off-speed split *within the filtered quality-contact subset* (NA-safe)
    pt_q = q.get("pitch_type", pd.Series(index=q.index, dtype=object)).astype(str).fillna("")
    is_fb_q = pt_q.str.startswith("FF")

    def _hit_rate(sub: pd.DataFrame) -> tuple[float, int]:
        pa = len(sub)
        if pa == 0:
            return 0.0, 0
        return float(sub["events"].isin(["single","double","triple","home_run"]).mean()), pa

    fb_rate,  fb_n  = _hit_rate(q[is_fb_q])
    off_rate, off_n = _hit_rate(q[~is_fb_q])

    raw_rate = fb_pct * fb_rate + off_pct * off_rate
    raw_n    = fb_n + off_n

    # Empirical-Bayes shrink toward league hit rate (avoid circular import)
    league_rate = _league_hit_rate()

    # Optional: if you want min_pa to matter, you can fold it into the prior:
    # prior_pa_eff = max(prior_pa, max(0, min_pa - raw_n))
    prior_pa_eff = prior_pa

    weight = raw_n / (raw_n + prior_pa_eff) if (raw_n + prior_pa_eff) > 0 else 0.0
    shrunk = weight * raw_rate + (1 - weight) * league_rate

    return {"H_rate": float(shrunk), "PA": int(raw_n)}

def pitcher_throws(pid:int)->str:
    if not pid: return "R"
    r=safe_get(f"https://statsapi.mlb.com/api/v1/people/{pid}")
    if not r: return "R"
    return r.json().get("people",[{}])[0].get("throws","R")

@lru_cache(None)
def batter_bats(pid:int) -> str:
    if not pid:
        return "R"
    r = safe_get(f"https://statsapi.mlb.com/api/v1/people/{pid}")
    if not r:
        return "R"
    try:
        code = r.json()["people"][0]["batSide"]["code"]
        return (code or "R").upper()[0]
    except Exception:
        return "R"

def empirical_bayes_shrink(k9s, ips, prior_ip:float=50.0):
    league=(k9s*(ips/9.0)).sum()/(ips.sum()/9.0) if ips.sum()>0 else 8.5
    post=(k9s*(ips/9.0)+league*(prior_ip/9.0))/(ips/9.0+prior_ip/9.0)
    return post.fillna(league)

def empirical_bayes_shrink_era(ers, ips, prior_ip:float=50.0):
    league=(ers*(ips/9.0)).sum()/(ips.sum()/9.0) if ips.sum()>0 else 4.5
    post=(ers*(ips/9.0)+league*(prior_ip/9.0))/(ips/9.0+prior_ip/9.0)
    return post.fillna(league)

def select_first_reliever(df_rel:pd.DataFrame, used_ids:set)->dict:
    if df_rel is None or df_rel.empty: return {}
    df=df_rel.copy().dropna(subset=["Season_IP","Season_ERA","Season_K9"])
    df["Shrunk_K9"]=empirical_bayes_shrink(df["Season_K9"],df["Season_IP"])
    df["Shrunk_ERA"]=empirical_bayes_shrink_era(df["Season_ERA"],df["Season_IP"])
    df["Score"]=df["Shrunk_K9"]*df["Season_IP"]/df["Shrunk_ERA"].replace(0,np.nan)
    df=df[~df.Pitcher_ID.isin(used_ids)]
    if df.empty: return {}
    return df.sort_values("Score",ascending=False).iloc[0].to_dict()

def select_high_leverage_reliever(df_rel:pd.DataFrame, used_ids:set, state:int)->dict:
    if df_rel is None or df_rel.empty: return {}
    df=df_rel[~df_rel.Pitcher_ID.isin(used_ids)].dropna(subset=["Recent_K9_30d","Recent_ERA_30d"])
    if df.empty: return {}
    if "Hard_Hit_%" in df.columns:
        hard=df["Hard_Hit_%"].fillna(0)
    elif "hard_hit_pct" in df.columns:
        hard=df["hard_hit_pct"].fillna(0)
    else:
        hard=0.0
    df["LI_Score"]=(df["Recent_K9_30d"]-df["Recent_ERA_30d"])*(1+hard)
    return df.sort_values("LI_Score",ascending=False).iloc[0].to_dict()

def select_closer(df_rel:pd.DataFrame, used_ids:set)->dict:
    if df_rel is None or df_rel.empty: return {}
    df=df_rel[~df_rel.Pitcher_ID.isin(used_ids)].dropna(subset=["Season_K9","Season_IP"])
    if df.empty: return {}
    df["Shrunk_K9"]=empirical_bayes_shrink(df["Season_K9"],df["Season_IP"])
    return df.sort_values("Shrunk_K9",ascending=False).iloc[0].to_dict()

from datetime import date
@functools.lru_cache(None)
def build_bullpen_dataframe(team_abbr:str)->pd.DataFrame:
    staff=full_pitching_staff(team_abbr)
    if not staff: return pd.DataFrame()
    all_stats=pitching_stats(YEAR,qual=0)
    all_stats["Name_norm"]=all_stats.Name.apply(clean_name)
    league_ip=all_stats.IP.mean()
    league_k9=all_stats["K/9"].replace([np.inf,np.nan],0).mean()
    league_era=all_stats.ERA.replace([np.inf,np.nan],0).mean()
    rows=[]
    for nm,pid in staff.items():
        rec=all_stats[all_stats.Name_norm==nm]
        if not rec.empty:
            r0=rec.iloc[0]
            sip=float(r0.IP or league_ip)
            sk9=float(r0["K/9"] or league_k9)
            ser=float(r0.ERA or league_era)
        else:
            sip,sk9,ser=league_ip,league_k9,league_era
        rec_k9=get_recent_pitcher_k9(pid) or sk9
        rec_er=get_recent_pitcher_era(pid) or ser
        # last 30 day raw
        df30=fetch_statcast_raw((datetime.today()-timedelta(days=30)).strftime("%Y-%m-%d"),datetime.today().strftime("%Y-%m-%d"))
        if df30 is None or df30.empty or "pitcher" not in df30.columns:
            days_rest=travel_days=ip_7d=ip_14d=apps_7d=apps_14d=0
        else:
            df30=df30[df30["pitcher"]==pid]
            if df30.empty:
                days_rest=travel_days=ip_7d=ip_14d=apps_7d=apps_14d=0
            else:
                df30["gdate"]=pd.to_datetime(df30.game_date).dt.date
                dates=sorted(df30.gdate.unique(),reverse=True)
                today_dt=date.today()
                days_rest=(today_dt-dates[0]).days
                travel_days=(dates[0]-dates[1]).days if len(dates)>1 else 0
                def window_stats(days):
                    cutoff=today_dt-timedelta(days=days)
                    sub=df30[df30.gdate>=cutoff]
                    if "innings_pitched" in sub.columns:
                        ip=sum(sub.innings_pitched)
                    elif "outs" in sub.columns:
                        ip=sum(sub.outs)/3.0
                    else:
                        ip=0.0
                    apps=len(sub.gdate.unique())
                    return ip,apps
                ip_7d,apps_7d=window_stats(7)
                ip_14d,apps_14d=window_stats(14)
        sc=get_statcast_pitcher_features(pid) or {}
        row={
            "Team":team_abbr,"Name":nm,"Pitcher_ID":pid,
            "Season_IP":sip,"Season_K9":sk9,"Season_ERA":ser,
            "Recent_K9_30d":rec_k9,"Recent_ERA_30d":rec_er,
            "Days_Rest":days_rest,"Travel_Days":travel_days,
            "IP_7d":ip_7d,"Apps_7d":apps_7d,"IP_14d":ip_14d,"Apps_14d":apps_14d
        }
        row.update({k:v or 0.0 for k,v in sc.items()})
        rows.append(row)
    return pd.DataFrame(rows)

def profile_bullpen(df_rel:pd.DataFrame)->dict:
    if df_rel is None or df_rel.empty:
        return {"Bullpen_ERA_Weighted":None,"Bullpen_K9_Weighted":None,
                "Bullpen_HighLeverage_Count":0,"Bullpen_Lefty_Count":0,
                "Bullpen_Righty_Count":0}
    df=df_rel.dropna(subset=["Season_IP"])
    total_ip=df.Season_IP.sum() or 1.0
    df["Shrunk_K9"]=empirical_bayes_shrink(df["Season_K9"],df["Season_IP"])
    df["Shrunk_ERA"]=empirical_bayes_shrink_era(df["Season_ERA"],df["Season_IP"])
    w_era=float((df.Shrunk_ERA*df.Season_IP).sum()/total_ip)
    w_k9=float((df.Shrunk_K9*df.Season_IP).sum()/total_ip)
    high=df[(df.Recent_ERA_30d<3.0)&(df.Recent_K9_30d>10)]
    left=sum(1 for pid in df.Pitcher_ID if pitcher_throws(pid)=="L")
    right=sum(1 for pid in df.Pitcher_ID if pitcher_throws(pid)=="R")
    return {"Bullpen_ERA_Weighted":round(w_era,2),"Bullpen_K9_Weighted":round(w_k9,2),
            "Bullpen_HighLeverage_Count":len(high),"Bullpen_Lefty_Count":left,
            "Bullpen_Righty_Count":right}

@functools.lru_cache(None)
def get_batted_ball_profile(bat_id:int, start_date:str, end_date:str)->dict:
    df=fetch_statcast_raw(start_date,end_date)
    if df is None or df.empty or "batter" not in df.columns:
        return {}
    df=df[df["batter"]==bat_id]
    bb=df[df.events.isin(['single','double','triple','home_run'])]
    total=len(bb)
    if total==0: return {}
    counts=bb.events.value_counts()
    return {
        '1B':counts.get('single',0)/total,
        '2B':counts.get('double',0)/total,
        '3B':counts.get('triple',0)/total,
        'HR':counts.get('home_run',0)/total
    }

@disk_cache("game_weather_{0}.pkl", ttl_days=1)
def get_game_weather(game_pk:int)->dict:
    url=f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    r=safe_get(url)
    if not r: return {}
    w = r.json().get("gameData",{}).get("weather",{}) or {}
    # Normalize temperature to a float if possible
    t = w.get("temperature")
    if isinstance(t, str):
        m = re.search(r"(\d+(\.\d+)?)", t)
        tval = float(m.group(1)) if m else None
    else:
        tval = float(t) if isinstance(t,(int,float)) else None
    return {
        "temp_f": tval,  # <- lowercase key to match simulator
        "wind_mph": w.get("windMiles"),
        "wind_dir": w.get("windDirection"),
        "conditions": w.get("condition"),
        "humidity": w.get("humidityPercent")
    }

def tail_wind_pct(game_pk: int) -> float:
    """
    Return a small [0..~0.3] tail-wind boost factor.
    Treat 'Out to ...' / 'blowing out' as tail wind.
    Robust to missing/None/str speeds & directions.
    """
    w = get_game_weather(game_pk) or {}
    # direction can be None or strings like "Out to LF", "In from RF", etc.
    d_raw = w.get("wind_dir")
    d = (d_raw or "").strip().lower()

    # speed can be int/float or strings like "12 mph"
    s_raw = w.get("wind_mph")
    if isinstance(s_raw, (int, float)):
        speed = float(s_raw)
    elif isinstance(s_raw, str):
        m = re.search(r"(\d+(\.\d+)?)", s_raw)
        speed = float(m.group(1)) if m else 0.0
    else:
        speed = 0.0

    # consider common “tailwind” phrasings
    tail_terms = ("out to", "out toward", "blowing out")
    in_terms   = ("in from", "in toward", "blowing in")

    is_tail = any(t in d for t in tail_terms)
    if not is_tail:
        # generic 'out' but avoid false-positive on 'south'
        if re.search(r"\bout\b", d) and not any(t in d for t in in_terms):
            is_tail = True

    return (speed / 100.0) if is_tail else 0.0


@functools.lru_cache(None)
def get_count_hr_rates(batter_id:int, pitcher_id:int)->dict:
    """Return {('0-0'):p0,('2-0'):p1,('2-2'):p2,('3-2'):p3} HR rates from last 365d."""
    df = fetch_statcast_raw((datetime.today()-timedelta(days=365)).strftime("%Y-%m-%d"),
                             datetime.today().strftime("%Y-%m-%d"))
    df = df[(df.batter==batter_id)&(df.pitcher==pitcher_id)]
    out = {}
    for balls in (0,1,2,3):
        for strikes in (0,1,2):
            sub = df[(df.balls==balls)&(df.strikes==strikes)]
            if sub.empty:
                out[f"{balls}-{strikes}"] = 0.0
            else:
                out[f"{balls}-{strikes}"] = (sub.events=="home_run").mean()
    return out

# ───────────────────────────────────────────────────────────────────────────────
# NEW: Matchup helpers — pitcher pitch mix & batter xISO by pitch type
# ───────────────────────────────────────────────────────────────────────────────
@lru_cache(None)
def pitcher_mix_last_starts(pid:int, days:int=30, max_games:int=3)->dict:
    """Return recent pitch-type mix {pitch_type: pct} for last few starts."""
    if not pid:
        return {}
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
    df = fetch_statcast_raw(start, end)
    if df is None or df.empty:
        return {}

    df = df.copy()
    if "pitcher" not in df.columns:
        return {}
    df = df[pd.to_numeric(df["pitcher"], errors="coerce") == int(pid)].copy()
    if df.empty or "game_pk" not in df.columns or "pitch_type" not in df.columns:
        return {}

    # Critical fix: normalize game_date before sorting/comparing.
    if "game_date" in df.columns:
        gdt = pd.to_datetime(df["game_date"], errors="coerce")
        tmp = df.loc[gdt.notna(), ["game_pk"]].copy()
        tmp["game_date"] = gdt[gdt.notna()].dt.normalize()
        last_games = (
            tmp.drop_duplicates()
               .sort_values("game_date", ascending=False, kind="stable")
               .head(max_games)["game_pk"]
               .tolist()
        )
    else:
        # Fallback if game_date is missing.
        last_games = pd.Series(df["game_pk"]).dropna().drop_duplicates().tail(max_games).tolist()

    if not last_games:
        return {}

    sub = df[df["game_pk"].isin(last_games)].copy()
    if sub.empty:
        return {}

    return sub["pitch_type"].astype(str).value_counts(normalize=True).to_dict()

@lru_cache(None)
def batter_xiso_by_pitch(bat_id:int, days:int=180)->dict:
    """Return batter ISO by pitch type over window: {pitch_type: ISO}."""
    if not bat_id: return {}
    start=(datetime.today()-timedelta(days=days)).strftime("%Y-%m-%d")
    end  =datetime.today().strftime("%Y-%m-%d")
    df=fetch_statcast_raw(start,end)
    if df is None or df.empty: return {}
    df=df[df.batter==bat_id]
    if df.empty or "pitch_type" not in df.columns: return {}
    bb=df[df.events.isin(["single","double","triple","home_run"])]
    if bb.empty: return {}
    tb=(1*(bb.events=="single")+2*(bb.events=="double")+
        3*(bb.events=="triple")+4*(bb.events=="home_run"))
    iso=tb.groupby(bb.pitch_type).sum()/bb.groupby(bb.pitch_type).size().clip(lower=1)
    return {pt: float(v) for pt,v in iso.items()}

# --- Bullpen HR/PA (team-level, exclude the day’s starter) --------------------
@lru_cache(None)
def team_bullpen_hrpa(team_abbr: str, exclude_pid: int | None = None, days: int = 180) -> tuple[float, int]:
    """
    Aggregate bullpen HR/PA allowed for a team over a rolling window.
    We classify relievers from current active staff by (very) simple GS rule.
    Returns (hr_per_pa, total_PA_used).
    """
    from datetime import datetime, timedelta
    end = datetime.today()
    start = end - timedelta(days=days)

    # one bulk pull (fast, cached), then slice by pitchers
    df = fetch_statcast_raw(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if df is None or df.empty or "pitcher" not in df.columns:
        return 0.025, 0  # conservative league-ish fallback

    staff = full_pitching_staff(team_abbr) or {}
    if not staff:
        return 0.025, 0

    # Identify relievers using season GS if available
    ps = pitching_stats(YEAR, qual=0).copy()
    ps["Name_norm"] = ps.Name.apply(clean_name)
    reliever_pids = []
    for nm, pid in staff.items():
        if exclude_pid and pid == exclude_pid:
            continue
        rec = ps[ps["Name_norm"] == nm]
        gs = int(rec["GS"].iloc[0]) if not rec.empty and "GS" in rec.columns else 0
        # treat low-GS arms as bullpen; if in doubt, include (we need a bullpen baseline)
        if gs <= 1:
            reliever_pids.append(pid)

    # if we detected none (odd roster scrape), just use everyone except starter
    if not reliever_pids:
        reliever_pids = [pid for pid in staff.values() if (exclude_pid is None or pid != exclude_pid)]

    if not reliever_pids:
        return 0.025, 0

    # compute HR/PA per pitcher and weight by PA faced
    rates, weights = [], []
    sub = df[df["pitcher"].isin(reliever_pids)]
    if sub.empty:
        return 0.025, 0

    for pid, g in sub.groupby("pitcher"):
        pa = int(g["events"].notna().sum())
        if pa <= 0:
            continue
        hr = int((g["events"] == "home_run").sum())
        rates.append(hr / pa)
        weights.append(pa)

    if not weights:
        return 0.025, 0

    hrpa = float(np.average(rates, weights=weights))
    return hrpa, int(sum(weights))

# ───────────────────────────────────────────────────────────────────────────────
# per-game batter logs (for get_season_hr_freq)
# ───────────────────────────────────────────────────────────────────────────────
try:
    # pybaseball ≥2.4
    from pybaseball import batting_game_logs as _pyb_game_logs
except ImportError:
    try:
        # older naming
        from pybaseball import batter_game_logs as _pyb_game_logs
    except ImportError:
        # fallback: team_game_logs(year, team)
        from pybaseball import team_game_logs

        def _pyb_game_logs(year: int) -> pd.DataFrame:
            """
            Fallback: call team_game_logs for each club and concat:
            returns columns ['Date','Name','HR']
            """
            dfs = []
            for tm in ABBR2ID.keys():
                df = team_game_logs(year, tm)   # no stat_type arg
                dfs.append(df[['Date','Name','HR']])
            return pd.concat(dfs, ignore_index=True)

def batting_game_logs(year: int) -> pd.DataFrame:
    """
    Unified wrapper: returns per-game rows with ['game_date','Name','HR'].
    """
    df = _pyb_game_logs(year)
    # pybaseball uses 'Date' for per-game logs
    df = df.rename(columns={'Date':'game_date'})
    return df[['game_date','Name','HR']]



# ───────────────────────────────────────────────────────────────────────────────
# STATE-DATA LAYER: COUNT / LOCATION / MATCHUP TENSORS / SURVIVAL CONTEXT
# ───────────────────────────────────────────────────────────────────────────────
STATE_PITCH_CORE_COLS = [
    "game_pk","game_date","pitcher","batter","pitch_type","description","events","bb_type",
    "balls","strikes","zone","plate_x","plate_z","p_throws","stand","home_team",
    "release_speed","launch_speed","launch_angle","barrel","pitch_number","at_bat_number",
]

SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "foul_bunt",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "missed_bunt", "foul_pitchout",
}
CONTACT_DESCRIPTIONS = {
    "foul", "foul_tip", "foul_bunt", "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
FOUL_DESCRIPTIONS = {"foul", "foul_tip", "foul_bunt", "foul_pitchout"}
CALLED_STRIKE_DESCRIPTIONS = {"called_strike", "pitchout"}
BALL_DESCRIPTIONS = {"ball", "blocked_ball", "ball_in_dirt", "automatic_ball", "intent_ball", "pitchout"}
INPLAY_DESCRIPTIONS = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}
HR_EVENTS = {"home_run"}
ONBASE_EVENTS = {"single", "double", "triple", "home_run", "walk", "intent_walk", "hit_by_pitch", "field_error"}
OUT_EVENTS = {
    "field_out", "force_out", "double_play", "grounded_into_double_play", "fielders_choice_out",
    "strikeout", "strikeout_double_play", "sac_bunt", "sac_fly", "fielders_choice", "other_out"
}


def _season_weather_bucket(game_date: object) -> str:
    try:
        dt = pd.to_datetime(game_date, errors="coerce")
        if pd.isna(dt):
            return "mild"
        m = int(dt.month)
    except Exception:
        return "mild"
    if m in (4, 10):
        return "cool"
    if m in (5, 9):
        return "mild"
    if m in (6, 7, 8):
        return "warm"
    return "cold"


def _normalize_pitch_type(pt: object) -> str:
    s = str(pt or "").strip().upper()
    return s if s else "UNK"


def _location_bucket(zone: object = None, plate_x: object = None, plate_z: object = None) -> str:
    try:
        z = int(float(zone)) if pd.notna(zone) else None
    except Exception:
        z = None
    if z in {1, 2, 3, 4, 5, 6, 7, 8, 9}:
        return "heart" if z in {2, 4, 5, 6, 8} else "shadow"

    try:
        px = float(plate_x)
    except Exception:
        px = np.nan
    try:
        pz = float(plate_z)
    except Exception:
        pz = np.nan

    if not np.isfinite(px) or not np.isfinite(pz):
        return "unknown"
    if abs(px) <= 0.28 and 2.0 <= pz <= 3.2:
        return "heart"
    if abs(px) <= 0.83 and 1.45 <= pz <= 3.55:
        return "shadow"
    if abs(px) <= 1.25 and 1.0 <= pz <= 4.0:
        return "chase"
    return "waste"


def _horiz_bucket(plate_x: object, pitcher_hand: str | None = None) -> str:
    try:
        px = float(plate_x)
    except Exception:
        return "middle"
    if not np.isfinite(px) or abs(px) <= 0.28:
        return "middle"
    side = "left" if px < 0 else "right"
    ph = str(pitcher_hand or "").upper()[:1]
    if ph == "L":
        return "glove_side" if side == "right" else "arm_side"
    return "glove_side" if side == "left" else "arm_side"


def _vert_bucket(plate_z: object) -> str:
    try:
        pz = float(plate_z)
    except Exception:
        return "middle"
    if not np.isfinite(pz):
        return "middle"
    if pz < 2.0:
        return "lower"
    if pz > 3.1:
        return "upper"
    return "middle"


def _launch_damage_bucket(launch_speed: object, launch_angle: object, barrel_flag: object = None) -> str:
    try:
        ev = float(launch_speed)
    except Exception:
        ev = np.nan
    try:
        la = float(launch_angle)
    except Exception:
        la = np.nan
    try:
        barrel = int(float(barrel_flag)) == 1 if pd.notna(barrel_flag) else False
    except Exception:
        barrel = False
    if barrel or (np.isfinite(ev) and np.isfinite(la) and ev >= 98 and 24 <= la <= 32):
        return "barrel"
    if np.isfinite(ev) and np.isfinite(la) and ev >= 95 and 10 <= la <= 35:
        return "damage_air"
    if np.isfinite(ev) and ev >= 95:
        return "hard_contact"
    if np.isfinite(la) and la < 0:
        return "ground_contact"
    return "weak_contact"


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    try:
        num = float(num)
        den = float(den)
    except Exception:
        return float(default)
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return float(default)
    return float(num / den)


def _state_rate_row(g: pd.DataFrame) -> dict:
    if g is None or g.empty:
        return {
            "n_pitches": 0, "n_swings": 0, "n_contacts": 0, "n_in_play": 0,
            "swing_pct": 0.0, "take_pct": 0.0, "contact_pct": 0.0, "whiff_pct": 0.0,
            "foul_pct": 0.0, "called_strike_pct": 0.0, "ball_pct": 0.0, "in_play_pct": 0.0,
            "hr_on_contact_pct": 0.0, "damage_on_contact_pct": 0.0,
        }
    n = len(g)
    swings = pd.to_numeric(g.get("is_swing", pd.Series(index=g.index, dtype=float)), errors="coerce").fillna(0)
    contacts = pd.to_numeric(g.get("is_contact", pd.Series(index=g.index, dtype=float)), errors="coerce").fillna(0)
    whiffs = pd.to_numeric(g.get("is_whiff", pd.Series(index=g.index, dtype=float)), errors="coerce").fillna(0)
    fouls = pd.to_numeric(g.get("is_foul", pd.Series(index=g.index, dtype=float)), errors="coerce").fillna(0)
    called = pd.to_numeric(g.get("is_called_strike", pd.Series(index=g.index, dtype=float)), errors="coerce").fillna(0)
    balls = pd.to_numeric(g.get("is_ball", pd.Series(index=g.index, dtype=float)), errors="coerce").fillna(0)
    in_play = pd.to_numeric(g.get("is_in_play", pd.Series(index=g.index, dtype=float)), errors="coerce").fillna(0)
    hrs = pd.to_numeric(g.get("is_hr", pd.Series(index=g.index, dtype=float)), errors="coerce").fillna(0)
    damage = pd.to_numeric(g.get("is_damage_contact", pd.Series(index=g.index, dtype=float)), errors="coerce").fillna(0)

    n_swings = float(swings.sum())
    n_contacts = float(contacts.sum())
    n_in_play = float(in_play.sum())
    return {
        "n_pitches": int(n),
        "n_swings": int(n_swings),
        "n_contacts": int(n_contacts),
        "n_in_play": int(n_in_play),
        "swing_pct": _safe_ratio(n_swings, n),
        "take_pct": _safe_ratio(n - n_swings, n),
        "contact_pct": _safe_ratio(n_contacts, n_swings, default=0.0),
        "whiff_pct": _safe_ratio(whiffs.sum(), n_swings, default=0.0),
        "foul_pct": _safe_ratio(fouls.sum(), n_contacts, default=0.0),
        "called_strike_pct": _safe_ratio(called.sum(), max(n - n_swings, 1.0), default=0.0),
        "ball_pct": _safe_ratio(balls.sum(), max(n - n_swings, 1.0), default=0.0),
        "in_play_pct": _safe_ratio(n_in_play, n_contacts, default=0.0),
        "hr_on_contact_pct": _safe_ratio(hrs.sum(), n_contacts, default=0.0),
        "damage_on_contact_pct": _safe_ratio(damage.sum(), n_contacts, default=0.0),
    }


def _prepare_state_pitch_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = _statcast_slice(df.copy(), STATE_PITCH_CORE_COLS)
    if out.empty:
        return pd.DataFrame()

    for c in ("balls", "strikes", "zone", "plate_x", "plate_z", "release_speed", "launch_speed", "launch_angle", "pitch_number", "at_bat_number"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ("pitcher", "batter"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["description_norm"] = out.get("description", pd.Series(index=out.index, dtype=object)).astype(str).str.lower().str.strip()
    out["event_norm"] = out.get("events", pd.Series(index=out.index, dtype=object)).astype(str).str.lower().str.strip()
    out["pitch_type_norm"] = out.get("pitch_type", pd.Series(index=out.index, dtype=object)).map(_normalize_pitch_type)
    out["side_bat"] = out.get("stand", pd.Series(index=out.index, dtype=object)).astype(str).str.upper().str[0].where(lambda s: s.isin(["L", "R"]), "R")
    out["p_hand"] = out.get("p_throws", pd.Series(index=out.index, dtype=object)).astype(str).str.upper().str[0].where(lambda s: s.isin(["L", "R"]), "R")
    out["count"] = pd.to_numeric(out.get("balls", 0), errors="coerce").fillna(0).astype(int).astype(str) + "-" + pd.to_numeric(out.get("strikes", 0), errors="coerce").fillna(0).astype(int).astype(str)
    out["zone_bucket"] = [
        _location_bucket(z, x, y) for z, x, y in zip(out.get("zone"), out.get("plate_x"), out.get("plate_z"))
    ]
    out["horiz_bucket"] = [
        _horiz_bucket(x, ph) for x, ph in zip(out.get("plate_x"), out.get("p_hand"))
    ]
    out["vert_bucket"] = [_vert_bucket(z) for z in out.get("plate_z")]
    out["location_combo"] = out["zone_bucket"].astype(str) + "|" + out["horiz_bucket"].astype(str) + "|" + out["vert_bucket"].astype(str)
    out["weather_bucket"] = out.get("game_date", pd.Series(index=out.index, dtype=object)).map(_season_weather_bucket)
    out["park_team"] = out.get("home_team", pd.Series(index=out.index, dtype=object)).astype(str).str.upper().replace({"": "UNK"})
    out["launch_bucket"] = [
        _launch_damage_bucket(ev, la, br) for ev, la, br in zip(out.get("launch_speed"), out.get("launch_angle"), out.get("barrel"))
    ]

    desc = out["description_norm"]
    out["is_swing"] = desc.isin(SWING_DESCRIPTIONS).astype(int)
    out["is_contact"] = desc.isin(CONTACT_DESCRIPTIONS).astype(int)
    out["is_whiff"] = desc.isin(WHIFF_DESCRIPTIONS).astype(int)
    out["is_foul"] = desc.isin(FOUL_DESCRIPTIONS).astype(int)
    out["is_called_strike"] = desc.isin(CALLED_STRIKE_DESCRIPTIONS).astype(int)
    out["is_ball"] = desc.isin(BALL_DESCRIPTIONS).astype(int)
    out["is_in_play"] = desc.isin(INPLAY_DESCRIPTIONS).astype(int)
    out["is_hr"] = out["event_norm"].isin(HR_EVENTS).astype(int)
    out["is_damage_contact"] = out["launch_bucket"].isin({"barrel", "damage_air", "hard_contact"}).astype(int)
    out["is_on_base"] = out["event_norm"].isin(ONBASE_EVENTS).astype(int)
    out["is_out_event"] = out["event_norm"].isin(OUT_EVENTS).astype(int)
    return out


def _aggregate_state_table(df: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=by_cols + [
            "n_pitches", "n_swings", "n_contacts", "n_in_play",
            "swing_pct", "take_pct", "contact_pct", "whiff_pct", "foul_pct",
            "called_strike_pct", "ball_pct", "in_play_pct", "hr_on_contact_pct", "damage_on_contact_pct",
        ])
    rows = []
    for keys, g in df.groupby(by_cols, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(by_cols, keys))
        row.update(_state_rate_row(g))
        rows.append(row)
    return pd.DataFrame(rows)


@parquet_cache_df("pitcher_count_hand_state_{0}.parquet", ttl_days=7)
def build_pitcher_count_hand_state_table(days: int = 365) -> pd.DataFrame:
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=int(days))).strftime("%Y-%m-%d")
    raw = fetch_statcast_raw(start, end)
    df = _prepare_state_pitch_df(raw)
    if df.empty or "pitcher" not in df.columns:
        return pd.DataFrame()
    df = df[df["pitcher"].notna()].copy()
    df["pitcher"] = df["pitcher"].astype(int)
    return _aggregate_state_table(df, ["pitcher", "count", "side_bat"])


@parquet_cache_df("batter_count_hand_state_{0}.parquet", ttl_days=7)
def build_batter_count_hand_state_table(days: int = 365) -> pd.DataFrame:
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=int(days))).strftime("%Y-%m-%d")
    raw = fetch_statcast_raw(start, end)
    df = _prepare_state_pitch_df(raw)
    if df.empty or "batter" not in df.columns:
        return pd.DataFrame()
    df = df[df["batter"].notna()].copy()
    df["batter"] = df["batter"].astype(int)
    return _aggregate_state_table(df, ["batter", "count", "p_hand"])


@parquet_cache_df("pitcher_pitchtype_count_zone_state_{0}.parquet", ttl_days=7)
def build_pitcher_pitchtype_count_zone_state_table(days: int = 365) -> pd.DataFrame:
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=int(days))).strftime("%Y-%m-%d")
    raw = fetch_statcast_raw(start, end)
    df = _prepare_state_pitch_df(raw)
    if df.empty or "pitcher" not in df.columns:
        return pd.DataFrame()
    df = df[df["pitcher"].notna()].copy()
    df["pitcher"] = df["pitcher"].astype(int)
    return _aggregate_state_table(df, ["pitcher", "pitch_type_norm", "count", "side_bat", "zone_bucket", "horiz_bucket", "vert_bucket"])


@parquet_cache_df("batter_pitchtype_count_zone_state_{0}.parquet", ttl_days=7)
def build_batter_pitchtype_count_zone_state_table(days: int = 365) -> pd.DataFrame:
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=int(days))).strftime("%Y-%m-%d")
    raw = fetch_statcast_raw(start, end)
    df = _prepare_state_pitch_df(raw)
    if df.empty or "batter" not in df.columns:
        return pd.DataFrame()
    df = df[df["batter"].notna()].copy()
    df["batter"] = df["batter"].astype(int)
    return _aggregate_state_table(df, ["batter", "pitch_type_norm", "count", "p_hand", "zone_bucket", "horiz_bucket", "vert_bucket"])


def _bucketize_pitcher_archetype(sub: pd.DataFrame) -> str:
    velo = _safe_mean(sub.get("release_speed", pd.Series(dtype=float)))
    whiff = _safe_frac(sub.get("is_whiff", pd.Series(dtype=float)))
    loc_heart = _safe_frac(sub.get("zone_bucket", pd.Series(dtype=object)).astype(str).eq("heart"))
    velo_b = "velo_hi" if velo >= 95 else "velo_mid" if velo >= 92 else "velo_lo"
    miss_b = "miss_hi" if whiff >= 0.15 else "miss_mid" if whiff >= 0.10 else "miss_lo"
    zone_b = "zone_hi" if loc_heart >= 0.35 else "zone_mid" if loc_heart >= 0.22 else "zone_lo"
    return f"{velo_b}|{miss_b}|{zone_b}"


def _bucketize_batter_archetype(sub: pd.DataFrame) -> str:
    ev = _safe_mean(sub.get("launch_speed", pd.Series(dtype=float)))
    contact = _safe_ratio(pd.to_numeric(sub.get("is_contact", pd.Series(dtype=float)), errors="coerce").fillna(0).sum(),
                          max(pd.to_numeric(sub.get("is_swing", pd.Series(dtype=float)), errors="coerce").fillna(0).sum(), 1.0), 0.0)
    damage = _safe_frac(sub.get("is_damage_contact", pd.Series(dtype=float)))
    pow_b = "pow_hi" if damage >= 0.08 or ev >= 92 else "pow_mid" if damage >= 0.04 or ev >= 89 else "pow_lo"
    con_b = "con_hi" if contact >= 0.78 else "con_mid" if contact >= 0.68 else "con_lo"
    return f"{pow_b}|{con_b}"


@disk_cache("matchup_state_tensors_{0}_{1}.pkl", ttl_days=7)
def build_matchup_state_tensors(days: int = 365, min_pair_pitches: int = 25) -> dict:
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=int(days))).strftime("%Y-%m-%d")
    raw = fetch_statcast_raw(start, end)
    df = _prepare_state_pitch_df(raw)
    if df.empty or "pitcher" not in df.columns or "batter" not in df.columns:
        return {"pair": pd.DataFrame(), "archetype": pd.DataFrame(), "global": pd.DataFrame(), "hr_contact": pd.DataFrame()}

    df = df[df["pitcher"].notna() & df["batter"].notna()].copy()
    df["pitcher"] = df["pitcher"].astype(int)
    df["batter"] = df["batter"].astype(int)

    p_arch = {int(pid): _bucketize_pitcher_archetype(g) for pid, g in df.groupby("pitcher")}
    b_arch = {int(bid): _bucketize_batter_archetype(g) for bid, g in df.groupby("batter")}
    df["pitcher_archetype"] = df["pitcher"].map(p_arch)
    df["batter_archetype"] = df["batter"].map(b_arch)

    pair = _aggregate_state_table(df, ["pitcher", "batter", "count", "pitch_type_norm", "side_bat"])
    arch = _aggregate_state_table(df, ["pitcher_archetype", "batter_archetype", "count", "pitch_type_norm", "side_bat"])
    global_tbl = _aggregate_state_table(df, ["count", "pitch_type_norm", "side_bat"])

    contact_df = df[df["is_contact"] == 1].copy()
    if contact_df.empty:
        hr_contact = pd.DataFrame(columns=["launch_bucket", "park_team", "weather_bucket", "hr_on_contact_pct", "damage_on_contact_pct", "n_contacts"])
    else:
        rows = []
        for keys, g in contact_df.groupby(["launch_bucket", "park_team", "weather_bucket"], dropna=False, observed=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rows.append({
                "launch_bucket": keys[0],
                "park_team": keys[1],
                "weather_bucket": keys[2],
                "n_contacts": int(len(g)),
                "hr_on_contact_pct": _safe_frac(g["is_hr"]),
                "damage_on_contact_pct": _safe_frac(g["is_damage_contact"]),
            })
        hr_contact = pd.DataFrame(rows)

    if not pair.empty and not arch.empty:
        arch_keyed = arch.set_index(["pitcher_archetype", "batter_archetype", "count", "pitch_type_norm", "side_bat"])
        smoothed = []
        for _, row in pair.iterrows():
            p_arc = p_arch.get(int(row["pitcher"]), "unknown")
            b_arc = b_arch.get(int(row["batter"]), "unknown")
            key = (p_arc, b_arc, row["count"], row["pitch_type_norm"], row["side_bat"])
            base = arch_keyed.loc[key] if key in arch_keyed.index else None
            n = float(row.get("n_pitches", 0.0) or 0.0)
            prior_w = 60.0
            new_row = row.to_dict()
            if base is not None is not False and base is not None:
                for c in ["contact_pct", "whiff_pct", "in_play_pct", "hr_on_contact_pct", "damage_on_contact_pct", "ball_pct", "called_strike_pct"]:
                    raw_v = float(row.get(c, 0.0) or 0.0)
                    base_v = float(base.get(c, raw_v) or raw_v)
                    new_row[c] = (raw_v * n + base_v * prior_w) / max(n + prior_w, 1.0)
            smoothed.append(new_row)
        pair = pd.DataFrame(smoothed)

    return {"pair": pair, "archetype": arch, "global": global_tbl, "hr_contact": hr_contact, "pitcher_archetype": p_arch, "batter_archetype": b_arch}


def get_matchup_tensor_features(
    pitcher_id: int | None,
    batter_id: int | None,
    count: str = "0-0",
    pitch_type: str = "FF",
    side: str = "R",
    days: int = 365,
    min_pair_pitches: int = 25,
) -> dict:
    tensors = build_matchup_state_tensors(days, min_pair_pitches)
    pair = tensors.get("pair", pd.DataFrame())
    arch = tensors.get("archetype", pd.DataFrame())
    glob = tensors.get("global", pd.DataFrame())
    p_arch = tensors.get("pitcher_archetype", {}) or {}
    b_arch = tensors.get("batter_archetype", {}) or {}

    pt = _normalize_pitch_type(pitch_type)
    side = str(side or "R").upper()[:1]
    out = {"source": "global", "contact_prob": np.nan, "whiff_prob": np.nan, "hr_on_contact_prob": np.nan, "damage_on_contact_prob": np.nan, "n_pitches": 0}

    if isinstance(pair, pd.DataFrame) and not pair.empty and pitcher_id and batter_id:
        hit = pair[
            (pair["pitcher"] == int(pitcher_id)) &
            (pair["batter"] == int(batter_id)) &
            (pair["count"] == count) &
            (pair["pitch_type_norm"] == pt) &
            (pair["side_bat"].astype(str).str.upper() == side)
        ]
        if not hit.empty:
            row = hit.sort_values("n_pitches", ascending=False).iloc[0]
            if float(row.get("n_pitches", 0.0) or 0.0) >= min_pair_pitches:
                return {
                    "source": "pair",
                    "contact_prob": float(row.get("contact_pct", np.nan)),
                    "whiff_prob": float(row.get("whiff_pct", np.nan)),
                    "hr_on_contact_prob": float(row.get("hr_on_contact_pct", np.nan)),
                    "damage_on_contact_prob": float(row.get("damage_on_contact_pct", np.nan)),
                    "n_pitches": int(row.get("n_pitches", 0)),
                }

    if isinstance(arch, pd.DataFrame) and not arch.empty and pitcher_id and batter_id:
        p_key = p_arch.get(int(pitcher_id))
        b_key = b_arch.get(int(batter_id))
        hit = arch[
            (arch["pitcher_archetype"] == p_key) &
            (arch["batter_archetype"] == b_key) &
            (arch["count"] == count) &
            (arch["pitch_type_norm"] == pt) &
            (arch["side_bat"].astype(str).str.upper() == side)
        ]
        if not hit.empty:
            row = hit.sort_values("n_pitches", ascending=False).iloc[0]
            return {
                "source": "archetype",
                "contact_prob": float(row.get("contact_pct", np.nan)),
                "whiff_prob": float(row.get("whiff_pct", np.nan)),
                "hr_on_contact_prob": float(row.get("hr_on_contact_pct", np.nan)),
                "damage_on_contact_prob": float(row.get("damage_on_contact_pct", np.nan)),
                "n_pitches": int(row.get("n_pitches", 0)),
            }

    if isinstance(glob, pd.DataFrame) and not glob.empty:
        hit = glob[
            (glob["count"] == count) &
            (glob["pitch_type_norm"] == pt) &
            (glob["side_bat"].astype(str).str.upper() == side)
        ]
        if not hit.empty:
            row = hit.sort_values("n_pitches", ascending=False).iloc[0]
            return {
                "source": "global",
                "contact_prob": float(row.get("contact_pct", np.nan)),
                "whiff_prob": float(row.get("whiff_pct", np.nan)),
                "hr_on_contact_prob": float(row.get("hr_on_contact_pct", np.nan)),
                "damage_on_contact_prob": float(row.get("damage_on_contact_pct", np.nan)),
                "n_pitches": int(row.get("n_pitches", 0)),
            }
    return out


def get_hr_contact_context_prob(launch_bucket: str, park_team: str, weather_bucket: str, days: int = 365) -> dict:
    tensors = build_matchup_state_tensors(days)
    tbl = tensors.get("hr_contact", pd.DataFrame())
    if not isinstance(tbl, pd.DataFrame) or tbl.empty:
        return {"source": "fallback", "hr_on_contact_prob": 0.06, "damage_on_contact_prob": 0.35, "n_contacts": 0}
    hit = tbl[
        (tbl["launch_bucket"].astype(str) == str(launch_bucket)) &
        (tbl["park_team"].astype(str).str.upper() == str(park_team).upper()) &
        (tbl["weather_bucket"].astype(str) == str(weather_bucket))
    ]
    if hit.empty:
        hit = tbl[tbl["launch_bucket"].astype(str) == str(launch_bucket)]
    if hit.empty:
        return {"source": "fallback", "hr_on_contact_prob": 0.06, "damage_on_contact_prob": 0.35, "n_contacts": 0}
    row = hit.sort_values("n_contacts", ascending=False).iloc[0]
    return {
        "source": "hr_contact_context",
        "hr_on_contact_prob": float(row.get("hr_on_contact_pct", 0.06)),
        "damage_on_contact_prob": float(row.get("damage_on_contact_pct", 0.35)),
        "n_contacts": int(row.get("n_contacts", 0)),
    }


def _pitcher_game_level_frame(pid: int, days: int = 120) -> pd.DataFrame:
    if not pid:
        return pd.DataFrame()
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=int(days))).strftime("%Y-%m-%d")
    raw = fetch_statcast_raw(start, end)
    df = _prepare_state_pitch_df(raw)
    if df.empty or "pitcher" not in df.columns:
        return pd.DataFrame()
    df = df[df["pitcher"] == int(pid)].copy()
    if df.empty or "game_pk" not in df.columns:
        return pd.DataFrame()

    rows = []
    for gpk, g in df.groupby("game_pk", sort=False):
        g = g.sort_values([c for c in ["game_date", "pitch_number"] if c in g.columns], kind="stable")
        game_date = pd.to_datetime(g.get("game_date", pd.Series(index=g.index, dtype=object)), errors="coerce").max()
        pitches = int(pd.to_numeric(g.get("pitch_number", pd.Series(index=g.index, dtype=float)), errors="coerce").max()) if "pitch_number" in g.columns else int(len(g))
        if not np.isfinite(pitches) or pitches <= 0:
            pitches = int(len(g))
        bf = int(g.get("events", pd.Series(index=g.index, dtype=object)).notna().sum())
        onbase = int(pd.to_numeric(g.get("is_on_base", pd.Series(index=g.index, dtype=float)), errors="coerce").fillna(0).sum())
        hard = int(pd.to_numeric(g.get("is_damage_contact", pd.Series(index=g.index, dtype=float)), errors="coerce").fillna(0).sum())
        hrs = int(pd.to_numeric(g.get("is_hr", pd.Series(index=g.index, dtype=float)), errors="coerce").fillna(0).sum())
        stress = float(0.02 * max(bf - 18, 0) + 0.10 * hrs + 0.04 * onbase + 0.015 * hard)
        rows.append({
            "game_pk": int(gpk),
            "game_date": game_date,
            "pitch_count": int(pitches),
            "batters_faced": int(bf),
            "on_base_events": int(onbase),
            "hard_damage_events": int(hard),
            "hr_allowed": int(hrs),
            "stress_score": float(stress),
            "times_through_order": float(bf / 9.0) if bf > 0 else 0.0,
            "start_like": int((pitches >= 50) or (bf >= 15)),
        })
    out = pd.DataFrame(rows)
    if not out.empty and "game_date" in out.columns:
        out = out.sort_values("game_date", ascending=False, kind="stable").reset_index(drop=True)
    return out


@disk_cache("pitcher_recent_start_context_{0}.pkl", ttl_days=1)
def get_pitcher_recent_start_context(pid: int, days: int = 120, max_games: int = 3) -> dict:
    gl = _pitcher_game_level_frame(int(pid), days=days)
    if gl.empty:
        return {
            "last3_pitch_counts": [], "last3_batters_faced": [], "last3_stress_scores": [],
            "avg_pitch_count_last3": np.nan, "avg_bf_last3": np.nan, "avg_stress_last3": np.nan,
            "avg_tto_exposure": np.nan, "pct_third_time": np.nan, "pct_fourth_time": np.nan,
        }
    starts = gl[gl["start_like"] == 1].head(max_games).copy()
    if starts.empty:
        starts = gl.head(max_games).copy()
    bf = pd.to_numeric(starts["batters_faced"], errors="coerce").fillna(0)
    pc = pd.to_numeric(starts["pitch_count"], errors="coerce").fillna(0)
    stress = pd.to_numeric(starts["stress_score"], errors="coerce").fillna(0)
    tto = pd.to_numeric(starts["times_through_order"], errors="coerce").fillna(0)
    return {
        "last3_pitch_counts": [int(x) for x in pc.tolist()],
        "last3_batters_faced": [int(x) for x in bf.tolist()],
        "last3_stress_scores": [float(x) for x in stress.tolist()],
        "avg_pitch_count_last3": float(pc.mean()) if len(pc) else np.nan,
        "avg_bf_last3": float(bf.mean()) if len(bf) else np.nan,
        "avg_stress_last3": float(stress.mean()) if len(stress) else np.nan,
        "avg_tto_exposure": float(tto.mean()) if len(tto) else np.nan,
        "pct_third_time": float((bf >= 19).mean()) if len(bf) else np.nan,
        "pct_fourth_time": float((bf >= 28).mean()) if len(bf) else np.nan,
    }


@disk_cache("bullpen_freshness_{0}.pkl", ttl_days=1)
def get_bullpen_freshness(team_abbr: str, days: int = 3) -> dict:
    ab = _canonical_team_abbr(team_abbr)
    staff = full_pitching_staff(ab) or {}
    if not staff:
        return {"bullpen_freshness_score": np.nan, "bullpen_recent_pitches": 0, "bullpen_recent_apps": 0, "bullpen_recent_bf": 0}

    ps = pitching_stats(YEAR, qual=0)
    if isinstance(ps, pd.DataFrame) and not ps.empty and "Name" in ps.columns:
        ps = ps.copy()
        ps["Name_norm"] = ps["Name"].apply(clean_name)
    reliever_ids = []
    for nm, pid in staff.items():
        try:
            rec = ps[ps["Name_norm"] == nm] if isinstance(ps, pd.DataFrame) and not ps.empty else pd.DataFrame()
            gs = int(rec["GS"].iloc[0]) if not rec.empty and "GS" in rec.columns and pd.notna(rec["GS"].iloc[0]) else 0
        except Exception:
            gs = 0
        if gs <= 1:
            reliever_ids.append(int(pid))
    if not reliever_ids:
        reliever_ids = [int(pid) for pid in staff.values()]

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=int(days))).strftime("%Y-%m-%d")
    raw = fetch_statcast_raw(start, end)
    df = _prepare_state_pitch_df(raw)
    if df.empty or "pitcher" not in df.columns:
        return {"bullpen_freshness_score": 1.0, "bullpen_recent_pitches": 0, "bullpen_recent_apps": 0, "bullpen_recent_bf": 0}
    df = df[df["pitcher"].isin(reliever_ids)].copy()
    if df.empty:
        return {"bullpen_freshness_score": 1.0, "bullpen_recent_pitches": 0, "bullpen_recent_apps": 0, "bullpen_recent_bf": 0}
    pitches = int(len(df))
    apps = int(df.groupby(["pitcher", "game_pk"], dropna=False).ngroups)
    bf = int(df.get("events", pd.Series(index=df.index, dtype=object)).notna().sum())
    pen_size = max(len(set(reliever_ids)), 1)
    pitch_load = pitches / max(55.0 * pen_size, 1.0)
    app_load = apps / max(2.0 * pen_size, 1.0)
    freshness = float(np.clip(1.0 - 0.55 * pitch_load - 0.45 * app_load, 0.0, 1.0))
    return {
        "bullpen_freshness_score": freshness,
        "bullpen_recent_pitches": pitches,
        "bullpen_recent_apps": apps,
        "bullpen_recent_bf": bf,
    }


@disk_cache("manager_hook_tendency_{0}.pkl", ttl_days=1)
def estimate_manager_hook_tendency(team_abbr: str, days: int = 120) -> dict:
    ab = _canonical_team_abbr(team_abbr)
    staff = full_pitching_staff(ab) or {}
    if not staff:
        return {"manager_hook_pitch_count": np.nan, "manager_hook_bf": np.nan, "manager_quick_hook_rate": np.nan}

    ps = pitching_stats(YEAR, qual=0)
    if isinstance(ps, pd.DataFrame) and not ps.empty and "Name" in ps.columns:
        ps = ps.copy()
        ps["Name_norm"] = ps["Name"].apply(clean_name)
    starter_ids = []
    for nm, pid in staff.items():
        try:
            rec = ps[ps["Name_norm"] == nm] if isinstance(ps, pd.DataFrame) and not ps.empty else pd.DataFrame()
            gs = int(rec["GS"].iloc[0]) if not rec.empty and "GS" in rec.columns and pd.notna(rec["GS"].iloc[0]) else 0
        except Exception:
            gs = 0
        if gs >= 3:
            starter_ids.append(int(pid))
    if not starter_ids:
        starter_ids = [int(pid) for pid in staff.values()][:5]

    pitch_means = []
    bf_means = []
    quick_flags = []
    for pid in starter_ids:
        ctx = get_pitcher_recent_start_context(pid, days=days, max_games=3)
        if np.isfinite(ctx.get("avg_pitch_count_last3", np.nan)):
            pitch_means.append(float(ctx["avg_pitch_count_last3"]))
        if np.isfinite(ctx.get("avg_bf_last3", np.nan)):
            bf_means.append(float(ctx["avg_bf_last3"]))
        pcs = ctx.get("last3_pitch_counts", []) or []
        bfs = ctx.get("last3_batters_faced", []) or []
        for pc, bf in zip(pcs, bfs):
            quick_flags.append(int((pc <= 85) or (bf <= 21)))

    return {
        "manager_hook_pitch_count": float(np.mean(pitch_means)) if pitch_means else np.nan,
        "manager_hook_bf": float(np.mean(bf_means)) if bf_means else np.nan,
        "manager_quick_hook_rate": float(np.mean(quick_flags)) if quick_flags else np.nan,
    }


@disk_cache("starter_survival_context_{0}_{1}.pkl", ttl_days=1)
def get_starter_survival_context(team_abbr: str, pitcher_pid: int, days: int = 120) -> dict:
    pid = int(pitcher_pid) if pitcher_pid else 0
    start_ctx = get_pitcher_recent_start_context(pid, days=days, max_games=3)
    pen_ctx = get_bullpen_freshness(team_abbr, days=3)
    hook_ctx = estimate_manager_hook_tendency(team_abbr, days=days)
    return {
        **start_ctx,
        **pen_ctx,
        **hook_ctx,
        "days_rest": get_days_rest(pid),
        "travel_days": get_travel_days(_canonical_team_abbr(team_abbr)),
    }

if __name__=="__main__":
    print("▶ Starting full fetch.py cache run...")
    start=time.time()
    _=load_teams_info()
    for ab in ABBR2ID: roster_map(ab); full_pitching_staff(ab)
    fetch_yearly_park_factors(YEAR); fetch_monthly_park_factors(YEAR)
    _=fetch_statcast_raw(f"{YEAR}-03-01",f"{YEAR}-11-01")
    print(f" fetch.py all cached in {time.time()-start:.1f}s")
