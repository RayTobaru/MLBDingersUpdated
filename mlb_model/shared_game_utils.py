from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Optional
import re

from fetch import (  # type: ignore
    safe_get,
    ID2ABBR,
    fetch_unofficial_lineup,
    fetch_lineup_and_starters,
    name_to_mlbam_id,
    strip_pos,
    clean_name,
    roster_map,
    ABBR2ID,
    active_hitter_names,
    pitcher_throws,
)

TEAM_ABBR_ALIASES = {
    'KC': 'KCR',
    'AZ': 'ARI',
    'CWS': 'CHW',
    'WSH': 'WSN',
    'TB': 'TBR',
    'SF': 'SFG',
    'SD': 'SDP',
}


@lru_cache(maxsize=1)
def _lineup_utils():
    from . import lineups as lu  # type: ignore
    return lu



def fetch_matchups_for_date(date_str: str) -> list[str]:
    resp = safe_get('https://statsapi.mlb.com/api/v1/schedule', {'sportId': 1, 'date': date_str})
    js = resp.json() if resp else {}
    out = []
    for d in js.get('dates', []):
        for g in d.get('games', []):
            try:
                away = ID2ABBR[g['teams']['away']['team']['id']]
                home = ID2ABBR[g['teams']['home']['team']['id']]
                out.append(f'{away} @ {home}')
            except Exception:
                continue
    return out


def _get_probables_for_date(date_str: str, away: str, home: str):
    resp = safe_get('https://statsapi.mlb.com/api/v1/schedule', {'sportId': 1, 'date': date_str})
    js = resp.json() if resp else {}
    game_pk = None
    away_name = home_name = None
    away_pid = home_pid = None
    for d in js.get('dates', []):
        for g in d.get('games', []):
            try:
                a = ID2ABBR[g['teams']['away']['team']['id']]
                h = ID2ABBR[g['teams']['home']['team']['id']]
            except Exception:
                continue
            if (a, h) != (away, home):
                continue
            game_pk = g.get('gamePk')
            pp = g.get('probablePitchers', {})
            away_name = strip_pos(pp.get('away', {}).get('fullName', '') or '') or None
            home_name = strip_pos(pp.get('home', {}).get('fullName', '') or '') or None
            away_pid = pp.get('away', {}).get('id')
            home_pid = pp.get('home', {}).get('id')
            return away_name, home_name, away_pid, home_pid, game_pk
    return away_name, home_name, away_pid, home_pid, game_pk


def _canonical_team_abbr(team_abbr: str) -> str:
    ab = str(team_abbr or '').strip().upper()
    if ab in ABBR2ID:
        return ab
    return TEAM_ABBR_ALIASES.get(ab, ab)


@lru_cache(maxsize=64)
def _team_roster_lookup(team_abbr: str) -> dict[str, int]:
    ab = _canonical_team_abbr(team_abbr)
    tid = ABBR2ID.get(ab)
    out: dict[str, int] = {}
    if not tid:
        return out

    try:
        rm = roster_map(ab) or {}
        for nm, pid in rm.items():
            if pid:
                out[clean_name(nm)] = int(pid)
    except Exception:
        pass

    endpoints = [
        f'https://statsapi.mlb.com/api/v1/teams/{tid}/roster/active',
        f'https://statsapi.mlb.com/api/v1/teams/{tid}/roster/40Man',
    ]
    for url in endpoints:
        try:
            resp = safe_get(url)
            js = resp.json() if resp else {}
            for p in js.get('roster', []):
                nm = p.get('person', {}).get('fullName')
                pid = p.get('person', {}).get('id')
                if nm and pid:
                    out[clean_name(nm)] = int(pid)
        except Exception:
            continue

    return out


def _team_roster_names(team_abbr: str) -> dict[str, int]:
    return _team_roster_lookup(_canonical_team_abbr(team_abbr))


def _active_roster_keyset(team_abbr: str) -> set[str]:
    try:
        return {clean_name(strip_pos(nm)) for nm in active_hitter_names(_canonical_team_abbr(team_abbr)) if str(nm).strip()}
    except Exception:
        return set()


def _search_people_by_name(hitter_name: str | None) -> int | None:
    nm = strip_pos(hitter_name or '').strip()
    if not nm or nm.upper() == 'TBD':
        return None
    try:
        resp = safe_get('https://statsapi.mlb.com/api/v1/people/search', {'sportId': 1, 'names': nm})
        js = resp.json() if resp else {}
        people = js.get('people', []) or []
        key = clean_name(nm)
        for p in people:
            full = p.get('fullName') or ''
            pid = p.get('id')
            if pid and clean_name(full) == key:
                return int(pid)
        for p in people:
            pid = p.get('id')
            if pid:
                return int(pid)
    except Exception:
        return None
    return None


def _resolve_batter_id(team_abbr: str, hitter_name: str | None):
    hitter_name = strip_pos(hitter_name or '').strip()
    if not hitter_name or hitter_name.upper() == 'TBD':
        return None

    key = clean_name(hitter_name)
    ab = _canonical_team_abbr(team_abbr)

    try:
        rm = _team_roster_lookup(ab)
        if key in rm:
            return int(rm[key])
    except Exception:
        pass

    alias = re.sub(r"\b(JR|SR|II|III|IV)\b", '', key).replace('.', ' ').strip()
    alias = re.sub(r'\s+', ' ', alias)
    try:
        rm = _team_roster_lookup(ab)
        if alias in rm:
            return int(rm[alias])
        for nm, pid in rm.items():
            nm2 = re.sub(r"\b(JR|SR|II|III|IV)\b", '', str(nm)).replace('.', ' ').strip()
            nm2 = re.sub(r'\s+', ' ', nm2)
            if nm2 == alias:
                return int(pid)
    except Exception:
        pass

    pid = _search_people_by_name(hitter_name)
    if pid:
        return int(pid)

    try:
        pid = name_to_mlbam_id(hitter_name)
        if pid:
            return int(pid)
    except Exception:
        pass

    return None


def _strict_resolve_batter(team_abbr: str, hitter_name: str | None):
    hitter_name = strip_pos(hitter_name or '').strip()
    if not hitter_name or hitter_name.upper() == 'TBD':
        return None, None

    roster = _team_roster_names(team_abbr)
    key = clean_name(hitter_name)
    if key in roster:
        return int(roster[key]), hitter_name

    alias = re.sub(r"\b(JR|SR|II|III|IV)\b", '', key).replace('.', ' ').strip()
    alias = re.sub(r'\s+', ' ', alias)
    for nm, pid in roster.items():
        nm2 = re.sub(r"\b(JR|SR|II|III|IV)\b", '', str(nm)).replace('.', ' ').strip()
        nm2 = re.sub(r'\s+', ' ', nm2)
        if nm2 == alias:
            return int(pid), hitter_name

    pid = _resolve_batter_id(team_abbr, hitter_name)
    if pid and int(pid) in set(roster.values()):
        return int(pid), hitter_name
    return None, None


def _sanitize_lineup_names(team_abbr: str, lineup: list[str]) -> list[str]:
    cleaned = []
    for nm in list(lineup or [])[:9]:
        raw = strip_pos(str(nm or '')).strip()
        if not raw or raw.upper() == 'TBD':
            cleaned.append('TBD')
        else:
            cleaned.append(raw)
    return cleaned


def _resolve_lineups(date_str: str, away: str, home: str, override_path: str | None = None):
    matchup_key = f'{away}@{home}'
    override = _lineup_utils().get_override_for_game(date_str, matchup_key, override_path)

    lineup_source = 'projected'
    ump_feats = {'ump_k9': 1.0, 'ump_bb9': 1.0}
    framing_feats = {'away_frame': 0.0, 'home_frame': 0.0}
    away_cid = home_cid = None

    if date_str == datetime.today().strftime('%Y-%m-%d'):
        try:
            a_name, h_name, a_line, h_line, a_pid, h_pid, away_cid, home_cid, ump_feats, framing_feats = fetch_lineup_and_starters(away, home)
            _, _, _, _, game_pk = _get_probables_for_date(date_str, away, home)
            lineup_source = 'confirmed' if len(a_line) >= 9 and len(h_line) >= 9 else 'hybrid'
        except Exception:
            a_name, h_name, a_pid, h_pid, game_pk = _get_probables_for_date(date_str, away, home)
            a_line = fetch_unofficial_lineup(away)
            h_line = fetch_unofficial_lineup(home)
            lineup_source = 'projected'
    else:
        a_name, h_name, a_pid, h_pid, game_pk = _get_probables_for_date(date_str, away, home)
        a_line = fetch_unofficial_lineup(away)
        h_line = fetch_unofficial_lineup(home)
        lineup_source = 'projected'

    if override:
        a_line = override.get('away_lineup', a_line)
        h_line = override.get('home_lineup', h_line)
        a_name = override.get('away_pitcher_name', a_name)
        h_name = override.get('home_pitcher_name', h_name)
        a_pid = override.get('away_pitcher_id', a_pid)
        h_pid = override.get('home_pitcher_id', h_pid)
        game_pk = override.get('game_pk', game_pk)
        lineup_source = str(override.get('lineup_source', 'hybrid' if lineup_source == 'confirmed' else lineup_source))

    if a_pid is None and a_name:
        a_pid = name_to_mlbam_id(a_name)
    if h_pid is None and h_name:
        h_pid = name_to_mlbam_id(h_name)

    away_opp_hand = pitcher_throws(int(h_pid)) if h_pid else 'R'
    home_opp_hand = pitcher_throws(int(a_pid)) if a_pid else 'R'

    a_line = _sanitize_lineup_names(away, a_line)
    h_line = _sanitize_lineup_names(home, h_line)

    away_before = list(a_line or [])
    home_before = list(h_line or [])
    if _lineup_utils().lineup_needs_projection(a_line):
        a_line = _lineup_utils().fill_tbd_lineup(a_line, away, pitcher_hand=away_opp_hand, n=9)
    if _lineup_utils().lineup_needs_projection(h_line):
        h_line = _lineup_utils().fill_tbd_lineup(h_line, home, pitcher_hand=home_opp_hand, n=9)

    if lineup_source == 'confirmed' and (_lineup_utils().lineup_needs_projection(away_before) or _lineup_utils().lineup_needs_projection(home_before)):
        lineup_source = 'hybrid_projected'
    elif lineup_source in {'projected', 'hybrid'}:
        away_changed = [strip_pos(str(x or '')).strip() for x in away_before[:9]] != [strip_pos(str(x or '')).strip() for x in a_line[:9]]
        home_changed = [strip_pos(str(x or '')).strip() for x in home_before[:9]] != [strip_pos(str(x or '')).strip() for x in h_line[:9]]
        if away_changed or home_changed:
            lineup_source = 'projected_roster' if lineup_source == 'projected' else 'hybrid_projected'

    lw = _lineup_utils().lineup_source_weight(lineup_source)
    return {
        'away_pitcher_name': a_name,
        'home_pitcher_name': h_name,
        'away_lineup': a_line,
        'home_lineup': h_line,
        'away_pitcher_id': a_pid,
        'home_pitcher_id': h_pid,
        'game_pk': game_pk,
        'lineup_source': lineup_source,
        'lineup_weight': lw,
        'lineup_certainty': _lineup_utils().lineup_certainty_score(lineup_source, len(a_line) >= 9 and len(h_line) >= 9),
        'ump_feats': ump_feats,
        'framing_feats': framing_feats,
        'away_catcher_id': away_cid,
        'home_catcher_id': home_cid,
    }


__all__ = [
    'TEAM_ABBR_ALIASES',
    'fetch_matchups_for_date',
    '_get_probables_for_date',
    '_canonical_team_abbr',
    '_team_roster_lookup',
    '_team_roster_names',
    '_active_roster_keyset',
    '_search_people_by_name',
    '_resolve_batter_id',
    '_strict_resolve_batter',
    '_sanitize_lineup_names',
    '_resolve_lineups',
]
