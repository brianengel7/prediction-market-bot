#!/usr/bin/env python3

"""BreakTheBank -> Kalshi observed-fill proportional cash copier (V30 sports extension).



    python3 breakthebank.py --paper

    python3 breakthebank.py --live



Every newly observed, eligible Polymarket BUY/SELL fill can be copied with 0.03%

proportional dollar allocations (60-second batching) and 0.01-contract

rounding; no arbitrary strategy purchase caps. Source-wallet SELLs reduce

our matching position by the same fraction. Uses Kalshi YES-side event orders (Sept 2026 V2).

PAPER uses current public Kalshi order-book depth; LIVE submits IOC orders.



NBA/NFL automatic matcher retained; conservative additional sports game-winner matcher in PAPER mode; universal reviewed cross-category mappings. No live order on catalog similarity alone.

Exceptional settlement rules remain unverified; use paper until reviewed.

The account must be funded and live API behaviour independently verified.

An initial poll only establishes a baseline. SQLite is persistent and separate

for paper/live. A live order with an ambiguous response permanently HALTS live

mode until manually reconciled. Do not run concurrent instances for one mode.



Requires pip install cryptography only for --live. Never print private keys.

"""

from __future__ import annotations

import argparse

import base64

from collections import Counter

import fcntl

from datetime import date, datetime, timedelta, timezone

from decimal import Decimal, ROUND_DOWN, ROUND_UP

import hashlib
from difflib import SequenceMatcher

import json

import math

import os

from pathlib import Path

import re

import sqlite3

import sys

import time

from urllib.error import HTTPError, URLError

from urllib.parse import quote, urlencode

from urllib.request import Request, urlopen

from uuid import uuid4



WALLET = '0xf0318c32136c2db7fec88b84869aee6a1106c80c'

POLY_DATA = 'https://data-api.polymarket.com/v2/trades'

POLY_CLOB = 'https://clob.polymarket.com'

POLY_GAMMA = 'https://gamma-api.polymarket.com'

KALSHI = 'https://external-api.kalshi.com/trade-api/v2'

SLUG = re.compile(r'^(nba|nfl)-([a-z0-9]+)-([a-z0-9]+)-(20\d{2}-\d{2}-\d{2})(?:-(.+))?$', re.I)

NBA = {

'atl':'Atlanta Hawks|Hawks','bos':'Boston Celtics|Celtics','bkn':'Brooklyn Nets|Nets',

'cha':'Charlotte Hornets|Hornets','chi':'Chicago Bulls|Bulls','cle':'Cleveland Cavaliers|Cavaliers',

'dal':'Dallas Mavericks|Mavericks','den':'Denver Nuggets|Nuggets','det':'Detroit Pistons|Pistons',

'gsw':'Golden State Warriors|Warriors','hou':'Houston Rockets|Rockets','ind':'Indiana Pacers|Pacers',

'lac':'Los Angeles Clippers|Clippers','lal':'Los Angeles Lakers|Lakers','mem':'Memphis Grizzlies|Grizzlies',

'mia':'Miami Heat|Heat','mil':'Milwaukee Bucks|Bucks','min':'Minnesota Timberwolves|Timberwolves',

'nop':'New Orleans Pelicans|Pelicans','nyk':'New York Knicks|Knicks','okc':'Oklahoma City Thunder|Thunder',

'orl':'Orlando Magic|Magic','phi':'Philadelphia 76ers|76ers|Sixers','phx':'Phoenix Suns|Suns',

'por':'Portland Trail Blazers|Trail Blazers|Blazers','sac':'Sacramento Kings|Kings',

'sas':'San Antonio Spurs|Spurs','tor':'Toronto Raptors|Raptors','uta':'Utah Jazz|Jazz',

'was':'Washington Wizards|Wizards'}

NFL = {

 'ari':'Arizona Cardinals|Cardinals','atl':'Atlanta Falcons|Falcons',

 'bal':'Baltimore Ravens|Ravens','buf':'Buffalo Bills|Bills',

 'car':'Carolina Panthers|Panthers','chi':'Chicago Bears|Bears',

 'cin':'Cincinnati Bengals|Bengals','cle':'Cleveland Browns|Browns',

 'dal':'Dallas Cowboys|Cowboys','den':'Denver Broncos|Broncos',

 'det':'Detroit Lions|Lions','gb':'Green Bay Packers|Packers',

 'hou':'Houston Texans|Texans','ind':'Indianapolis Colts|Colts',

 'jax':'Jacksonville Jaguars|Jaguars','kc':'Kansas City Chiefs|Chiefs',

 'lv':'Las Vegas Raiders|Raiders','lac':'Los Angeles Chargers|Chargers',

 'lar':'Los Angeles Rams|Rams','mia':'Miami Dolphins|Dolphins',

 'min':'Minnesota Vikings|Vikings','ne':'New England Patriots|Patriots',

 'no':'New Orleans Saints|Saints','nyg':'New York Giants|Giants',

 'nyj':'New York Jets|Jets','phi':'Philadelphia Eagles|Eagles',

 'pit':'Pittsburgh Steelers|Steelers','sf':'San Francisco 49ers|49ers',

 'sea':'Seattle Seahawks|Seahawks','tb':'Tampa Bay Buccaneers|Buccaneers',

 'ten':'Tennessee Titans|Titans','was':'Washington Commanders|Commanders'

}

TEAMS = {'nba': NBA, 'nfl': NFL}

KAL_CODES = {'jax':'JAC'}

# V30: explicit league-level sports discovery. The three established NBA/NFL
# game/spread/total matchers below are preserved. The cross-sport extension is
# intentionally PAPER-ONLY unless an outcome has a reviewed mapping entry.
# New league aliases do NOT imply every prop or special outcome is equivalent.
NHL = {
 'ana':'Anaheim Ducks|Ducks','bos':'Boston Bruins|Bruins','buf':'Buffalo Sabres|Sabres',
 'cgy':'Calgary Flames|Flames','car':'Carolina Hurricanes|Hurricanes',
 'chi':'Chicago Blackhawks|Blackhawks','col':'Colorado Avalanche|Avalanche',
 'cbj':'Columbus Blue Jackets|Blue Jackets','dal':'Dallas Stars|Stars',
 'det':'Detroit Red Wings|Red Wings','edm':'Edmonton Oilers|Oilers',
 'fla':'Florida Panthers|Panthers','lak':'Los Angeles Kings|LA Kings|Kings',
 'min':'Minnesota Wild|Wild','mtl':'Montreal Canadiens|Canadiens',
 'mon':'Montreal Canadiens|Canadiens','njd':'New Jersey Devils|Devils',
 'nsh':'Nashville Predators|Predators','nyi':'New York Islanders|Islanders',
 'nyr':'New York Rangers|Rangers','ott':'Ottawa Senators|Senators',
 'phi':'Philadelphia Flyers|Flyers','pit':'Pittsburgh Penguins|Penguins',
 'sjs':'San Jose Sharks|Sharks','sea':'Seattle Kraken|Kraken',
 'stl':'St. Louis Blues|St Louis Blues|Blues','tbl':'Tampa Bay Lightning|Lightning',
 'tor':'Toronto Maple Leafs|Maple Leafs','uta':'Utah Mammoth|Utah Hockey Club|Mammoth',
 'utah':'Utah Mammoth|Utah Hockey Club|Mammoth','van':'Vancouver Canucks|Canucks',
 'vgk':'Vegas Golden Knights|Golden Knights','wsh':'Washington Capitals|Capitals',
 'wpg':'Winnipeg Jets|Jets'
}
MLB = {
 'ari':'Arizona Diamondbacks|Diamondbacks','atl':'Atlanta Braves|Braves',
 'bal':'Baltimore Orioles|Orioles','bos':'Boston Red Sox|Red Sox',
 'chc':'Chicago Cubs|Cubs','chw':'Chicago White Sox|White Sox',
 'cin':'Cincinnati Reds|Reds','cle':'Cleveland Guardians|Guardians',
 'col':'Colorado Rockies|Rockies','det':'Detroit Tigers|Tigers',
 'hou':'Houston Astros|Astros','kc':'Kansas City Royals|Royals',
 'laa':'Los Angeles Angels|Angels','lad':'Los Angeles Dodgers|Dodgers',
 'mia':'Miami Marlins|Marlins','mil':'Milwaukee Brewers|Brewers',
 'min':'Minnesota Twins|Twins','nym':'New York Mets|Mets',
 'nyy':'New York Yankees|Yankees','oak':'Athletics|Oakland Athletics',
 'phi':'Philadelphia Phillies|Phillies','pit':'Pittsburgh Pirates|Pirates',
 'sd':'San Diego Padres|Padres','sea':'Seattle Mariners|Mariners',
 'sf':'San Francisco Giants|Giants','stl':'St. Louis Cardinals|Cardinals',
 'tb':'Tampa Bay Rays|Rays','tex':'Texas Rangers|Rangers',
 'tor':'Toronto Blue Jays|Blue Jays','wsh':'Washington Nationals|Nationals'
}
WNBA = {
 'atl':'Atlanta Dream|Dream','chi':'Chicago Sky|Sky',
 'con':'Connecticut Sun|Sun','dal':'Dallas Wings|Wings',
 'ind':'Indiana Fever|Fever','las':'Los Angeles Sparks|Sparks',
 'lv':'Las Vegas Aces|Aces','min':'Minnesota Lynx|Lynx',
 'nyl':'New York Liberty|Liberty','phx':'Phoenix Mercury|Mercury',
 'sea':'Seattle Storm|Storm','was':'Washington Mystics|Mystics',
 'gs':'Golden State Valkyries|Valkyries','por':'Portland Fire|Fire',
 'tor':'Toronto Tempo|Tempo'
}
SPORT_LEAGUES = {
 'nba':('KXNBAGAME',NBA), 'nfl':('KXNFLGAME',NFL),
 'nhl':('KXNHLGAME',NHL), 'mlb':('KXMLBGAME',MLB),
 'wnba':('KXWNBAGAME',WNBA),
 'cfb':('KXNCAAFGAME',{}), 'ncaaf':('KXNCAAFGAME',{}),
 'cbb':('KXNCAAMBGAME',{}), 'ncaab':('KXNCAAMBGAME',{}),
 'ncaamb':('KXNCAAMBGAME',{}), 'ncaaw':('KXNCAAWBGAME',{}),
 'atp':('KXATPMATCH',{}), 'wta':('KXWTAMATCH',{}),
 'ufc':('KXUFCFIGHT',{}), 'mma':('KXUFCFIGHT',{}),
 'epl':('KXEPLGAME',{}), 'ucl':('KXUCLGAME',{}),
 'mls':('KXMLSGAME',{}), 'laliga':('KXLALIGAGAME',{}),
 'bundesliga':('KXBUNDESLIGAGAME',{}), 'seriea':('KXSERIEAGAME',{}),
 'ligue1':('KXLIGUE1GAME',{}),
 'lol':('KXLOLMATCH',{}), 'cs2':('KXCS2MATCH',{}),
 'dota2':('KXDOTA2MATCH',{}), 'valorant':('KXVALORANTMATCH',{}),
}
# Names in the source are aliases of the same team, but winning lines for
# unsupported leagues/periods are not inferred from a guessed shorthand.
SPORT_CODE_ALIASES = {
 ('nhl','mon'):'MTL', ('nhl','utah'):'UTA', ('nhl','nj'):'NJD',
 ('nhl','tb'):'TBL', ('nhl','la'):'LAK', ('nhl','sj'):'SJS',
 ('mlb','cws'):'CHW', ('mlb','wsn'):'WSH', ('mlb','sdg'):'SD',
 ('mlb','sfg'):'SF', ('mlb','kan'):'KC', ('mlb','tam'):'TB',
 ('wnba','lva'):'LV', ('wnba','la'):'LAS'
}
SPORTS_AUTO_ENABLED = False


def sports_source(slug):
    """Extract a dated matchup, never confuse an event future/prop with a game."""
    m=re.fullmatch(r'([a-z0-9]+)-([a-z0-9]+)-([a-z0-9]+)-(20\d{2}-\d{2}-\d{2})(?:-(.*))?',
                   slug, re.I)
    if not m:return None
    league,a,b,day,extra=m.groups()
    league=league.lower()
    if league not in SPORT_LEAGUES or a==b:return None
    try:date.fromisoformat(day)
    except ValueError:return None
    return league,a.lower(),b.lower(),day,extra or ''


def sports_aliases(league,code):
    aliases=SPORT_LEAGUES[league][1].get(code, '').split('|')
    return [x for x in [code, *aliases] if x]


def sports_code(league,code):
    return SPORT_CODE_ALIASES.get((league,code),KAL_CODES.get(code,code.upper()))


def sports_outcome_code(label,league,a,b):
    """Only identify unambiguous code/known-name labels (not 'Yes'/'No')."""
    hit=[]
    for code in (a,b):
        if any(norm(label)==norm(alias) for alias in sports_aliases(league,code)):
            hit.append(code)
    return hit[0] if len(hit)==1 else None


def sports_event_date(event):
    m=re.search(r'-(\d{2}[A-Z]{3}\d{2})',str(event).upper())
    if not m:return None
    try:return datetime.strptime(m.group(1),'%y%b%d').date()
    except ValueError:return None


def sports_event_date_ok(event, day):
    d=sports_event_date(event)
    return d is not None and abs((d-date.fromisoformat(day)).days)<=1


def sports_catalog_candidates(series,day,league,a,b,kind='GAME'):
    """Narrow local complete Kalshi index; never use a title alone as a match."""
    dates={(date.fromisoformat(day)+timedelta(days=i)).strftime('%y%b%d').upper()
           for i in (-1,0,1)}
    aa,bb=sports_code(league,a),sports_code(league,b)
    out=[]
    for m in CANDIDATE_CATALOG:
        ticker=str(m.get('ticker','')).upper()
        event=str(m.get('event_ticker','')).upper()
        if not ticker.startswith(series+'-') or not event.startswith(series+'-'):
            continue
        if not any('-'+d in event for d in dates):continue
        # For professional North American teams, identify BOTH clubs in the
        # encoded event ticker; avoid potentially ambiguous display names.
        if SPORT_LEAGUES[league][1]:
            tail=event.split('-',1)[1]
            if aa+bb not in tail and bb+aa not in tail:
                continue
        out.append(m)
    return out


def sports_event_candidates_direct(series,day,league,a,b,kind='GAME'):
    """Small exact API lookup for known pro team codes without a catalog."""
    if not SPORT_LEAGUES[league][1]:return []
    aa,bb=sports_code(league,a),sports_code(league,b)
    result=[]
    for shift in (0,-1,1):
        datecode=(date.fromisoformat(day)+timedelta(days=shift)).strftime('%y%b%d').upper()
        for pair in (aa+bb,bb+aa):
            result.append(series+'-'+datecode+pair)
    return list(dict.fromkeys(result))


def sports_source_question(cid,poly):
    question=str(poly.get('question') or '').strip()
    if not question:
        meta=gamma_market(cid)
        question=str((meta or {}).get('question') or '').strip()
    return question


def sports_outcome_matches_market(label,code,league,market):
    """Kalshi YES must name source winner; never substitute an unrelated NO."""
    yes=str(market.get('yes_sub_title') or market.get('subtitle') or '').strip()
    if not yes:return False
    names=sports_aliases(league,code)
    return (any(norm(yes)==norm(x) for x in names) or
            any(norm(yes)==norm(sports_code(league,code)+' '+x) for x in names[1:]))


def sports_game_rules_ok(m,league):
    text=str(m.get('rules_primary') or '')
    if not text:return False
    # Match/sports outcome, not team props, first periods, series or futures.
    if re.search(r'\b(first|second|third|fourth)\s+(half|period|quarter|set|map)\b', text, re.I):
        return False
    if not re.search(r'\b(wins?|winner|victory|defeats?)\b',text,re.I):
        return False
    return True


# Confirmed full-game ladder series. Separate first-half/period/team-prop
# contracts are NEVER considered equivalent to these whole-game totals.
SPORT_LINE_SERIES = {
    'nhl':('KXNHLSPREAD','KXNHLTOTAL','goals'),
    'mlb':('KXMLBSPREAD','KXMLBTOTAL','runs'),
    'wnba':('KXWNBASPREAD','KXWNBATOTAL','points'),
    'cfb':('KXNCAAFSPREAD','KXNCAAFTOTAL','points'),
    'ncaaf':('KXNCAAFSPREAD','KXNCAAFTOTAL','points'),
    'cbb':('KXNCAAMBSPREAD','KXNCAAMBTOTAL','points'),
    'ncaab':('KXNCAAMBSPREAD','KXNCAAMBTOTAL','points'),
    'ncaamb':('KXNCAAMBSPREAD','KXNCAAMBTOTAL','points'),
}


def sport_name_is(label,league,code,market_name):
    if not market_name:return False
    names=sports_aliases(league,code)
    if len(names)>1:
        return any(norm(market_name)==norm(n) for n in names)
    return norm(label)==norm(market_name)


def sports_line_rule_matches(m,kind,line,unit,side_label=None):
    """Validate direction, exact half-unit threshold, and full-game unit."""
    text=str(m.get('rules_primary') or '')
    if not text:return False
    if re.search(r'\b(first|second|third|fourth)\s+(half|period|quarter|set|map)\b',text,re.I):
        return False
    amount=re.escape(format(line,'.1f'))
    if kind=='SPREAD':
        patt=r'\bwin(?:s)?\s+by\s+(?:more\s+than|over)\s+'+amount+r'\b'
        if not re.search(patt,text,re.I):return False
        # Don't silently map a source team's win-by-X onto the opponent.
        mteam=re.search(r'\bif\s+(.+?)\s+wins?\s+by\b',text,re.I)
        if not mteam or not side_label:return False
        return any(norm(mteam.group(1))==norm(x) for x in side_label)
    patt=r'\b(?:over|more\s+than)\s+(?:a\s+total\s+of\s+)?'+amount+r'\s+(?:total\s+)?'+re.escape(unit)+r'\b'
    return bool(re.search(patt,text,re.I))


def sports_discover_line(poly,token,cid,parts):
    """PAPER-ONLY: strict named line plus Kalshi threshold and side/rules."""
    league,a,b,day,extra=parts
    if league not in SPORT_LINE_SERIES:
        return None,'sports_line_series_needs_reviewed_mapping'
    spread=re.fullmatch(r'spread-(?:home|away)-(\d+)pt5',extra,re.I)
    total=re.fullmatch(r'total-(\d+)pt5',extra,re.I)
    if not spread and not total:
        return None,'sports_special_market_requires_reviewed_mapping'
    kind='SPREAD' if spread else 'TOTAL'
    line=Decimal((spread or total).group(1))+Decimal('.5')
    tokens=poly.get('tokens')
    if not isinstance(tokens,list) or len(tokens)!=2:
        return None,'sports_line_source_nonbinary'
    matches=[t for t in tokens if isinstance(t,dict) and str(t.get('token_id'))==token]
    if len(matches)!=1:return None,'sports_line_source_token_not_found'
    label=str(matches[0].get('outcome') or '')
    question=sports_source_question(cid,poly)
    if not question or not re.search(r'(?<!\d)'+re.escape(str(line))+r'(?!\d)',question):
        return None,'sports_line_source_question_threshold_not_verified'
    spread_series,total_series,unit=SPORT_LINE_SERIES[league]
    series=spread_series if spread else total_series
    wanted_side=None
    team_label=None
    if spread:
        q=re.fullmatch(r'Spread:\s*(.+?)\s*\(([+-])\s*(\d+(?:\.5)?)\)',question,re.I)
        if not q or Decimal(q.group(3))!=line:
            return None,'sports_spread_source_sign_or_team_unverified'
        if SPORT_LEAGUES[league][1]:
            named=None
            for code in (a,b):
                if any(norm(q.group(1))==norm(n) for n in sports_aliases(league,code)):
                    named=code;break
            if named is None:return None,'sports_spread_source_team_ambiguous'
            selected=sports_outcome_code(label,league,a,b)
            if selected is None:return None,'sports_spread_source_outcome_ambiguous'
            winner_side=named if q.group(2)=='-' else (b if named==a else a)
            wanted_side='YES' if selected==winner_side else 'NO'
            team_label=sports_aliases(league,winner_side)
        else:
            # Without an authoritative university abbreviation table, map only
            # the explicit full outcome names printed in the source question.
            names=[str(t.get('outcome') or '').strip() for t in tokens if isinstance(t,dict)]
            if (len(names)!=2 or any(len(norm(n))<5 for n in names) or
                norm(names[0])==norm(names[1])):
                return None,'sports_spread_college_participants_not_verified'
            named_index=[i for i,n in enumerate(names) if norm(n)==norm(q.group(1))]
            if len(named_index)!=1:
                return None,'sports_spread_college_signed_team_not_verified'
            winner_index=named_index[0] if q.group(2)=='-' else 1-named_index[0]
            team_label=[names[winner_index]]
            wanted_side='YES' if norm(label)==norm(team_label[0]) else 'NO'
            winner_side=None
    else:
        if not re.search(r'\b(?:O\s*/\s*U|over|under|total)\b',question,re.I):
            return None,'sports_total_source_question_unverified'
        if norm(label) in ('over','o'):wanted_side='YES'
        elif norm(label) in ('under','u'):wanted_side='NO'
        else:return None,'sports_total_source_outcome_unverified'
    cat=sports_catalog_candidates(series,day,league,a,b,kind)
    if not cat and SPORT_LEAGUES[league][1]:
        code=sports_code(league,winner_side) if spread else None
        suffix=code+str(int(line+Decimal('.5'))) if spread else str(int(line+Decimal('.5')))
        for event in sports_event_candidates_direct(series,day,league,a,b):
            cat.append({'ticker':event+'-'+suffix,'event_ticker':event})
    if not cat:return None,'sports_line_no_catalog_candidate'
    trimmed=[]
    wanted_suffix=(sports_code(league,winner_side)+str(int(line+Decimal('.5')))) if spread and SPORT_LEAGUES[league][1] else str(int(line+Decimal('.5'))) if total else None
    for m in cat:
        t=str(m.get('ticker') or '').upper()
        if wanted_suffix and t.rsplit('-',1)[-1]!=wanted_suffix:continue
        if not SPORT_LEAGUES[league][1]:
            # College school aliases differ between venues. For TEAM spreads
            # require both full source outcomes in the Kalshi event title.
            # For totals, both encoded source team codes must exactly match.
            names=[str(tk.get('outcome') or '').strip() for tk in tokens if isinstance(tk,dict)]
            preview=' '.join(str(m.get(k) or '') for k in ('title','subtitle'))
            if kind=='SPREAD':
                if not all(len(norm(n))>4 and norm(n) in norm(preview) for n in names):continue
            else:
                ev=str(m.get('event_ticker') or '').upper().split('-',1)[-1]
                if min(len(a),len(b))<3 or (a.upper()+b.upper() not in ev and b.upper()+a.upper() not in ev):
                    continue
        trimmed.append(t)
    trimmed=list(dict.fromkeys(trimmed))
    if len(trimmed)>15:return None,f'sports_line_candidates_too_many_{len(trimmed)}'
    hits=[]
    for t in trimmed:
        m=market_get(t)
        if not open_market(m,t) or not str(m.get('event_ticker') or '').startswith(series+'-'):
            continue
        if not sports_event_date_ok(m['event_ticker'],day):continue
        if SPORT_LEAGUES[league][1] or kind=='TOTAL':
            tail=m['event_ticker'].split('-',1)[1]
            x,y=sports_code(league,a),sports_code(league,b)
            if x+y not in tail and y+x not in tail:continue
        if not sports_line_rule_matches(m,kind,line,unit,team_label):continue
        hits.append(t)
    if len(hits)!=1:return None,f'sports_line_verified_candidates_{len(hits)}'
    return (hits[0],str(poly.get('market_slug') or poly.get('slug') or ''),wanted_side), 'sports_full_game_line_paper_only_exceptional_rule_review_needed'


def sports_discover_winner(poly,token,cid,parts):
    """Conservative automatic GAME/MATCH/FIGHT winner for new sports in PAPER only.

    An exact two-sided dated event, outcome label, series, event participants,
    Kalshi outcome and win rule must all verify. For college/tennis/fights,
    participants must be present in full in both venue questions/titles.
    """
    league,a,b,day,extra=parts
    if extra:return sports_discover_line(poly,token,cid,parts)
    series,known=SPORT_LEAGUES[league]
    if league in ('nba','nfl'):
        return None,'existing_nba_nfl_matcher'
    tokens=poly.get('tokens')
    if not isinstance(tokens,list) or len(tokens)!=2:
        return None,'sports_nonbinary_or_missing_tokens'
    found=[r for r in tokens if isinstance(r,dict) and str(r.get('token_id'))==token]
    if len(found)!=1:return None,'sports_token_unverified'
    label=str(found[0].get('outcome') or '')
    selected=sports_outcome_code(label,league,a,b)
    question=sports_source_question(cid,poly)
    # Unknown team aliases: require full athlete/team labels from SOURCE outcomes,
    # and exact participant names embedded in both source and destination text.
    if not known:
        names=[str(t.get('outcome') or '').strip() for t in tokens if isinstance(t,dict)]
        if (len(names)!=2 or any(len(norm(n))<5 for n in names) or
            norm(names[0])==norm(names[1]) or not question or
            not all(norm(n) in norm(question) for n in names)):
            return None,'sports_participants_need_verified_names'
    elif selected is None:
        return None,'sports_source_winner_label_unknown'
    catalog=sports_catalog_candidates(series,day,league,a,b)
    # Index is advisory and can be stale; every shortlisted match refetched.
    if not catalog and known:
        # Avoid enumerating unknown team outcomes; fetch known team's market only.
        if selected is None:return None,'sports_source_outcome_unverified'
        suffix=sports_code(league,selected)
        for event in sports_event_candidates_direct(series,day,league,a,b):
            catalog.append({'ticker':event+'-'+suffix,'event_ticker':event})
    if not catalog:
        return None,'sports_no_catalog_candidate'
    hits=[]
    # Limit network scans per signal; broad unresolved searches do not block
    # a latency-sensitive wallet poll indefinitely.
    narrowed=[]
    for m in catalog:
        ticker=str(m.get('ticker') or '').upper()
        if known:
            if ticker.rsplit('-',1)[-1]!=sports_code(league,selected):continue
        else:
            preview=' '.join(str(m.get(x) or '') for x in ('title','subtitle','yes_sub_title'))
            if not all(norm(n) in norm(preview) for n in names):continue
            if norm(str(m.get('yes_sub_title') or ''))!=norm(label):continue
        narrowed.append(ticker)
    narrowed=list(dict.fromkeys(narrowed))
    if len(narrowed)>12:return None,f'sports_ambiguous_catalog_{len(narrowed)}'
    for ticker in narrowed:
        m=market_get(ticker)
        if not open_market(m,ticker):continue
        if not str(m.get('event_ticker') or '').startswith(series+'-'):continue
        if not sports_event_date_ok(m.get('event_ticker'),day):continue
        if not sports_game_rules_ok(m,league):continue
        if known:
            ev=m['event_ticker'].split('-',1)[1]
            left,right=sports_code(league,a),sports_code(league,b)
            if left+right not in ev and right+left not in ev:continue
            if not sports_outcome_matches_market(label,selected,league,m):continue
        else:
            text=' '.join(str(m.get(x) or '') for x in ('title','subtitle','rules_primary'))
            if not all(norm(n) in norm(text) for n in names):continue
            if norm(str(m.get('yes_sub_title') or ''))!=norm(label):continue
        hits.append(ticker)
    if len(hits)!=1:return None,f'sports_verified_winner_candidates_{len(hits)}'
    return (hits[0],str(poly.get('market_slug') or poly.get('slug') or ''),'YES'), 'sports_game_winner_paper_only_exceptional_rule_review_needed'



# External approved mappings are loaded once, and their SHA-256 is bound to the DB.
# They can cover ANY exchange market, not only NBA/NFL.
APPROVED_MATCHES = {}
CANDIDATE_CATALOG = []
MAPPING_FILE_SHA256 = 'none'





HEADERS = {'Accept':'application/json','User-Agent':'breakthebank-kalshi/1.0'}





def clock(): return time.time()

def utc(t): return datetime.fromtimestamp(t,timezone.utc).isoformat(timespec='seconds')

def dec(x):

    v=Decimal(str(x))

    if not v.is_finite(): raise ValueError('nonfinite amount')

    return v



def http_json(url, *, method='GET', body=None, headers=None, timeout=15):

    h=dict(HEADERS)

    if headers: h.update(headers)

    if body is not None: h['Content-Type']='application/json'

    raw=json.dumps(body,separators=(',',':')).encode() if body is not None else None

    req=Request(url,data=raw,headers=h,method=method)

    with urlopen(req,timeout=timeout) as r:

        result=json.load(r)

    if not isinstance(result,dict): raise ValueError('Expected JSON object')

    return result





def get_poly_trades(taker, start_ts):

    """Read EVERY public wallet fill in a short recent block-time window.



    A full page does NOT prove feed saturation. The Data API v2 uses

    keyset pagination; follow its opaque next_cursor until exhausted.

    Fail closed if a cursor is missing or pagination becomes excessive.

    """

    params = {

        'user': WALLET,

        'limit': 500,

        'taker_only': str(taker).lower(),

        'start': int(start_ts),

    }

    rows = []

    cursor = None

    seen_cursors = set()

    for page in range(1, 51):

        query = dict(params)

        if cursor is not None:

            query['cursor'] = cursor

        res = http_json(POLY_DATA + '?' + urlencode(query))

        batch = res.get('data')

        pagination = res.get('pagination')

        if not isinstance(batch, list) or not isinstance(pagination, dict):

            raise RuntimeError('Malformed Polymarket paginated trade feed')

        for r in batch:

            if not isinstance(r, dict) or str(r.get('proxy_wallet', '')).lower() != WALLET:

                raise RuntimeError('Wallet mismatch in public data feed')

            if float(r.get('timestamp', 0)) < int(start_ts):

                raise RuntimeError('Polymarket returned a trade before requested start')

        rows.extend(batch)

        next_cursor = pagination.get('next_cursor')

        has_more = pagination.get('has_more')

        if not next_cursor:

            if has_more:

                raise RuntimeError('Incomplete wallet trade feed: has_more but no cursor')

            return rows

        if not isinstance(next_cursor, str) or next_cursor in seen_cursors:

            raise RuntimeError('Invalid or repeating wallet feed cursor')

        seen_cursors.add(next_cursor)

        cursor = next_cursor

    raise RuntimeError('Wallet feed exceeded 50 pages within recent window; fail closed')







def first_taker_is_first(cid, incoming):

    """Full-history existence test: was this the wallet's first taker SECOND?



    Query for any older wallet taker in this condition, NOT all takers

    globally and NOT the first market trade. Same-second fills remain

    ambiguous and are never copied because ingest requires trigger+1s.

    """

    ts=int(float(incoming['timestamp']))

    if ts<=1:raise RuntimeError('Invalid source taker timestamp')

    params={'user':WALLET,'condition':cid,'limit':1,

            'taker_only':'true','start':1,'end':ts-1}

    res=http_json(POLY_DATA+'?'+urlencode(params))

    rows=res.get('data')

    pg=res.get('pagination')

    if not isinstance(rows,list) or not isinstance(pg,dict):

        raise RuntimeError('Cannot verify historical first taker; fail closed')

    for r in rows:

        if not isinstance(r,dict) or str(r.get('proxy_wallet','')).lower()!=WALLET:

            raise RuntimeError('Invalid taker-history wallet')

        if str(r.get('condition_id','')).lower()!=cid or float(r.get('timestamp',0))>=ts:

            raise RuntimeError('Unexpected taker-history market or timestamp')

    if rows:return False

    if pg.get('has_more') or pg.get('next_cursor'):

        raise RuntimeError('Ambiguous prior-taker pagination; fail closed')

    return True





def fingerprint(r):

    fields=('transaction_hash','condition_id','token_id','side','timestamp','price','size')

    return hashlib.sha256(json.dumps([str(r.get(k,'')) for k in fields],separators=(',',':')).encode()).hexdigest()





def register(conn, feed, rows, received):

    # Preserve multiplicity of otherwise-identical on-chain fills.

    seen=Counter(); added=[]

    for r in sorted(rows,key=lambda x:(float(x['timestamp']),fingerprint(x))):

        fp=fingerprint(r);occ=seen[fp];seen[fp]+=1

        ts=float(r['timestamp'])

        if ts>received+3 or ts<0: raise ValueError('Wallet timestamp invalid')

        cur=conn.execute('INSERT OR IGNORE INTO seen(feed,fp,occ,source_ts,first_seen) VALUES (?,?,?,?,?)',

                         (feed,fp,occ,ts,received))

        if cur.rowcount: added.append((r,received,fp,occ))

    return added





def connect_db(path):

    path.parent.mkdir(parents=True,exist_ok=True)

    c=sqlite3.connect(path)

    c.execute('PRAGMA busy_timeout=5000')

    c.execute('PRAGMA journal_mode=WAL')

    c.executescript('''

    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);

    CREATE TABLE IF NOT EXISTS seen(feed TEXT,fp TEXT,occ INTEGER,source_ts REAL,first_seen REAL,

        PRIMARY KEY(feed,fp,occ));

    CREATE TABLE IF NOT EXISTS blocked(cid TEXT PRIMARY KEY);
    CREATE TABLE IF NOT EXISTS signal_audit(id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_ts REAL, seen_ts REAL, cid TEXT, token TEXT, slug TEXT, side TEXT,
        source_qty REAL, source_price REAL, decision TEXT, detail TEXT);
    CREATE INDEX IF NOT EXISTS idx_signal_audit_decision ON signal_audit(decision);


    CREATE TABLE IF NOT EXISTS activated(cid TEXT PRIMARY KEY,source_ts REAL,seen_ts REAL,slug TEXT);

    CREATE TABLE IF NOT EXISTS mapping(token TEXT PRIMARY KEY,cid TEXT,ticker TEXT,slug TEXT,kalshi_side TEXT,created_ts REAL,review_sha TEXT);

    CREATE TABLE IF NOT EXISTS pending(token TEXT PRIMARY KEY,cid TEXT,side TEXT,qty REAL,ref REAL,

        first_ts REAL,last_seen REAL,nfills INTEGER);

    CREATE TABLE IF NOT EXISTS positions(ticker TEXT PRIMARY KEY,qty REAL NOT NULL,cost REAL NOT NULL);

    CREATE TABLE IF NOT EXISTS source_inventory(token TEXT PRIMARY KEY,qty TEXT NOT NULL);

    CREATE TABLE IF NOT EXISTS dust(token TEXT PRIMARY KEY,usd TEXT NOT NULL);

    CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY AUTOINCREMENT,created_ts REAL,mode TEXT,

        cid TEXT,ticker TEXT,side TEXT,requested REAL,filled REAL,limit_px REAL,fill_px REAL,

        fee_est REAL,status TEXT,client_id TEXT UNIQUE,exchange_id TEXT,note TEXT);

    CREATE TABLE IF NOT EXISTS errors(ts REAL,message TEXT);

    CREATE TABLE IF NOT EXISTS settlements(ticker TEXT PRIMARY KEY, ts REAL, qty REAL, cost REAL, payout REAL, pnl REAL);

    ''')

    # Migration supports older databases for READ operations, but main blocks
    # version mixing if a copier has already bootstrapped under another version.
    columns={r[1] for r in c.execute('PRAGMA table_info(mapping)')}
    if 'review_sha' not in columns:c.execute('ALTER TABLE mapping ADD COLUMN review_sha TEXT')
    c.commit(); return c





def norm(s): return re.sub('[^a-z0-9]','',str(s).lower())

def outcome_team(label, league, a, b):

    catalog=TEAMS[league]

    found=[]

    for code in (a,b):

        if any(norm(label)==norm(z) for z in [code]+catalog[code].split('|')):

            found.append(code)

    return found[0] if len(found)==1 else None





def candidate_tickers(a,b,day,series,suffix):

    dt=date.fromisoformat(day)

    ac=KAL_CODES.get(a,a.upper());bc=KAL_CODES.get(b,b.upper())

    for shift in (0,-1,1):

        datecode=(dt+timedelta(days=shift)).strftime('%y%b%d').upper()

        for pair in (ac+bc,bc+ac):

            ev=f'{series}-{datecode}{pair}'

            yield ev, f'{ev}-{suffix}'





def market_get(ticker):

    try:

        return http_json(KALSHI+'/markets/'+quote(ticker,safe='')).get('market')

    except HTTPError as e:

        if e.code==404:return None

        raise





def open_market(m,ticker,event=None):

    return (isinstance(m,dict) and m.get('ticker')==ticker and

            (event is None or m.get('event_ticker')==event) and

            str(m.get('status','')).lower() in ('active','open') and

            str(m.get('result') or '').lower() not in ('yes','no'))





def gamma_market(cid):

    # Public Gamma is needed for spread/total question and signed handicap.

    url=POLY_GAMMA+'/markets?'+urlencode({'condition_ids':cid,'limit':5})

    req=Request(url,headers=HEADERS)

    with urlopen(req,timeout=15) as r: data=json.load(r)

    if not isinstance(data,list):raise RuntimeError('Malformed Gamma markets')

    valid=[x for x in data if isinstance(x,dict) and str(x.get('conditionId','')).lower()==cid]

    return valid[0] if len(valid)==1 else None





def rule_digest(m):
    """Hash settlement-critical published Kalshi terms for review/recheck."""
    fields=('ticker','event_ticker','title','subtitle','yes_sub_title','no_sub_title',
            'rules_primary','rules_secondary','early_close_condition',
            'settlement_bounds_type','strike_type','floor_strike','cap_strike',
            'functional_strike','expiration_value','can_close_early')
    raw=json.dumps({k:m.get(k) for k in fields},sort_keys=True,separators=(',',':'),ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def approved_entry_digest(entry):
    return hashlib.sha256(json.dumps(entry,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def load_approved_matches(path):
    """Load audited, explicit condition+token -> ticker/outcome links.

    The mapping may belong to ANY category. Each reviewed outcome is distinct.
    Rules hashes prevent stale or accidentally reassigned tickers from trading.
    """
    global MAPPING_FILE_SHA256
    if path is None:
        MAPPING_FILE_SHA256='none';return {}
    raw=path.expanduser().read_bytes()
    MAPPING_FILE_SHA256=hashlib.sha256(raw).hexdigest()
    content=json.loads(raw)
    if not isinstance(content,dict) or content.get('version')!=1 or not isinstance(content.get('matches'),list):
        raise ValueError('Mapping file must be {"version":1,"matches":[...]}')
    approved={}
    for index,row in enumerate(content['matches']):
        if not isinstance(row,dict):raise ValueError(f'Mapping #{index} is not an object')
        cid=str(row.get('condition_id','')).lower()
        token=str(row.get('token_id',''))
        ticker=str(row.get('kalshi_ticker',''))
        kout=str(row.get('kalshi_outcome','')).upper()
        outcome=str(row.get('poly_outcome',''))
        question=str(row.get('poly_question',''))
        digest=str(row.get('kalshi_rules_sha256','')).lower()
        if (not re.fullmatch(r'0x[0-9a-f]{64}',cid) or not token.isdigit() or
            not re.fullmatch(r'[A-Z0-9-]+',ticker) or kout not in ('YES','NO') or
            not outcome.strip() or not question.strip() or
            not re.fullmatch(r'[0-9a-f]{64}',digest) or row.get('reviewed') is not True):
            raise ValueError(f'Mapping #{index} incomplete/unreviewed. No order allowed.')
        if (cid,token) in approved:raise ValueError(f'Duplicate mapping for {(cid,token)}')
        approved[(cid,token)]=dict(row, condition_id=cid,token_id=token,
                                   kalshi_ticker=ticker,kalshi_outcome=kout,
                                   kalshi_rules_sha256=digest)
    return approved


def discover_approved(token,cid, poly=None):
    """Verify BOTH venue IDs, actual token outcome, source question and Kalshi rules."""
    entry=APPROVED_MATCHES.get((cid,token))
    if entry is None:return None,None
    if poly is None:poly=http_json(POLY_CLOB+'/markets/'+quote(cid,safe=''))
    if str(poly.get('condition_id','')).lower()!=cid:
        raise RuntimeError('Mapped Polymarket condition mismatch')
    tokens=poly.get('tokens')
    if not isinstance(tokens,list):return None,'mapped_poly_tokens_missing'
    matches=[t for t in tokens if isinstance(t,dict) and str(t.get('token_id'))==token]
    if len(matches)!=1:return None,'mapped_poly_token_unverified'
    if norm(matches[0].get('outcome',''))!=norm(entry['poly_outcome']):
        return None,'mapped_poly_outcome_changed'
    meta=gamma_market(cid)
    if not meta or str(meta.get('question','')).strip()!=entry['poly_question'].strip():
        return None,'mapped_poly_question_changed_or_unavailable'
    market=market_get(entry['kalshi_ticker'])
    if not open_market(market,entry['kalshi_ticker'],None):
        return None,'mapped_kalshi_market_not_open'
    if rule_digest(market)!=entry['kalshi_rules_sha256']:
        return None,'mapped_kalshi_settlement_rules_changed'
    return (entry['kalshi_ticker'],str(poly.get('market_slug') or poly.get('slug') or '').lower(),
            entry['kalshi_outcome']),'verified_reviewed_crossvenue_mapping'


def catalog_load(path):
    if path is None:return []
    content=json.loads(path.expanduser().read_text())
    markets=content.get('markets') if isinstance(content,dict) else content
    if not isinstance(markets,list):raise ValueError('Bad catalog: expected list of markets')
    # Catalog is ADVISORY. It can never cause an order by itself.
    return [m for m in markets if isinstance(m,dict) and
            str(m.get('status','')).lower() in ('active','open') and m.get('ticker')]


def proposed_candidates(question,limit=3):
    """Untrusted TITLE-ONLY suggestions for manual review, never executable."""
    if not question or not CANDIDATE_CATALOG:return []
    canon=lambda x: re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]',' ',str(x).lower())).strip()
    q=canon(question)
    if len(q)<15:return []
    qwords={w for w in q.split() if len(w)>2}
    if not qwords:return []
    ranked=[]
    for m in CANDIDATE_CATALOG:
        text=' '.join(str(m.get(k) or '') for k in ('title','subtitle','yes_sub_title'))
        toks={w for w in canon(text).split() if len(w)>2}
        overlap=len(qwords&toks)
        if overlap<max(2,(len(qwords)+2)//3):continue
        score=SequenceMatcher(None,q,canon(text)).ratio()
        if score>=.40:
            ranked.append((score,str(m['ticker'])))
    return [ticker for score,ticker in sorted(ranked,reverse=True)[:limit]]


def sync_kalshi_catalog(path, max_pages=5000, mve_filter='exclude'):
    """Checkpointed, read-only Kalshi market index for non-executable suggestions.

    The live copier never consumes this index as a verified market mapping.
    Persist an SQLite staging cursor after EVERY successful page so an HTTP
    outage or process interruption does not discard hours of pagination.

    Default excludes multivariate combo/parlay markets, but NOT any ordinary
    market category. Use --catalog-mve all to index combos too (much bigger).
    """
    if mve_filter not in ('exclude', 'only', 'all'):
        raise ValueError('Invalid catalog MVE filter')
    if max_pages < 1:
        raise ValueError('catalog-max-pages must be positive')
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(str(path) + '.sync.sqlite3')
    config = json.dumps({'schema': 1, 'status': 'open', 'limit': 1000,
                         'mve_filter': mve_filter}, sort_keys=True)
    db = sqlite3.connect(stage)
    try:
        db.execute('PRAGMA busy_timeout=5000')
        db.executescript("""
        CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS markets (ticker TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cursors (cursor TEXT PRIMARY KEY);
        """)
        existing = db.execute("SELECT value FROM state WHERE key='config'").fetchone()
        if existing and existing[0] != config:
            raise RuntimeError('Catalog staging parameters have changed. Resume with the original '
                               '--catalog-mve setting, or remove '+str(stage)+' to start over.')
        if not existing:
            with db:
                db.execute("INSERT INTO state VALUES ('config', ?)", (config,))
                db.execute("INSERT INTO state VALUES ('pages', '0')")
                db.execute("INSERT INTO state VALUES ('next_cursor', '')")
        pages = int(db.execute("SELECT value FROM state WHERE key='pages'").fetchone()[0])
        cursor = db.execute("SELECT value FROM state WHERE key='next_cursor'").fetchone()[0]
        rows_so_far = db.execute('SELECT COUNT(*) FROM markets').fetchone()[0]
        if pages:
            print(f'CATALOG RESUMING: pages={pages} markets={rows_so_far} '
                  f'mve={mve_filter}', flush=True)
        else:
            print(f'CATALOG START: all categories, status=open, '
                  f'mve={mve_filter}, max_pages={max_pages}', flush=True)
        while True:
            if pages >= max_pages:
                raise RuntimeError(f'Catalog still has more pages after {pages}; '
                                   f'{rows_so_far} unique markets checkpointed at {stage}. '
                                   'Rerun with --catalog-max-pages higher to resume.')
            params = {'status': 'open', 'limit': 1000}
            if mve_filter != 'all':
                params['mve_filter'] = mve_filter
            if cursor:
                params['cursor'] = cursor
            res = http_json(KALSHI+'/markets?'+urlencode(params), timeout=30)
            batch = res.get('markets')
            next_cursor = res.get('cursor') or ''
            if not isinstance(batch, list) or not isinstance(next_cursor, str):
                raise RuntimeError('Malformed Kalshi catalog response; checkpoint preserved')
            # Kalshi documents that an undecodable cursor can silently restart at
            # page 1. Detect any repeated cursor, not just the immediately prior one.
            if next_cursor and db.execute('SELECT 1 FROM cursors WHERE cursor=?',
                                          (next_cursor,)).fetchone():
                raise RuntimeError('Kalshi catalog cursor loop detected; checkpoint preserved')
            reduced = []
            for market in batch:
                if not isinstance(market, dict) or not isinstance(market.get('ticker'), str):
                    raise RuntimeError('Invalid Kalshi catalog market; checkpoint preserved')
                # This file is only an ADVISORY title index; full settlement rules
                # are re-fetched from Kalshi for any reviewed executable mapping.
                item = {k: market[k] for k in ('ticker', 'event_ticker', 'title',
                            'subtitle', 'yes_sub_title', 'no_sub_title', 'status',
                            'close_time') if k in market}
                reduced.append((market['ticker'], json.dumps(item, separators=(',', ':'))))
            with db:
                db.executemany('INSERT OR IGNORE INTO markets (ticker,payload) VALUES (?,?)', reduced)
                if next_cursor:
                    db.execute('INSERT INTO cursors(cursor) VALUES (?)', (next_cursor,))
                pages += 1
                db.execute("UPDATE state SET value=? WHERE key='pages'", (str(pages),))
                db.execute("UPDATE state SET value=? WHERE key='next_cursor'", (next_cursor,))
            cursor = next_cursor
            if pages % 10 == 0 or not cursor:
                rows_so_far = db.execute('SELECT COUNT(*) FROM markets').fetchone()[0]
                print(f'CATALOG PROGRESS: pages={pages}, markets={rows_so_far}, '
                      f'has_more={bool(cursor)}', flush=True)
            if not cursor:
                break
        # Build a completed, valid JSON file and publish atomically. Never load
        # all pages into RAM, never expose an incomplete cache to the copier.
        tmp = Path(str(path)+'.tmp')
        with tmp.open('w', encoding='utf-8') as handle:
            handle.write('{"updated_utc":')
            handle.write(json.dumps(utc(clock())))
            handle.write(',"mve_filter":')
            handle.write(json.dumps(mve_filter))
            handle.write(',"markets":[')
            first = True
            for (payload,) in db.execute('SELECT payload FROM markets ORDER BY ticker'):
                if not first: handle.write(',')
                handle.write(payload)
                first = False
            handle.write(']}\n')
        tmp.replace(path)
        rows_so_far = db.execute('SELECT COUNT(*) FROM markets').fetchone()[0]
        print(f'CATALOG SAVED: {rows_so_far} open markets / {pages} pages -> {path}', flush=True)
    finally:
        db.close()
    stage.unlink(missing_ok=True)



def sports_catalog_request(url, retries=7):
    """Catalog-only throttling. Never add sleeps/retries to the live fill path.

    On HTTP 429, honor Retry-After when numeric, otherwise use bounded
    exponential backoff. A terminal failure keeps all SQLite checkpoints.
    """
    from urllib.error import HTTPError
    for attempt in range(retries + 1):
        try:
            return http_json(url, timeout=30)
        except HTTPError as exc:
            if exc.code != 429 or attempt == retries:
                raise
            raw = exc.headers.get('Retry-After') if exc.headers else None
            try:
                advised = float(raw)
                if not math.isfinite(advised) or advised < 0:
                    raise ValueError('invalid Retry-After')
            except (TypeError, ValueError):
                advised = 0
            seconds = min(120.0, max(advised, min(60.0, 3.0 * (2 ** attempt))))
            print(f'SPORTS CATALOG RATE LIMITED (HTTP 429): '
                  f'retrying page after {seconds:g}s; '
                  f'attempt {attempt+1}/{retries}', flush=True)
            time.sleep(seconds)
    raise AssertionError('unreachable: exhausted retries must raise')


def sync_sports_catalog(path, delay=1.0, retries=7):
    """Resumeable, targeted PUBLIC sports-market index, not full-exchange crawl.

    Only full game winner, spread and total, plus match/fight series our sport
    matcher understands. The resulting cache is *advisory*; trading always
    refetches and independently checks a proposed Kalshi contract.
    """
    if not math.isfinite(delay) or delay < 0 or delay > 30 or not 0 <= retries <= 15:
        raise ValueError('Sports catalog delay must be 0..30 seconds; retries 0..15')
    path=Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    series=sorted(set(v[0] for v in SPORT_LEAGUES.values()) |
                  set(x for v in SPORT_LINE_SERIES.values() for x in v[:2]))
    stage=Path(str(path)+'.sync.sqlite3')
    db=sqlite3.connect(stage)
    try:
        db.execute('PRAGMA busy_timeout=5000')
        db.executescript("""CREATE TABLE IF NOT EXISTS markets(ticker TEXT PRIMARY KEY, payload TEXT NOT NULL);
                          CREATE TABLE IF NOT EXISTS completed(series TEXT PRIMARY KEY);
                          CREATE TABLE IF NOT EXISTS progress(series TEXT PRIMARY KEY,cursor TEXT NOT NULL, pages INTEGER NOT NULL);
                          CREATE TABLE IF NOT EXISTS cursor_seen(series TEXT, cursor TEXT, PRIMARY KEY(series,cursor));""")
        for ser in series:
            if db.execute('SELECT 1 FROM completed WHERE series=?',(ser,)).fetchone():continue
            row=db.execute('SELECT cursor,pages FROM progress WHERE series=?',(ser,)).fetchone()
            cursor,pages=(row[0],row[1]) if row else ('',0)
            print(f'SPORTS CATALOG {ser}: resume_pages={pages}',flush=True)
            while True:
                if pages>=500:
                    raise RuntimeError(f'Sports series {ser} exceeded 500 pages, state saved at {stage}')
                params={'series_ticker':ser,'status':'open','limit':1000,'mve_filter':'exclude'}
                if cursor:params['cursor']=cursor
                response=sports_catalog_request(KALSHI+'/markets?'+urlencode(params),retries=retries)
                markets=response.get('markets');nxt=response.get('cursor') or ''
                if not isinstance(markets,list) or not isinstance(nxt,str):
                    raise RuntimeError('Invalid sports series response, progress preserved')
                if nxt and db.execute('SELECT 1 FROM cursor_seen WHERE series=? AND cursor=?',(ser,nxt)).fetchone():
                    raise RuntimeError('Sports cursor loop: progress preserved')
                saved=[]
                for market in markets:
                    if not isinstance(market,dict) or not isinstance(market.get('ticker'),str):
                        raise RuntimeError('Malformed sports market payload')
                    item={k:market[k] for k in ('ticker','event_ticker','title','subtitle',
                           'yes_sub_title','no_sub_title','status','close_time') if k in market}
                    saved.append((market['ticker'],json.dumps(item,separators=(',',':'))))
                with db:
                    db.executemany('INSERT OR IGNORE INTO markets VALUES (?,?)',saved)
                    if nxt:
                        db.execute('INSERT INTO cursor_seen VALUES (?,?)',(ser,nxt))
                    pages+=1
                    db.execute('INSERT INTO progress(series,cursor,pages) VALUES (?,?,?) '
                               'ON CONFLICT(series) DO UPDATE SET cursor=excluded.cursor,pages=excluded.pages',
                               (ser,nxt,pages))
                    if not nxt:
                        db.execute('INSERT OR IGNORE INTO completed VALUES (?)',(ser,))
                cursor=nxt
                if pages%10==0 or not cursor:
                    print(f'  {ser}: pages={pages}, batch={len(markets)}, next={bool(cursor)}',flush=True)
                # Gentle pacing across all series, not just multi-page markets.
                # The successful page was committed before sleeping, so Ctrl+C
                # and HTTP errors still leave a resumable checkpoint.
                if delay:time.sleep(delay)
                if not cursor:break
        tmp=Path(str(path)+'.tmp')
        with tmp.open('w',encoding='utf-8') as f:
            f.write('{"updated_utc":'+json.dumps(utc(clock()))+',"sports_only":true,"markets":[')
            first=True
            for (raw,) in db.execute('SELECT payload FROM markets ORDER BY ticker'):
                if not first:f.write(',')
                first=False;f.write(raw)
            f.write(']}\n')
        tmp.replace(path)
        count=db.execute('SELECT COUNT(*) FROM markets').fetchone()[0]
        print(f'SPORTS CATALOG COMPLETE: {count} markets in {len(series)} series -> {path}',flush=True)
    finally:db.close()
    stage.unlink(missing_ok=True)


def inspect_pair(cid,ticker):
    """Print review material to create manually approved cross-category mapping."""
    cid=cid.lower();ticker=ticker.upper()
    poly=http_json(POLY_CLOB+'/markets/'+quote(cid,safe=''))
    meta=gamma_market(cid)
    market=market_get(ticker)
    if not isinstance(market,dict) or not meta:
        raise RuntimeError('Could not load both markets for review')
    if str(poly.get('condition_id','')).lower()!=cid:
        raise RuntimeError('Source condition mismatch')
    print('SOURCE POLYMARKET:')
    print(json.dumps({'condition_id':cid,'question':meta.get('question'),
          'description':meta.get('description'),'resolutionSource':meta.get('resolutionSource'),
          'tokens':poly.get('tokens')},indent=2))
    print('DESTINATION KALSHI:')
    print(json.dumps({k:market.get(k) for k in ('ticker','event_ticker','title',
          'subtitle','yes_sub_title','no_sub_title','rules_primary','rules_secondary',
          'status','result','can_close_early','early_close_condition','settlement_bounds_type')},indent=2))
    print('KALSHI RULE DIGEST:',rule_digest(market))
    print('IMPORTANT: Similar text is NOT proof of settlement equivalence. Review both')
    print('resolution sources, edge cases, timing, extraordinary event/tie/void policies.')
    print('ONLY manually add an entry to the map file with reviewed=true if equivalent.')


def rule_has_line(text, kind, line):

    text=' '.join(str(text or '').split())

    if kind=='SPREAD':

        matches=re.findall(r'win(?:s)?\s+by\s+(?:more\s+than|over)\s+(\d+(?:\.5)?)',text,re.I)

    else:

        matches=re.findall(r'(?:over|more\s+than)\s+(\d+(?:\.5)?)\s+(?:total\s+)?points',text,re.I)

    return any(abs(float(x)-float(line))<1e-8 for x in matches)





def discover(token,cid):

    # Universal path: explicitly audited token -> market/outcome mappings, any category.
    # CLOB + Gamma + Kalshi identities/rules are checked before trusting a mapping.
    if (cid,token) in APPROVED_MATCHES:
        return discover_approved(token,cid)

    poly=http_json(POLY_CLOB+'/markets/'+quote(cid,safe=''))

    if str(poly.get('condition_id','')).lower()!=cid:

        raise RuntimeError('Polymarket condition verification failed')

    slug=str(poly.get('market_slug') or poly.get('slug') or '').lower()

    # Every supported sport is searched; only high-confidence game-winner
    # pairs auto-queue in PAPER mode. Remaining sports still get suggestions
    # and can be traded via explicitly reviewed per-token mappings.
    parts=sports_source(slug)
    if parts is not None and SPORTS_AUTO_ENABLED:
        if parts[0] not in ('nba','nfl'):
            mapped,reason=sports_discover_winner(poly,token,cid,parts)
            if mapped:return mapped,reason
            # Do not misclassify a recognized sports source as a non-sport.
            return None,reason

    z=SLUG.fullmatch(slug)

    if not z:
        if parts is not None and not SPORTS_AUTO_ENABLED:
            return None,'additional_sports_require_reviewed_mapping_for_live'
        question=str(poly.get('question') or '')
        if not question and CANDIDATE_CATALOG:
            meta=gamma_market(cid)
            question=str((meta or {}).get('question') or '')
        suggested=proposed_candidates(question)
        if suggested:
            print(f'UNREVIEWED KALSHI CANDIDATES for {slug}: {suggested}; NOT TRADED',flush=True)
        return None,('unreviewed_cross_category_mapping:' + ','.join(suggested)) if suggested else 'unmapped_needs_reviewed_cross_category_mapping'

    league,a,b,day,extra=z.groups()

    if a==b or a not in TEAMS[league] or b not in TEAMS[league]:

        return None,'unknown_team'

    tokens=poly.get('tokens')

    if not isinstance(tokens,list) or len(tokens)!=2:

        return None,'nonbinary_polymarket'

    labels={str(t.get('token_id')):str(t.get('outcome','')) for t in tokens if isinstance(t,dict)}

    if token not in labels:return None,'missing_outcome_token'

    label=labels[token]

    sport=league.upper()

    kind='GAME'; yesno='YES'; side_team=None; line=None

    if not extra:

        side_team=outcome_team(label,league,a,b)

        if side_team is None:return None,'unknown_winner_outcome'

        suffix=KAL_CODES.get(side_team,side_team.upper())

    elif (sm:=re.fullmatch(r'spread-(home|away)-(\d+)pt5',extra)):

        kind='SPREAD';line=int(sm.group(2))+0.5

        meta=gamma_market(cid)

        question=str(meta.get('question') or '') if meta else str(poly.get('question') or '')

        q=re.fullmatch(r'Spread:\s*(.+?)\s*\(([+-])\s*(\d+(?:\.5)?)\)',question,re.I)

        if not q or abs(float(q.group(3))-line)>1e-8:

            return None,'spread_source_question_unverified'

        named=outcome_team(q.group(1),league,a,b)

        chosen=outcome_team(label,league,a,b)

        if not named or not chosen:return None,'spread_outcome_team_unverified'

        # Negative handicap => named team must win by more than line;

        # positive handicap => opponent must NOT win by more than line.

        side_team=named if q.group(2)=='-' else (b if named==a else a)

        yesno='YES' if chosen==side_team else 'NO'

        suffix=KAL_CODES.get(side_team,side_team.upper())+str(int(line+0.5))

    elif (sm:=re.fullmatch(r'total-(\d+)pt5',extra)):

        kind='TOTAL';line=int(sm.group(1))+0.5

        if norm(label) in ('over','o'):yesno='YES'

        elif norm(label) in ('under','u'):yesno='NO'

        else:return None,'total_outcome_unverified'

        meta=gamma_market(cid)

        question=str(meta.get('question') or '') if meta else str(poly.get('question') or '')

        if not re.search(r'\b(?:O\s*/\s*U|Over\s*/\s*Under|Total(?:\s+Points)?)\b',question,re.I):

            return None,'total_question_unverified'

        nums=re.findall(r'\b(\d+\.5)\b',question)

        if not any(abs(float(x)-line)<1e-8 for x in nums):

            return None,'total_threshold_unverified'

        suffix=str(int(line+0.5))

    else:

        return None,'unsupported_contract_type'

    series='KX'+sport+kind

    hits=[]

    for event,ticker in candidate_tickers(a,b,day,series,suffix):

        market=market_get(ticker)

        if not open_market(market,ticker,event):continue

        if kind!='GAME':

            rules=str(market.get('rules_primary') or '')

            if not rule_has_line(rules,kind,line):continue

            if kind=='SPREAD':

                full=TEAMS[league][side_team].split('|')[0]

                aliases=[full,full.rsplit(' ',1)[0],full.split()[0]]

                mm=re.search(r'\bif\s+(.+?)\s+wins?\s+by\b',rules,re.I)

                if not mm or norm(mm.group(1)) not in {norm(x) for x in aliases}:

                    continue

        hits.append(ticker)

    hits=set(hits)

    if len(hits)!=1:return None,f'kalshi_equivalent_candidates_{len(hits)}'

    return (next(iter(hits)),slug,yesno),'matched_pending_exceptional_rule_review'





def book(ticker, outcome='YES'):

    start=clock()

    result=http_json(KALSHI+'/markets/'+quote(ticker,safe='')+'/orderbook')

    end=clock()

    if end-start>8: raise RuntimeError('Kalshi order book HTTP response too slow')

    fp=result.get('orderbook_fp')

    if not isinstance(fp,dict):raise ValueError('Kalshi orderbook_fp missing')

    def read(side):

        entries=fp.get(side)

        if not isinstance(entries,list):raise ValueError('Missing Kalshi '+side)

        out=[]

        for entry in entries:

            if not isinstance(entry,(list,tuple)) or len(entry)!=2:raise ValueError('Bad order book level')

            p,q=dec(entry[0]),dec(entry[1])

            if not (0<p<1 and q>0):raise ValueError('Invalid price or size in order book')

            out.append((p,q))

        return out

    bids=sorted(read('yes_dollars'),reverse=True)

    asks=sorted([(Decimal(1)-p,q) for p,q in read('no_dollars')])

    if bids and asks and bids[0][0]>=asks[0][0]:raise RuntimeError('Crossed Kalshi order book')

    if outcome=='NO':

        no_bids=sorted(read('no_dollars'),reverse=True)

        no_asks=sorted([(Decimal(1)-p,q) for p,q in read('yes_dollars')])

        if no_bids and no_asks and no_bids[0][0]>=no_asks[0][0]:

            raise RuntimeError('Crossed NO book')

        return no_bids,no_asks,end

    if outcome!='YES':raise ValueError('Bad Kalshi outcome')

    return bids,asks,end





def take_levels(levels,qty,lim,side):

    have=Decimal(0);gross=Decimal(0)

    for p,available in levels:

        if side=='BUY' and p>lim: break

        if side=='SELL' and p<lim: break

        n=min(qty-have,available)

        if n<=0:break

        gross+=p*n;have+=n

        if have>=qty:break

    return have,(gross/have if have else None),gross





def floor_qty(q):return dec(q).quantize(Decimal('.01'),rounding=ROUND_DOWN)



def limit_for(ref,side,cents):

    p=dec(ref)+(dec(cents)/100 if side=='BUY' else -dec(cents)/100)

    # Avoid crossing beyond the source-price guard due to tick rounding.

    step=Decimal('.01')

    p=p.quantize(step,rounding=ROUND_DOWN if side=='BUY' else ROUND_UP)

    return max(Decimal('.01'),min(Decimal('.99'),p))





def estimated_fee(q,p,coeff):

    return (q*dec(coeff)*p*(1-p)).quantize(Decimal('.0001'),rounding=ROUND_UP)





def position(c,ticker):

    r=c.execute('SELECT qty,cost FROM positions WHERE ticker=?',(ticker,)).fetchone()

    return (dec(r[0]),dec(r[1])) if r else (Decimal(0),Decimal(0))





def daily_spend(c):

    begin=datetime.now(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()

    r=c.execute("SELECT COALESCE(SUM(filled*fill_px+fee_est),0) FROM orders WHERE side='BUY' AND status IN ('paper_fill','live_fill') AND created_ts>=?",(begin,)).fetchone()

    return dec(r[0])





def total_exposure(c):

    return dec(c.execute('SELECT COALESCE(SUM(cost),0) FROM positions').fetchone()[0])





def apply_fill(c,ticker,side,qty,px,fee):

    own,cost=position(c,ticker)

    if side=='BUY':n=own+qty;basis=cost+qty*px+fee

    else:

        if qty>own+Decimal('.00000001'):raise RuntimeError('Sell exceeds own recorded inventory')

        n=max(Decimal(0),own-qty)

        basis=cost*n/own if own else Decimal(0)

    c.execute('INSERT INTO positions(ticker,qty,cost) VALUES (?,?,?) ON CONFLICT(ticker) DO UPDATE SET qty=excluded.qty,cost=excluded.cost',

              (ticker,float(n),float(basis)))





def signer():

    key_id=os.environ.get('KALSHI_API_KEY_ID','').strip()

    p=os.environ.get('KALSHI_PRIVATE_KEY_PATH','').strip()

    if not key_id or not p: raise RuntimeError('--live requires KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH environment variables')

    from cryptography.hazmat.primitives import hashes,serialization

    from cryptography.hazmat.primitives.asymmetric import padding, ed25519

    key=serialization.load_pem_private_key(Path(p).expanduser().read_bytes(),password=None)

    def sign(method,path):

        timestamp=str(int(clock()*1000))

        message=(timestamp+method+path).encode()

        if isinstance(key,ed25519.Ed25519PrivateKey): sig=key.sign(message)

        else: sig=key.sign(message,padding.PSS(mgf=padding.MGF1(hashes.SHA256()),salt_length=padding.PSS.DIGEST_LENGTH),hashes.SHA256())

        return {'KALSHI-ACCESS-KEY':key_id,'KALSHI-ACCESS-TIMESTAMP':timestamp,

                'KALSHI-ACCESS-SIGNATURE':base64.b64encode(sig).decode()}

    return sign





def submit_live(a,body):

    path='/trade-api/v2/portfolio/events/orders'

    url=KALSHI+'/portfolio/events/orders'

    return http_json(url,method='POST',body=body,headers=a.sign('POST',path),timeout=12)





def store_attempt(c,mode,cid,ticker,side,requested,lim,note):

    cid_order='btb_'+uuid4().hex

    cur=c.execute('''INSERT INTO orders(created_ts,mode,cid,ticker,side,requested,filled,limit_px,fill_px,

        fee_est,status,client_id,note) VALUES (?,?,?,?,?,?,0,?,NULL,0,?,?,?)''',

        (clock(),mode,cid,ticker,side,float(requested),float(lim),

         'SUBMITTING' if mode=='live' else 'paper_pending',cid_order,note))

    c.commit()

    return cur.lastrowid,cid_order





def mark_attempt(c,oid,status,filled=0,px=None,fee=0,exchange_id=None,note=''):

    c.execute('UPDATE orders SET status=?,filled=?,fill_px=?,fee_est=?,exchange_id=?,note=? WHERE id=?',

        (status,float(filled),None if px is None else float(px),float(fee),exchange_id,note,oid))

    c.commit()





def drop(c,cid,token,side,qty,reason):

    c.execute('INSERT INTO orders(created_ts,mode,cid,ticker,side,requested,filled,status,note) VALUES (?,?,?,?,?, ?,0,?,?)',

        (clock(),'skip',cid,None,side,float(qty),'skip',reason))

    c.commit()

    print(f'{utc(clock())} SKIP {side} {cid[:12]} qty={qty:.2f}: {reason}',flush=True)





def buy_contracts_for_budget(budget, limit, fee_coeff, max_depth):

    """0.01-contract multiples; NEVER exceed source-dollar allocation.



    Price and fee evaluated at limit (conservative vs book VWAP).

    """

    budget,limit,max_depth=dec(budget),dec(limit),floor_qty(max_depth)

    if budget<=0 or limit<=0 or max_depth<Decimal('.01'):

        return Decimal(0)

    low,high=0,min(int(max_depth*100),int(budget/(limit*Decimal('.01'))))

    while low<high:

        mid=(low+high+1)//2

        q=Decimal(mid)/100

        total=q*limit+estimated_fee(q,limit,fee_coeff)

        if total<=budget:low=mid

        else:high=mid-1

    return Decimal(low)/100





def dust_get(c,token):

    row=c.execute('SELECT usd FROM dust WHERE token=?',(token,)).fetchone()

    return dec(row[0]) if row else Decimal(0)





def dust_set(c,token,usd):

    assert dec(usd)>=0

    c.execute('INSERT INTO dust(token,usd) VALUES (?,?) ON CONFLICT(token) DO UPDATE SET usd=excluded.usd',

              (token,str(dec(usd))))

    c.commit()





def source_position(c,token):

    row=c.execute('SELECT qty FROM source_inventory WHERE token=?',(token,)).fetchone()

    return dec(row[0]) if row else Decimal(0)





def update_source_position(c,token,qty):

    c.execute('INSERT INTO source_inventory(token,qty) VALUES (?,?) ON CONFLICT(token) DO UPDATE SET qty=excluded.qty',

              (token,str(max(Decimal(0),dec(qty)))))

    c.commit()





def flush(c,token,a,reason):

    r=c.execute('SELECT cid,side,qty,ref,first_ts,last_seen,nfills FROM pending WHERE token=?',(token,)).fetchone()

    if not r:return

    cid,side,raw,ref,started,last_seen,nfills=r

    raw,ref=dec(raw),dec(ref)

    # For BUY raw=source shares, ref=source weighted price; SELL ref=fraction

    # of our position to liquidate, compounded across the pending source SELLs.

    max_lag=max(a.max_signal_age,a.batch_window+2*a.poll+5)
    if clock()-last_seen>max_lag:

        # A stale signal may not be revived as a future purchase.

        c.execute('DELETE FROM pending WHERE token=?',(token,));c.commit()

        return drop(c,cid,token,side,raw,'expired_buy' if side=='BUY' else 'expired_sell')

    m=c.execute('SELECT ticker,kalshi_side FROM mapping WHERE token=?',(token,)).fetchone()

    if not m:
        c.execute('DELETE FROM pending WHERE token=?',(token,));c.commit()
        return drop(c,cid,token,side,raw,'market_mapping_missing')

    ticker,kalshi_side=m

    key=ticker+'|'+kalshi_side

    own,cost=position(c,key)

    budget=raw*ref*dec(a.ratio)+dust_get(c,token) if side=='BUY' else Decimal(0)

    # At this point consume the pending batch once. Preserved deficit is in dust.

    c.execute('DELETE FROM pending WHERE token=?',(token,));c.commit()

    try:

        state=market_get(ticker)

        if not open_market(state,ticker,None):

            if side=='BUY':dust_set(c,token,budget)

            return drop(c,cid,token,side,raw,'Kalshi_market_not_open')

        bids,asks,received=book(ticker,kalshi_side)

        levels=asks if side=='BUY' else bids

        if not levels:

            if side=='BUY':dust_set(c,token,budget)

            return drop(c,cid,token,side,raw,'empty_opposing_book')

        # Cross-venue prices differ: guard SLIPPAGE from current Kalshi top,

        # not differences from Polymarket wallet price.

        best=levels[0][0]

        lim=limit_for(best,side,a.max_book_slippage_cents)

        depth,_,_=take_levels(levels,Decimal('999999999'),lim,side)

        if side=='BUY':

            qty=buy_contracts_for_budget(budget,lim,a.fee_coeff,depth)

        else:

            qty=min(floor_qty(own*ref),floor_qty(depth))

        if qty<Decimal('.01'):

            if side=='BUY':dust_set(c,token,budget)

            return drop(c,cid,token,side,raw,'unexecutable_0.01_contract_or_fee_or_depth')

        got,book_px,_=take_levels(levels,qty,lim,side)

        if got<qty or book_px is None:raise RuntimeError('Depth changed unexpectedly within book response')

        if side=='BUY':

            # The source allocation, not an arbitrary strategy cap, limits buys.

            conservative=qty*lim+estimated_fee(qty,lim,a.fee_coeff)

            assert conservative<=budget+Decimal('.000000001')

        note=(f'{reason}; kalshi_outcome={kalshi_side}; ratio={a.ratio}; source_qty={raw}; source_vwap_or_sell_fraction={ref}; '

              f'wallet_batch_fills={nfills}; book_ts={utc(received)}; '

              f'budget={budget}; best={best}')

        mode='live' if a.live else 'paper'

        oid,client_id=store_attempt(c,mode,cid,ticker,side,qty,lim,note)

        if not a.live:

            bill=qty*book_px+estimated_fee(qty,book_px,a.fee_coeff) if side=='BUY' else Decimal(0)

            fee=estimated_fee(qty,book_px,a.fee_coeff)

            apply_fill(c,key,side,qty,book_px,fee)

            mark_attempt(c,oid,'paper_fill',qty,book_px,fee,note=note)

            if side=='BUY':dust_set(c,token,max(Decimal(0),budget-bill))

            print(f'{utc(clock())} PAPER {side} {ticker} qty={qty} px={book_px:.4f} '

                  f'fee~${fee:.2f} source_cash_ratio={a.ratio} carry=${dust_get(c,token):.4f}',flush=True)

            return

        yes_limit=lim if kalshi_side=='YES' else Decimal(1)-lim

        kalshi_direction=('bid' if side=='BUY' else 'ask') if kalshi_side=='YES' else ('ask' if side=='BUY' else 'bid')

        body={'ticker':ticker,'client_order_id':client_id,'side':kalshi_direction,

              'count':f'{qty:.2f}','price':f'{yes_limit:.4f}','time_in_force':'immediate_or_cancel',

              'self_trade_prevention_type':'taker_at_cross','post_only':False,

              'cancel_order_on_pause':True,'reduce_only':side=='SELL'}

        if a.subaccount is not None:body['subaccount']=a.subaccount

        try:

            res=submit_live(a,body)

            filled=dec(res['fill_count']);remaining=dec(res['remaining_count'])

            if (filled<0 or remaining<0 or filled>qty or not isinstance(res.get('order_id'),str)):

                raise RuntimeError('Malformed Kalshi accepted-order response')

            yes_px=dec(res['average_fill_price']) if res.get('average_fill_price') is not None else (yes_limit if filled else None)

            px=(yes_px if kalshi_side=='YES' else Decimal(1)-yes_px) if yes_px is not None else None

            if filled and (px is None or px<=0 or px>=1):raise RuntimeError('Invalid fill price')

            fee=estimated_fee(filled,px,a.fee_coeff) if filled else Decimal(0)  # estimated until fills reconciled

            if filled:apply_fill(c,key,side,filled,px,fee)

            mark_attempt(c,oid,'live_fill' if filled else 'live_no_fill',filled,px,fee,res['order_id'],

                         note+'; fee_estimate_only; live_reply_not_exchange_reconciled')

            if side=='BUY':

                actual=filled*px+fee if filled else Decimal(0)

                dust_set(c,token,max(Decimal(0),budget-actual))

            print(f'{utc(clock())} LIVE {side} {ticker} filled={filled}/{qty} px=${px} '

                  f'fee_est=${fee:.2f}',flush=True)

        except Exception as ex:

            mark_attempt(c,oid,'UNKNOWN',note=f'UNRECONCILED LIVE POST: {type(ex).__name__}: {ex}')

            raise RuntimeError('AMBIGUOUS LIVE ORDER; HALTED. Reconcile exchange account before restart.') from ex

    except (HTTPError,URLError,TimeoutError,OSError,ValueError) as e:

        if side=='BUY':dust_set(c,token,budget)

        drop(c,cid,token,side,raw,f'public_GET_error:{type(e).__name__}:{e}')





def audit_signal(c, r, seen, decision, detail=''):
    # Persist exactly one audit row per newly received wallet fill.
    src_ts = float(r.get('timestamp') or 0)
    try: qty=float(r.get('size') or 0)
    except (TypeError, ValueError): qty=0.0
    try: px=float(r.get('price') or 0)
    except (TypeError, ValueError): px=0.0
    cid=str(r.get('condition_id') or '').lower()
    token=str(r.get('token_id') or '')
    side=str(r.get('side') or '').upper()
    slug=str(r.get('slug') or '')
    c.execute("INSERT INTO signal_audit(source_ts,seen_ts,cid,token,slug,side,source_qty,source_price,decision,detail) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (src_ts,seen,cid,token,slug,side,qty,px,decision,str(detail)[:400]))
    c.commit()
    print(f'{utc(clock())} WALLET {side} slug={slug or cid[:14]} '
          f'shares={qty:.4f} px={px:.4f} age={seen-src_ts:.1f}s '
          f'=> {decision}{(": " + detail) if detail else ""}',flush=True)


def ingest(c,r,seen,a):

    cid=str(r.get('condition_id','')).lower()

    event_ts=float(r.get('timestamp',0))

    # ALL copies newly observed wallet fills without the first-taker gate.
    # FIRST-TAKER retains the V25 eligibility cohort for parity testing.
    if a.activation=='first-taker':
        activated=c.execute('SELECT source_ts,seen_ts FROM activated WHERE cid=?',(cid,)).fetchone()
        if not activated:
            return audit_signal(c,r,seen,'GATE_NO_FIRST_TAKER')
        if event_ts<float(activated[0])+1 or seen<float(activated[1]):
            return audit_signal(c,r,seen,'GATE_BEFORE_FIRST_TAKER_PLUS_1S')
    if seen-event_ts>a.max_signal_age:
        return audit_signal(c,r,seen,'STALE_SOURCE_FILL',f'limit={a.max_signal_age}s')
    if event_ts>seen+3:
        return audit_signal(c,r,seen,'FUTURE_SOURCE_TIMESTAMP')

    side=str(r.get('side','')).upper()

    if side not in ('BUY','SELL'):
        return audit_signal(c,r,seen,'BAD_SOURCE_SIDE')

    token=str(r.get('token_id',''))

    qty=dec(r.get('size',0));price=dec(r.get('price',0))

    if not token.isdigit() or qty<=0 or not 0<price<1:
        return audit_signal(c,r,seen,'BAD_SOURCE_FIELDS')

    # Any category may have a reviewed equivalent; never skip based on slug.
    maprow=c.execute('SELECT cid,ticker,kalshi_side,review_sha FROM mapping WHERE token=?',(token,)).fetchone()
    if maprow is not None:
        recorded_cid,ticker,k_side,review_sha=maprow
        if recorded_cid!=cid:
            return audit_signal(c,r,seen,'STORED_MAPPING_CID_MISMATCH','manual investigation required')
        if review_sha:
            entry=APPROVED_MATCHES.get((cid,token))
            if (not entry or approved_entry_digest(entry)!=review_sha or
                entry['kalshi_ticker']!=ticker or entry['kalshi_outcome']!=k_side):
                return audit_signal(c,r,seen,'REVIEWED_MAPPING_REMOVED_OR_CHANGED','refusing old mapping')
            verified,why=discover_approved(token,cid)
            if not verified:
                return audit_signal(c,r,seen,'REVIEWED_MAPPING_REVALIDATION_FAILED',str(why))

    if maprow is None:

        mapped,why=discover(token,cid)

        if not mapped:

            audit_signal(c,r,seen,'NO_MARKET_MATCH',why)
            drop(c,cid,token,side,qty,'NO_MARKET_MATCH:'+why)

            return

        ticker,slug,k_side=mapped

        entry=APPROVED_MATCHES.get((cid,token))
        revsha=approved_entry_digest(entry) if entry else None
        c.execute('INSERT INTO mapping(token,cid,ticker,slug,kalshi_side,created_ts,review_sha) VALUES (?,?,?,?,?,?,?)',
                  (token,cid,ticker,slug,k_side,clock(),revsha))

        c.commit();print(f'MATCH {slug} -> {ticker} {k_side}',flush=True)

    previous=source_position(c,token)

    # Source holds are tracked even when a tiny BUY cannot be represented.

    # Only the eligible post-taker period is known; no preactivation snapshot.

    fraction=min(Decimal(1),qty/previous) if side=='SELL' and previous>0 else Decimal(0)

    update_source_position(c,token,previous+qty if side=='BUY' else max(Decimal(0),previous-qty))

    if side=='SELL' and fraction>0:

        dust_set(c,token,dust_get(c,token)*(Decimal(1)-fraction))

    prior=c.execute('SELECT side,qty,ref,first_ts,nfills FROM pending WHERE token=?',(token,)).fetchone()

    if prior and (prior[0]!=side or seen-prior[3]>=a.batch_window):

        flush(c,token,a,'opposite_side_or_window')

        prior=None

    if side=='SELL' and fraction<=0:

        audit_signal(c,r,seen,'UNTRACKED_SELL','No source inventory observed since baseline')
        return drop(c,cid,token,side,qty,'source_sell_without_recorded_wallet_inventory')

    if prior:

        oldq,oldref=dec(prior[1]),dec(prior[2])

        if side=='BUY':

            total=oldq+qty

            newref=(oldq*oldref+qty*price)/total

        else:

            total=oldq+qty

            newref=Decimal(1)-(Decimal(1)-oldref)*(Decimal(1)-fraction)

        c.execute('UPDATE pending SET qty=?,ref=?,last_seen=?,nfills=? WHERE token=?',

                  (float(total),float(newref),seen,prior[4]+1,token))

    else:

        total=qty;newref=price if side=='BUY' else fraction

        c.execute('INSERT INTO pending VALUES (?,?,?,?,?,?,?,?)',

                  (token,cid,side,float(total),float(newref),seen,seen,1))

    c.commit()
    audit_signal(c,r,seen,'ELIGIBLE_QUEUED',f'mapping={c.execute("SELECT ticker FROM mapping WHERE token=?",(token,)).fetchone()[0]}')

    # No minimum-dollar strategy filter; send once feasible after fee,

    # or at the end of the 60-second accumulation window.

    if side=='BUY' and total*newref*dec(a.ratio)+dust_get(c,token)>=Decimal('.02'):

        flush(c,token,a,'microbatch_sufficient')

    elif side=='SELL' and position(c,(lambda row:row[0]+'|'+row[1])(c.execute('SELECT ticker,kalshi_side FROM mapping WHERE token=?',(token,)).fetchone()))[0]*newref>=Decimal('.01'):

        flush(c,token,a,'sell_sufficient')





def cycle(c,a,initial):

    begin=clock()

    prev=c.execute("SELECT value FROM meta WHERE key='last_good_wallet_poll'").fetchone()

    now=clock()

    since=int(max(now-180,float(prev[0])-45)) if prev else int(now-180)

    # Always poll BOTH feeds. Full feed includes makers; taker feed establishes

    # activation. Fetch takers first so all fills have received_ts after gate.

    takers=get_poly_trades(True,since) if a.activation=='first-taker' else []
    received_t=clock()
    all_trades=get_poly_trades(False,since)

    received_all=clock()

    if initial:

        register(c,'taker',takers,received_t)

        register(c,'all',all_trades,received_all)

        # Never enter markets whose taker activation predates our startup.

        for r in takers:

            cid=str(r.get('condition_id','')).lower()

            if cid:c.execute('INSERT OR IGNORE INTO blocked(cid) VALUES (?)',(cid,))

        c.execute("INSERT INTO meta(key,value) VALUES ('bootstrapped',?)",(utc(received_all),))

        c.execute("INSERT INTO meta(key,value) VALUES ('last_good_wallet_poll',?)",(str(received_all),))

        c.commit()

        print(f'{utc(clock())} BASELINED {len(takers)} taker / {len(all_trades)} all fills; NO ORDERS',flush=True)

        return

    taker_new=register(c,'taker',takers,received_t)

    all_new=register(c,'all',all_trades,received_all)

    for r,seen,fp,occ in sorted(taker_new,key=lambda x:(float(x[0]['timestamp']),fingerprint(x[0]))):

        cid=str(r.get('condition_id','')).lower()

        if not cid:continue

        if c.execute('SELECT 1 FROM blocked WHERE cid=?',(cid,)).fetchone():continue

        if c.execute('SELECT 1 FROM activated WHERE cid=?',(cid,)).fetchone():continue

        slug=str(r.get('slug') or '').lower()

        if not slug:

            poly=http_json(POLY_CLOB+'/markets/'+quote(cid,safe=''))

            if str(poly.get('condition_id','')).lower()!=cid:

                raise RuntimeError('Cannot verify activation market identity')

            slug=str(poly.get('market_slug') or poly.get('slug') or '').lower()

        # The first-taker gate is category-independent; the separately
        # verified market mapper decides whether an eventual order can trade.
        if not first_taker_is_first(cid,r):

            c.execute('INSERT OR IGNORE INTO blocked(cid) VALUES (?)',(cid,))

            c.commit()

            print(f'BLOCK {slug}: earlier taker exists',flush=True)

            continue

        c.execute('INSERT INTO activated(cid,source_ts,seen_ts,slug) VALUES (?,?,?,?)',

                  (cid,float(r['timestamp']),seen,slug))

        c.commit()

        print(f'{utc(clock())} ACTIVATED first wallet taker {slug}; source_age={seen-float(r["timestamp"]):.1f}s',flush=True)

    for r,seen,fp,occ in all_new:
        try:
            ingest(c,r,seen,a)
        except (HTTPError,URLError,TimeoutError,OSError) as e:
            # A transient lookup failure before the market is mapped must not
            # permanently mark the signal as processed. Retry on next poll.
            # HTTP failures during flush are already handled inside flush().
            c.execute("DELETE FROM seen WHERE feed='all' AND fp=? AND occ=?",(fp,occ))
            c.commit()
            print(f'{utc(clock())} RETRYABLE_LOOKUP {r.get("side")} '
                  f'{r.get("slug",r.get("condition_id",""))} {type(e).__name__}: {e}',flush=True)

    due=list(c.execute('SELECT token FROM pending WHERE first_ts<=?',(clock()-a.batch_window,)))

    for (tok,) in due:flush(c,tok,a,'batch_window_elapsed')

    settle_positions(c)

    c.execute("INSERT INTO meta(key,value) VALUES ('last_good_wallet_poll',?) "

              "ON CONFLICT(key) DO UPDATE SET value=excluded.value",(str(received_all),))

    c.commit()

    print(f'{utc(clock())} poll: taker_new={len(taker_new)} all_new={len(all_new)} '

          f'gate={a.activation} first_taker_active={c.execute("SELECT COUNT(*) FROM activated").fetchone()[0] if a.activation=="first-taker" else "disabled"} '

          f'pending={c.execute("SELECT COUNT(*) FROM pending").fetchone()[0]} '

          f'cycle_seconds={clock()-begin:.2f}',flush=True)





def settle_positions(c):

    """Only finalize local inventory after official Kalshi finalization.

    Actual Kalshi payout is handled by the exchange, not by this program.

    """

    last=c.execute("SELECT value FROM meta WHERE key='settlement_check'").fetchone()

    if last is not None and clock()-float(last[0])<300:return

    openpos=list(c.execute('SELECT ticker,qty,cost FROM positions WHERE qty>0'))

    # Save poll timestamp only after all public requests succeed.

    for key,qty,cost in openpos:

        ticker,side=key.rsplit('|',1)

        m=market_get(ticker)

        if not isinstance(m,dict):continue

        if str(m.get('status','')).lower() not in ('finalized','settled'):continue

        result=str(m.get('result') or '').lower()

        if result not in ('yes','no'):

            raise RuntimeError('Finalized Kalshi contract with nonbinary result; manual check needed: '+ticker)

        payout=dec(qty) if result==side.lower() else Decimal(0)

        pnl=payout-dec(cost)

        c.execute('INSERT OR IGNORE INTO settlements(ticker,ts,qty,cost,payout,pnl) VALUES (?,?,?,?,?,?)',

                  (key,clock(),qty,cost,float(payout),float(pnl)))

        c.execute('UPDATE positions SET qty=0,cost=0 WHERE ticker=?',(key,))

        c.execute('DELETE FROM dust WHERE token IN (SELECT token FROM mapping WHERE ticker=?)',(ticker,))

        c.commit()

        print(f'{utc(clock())} SETTLED {ticker} result={result} modeled_remaining_pnl=${pnl:.2f}',flush=True)

    c.execute("INSERT INTO meta(key,value) VALUES ('settlement_check',?) "

              "ON CONFLICT(key) DO UPDATE SET value=excluded.value",(str(clock()),))

    c.commit()





def report(c, audit_limit=15):

    print('=== BREAKTHEBANK KALSHI LOCAL REPORT ===')

    print('Source fills seen:',c.execute('SELECT COUNT(*) FROM seen').fetchone()[0])
    print('Observed fill decisions:',dict(c.execute('SELECT decision, COUNT(*) FROM signal_audit GROUP BY decision')))
    print('Latest wallet fill decisions:')
    for src,slug,side,q,px,decision,detail in c.execute('SELECT source_ts,slug,side,source_qty,source_price,decision,detail FROM signal_audit ORDER BY id DESC LIMIT ?', (audit_limit,)):
        print(f'  {utc(src)} {side:4} {str(slug)[:36]:36} qty={q:.4f} px={px:.4f} {decision} {detail}')

    print('Mappings:',c.execute('SELECT COUNT(*) FROM mapping').fetchone()[0]);print('Approved universal token mappings:',len(APPROVED_MATCHES))

    print('Intent status:',dict(c.execute('SELECT status,COUNT(*) FROM orders GROUP BY status')))

    print('Outstanding positions:')

    for t,q,k in c.execute('SELECT ticker,qty,cost FROM positions WHERE qty>0 ORDER BY cost DESC'):

        print(f'  {t}: {q:.2f} contracts, modeled cost ${k:.2f}')

    print('Today buy spend:',round(float(daily_spend(c)),2),'Gross open cost:',round(float(total_exposure(c)),2))

    print('Settled positions:',c.execute('SELECT COUNT(*) FROM settlements').fetchone()[0])

    print('Modeled settlement-only remaining P&L:',round(float(c.execute('SELECT COALESCE(SUM(pnl),0) FROM settlements').fetchone()[0]),2))

    print('WARNING: local fee and cost estimates; not a reconciliation with actual account P&L.')





def selftest():

    # Offline two-page feed and full-page termination tests.

    original_http = globals()['http_json']

    calls = []

    def fake_http(url, **kw):

        from urllib.parse import urlsplit, parse_qs

        q = parse_qs(urlsplit(url).query)

        calls.append(q)

        rows = [{'proxy_wallet':WALLET,'timestamp':1700000000,'side':'BUY'}]

        if 'cursor' not in q:

            return {'data':rows,'pagination':{'has_more':True,'next_cursor':'page2'}}

        assert q['cursor'] == ['page2']

        return {'data':rows,'pagination':{'has_more':False,'next_cursor':None}}

    try:

        globals()['http_json'] = fake_http

        result = get_poly_trades(False,1699999999)

        assert len(result)==2 and len(calls)==2

        assert all(q['start']==['1699999999'] for q in calls)

        assert all(q['taker_only']==['false'] for q in calls)

    finally:

        globals()['http_json'] = original_http

    assert outcome_team('Boston Celtics','nba','bos','nyk')=='bos'

    assert outcome_team('Browns','nfl','bal','cle')=='cle'

    assert outcome_team('No','nba','bos','nyk') is None

    assert SLUG.fullmatch('nba-nyk-bos-2026-10-23-spread-home-1pt5')

    assert next(candidate_tickers('bos','nyk','2026-10-23','KXNBAGAME','NYK'))[1]=='KXNBAGAME-26OCT23BOSNYK-NYK'

    raw={'orderbook_fp':{'yes_dollars':[['0.4400','3.00']], 'no_dollars':[['0.4500','7.00']]}}

    asks=sorted([(Decimal(1)-dec(p),dec(q)) for p,q in raw['orderbook_fp']['no_dollars']])

    assert asks==[(Decimal('.55'),Decimal('7'))]

    assert take_levels(asks,Decimal(10),Decimal('.56'),'BUY')[0]==7

    assert limit_for('.503','BUY',2)==Decimal('.52')

    assert limit_for('.503','SELL',2)==Decimal('.49')

    assert floor_qty('2.999')==Decimal('2.99')

    assert buy_contracts_for_budget(Decimal('.01'),Decimal('.5'),Decimal('.07'),Decimal(100))==Decimal('.01')

    assert buy_contracts_for_budget(Decimal('.02'),Decimal('.5'),Decimal('.07'),Decimal(100))==Decimal('.03')

    assert buy_contracts_for_budget(Decimal('1'),Decimal('.5'),Decimal('.07'),Decimal(100))==Decimal('1.93')

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as d:

        c=connect_db(Path(d)/'test.sqlite')

        r={'timestamp':100,'side':'BUY','size':30,'price':.5,'token_id':'123','condition_id':'0xabc'}

        assert len(register(c,'all',[r,r],105))==2

        assert len(register(c,'all',[r,r],106))==0

        apply_fill(c,'KX', 'BUY',Decimal(3),Decimal('.5'),Decimal('.02'))

        apply_fill(c,'KX', 'SELL',Decimal(2),Decimal('.6'),Decimal('.01'))

        assert position(c,'KX')[0]==1

        c.close()

    print('PASS: pagination, 0.01-contract cash sizing, NFL/NBA classification, book, persistence; no network/orders')






def selftest_activation():
    """Mock-only end-to-end test: zero-taker maker BUY/SELL and optional V25 gate."""
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    now=int(clock())
    cid='0x'+'a'*64; token='12345678'
    slug='nba-bos-nyk-2026-11-17'
    ticker='KXNBAGAME-26NOV17BOSNYK-BOS'
    rows=[]
    saved={k:globals()[k] for k in
           ('get_poly_trades','discover','market_get','book','settle_positions')}
    def row(side,qty,tx):
        return {'proxy_wallet':WALLET,'condition_id':cid,'token_id':token,
                'timestamp':now,'side':side,'size':qty,'price':0.5,
                'transaction_hash':tx,'slug':slug}
    args=lambda mode:SimpleNamespace(activation=mode,ratio='0.0003',
        max_signal_age=15,batch_window=60,poll=2,
        max_book_slippage_cents=2,fee_coeff=.07,live=False)
    try:
        globals()['get_poly_trades']=lambda taker,start: [] if taker else list(rows)
        globals()['discover']=lambda t,c: ((ticker,slug,'YES'),'matched')
        globals()['market_get']=lambda t: {'ticker':t,'event_ticker':t.rsplit('-',1)[0],'status':'open'}
        globals()['book']=lambda t,o: ([(Decimal('.49'),Decimal(10000))],
                                         [(Decimal('.50'),Decimal(10000))],clock())
        globals()['settle_positions']=lambda c: None
        with TemporaryDirectory() as td:
            c=connect_db(Path(td)/'all.sqlite3')
            cycle(c,args('all'),True)
            rows.append(row('BUY',300,'0x1'))
            cycle(c,args('all'),False)
            first=position(c,ticker+'|YES')[0]
            assert first>0 and c.execute('SELECT COUNT(*) FROM activated').fetchone()[0]==0
            rows.append(row('SELL',150,'0x2'))
            cycle(c,args('all'),False)
            second=position(c,ticker+'|YES')[0]
            assert 0<second<first
            assert c.execute("SELECT COUNT(*) FROM orders WHERE status='paper_fill'").fetchone()[0]==2
            assert c.execute("SELECT COUNT(*) FROM signal_audit WHERE decision='ELIGIBLE_QUEUED'").fetchone()[0]==2
            c.close()
            c=connect_db(Path(td)/'first.sqlite3')
            cycle(c,args('first-taker'),True)
            rows.append(row('BUY',300,'0x3'))
            cycle(c,args('first-taker'),False)
            assert c.execute('SELECT COUNT(*) FROM orders').fetchone()[0]==0
            assert c.execute("SELECT COUNT(*) FROM signal_audit WHERE decision='GATE_NO_FIRST_TAKER'").fetchone()[0]==1
            c.close()
    finally:
        globals().update(saved)
    print('PASS: ALL copies maker BUY and proportional SELL with no taker; FIRST-TAKER blocks; no network/orders')


def selftest_universal():
    """Prove non-NBA mapping verification + no accidental title-only execution."""
    from tempfile import TemporaryDirectory
    cid='0x'+'b'*64
    token='123456789123'
    ticker='KXODD-26NOV17-YES'
    kal={'ticker':ticker,'event_ticker':'KXODD-26NOV17','status':'open',
         'title':'Will the satellite launch?', 'rules_primary':'Only official FAA record',
         'rules_secondary':'No launch = NO', 'result':''}
    poly={'condition_id':cid,'market_slug':'satellite-launch-2026','question':'Will the satellite launch?',
          'tokens':[{'token_id':token,'outcome':'Yes'},{'token_id':'444','outcome':'No'}]}
    meta={'question':'Will the satellite launch?'}
    orig={k:globals()[k] for k in ('http_json','gamma_market','market_get','book')}
    orig_maps=APPROVED_MATCHES.copy()
    orig_catalog=list(CANDIDATE_CATALOG)
    def fake_http(url,**kw):
        if url.startswith(POLY_CLOB):return poly
        raise AssertionError('Unexpected HTTP in universal test: '+url)
    try:
        with TemporaryDirectory() as td:
            path=Path(td)/'maps.json'
            row={'condition_id':cid,'token_id':token,'poly_outcome':'Yes',
                 'poly_question':meta['question'],'kalshi_ticker':ticker,
                 'kalshi_outcome':'YES','kalshi_rules_sha256':rule_digest(kal),'reviewed':True}
            path.write_text(json.dumps({'version':1,'matches':[row]}))
            globals()['APPROVED_MATCHES']=load_approved_matches(path)
            globals()['http_json']=fake_http
            globals()['gamma_market']=lambda c: meta
            globals()['market_get']=lambda t: kal
            matched,reason=discover(token,cid)
            assert matched==(ticker,'satellite-launch-2026','YES') and reason.startswith('verified_')
            kal['rules_primary']='Changed rules'
            assert discover(token,cid)[0] is None
            kal['rules_primary']='Only official FAA record'
            poly['tokens'][0]['outcome']='No'
            assert discover(token,cid)[0] is None
            poly['tokens'][0]['outcome']='Yes'
            globals()['APPROVED_MATCHES']={}
            assert discover(token,cid)[0] is None
            globals()['CANDIDATE_CATALOG']=[kal]
            assert 'KXODD' in ','.join(proposed_candidates(meta['question']))
            try:
                row['reviewed']=False
                path.write_text(json.dumps({'version':1,'matches':[row]}))
                load_approved_matches(path)
                assert False,'Unreviewed map accepted'
            except ValueError:pass
            # End-to-end simulated non-NBA BUY then SELL (no activation rule).
            row['reviewed']=True
            path.write_text(json.dumps({'version':1,'matches':[row]}))
            globals()['APPROVED_MATCHES']=load_approved_matches(path)
            globals()['book']=lambda t,o: ([(Decimal('.49'),Decimal(1000))],
                                           [(Decimal('.50'),Decimal(1000))],clock())
            from types import SimpleNamespace
            args=SimpleNamespace(activation='all',ratio='0.0003',max_signal_age=15,
                                 batch_window=60,poll=2,max_book_slippage_cents=2,
                                 fee_coeff=.07,live=False)
            c=connect_db(Path(td)/'universal_paper.sqlite')
            tnow=clock()
            buy={'condition_id':cid,'token_id':token,'timestamp':tnow,'side':'BUY',
                 'price':.5,'size':400,'slug':'satellite-launch-2026'}
            ingest(c,buy,tnow,args)
            owned_before=position(c,ticker+'|YES')[0]
            assert owned_before>Decimal(0)
            sell=dict(buy,side='SELL',size=200)
            ingest(c,sell,tnow,args)
            owned_after=position(c,ticker+'|YES')[0]
            assert Decimal(0)<owned_after<owned_before
            assert c.execute("SELECT COUNT(*) FROM orders WHERE status='paper_fill'").fetchone()[0]==2
            c.close()
    finally:
        globals().update(orig)
        globals()['APPROVED_MATCHES']=orig_maps
        globals()['CANDIDATE_CATALOG']=orig_catalog
    print('PASS: universal cross-category reviewed mapping, rule/outcome changes fail closed, title-only never trades; no network/orders')


def selftest_sports_v30():
    """No network: NHL, MLB timestamped doubleheaders, college, exclusions."""
    from tempfile import TemporaryDirectory
    global CANDIDATE_CATALOG, SPORTS_AUTO_ENABLED
    orig={k:globals()[k] for k in ('market_get','gamma_market','http_json')}
    orig_cat=list(CANDIDATE_CATALOG)
    orig_auto=SPORTS_AUTO_ENABLED
    now='2026-11-17'
    def src(league,a,b,labels,extra=''):
        return {'condition_id':'0x'+'a'*64,
                'market_slug':f'{league}-{a}-{b}-{now}'+(f'-{extra}' if extra else ''),
                'question':f'{labels[0]} vs {labels[1]}',
                'tokens':[{'token_id':'111','outcome':labels[0]},
                          {'token_id':'222','outcome':labels[1]}]}
    def km(t,ev,team,text):
        return {'ticker':t,'event_ticker':ev,'status':'active','result':'',
                'yes_sub_title':team,'title':text,
                'rules_primary':f'Resolves Yes if {team} wins the {text} game'}
    try:
        SPORTS_AUTO_ENABLED=True
        nhl=src('nhl','mon','nyr',['Montreal Canadiens','New York Rangers'])
        ev='KXNHLGAME-26NOV17MTLNYR'
        ticker=ev+'-MTL'
        kal=km(ticker,ev,'Montreal Canadiens','Montreal Canadiens vs New York Rangers')
        CANDIDATE_CATALOG=[dict(kal)]
        globals()['market_get']=lambda t: kal if t==ticker else None
        globals()['gamma_market']=lambda c: {'question':'Montreal Canadiens vs New York Rangers'}
        globals()['http_json']=lambda url,**kw: nhl
        match,reason=discover('111',nhl['condition_id'])
        assert match==(ticker,nhl['market_slug'],'YES'),(match,reason)
        assert 'paper_only' in reason
        assert discover('222',nhl['condition_id'])[0] is None  # Not Rangers market
        # Outcome identity, source-vs-destination mismatches and wrong dates all fail.
        kal['yes_sub_title']='New York Rangers'
        assert discover('111',nhl['condition_id'])[0] is None
        kal['yes_sub_title']='Montreal Canadiens'
        kal['event_ticker']='KXNHLGAME-26NOV23MTLNYR'
        assert discover('111',nhl['condition_id'])[0] is None
        kal['event_ticker']=ev
        # A game-spread slug must NEVER be misread as a game winner.
        nhl['market_slug']=nhl['market_slug']+'-spread-home-1pt5'
        assert discover('111',nhl['condition_id'])[0] is None
        # A timestamped MLB game event is found using the market catalog.
        mlb=src('mlb','nyy','bos',['New York Yankees','Boston Red Sox'])
        ev2='KXMLBGAME-26NOV171905NYYBOS'
        t2=ev2+'-NYY'
        m2=km(t2,ev2,'New York Yankees','New York Yankees vs Boston Red Sox')
        CANDIDATE_CATALOG=[dict(m2)]
        globals()['market_get']=lambda t: m2 if t==t2 else None
        globals()['http_json']=lambda url,**kw: mlb
        match,_=discover('111',mlb['condition_id'])
        assert match and match[0]==t2
        # Doubleheader: ambiguous games must not be auto-matched.
        t3='KXMLBGAME-26NOV172110NYYBOS-NYY'
        m3=km(t3,t3.rsplit('-',1)[0],'New York Yankees','New York Yankees vs Boston Red Sox')
        CANDIDATE_CATALOG=[dict(m2),dict(m3)]
        globals()['market_get']=lambda t: {t2:m2,t3:m3}.get(t)
        assert discover('111',mlb['condition_id'])[0] is None
        # College named matchup: must have both full participants and one outcome.
        cbb=src('cbb','murst','hofst',['Murray State','Hofstra'])
        ev4='KXNCAAMBGAME-26NOV17MURSTHOFST'
        t4=ev4+'-MURST'
        m4=km(t4,ev4,'Murray State','Murray State vs Hofstra')
        CANDIDATE_CATALOG=[dict(m4)]
        globals()['market_get']=lambda t: m4 if t==t4 else None
        globals()['http_json']=lambda url,**kw: cbb
        assert discover('111',cbb['condition_id'])[0][0]==t4
        # Reject college ambiguities and period-specific derivative descriptions.
        m4['rules_primary']='Murray State wins in the first half'
        assert discover('111',cbb['condition_id'])[0] is None
        # A matched NHL total maps Under to the Kalshi NO outcome.
        total=src('nhl','mon','nyr',['Over','Under'],'total-5pt5')
        total['question']='O/U: 5.5 total goals'
        ev5='KXNHLTOTAL-26NOV17MTLNYR'
        t5=ev5+'-6'
        m5={'ticker':t5,'event_ticker':ev5,'status':'open','result':'',
            'title':'Montreal Canadiens vs New York Rangers',
            'rules_primary':'Resolves Yes if the teams score over 5.5 total goals'}
        CANDIDATE_CATALOG=[dict(m5)]
        globals()['market_get']=lambda t: m5 if t==t5 else None
        globals()['http_json']=lambda url,**kw: total
        matched,why=discover('222',total['condition_id'])
        assert matched==(t5,total['market_slug'],'NO'),(matched,why)
        m5['rules_primary']='Resolves Yes if the teams score over 6.5 total goals'
        assert discover('222',total['condition_id'])[0] is None
        # A signed spread +1.5 must be expressed as the opposite team's win
        # margin exceeding 1.5, and the selected home side maps to NO.
        spread=src('nhl','mon','nyr',['Montreal Canadiens','New York Rangers'],
                   'spread-home-1pt5')
        spread['question']='Spread: Montreal Canadiens (+1.5)'
        ev6='KXNHLSPREAD-26NOV17MTLNYR'
        t6=ev6+'-NYR2'
        m6={'ticker':t6,'event_ticker':ev6,'status':'open','result':'',
            'title':'Montreal Canadiens vs New York Rangers',
            'rules_primary':'Resolves Yes if New York Rangers wins by more than 1.5 goals'}
        CANDIDATE_CATALOG=[dict(m6)]
        globals()['market_get']=lambda t: m6 if t==t6 else None
        globals()['http_json']=lambda url,**kw: spread
        match,why=discover('111',spread['condition_id'])
        assert match==(t6,spread['market_slug'],'NO'),(match,why)
        # University names are not guessed from arbitrary codes. Full names in
        # source questions + full names in Kalshi titles are required.
        coll=src('cbb','murst','hofst',['Murray State','Hofstra'],'spread-home-2pt5')
        coll['question']='Spread: Murray State (-2.5) | Murray State vs Hofstra'
        ev7='KXNCAAMBSPREAD-26NOV17MURSTHOFST'
        t7=ev7+'-MURST3'
        m7={'ticker':t7,'event_ticker':ev7,'status':'open','result':'',
            'title':'Murray State vs Hofstra',
            'rules_primary':'Resolves Yes if Murray State wins by over 2.5 points'}
        CANDIDATE_CATALOG=[dict(m7)]
        globals()['market_get']=lambda t: m7 if t==t7 else None
        globals()['http_json']=lambda url,**kw: coll
        assert discover('111',coll['condition_id'])[0] is None  # noncanonical source question
        coll['question']='Spread: Murray State (-2.5)'
        match,_=discover('111',coll['condition_id'])
        assert match==(t7,coll['market_slug'],'YES'),match
        coll['question']='Spread: Hofstra (-2.5)'
        assert discover('111',coll['condition_id'])[0] is None  # source team changed
        # A non-sports slug never reaches the automatic sports matcher.
        assert sports_source('will-anthropic-ipo-have-market-cap-over-1t') is None
    finally:
        globals().update(orig)
        CANDIDATE_CATALOG=orig_cat
        SPORTS_AUTO_ENABLED=orig_auto
    print('PASS: NHL, MLB time-coded event, college winner/spread, NHL spread/total NO, doubleheader, wrong side/date/period; offline')


def main():

    p=argparse.ArgumentParser(description='Copy newly observed eligible BreakTheBank fills, or choose first-taker V25 cohort.')

    modes=p.add_mutually_exclusive_group()

    modes.add_argument('--paper',action='store_true',help='Paper Kalshi depth-constrained IOC simulation')

    modes.add_argument('--live',action='store_true',help='REAL Kalshi IOC orders, explicit opt-in')

    p.add_argument('--once',action='store_true',help='One poll (first poll only establishes baseline)')

    p.add_argument('--report',action='store_true',help='Local report only; no public requests')

    p.add_argument('--confirm-live',action='store_true',help='Required additional opt-in for REAL orders')

    p.add_argument('--self-test',action='store_true',help='Offline tests; no network or orders')

    p.add_argument('--poll',type=float,default=2)
    p.add_argument('--sports-auto',choices=('paper','off'),default='paper',
                   help='Extra sports game-winner auto matching: paper-only or off; live requires reviewed JSON')
    p.add_argument('--activation',choices=('all','first-taker'),default='all',
                   help='all = every fresh observed wallet fill; first-taker = V25 gate')
    p.add_argument('--audit-limit',type=int,default=15,help='Number of latest fill decisions in report')

    p.add_argument('--ratio',type=str,default='0.0003',help='Proportion of source wallet BUY dollar spend; 0.0003=0.03%%')

    p.add_argument('--batch-window',type=float,default=60)

    # No minimum source spend filter: micro-buys accumulate toward exchange min

    p.add_argument('--max-book-slippage-cents',type=float,default=2,help='Maximum depth slippage vs current Kalshi best price')

    p.add_argument('--max-signal-age',type=float,default=15)

    p.add_argument('--fee-coeff',type=float,default=.07,help='Hypothetical quadratic Kalshi fee coefficient')

    p.add_argument('--db',type=Path,help='Optional SQLite path; defaults to alongside this script in sports-copy')
    p.add_argument('--match-file',type=Path,help='Reviewed cross-category mapping JSON (any market)')
    p.add_argument('--catalog-file',type=Path,help='Locally cached Kalshi open-market catalog for UNVERIFIED candidate suggestions only')
    p.add_argument('--sync-sports-catalog',type=Path,help='Download only supported sports game/line series into a small, resumable JSON index; READ ONLY')
    p.add_argument('--sports-catalog-delay',type=float,default=1.0,help='Seconds between sports catalog pages (default 1.0)')
    p.add_argument('--sports-catalog-retries',type=int,default=7,help='HTTP 429 exponential-backoff retries for catalog (default 7)')
    p.add_argument('--sync-catalog',type=Path,help='Download ALL open Kalshi markets to catalog JSON, then exit; READ ONLY')
    p.add_argument('--catalog-max-pages', type=int, default=5000,
                   help='Safety ceiling for index sync; persisted progress resumes on rerun')
    p.add_argument('--catalog-mve', choices=('exclude', 'only', 'all'), default='exclude',
                   help='Index combo/parlay markets: exclude (fast default), only, or all (large)')
    p.add_argument('--inspect-pair',nargs=2,metavar=('POLY_CONDITION','KALSHI_TICKER'),
                   help='Print source outcomes and Kalshi rules for human review, then exit; READ ONLY')


    a=p.parse_args()

    if a.self_test:
        selftest();selftest_activation();selftest_universal();selftest_sports_v30();return
    if a.sync_sports_catalog:
        sync_sports_catalog(a.sync_sports_catalog,delay=a.sports_catalog_delay,retries=a.sports_catalog_retries);return
    if a.sync_catalog:
        sync_kalshi_catalog(a.sync_catalog.expanduser().resolve(),
                            max_pages=a.catalog_max_pages, mve_filter=a.catalog_mve);return
    if a.inspect_pair:
        inspect_pair(*a.inspect_pair);return
    global APPROVED_MATCHES, CANDIDATE_CATALOG, SPORTS_AUTO_ENABLED
    SPORTS_AUTO_ENABLED = a.paper and a.sports_auto == 'paper'
    APPROVED_MATCHES=load_approved_matches(a.match_file)
    CANDIDATE_CATALOG=catalog_load(a.catalog_file)
    print(f'SPORTS AUTO: {"paper-only" if SPORTS_AUTO_ENABLED else "off (reviewed mappings only)"}; indexed markets={len(CANDIDATE_CATALOG)}',flush=True)


    if not a.paper and not a.live:p.error('Specify exactly one of --paper or --live')

    if not (a.poll>=2 and 0<dec(a.ratio)<=1 and a.batch_window>0 and a.audit_limit>=0 and

            0<=a.max_book_slippage_cents<=10 and a.max_signal_age>0 and

            0<=a.fee_coeff<1):p.error('Invalid ratio/timing/fee arguments')

    if a.db is None:
        label='all' if a.activation=='all' else 'first_taker'
        a.db=Path(__file__).resolve().parent/f'breakthebank_v29_{label}_{"live" if a.live else "paper"}.sqlite3'

    a.db=a.db.expanduser().resolve()

    # Keep state beside the script when requested. Exclude database/lock files
    # from Git; trading state must remain persistent across deployments.
    a.db.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive OS file lock ensures one copier per DB. SQLite alone does not

    # prevent two independent processes from simultaneously submitting orders.

    lock_path=str(a.db)+'.lock'

    a.lock_handle=open(lock_path,'a+')

    try:fcntl.flock(a.lock_handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)

    except BlockingIOError:raise SystemExit('Another BreakTheBank instance already owns this database; stopped')

    c=connect_db(a.db)

    # A database must never silently change activation rule, proportion or mode.
    # Unlike a whole-file hash, per-token hashes let you ADD verified mappings
    # without discarding historic inventory or accumulated SELL ratios.
    if (c.execute("SELECT 1 FROM meta WHERE key='bootstrapped'").fetchone() and
        c.execute("SELECT 1 FROM meta WHERE key='bound_mapping_engine_version'").fetchone() is None):
        raise SystemExit('LEGACY DB: use separate V29 --db; cannot import old strategy inventory safely')
    bindings={'activation':a.activation, 'ratio':str(dec(a.ratio)), 'execution_mode':'live' if a.live else 'paper', 'mapping_engine_version':'v29_universal_reviewed_1'}
    for k,v in bindings.items():
        row=c.execute('SELECT value FROM meta WHERE key=?',('bound_'+k,)).fetchone()
        if row is not None and row[0]!=v:
            raise SystemExit(f'DATABASE CONFIG MISMATCH {k}: db={row[0]}, requested={v}; choose a NEW --db.')
    for k,v in bindings.items():
        c.execute('INSERT OR IGNORE INTO meta(key,value) VALUES (?,?)',('bound_'+k,v))
    c.commit()
    for token,cid,ticker,k_side,sha in c.execute(
            'SELECT token,cid,ticker,kalshi_side,review_sha FROM mapping WHERE review_sha IS NOT NULL'):
        entry=APPROVED_MATCHES.get((cid,token))
        if (not entry or approved_entry_digest(entry)!=sha or
            entry['kalshi_ticker']!=ticker or entry['kalshi_outcome']!=k_side):
            raise SystemExit(f'PREVIOUSLY REVIEWED MAPPING CHANGED/MISSING for {cid} token={token}; HALTED')
    if a.report:report(c,a.audit_limit);return

    if a.live:

        if not a.confirm_live:raise SystemExit('LIVE LOCKED: requires --confirm-live after independent verification')

        sub=os.environ.get('KALSHI_SUBACCOUNT','').strip()

        if sub and (not sub.isdigit() or not 0<=int(sub)<=63):

            raise SystemExit('KALSHI_SUBACCOUNT must be an integer 0..63')

        a.subaccount=int(sub) if sub else None

        sub_label=sub if sub else 'default'

        stored=c.execute("SELECT value FROM meta WHERE key='kalshi_subaccount'").fetchone()

        if stored and stored[0]!=sub_label:

            raise SystemExit('Database bound to a DIFFERENT Kalshi subaccount. Refusing to trade.')

        if not stored:

            c.execute("INSERT INTO meta(key,value) VALUES ('kalshi_subaccount',?)",(sub_label,));c.commit()

        unknown=c.execute("SELECT COUNT(*) FROM orders WHERE status IN ('SUBMITTING','UNKNOWN')").fetchone()[0]

        if unknown:raise SystemExit(f'LIVE HALTED: {unknown} unresolved orders. Reconcile manually before restarting.')

        a.sign=signer()

        print('LIVE MODE: REAL KALSHI ORDERS ENABLED. No strategy caps; source cash ratio ONLY.',flush=True)

    else:

        a.subaccount=None

        print('PAPER MODE: NEVER SENDS ORDERS',flush=True)

    initial=c.execute("SELECT value FROM meta WHERE key='bootstrapped'").fetchone() is None

    print(f'Wallet={WALLET} NBA/NFL automatic candidates + all-category approved maps={len(APPROVED_MATCHES)}; activation={a.activation}; cash_ratio={a.ratio} ({dec(a.ratio)*100}%); feed age<={a.max_signal_age}s; '

          f'book slippage<={a.max_book_slippage_cents}c; db={a.db} initial_baseline={initial}',flush=True)

    errors=0

    while True:

        try:

            cycle(c,a,initial)

            errors=0;initial=False

        except (HTTPError,URLError,TimeoutError,ValueError,RuntimeError,OSError,sqlite3.Error) as e:

            print(f'{utc(clock())} FAIL CLOSED: {type(e).__name__}: {e}',flush=True)

            c.execute('INSERT INTO errors VALUES (?,?)',(clock(),str(e)));c.commit()

            if a.live and c.execute("SELECT 1 FROM orders WHERE status IN ('SUBMITTING','UNKNOWN') LIMIT 1").fetchone():

                raise SystemExit('LIVE HALTED: order state unknown. Do not automatically restart.') from e

            errors+=1

            if errors>=3:raise SystemExit('3 consecutive errors; stopped') from e

        if a.once:break

        time.sleep(a.poll)



if __name__=='__main__':main()
