#!/usr/bin/env python3
"""Offline copyability triage of cached Polymarket wallet trades.

Purely descriptive: no API requests, exchange orders, credentials, existing-file writes,
or assertions that copied fills were executable.  Data API timestamps have 1s precision.

Read caches created by poly_wallet_copy_audit_readonly.py.  Pairing uses same-market,
complementary outcome-index fills, sequential FIFO quantity matching, and ONLY
markets with buys (no sells).  The paired profit assumes both legs were held to
settlement and excludes fees, rebate effects, unmatched directional exposure,
funding, and settlement discrepancies. Slippage scenarios are hypotheticals,
not backtests of executable delayed quotes.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path
import re

WINDOW_RE = re.compile(r'(?:^|[-_])(btc|bitcoin|eth|ethereum|sol|solana|xrp)[-_]updown[-_](5m|15m)[-_]\d{10}(?:$|[-_])', re.I)
SENSITIVE = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03) # combined extra cost per paired share


def category(slug, title):
    text = (slug + ' ' + title).lower()
    m = WINDOW_RE.search(slug)
    if m: return ('crypto_'+m.group(2).lower())
    if 'highest temperature' in text or 'highest-temperature' in text: return 'weather_high'
    if 'lowest temperature' in text or 'lowest-temperature' in text: return 'weather_low'
    if any(s in text for s in ('bitcoin','btc-', 'ethereum','eth-', 'solana','sol-', 'crypto')): return 'crypto_other'
    if any(s in text for s in ('temperature', 'weather', 'snow', 'rain')): return 'weather_other'
    return 'other'


def data(path):
    markets = defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line_num, line in enumerate(f,1):
            if not line.strip(): continue
            r = json.loads(line)
            side = str(r.get('side','')).upper()
            if side not in ('BUY','SELL'): continue
            p = float(r['price']); q = float(r['size'])
            if not (0 < p < 1 and q > 0): continue
            cid, tok = str(r.get('condition_id') or ''), str(r.get('token_id') or '')
            if not (cid and tok): continue
            idx = str(r.get('outcome_index',''))
            markets[cid].append((int(r['timestamp']),line_num,side,idx,tok,p,q,str(r.get('slug') or ''),str(r.get('title') or '')))
    return markets


@dataclass
class PairResult:
    matched: float=0.0
    cost: float=0.0
    gross: float=0.0
    within5: float=0.0
    within60: float=0.0
    profitable: float=0.0
    positive_markets: int=0


def analyze_pairs(markets):
    groups = defaultdict(lambda: {'markets':0,'fills':0,'buy_notional':0.0,'pair':PairResult(),'skip_sells':0,'skip_tokens':0})
    for rows in markets.values():
        rows.sort(key=lambda x:(x[0],x[1]))
        kind = category(rows[0][7], rows[0][8]); a = groups[kind]
        a['markets']+=1; a['fills']+=len(rows)
        a['buy_notional'] += sum(r[5]*r[6] for r in rows if r[2]=='BUY')
        if any(r[2]=='SELL' for r in rows):
            a['skip_sells']+=1; continue
        tokens = defaultdict(set)
        for r in rows:
            tokens[r[3]].add(r[4])
        if any(k not in ('0','1') for k in tokens) or len(tokens.get('0',[]))!=1 or len(tokens.get('1',[]))!=1 or tokens['0']==tokens['1']:
            a['skip_tokens']+=1; continue
        pending = {'0':deque(),'1':deque()}
        p = a['pair']; market_gross=0.0
        for ts,_,_,idx,tok,price,qty,slug,title in rows:
            other = pending['1' if idx=='0' else '0']; rem = qty
            while rem > 1e-9 and other:
                v=other[0]; take=min(rem,v[1])
                cost=take*(price+v[2]); gross=take-cost
                lag=abs(ts-v[0]); p.matched+=take; p.cost+=cost; p.gross+=gross
                market_gross+=gross
                if lag<=5: p.within5+=take
                if lag<=60: p.within60+=take
                if price+v[2]<1: p.profitable+=take
                rem-=take; v[1]-=take
                if v[1] <= 1e-9: other.popleft()
            if rem > 1e-9: pending[idx].append([ts,rem,price])
        if market_gross>0: p.positive_markets+=1
    return groups


def analyze_exits(markets):
    times = ((5,'<=5s'),(15,'<=15s'),(60,'<=60s'),(300,'<=300s'))
    summary=defaultdict(lambda:{'matched_shares':0.0,'gross':0.0,'within':defaultdict(float),'within_gross':defaultdict(float),'market_count':0,'uncovered_shares':0.0})
    for rows in markets.values():
        rows.sort(key=lambda x:(x[0],x[1]))
        kind=category(rows[0][7],rows[0][8]); a=summary[kind]; a['market_count']+=1
        pending=defaultdict(deque)
        for ts,_,side,idx,tok,p,qty,_,_ in rows:
            if side=='BUY': pending[tok].append([ts,qty,p]); continue
            rem=qty
            while rem>1e-9 and pending[tok]:
                first=pending[tok][0]; take=min(rem,first[1]); hold=max(0,ts-first[0]); gross=take*(p-first[2])
                a['matched_shares']+=take; a['gross']+=gross
                for t,label in times:
                    if hold<=t: a['within'][label]+=take; a['within_gross'][label]+=gross
                first[1]-=take; rem-=take
                if first[1]<=1e-9: pending[tok].popleft()
            if rem>1e-9: a['uncovered_shares']+=rem
    return summary


def percent(num,den): return 100*num/den if den else 0.0


def show_pairs(name,groups):
    print('\n'+'='*83+'\n'+name+' — PAIRED INVENTORY; NOT FOLLOWER P&L\n'+'='*83)
    aggregate=PairResult()
    for k,v in sorted(groups.items(),key=lambda x:-x[1]['buy_notional']):
        p=v['pair']; aggregate.matched+=p.matched; aggregate.cost+=p.cost; aggregate.gross+=p.gross
        aggregate.within5+=p.within5; aggregate.within60+=p.within60
        print(f"{k:17} markets={v['markets']:6,d} fills={v['fills']:7,d} buy_USDC=${v['buy_notional']:>12,.2f} paired_qty={p.matched:>13,.1f} gross=${p.gross:>11,.2f} ROI={percent(p.gross,p.cost):>6.2f}%")
        if p.matched:
            print(f"   positive_margin_qty={percent(p.profitable,p.matched):.1f}%  legs_within_5s={percent(p.within5,p.matched):.1f}%  legs_within_60s={percent(p.within60,p.matched):.1f}%  paired_markets_positive_gross={p.positive_markets}")
        if v['skip_sells'] or v['skip_tokens']:
            print(f"   excluded_markets_with_sells={v['skip_sells']}, excluded_non_binary_or_incomplete_tokens={v['skip_tokens']}")
    print(f'\nTOTAL matched_qty={aggregate.matched:,.1f}, gross=${aggregate.gross:,.2f}, combined_margin={100*aggregate.gross/aggregate.matched if aggregate.matched else 0:.3f} cents per paired share')
    print('SENSITIVITY: extra cost on EACH matched two-leg pair, before any other fees:')
    for cost in SENSITIVE:
        print(f'   combined cost +{cost*100:.1f}c  => projected gross ${aggregate.gross-aggregate.matched*cost:,.2f}')
    print('NOTE: unmatched positions, incomplete token pairs, real delay/slippage and settlement effects excluded.')


def show_exits(name,groups):
    print('\n'+'='*83+'\n'+name+' — OBSERVED FIFO SELL TIMING; NOT EXECUTABLE COPY TEST\n'+'='*83)
    for k,v in sorted(groups.items(),key=lambda x:-x[1]['matched_shares']):
        if not v['matched_shares'] and not v['uncovered_shares']: continue
        q=v['matched_shares']; print(f"{k}: matched={q:,.1f} shares gross=${v['gross']:,.2f}, uncovered_sell_shares={v['uncovered_shares']:,.1f}")
        for label in ('<=5s','<=15s','<=60s','<=300s'):
            print(f"  {label:6} : {percent(v['within'][label],q):5.1f}% of sold shares ; gross from this duration ${v['within_gross'][label]:,.2f}")
    print('NOTE: 1-second timestamps; inside-second sequencing may be wrong. FIFO is an assumption.')


def self_test():
    def r(ts,line,side,idx,price,qty,slug='btc-updown-5m-1792000000'):
        return (ts,line,side,str(idx),'token'+str(idx),price,qty,slug,'Bitcoin Up or Down')
    market={'a':[r(100,1,'BUY',0,.48,5),r(110,2,'BUY',1,.49,3)]}
    d=analyze_pairs(market)['crypto_5m']['pair']
    assert round(d.matched,3)==3 and round(d.gross,3)==.09 and d.within60==3
    h={'b':[r(100,1,'BUY',0,.4,10),r(102,2,'SELL',0,.6,5),r(202,3,'SELL',0,.65,5)]}
    v=analyze_exits(h)['crypto_5m']
    assert v['matched_shares']==10 and round(v['gross'],2)==2.25 and v['within']['<=5s']==5
    print('PASS: offline synthetic pairing and exit-timing tests')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=Path('data/research/poly_wallet_copy_audit_20261111'))
    p.add_argument('--wallets',nargs='+',default=['asdcaxz','wallet_954b','HighTempTation'])
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test: return self_test()
    for name in args.wallets:
        fp=args.input_dir/f'{name}_trades.jsonl'; meta=args.input_dir/f'{name}_trades_meta.json'
        if not fp.is_file() or not meta.is_file(): p.error(f'missing cached files: {fp} and {meta}')
        if json.loads(meta.read_text()).get('complete') is not True: p.error(f'incomplete wallet cache {name}')
        markets=data(fp)
        if name.lower()=='hightemptation': show_exits(name,analyze_exits(markets))
        else: show_pairs(name,analyze_pairs(markets))
    print('\nREAD ONLY. No API requests, no changes to existing files or orders.')

if __name__=='__main__': main()
