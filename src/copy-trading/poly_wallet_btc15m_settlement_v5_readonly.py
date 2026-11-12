#!/usr/bin/env python3
"""Resolve Polymarket BTC 15-minute wallet inventory by actual condition payouts.

Input: unmodified market_inventory.csv + summary.json from poly_wallet_15m_risk_audit_v4.py.
Data: public GET https://data-api.polymarket.com/v2/resolutions?condition=<up-to-20-IDs>
Output: append-only resolution cache, market_settlements.csv, daily.csv, summary.json.

No order placement, API keys, wallet signing, or private endpoints. This does NOT
simulate copy executions. Fees are not inferred; all settlement P&L is gross.

Strict coverage: missing, unresolved, ambiguous, non-complementary, or invalid
payouts are excluded from realized totals and shown with explicit counts/cost.
The v4 input includes all PURCHASES but not possible external transfers or
positions outside its trade window. Its buy-only cache is trusted only for the
wallet's selected market cohort, not for all-time wallet realized P&L.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE = 'https://data-api.polymarket.com/v2/resolutions'
CID_RE = re.compile(r'^0x[a-fA-F0-9]{64}$')
EPS = 1e-6
BATCH = 20
CENTS = (0, .25, .5, 1, 1.5, 2)


def nfloat(x, context):
    try:
        x = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid float {context}: {x!r}') from exc
    if not math.isfinite(x):
        raise ValueError(f'Nonfinite float {context}: {x!r}')
    return x


def get_json(ids, retries=6, timeout=35):
    """Public GET. Don't turn transient HTTP errors into empty results."""
    assert 1 <= len(ids) <= BATCH
    url = BASE + '?' + urlencode({'condition': ','.join(ids)})
    req = Request(url, headers={'Accept': 'application/json', 'User-Agent': 'copy-risk-audit/5.0'})
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout) as f:
                data = json.load(f)
            if not isinstance(data, dict) or not isinstance(data.get('data'), list):
                raise ValueError('Unexpected resolution API envelope, refusing to guess')
            return data['data']
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries-1:
                raise
            header = exc.headers.get('Retry-After')
            try:
                delay = min(90., max(.5, float(header))) if header else min(60., 2.**attempt)
            except ValueError:
                delay = min(60., 2.**attempt)
            time.sleep(delay)
        except (URLError, TimeoutError):
            if attempt == retries-1:
                raise
            time.sleep(min(60., 2.**attempt))
    raise RuntimeError('retries exhausted')


def parse_payout(row):
    """Return normalized payout tuple for resolved *condition* rows only.

    The public resolution route expresses payouts in micro-USD per share,
    so [1000000, 0] is [1.0, 0.0]. A status or a stale market price is not
    a settlement. We reject absent, malformed or non-complementary payouts.
    """
    if not isinstance(row, dict) or str(row.get('status') or '').lower() != 'resolved':
        return None, 'not_resolved'
    p = row.get('payouts')
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except json.JSONDecodeError:
            return None, 'invalid_payouts'
    if not isinstance(p, list) or len(p) != 2:
        return None, 'invalid_payouts'
    try:
        vals = (nfloat(p[0], 'payout_0'), nfloat(p[1], 'payout_1'))
    except ValueError:
        return None, 'invalid_payouts'
    # Do not assume units if the endpoint returns another scale.
    if any(v < 0 or v > 1_000_000 for v in vals):
        return None, 'invalid_payouts'
    if abs(sum(vals)-1_000_000) > .01:
        return None, 'noncomplementary_payouts'
    return (vals[0]/1_000_000., vals[1]/1_000_000.), None


def normalize_rows(wanted, response_rows):
    """Ensure resolution rows match requested condition IDs exactly."""
    want = {c.lower(): c for c in wanted}
    found = defaultdict(list)
    for row in response_rows:
        if not isinstance(row, dict):
            raise ValueError('Malformed resolution row')
        cid = str(row.get('condition_id') or row.get('conditionId') or '').lower()
        if not cid:
            # Question-only rows do not give independently verifiable condition mapping.
            continue
        if cid not in want:
            raise ValueError(f'API returned unrequested condition {cid}')
        found[cid].append(row)
    result = {}
    for low, original in want.items():
        candidates = found[low]
        if not candidates:
            result[original] = {'status':'missing', 'payouts':None}
            continue
        payouts = []
        bad_resolved = []
        for row in candidates:
            if str(row.get('status') or '').lower() == 'resolved':
                p, err = parse_payout(row)
                if p is not None:
                    payouts.append(p)
                else:
                    bad_resolved.append(err)
        if bad_resolved:
            result[original] = {'status': 'invalid_resolved', 'reason': ','.join(sorted(set(bad_resolved))), 'payouts':None}
        elif payouts and all(all(abs(a-b) < 1e-9 for a,b in zip(payouts[0],p)) for p in payouts):
            result[original] = {'status':'resolved', 'payouts':list(payouts[0])}
        elif payouts:
            result[original] = {'status':'conflicting_resolutions', 'payouts':None}
        else:
            result[original] = {'status':'unresolved', 'payouts':None}
    return result


def load_input(srcdir):
    summary_path, csv_path = srcdir/'summary.json', srcdir/'market_inventory.csv'
    if not summary_path.is_file() or not csv_path.is_file():
        raise ValueError('Need V4 summary.json and market_inventory.csv in --input-dir')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('market_window') != '15m' or summary.get('asset_filter') != 'btc':
        raise ValueError('V5 requires BTC-only 15m v4 inventory; run V4 --window 15m --asset btc')
    with csv_path.open(newline='',encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows or len(rows) != int(summary['analyzed_markets']):
        raise ValueError('CSV row count does not match V4 summary')
    ids=set()
    numeric = ['bought_shares_0','bought_shares_1','total_cost_usdc',
               'paired_shares','paired_gross_usdc','unpaired_cost_usdc','total_shares_bought']
    for i,r in enumerate(rows):
        cid = str(r.get('condition_id') or '')
        if not CID_RE.fullmatch(cid) or cid.lower() in ids:
            raise ValueError(f'invalid/duplicate condition id at row {i+1}: {cid}')
        ids.add(cid.lower())
        if not str(r.get('slug','')).startswith('btc-updown-15m-'):
            raise ValueError('Non-BTC-15m slug in input: '+str(r.get('slug')))
        for k in numeric:
            r[k] = nfloat(r.get(k),k)
            if r[k] < -EPS:
                raise ValueError(f'negative {k} at {cid}')
        if abs(r['bought_shares_0'] + r['bought_shares_1'] - r['total_shares_bought']) > 1e-3:
            raise ValueError('Share conservation failed: '+cid)
        if abs(r['paired_shares'] - min(r['bought_shares_0'],r['bought_shares_1']))>1e-3:
            raise ValueError('Paired shares mismatch: '+cid)
        if abs(r['paired_gross_usdc'] -(r['paired_shares']-(r['total_cost_usdc']-r['unpaired_cost_usdc']))) > 1e-3:
            raise ValueError('Pair gross mismatch: '+cid)
        r['market_start_epoch'] = int(r['market_start_epoch'])
    for k,csvk in [('wallet_spend_usdc','total_cost_usdc'),('paired_qty','paired_shares')]:
        actual = sum(r[csvk] for r in rows)
        if abs(actual-nfloat(summary[k],k))>1e-2:
            raise ValueError(f'V4 summary {k} does not match input CSV')
    return rows,summary


def load_cache(path):
    found = {}
    if path.is_file():
        with path.open(encoding='utf-8') as f:
            for i, line in enumerate(f,1):
                if not line.strip(): continue
                j=json.loads(line)
                cid=j.get('condition_id')
                if not isinstance(cid,str) or not CID_RE.fullmatch(cid):
                    raise ValueError(f'invalid cache row {i}')
                if not isinstance(j.get('result'),dict):
                    raise ValueError(f'invalid cache result {i}')
                found[cid.lower()]=j['result']
    return found


def cache_fetch(ids, path, offline=False, pause=.15, refresh_resolved=False):
    existing=load_cache(path)
    todo=[cid for cid in ids if (refresh_resolved or existing.get(cid.lower(),{}).get('status')!='resolved')]
    if offline:
        print(f'OFFLINE: {len(ids)-len(todo)} settled resolutions from cache; {len(todo)} require refresh (not fetched).')
        return existing
    if not todo:
        print('All requested markets already have cached finalized payouts.')
        return existing
    path.parent.mkdir(parents=True,exist_ok=True)
    pages=(len(todo)+BATCH-1)//BATCH
    print(f'Public resolution GET: {len(todo):,} conditions in {pages:,} batches of <=20 (read-only).',flush=True)
    with path.open('a',encoding='utf-8') as f:
        for batch in range(pages):
            arr=todo[batch*BATCH:(batch+1)*BATCH]
            raw=get_json(arr)
            records=normalize_rows(arr,raw)
            for cid in arr:
                outcome=records[cid]
                f.write(json.dumps({'condition_id':cid,'checked_at':int(time.time()), 'result':outcome},separators=(',',':'))+'\n')
                existing[cid.lower()]=outcome
            f.flush()
            if (batch+1)%10==0 or batch==pages-1:
                c=Counter(existing[c.lower()].get('status','unknown') for c in ids if c.lower() in existing)
                print(f'  batch {batch+1:,}/{pages:,} | resolved={c["resolved"]:,}, unresolved={sum(v for k,v in c.items() if k!="resolved"):,}',flush=True)
            if pause and batch < pages-1:
                time.sleep(pause)
    return existing


def make_report(rows, resolutions):
    results=[]
    for r in rows:
        cid=r['condition_id']
        state=resolutions.get(cid.lower(),{'status':'not_fetched','payouts':None})
        status=state.get('status') or 'unknown'
        p=state.get('payouts') if status=='resolved' else None
        q0,q1=r['bought_shares_0'],r['bought_shares_1']
        payout=profit=unmatched_profit=None
        if p is not None:
            if not isinstance(p,list) or len(p)!=2 or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 or x>1 for x in p) or abs(sum(p)-1)>1e-9:
                raise ValueError('Invalid cached normalized payouts for '+cid)
            payout=p[0]*q0+p[1]*q1
            profit=payout-r['total_cost_usdc']
            unmatched_profit=profit-r['paired_gross_usdc']
        results.append({
            'condition_id':cid, 'slug':r['slug'], 'utc_market_date':r['utc_market_date'],
            'status':status,'payout_0':p[0] if p is not None else None,
            'payout_1':p[1] if p is not None else None,
            'spent_usdc':r['total_cost_usdc'],
            'bought_shares_0':q0,'bought_shares_1':q1,
            'total_shares_bought':r['total_shares_bought'],
            'paired_shares':r['paired_shares'],
            'paired_gross_before_fees_usdc':r['paired_gross_usdc'],
            'unmatched_cost_usdc':r['unpaired_cost_usdc'],
            'settlement_payout_usdc':payout,
            'settled_whole_book_gross_usdc':profit,
            'settled_residual_gross_usdc':unmatched_profit,
            'first_buy_s_from_market_start':r.get('first_buy_s_from_market_start'),
            'first_positive_floor_s_from_market_start':r.get('first_positive_floor_s_from_market_start'),
        })
    good=[r for r in results if r['status']=='resolved']
    skipped=[r for r in results if r['status']!='resolved']
    def sm(rs,k):return sum(r[k] for r in rs)
    spent=sm(good,'spent_usdc')
    gross=sm(good,'settled_whole_book_gross_usdc')
    paired=sm(good,'paired_gross_before_fees_usdc')
    residual=sm(good,'settled_residual_gross_usdc')
    qty=sm(good,'total_shares_bought')
    assert abs(gross-paired-residual)<1e-3
    summary={
        'interpretation':'Gross settlement payoff for the observed buy-only BTC15m cohort, not wallet net P&L and not executable follower P&L. Subtract real fees, transfers/merges, and funding elsewhere. No quote replay performed.',
        'market_count_total':len(results),
        'market_count_resolved':len(good),
        'market_count_not_resolved':len(skipped),
        'not_resolved_by_reason':dict(Counter(r['status'] for r in skipped)),
        'settled_coverage_pct':100*len(good)/len(results),
        'observed_spend_usdc':sm(results,'spent_usdc'),
        'resolved_spend_usdc':spent,
        'unresolved_spend_usdc':sm(skipped,'spent_usdc'),
        'resolved_settlement_payout_usdc':sm(good,'settlement_payout_usdc'),
        'resolved_full_inventory_gross_usdc_before_fees':gross,
        'resolved_full_inventory_gross_roi_pct':100*gross/spent if spent else None,
        'resolved_paired_gross_usdc_before_fees':paired,
        'resolved_unmatched_residual_gross_usdc_before_fees':residual,
        'resolved_winning_markets':sum(r['settled_whole_book_gross_usdc']>EPS for r in good),
        'resolved_losing_markets':sum(r['settled_whole_book_gross_usdc']<-EPS for r in good),
        'resolved_flat_markets':sum(abs(r['settled_whole_book_gross_usdc'])<=EPS for r in good),
        'resolved_gross_profit_after_added_purchase_cost_c_per_share':{
            f'{c:.2f}':gross-qty*c/100 for c in CENTS
        },
        'additional_purchase_cost_c_per_share_for_breakeven': 100*gross/qty if qty else None,
        'resolved_payout_split':dict(Counter('0' if r['payout_0']==1 else ('1' if r['payout_1']==1 else 'partial') for r in good)),
    }
    days=defaultdict(list)
    for r in good:days[r['utc_market_date']].append(r)
    daily=[]
    for d,v in sorted(days.items()):
        daily.append({
            'utc_market_date':d,'resolved_markets':len(v),'spend_usdc':sm(v,'spent_usdc'),
            'settlement_gross_usdc':sm(v,'settled_whole_book_gross_usdc'),
            'paired_gross_usdc':sm(v,'paired_gross_before_fees_usdc'),
            'residual_gross_usdc':sm(v,'settled_residual_gross_usdc'),
            'win_markets':sum(r['settled_whole_book_gross_usdc']>EPS for r in v)
        })
    return results,summary,daily


def write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def self_test():
    a,b='0x'+'a'*64,'0x'+'b'*64
    resp=[{'condition_id':a, 'status':'resolved','payouts':[1000000,0]},
          {'condition_id':b, 'status':'initialized','payouts':None}]
    mapped=normalize_rows([a,b],resp)
    assert mapped[a]['payouts']==[1.0,0.0] and mapped[b]['status']=='unresolved'
    assert normalize_rows([a],[])[a]['status']=='missing'
    assert normalize_rows([a],[{'condition_id':a,'status':'resolved','payouts':[500000,500000]}])[a]['payouts']==[.5,.5]
    assert normalize_rows([a],[{'condition_id':a,'status':'resolved','payouts':[900000,500000]}])[a]['status']=='invalid_resolved'
    assert normalize_rows([a],[{'condition_id':a,'status':'resolved','payouts':None}])[a]['status']=='invalid_resolved'
    assert normalize_rows([a],[{'condition_id':a,'status':'resolved','payouts':[1000000,0]}, {'condition_id':a,'status':'resolved','payouts':[0,1000000]}])[a]['status']=='conflicting_resolutions'
    base={'slug':'btc-updown-15m-1792000000','utc_market_date':'2026-10-14',
          'bought_shares_0':20.0,'bought_shares_1':10.0,'total_shares_bought':30.,
          'total_cost_usdc':13.5,'paired_shares':10.,'paired_gross_usdc':3.,
          'unpaired_cost_usdc':6.5}
    arow={'condition_id':a,**base}
    brow={'condition_id':b,**base,'bought_shares_0':10.,'bought_shares_1':20.}
    rows,summary,daily=make_report([arow,brow],{a.lower():mapped[a],b.lower():mapped[b]})
    assert summary['market_count_resolved']==1 and summary['market_count_not_resolved']==1
    # full settled payoff = 20 - 13.5 = +6.5, versus paired +3, residual +3.5
    assert abs(summary['resolved_full_inventory_gross_usdc_before_fees']-6.5)<EPS
    assert abs(summary['resolved_unmatched_residual_gross_usdc_before_fees']-3.5)<EPS
    assert abs(summary['unresolved_spend_usdc']-13.5)<EPS
    print('PASS: exact condition matching, normalized payouts, unsettled coverage, residual settlement, fee sensitivity; NO network.')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-dir',type=Path,default=Path('data/research/asdcaxz_btc15m_risk_v4'),help='V4 BTC-only output directory')
    ap.add_argument('--output-dir',type=Path,default=Path('data/research/asdcaxz_btc15m_settlement_v5'),help='New reports + append-only resolution cache')
    ap.add_argument('--offline',action='store_true',help='Only use existing cached resolutions')
    ap.add_argument('--refresh-resolved',action='store_true',help='Recheck even finalized cached payouts')
    ap.add_argument('--pause',type=float,default=.15,help='Public GET pause per batch (seconds)')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:return self_test()
    if args.pause<0:ap.error('pause must be nonnegative')
    if args.output_dir.resolve()==args.input_dir.resolve():ap.error('Use a NEW --output-dir, not V4 input directory')
    inputrows,old=load_input(args.input_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    ids=[r['condition_id'] for r in inputrows]
    cache=args.output_dir/'resolution_cache.jsonl'
    res=cache_fetch(ids,cache,offline=args.offline,pause=args.pause,refresh_resolved=args.refresh_resolved)
    rows,summary,days=make_report(inputrows,res)
    write_csv(args.output_dir/'market_settlements.csv',rows)
    write_csv(args.output_dir/'daily.csv',days)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('=== BTC15M WALLET INVENTORY: REAL SETTLEMENT PAYOUTS (NOT COPY P&L) ===')
    print(f"Markets: {summary['market_count_total']:,}; settled={summary['market_count_resolved']:,} ({summary['settled_coverage_pct']:.1f}%) ; not settled/verified={summary['market_count_not_resolved']:,}")
    print(f"Unverified spend=${summary['unresolved_spend_usdc']:,.2f}; resolved spend=${summary['resolved_spend_usdc']:,.2f}")
    print(f"Settled matched-pair gross=${summary['resolved_paired_gross_usdc_before_fees']:,.2f}; unmatched-residual gross=${summary['resolved_unmatched_residual_gross_usdc_before_fees']:,.2f}")
    print(f"WHOLE BOOK settled gross=${summary['resolved_full_inventory_gross_usdc_before_fees']:,.2f}, ROI={summary['resolved_full_inventory_gross_roi_pct']:.2f}% BEFORE actual fees") if summary['resolved_full_inventory_gross_roi_pct'] is not None else print('No resolved markets; no ROI calculated')
    print('Extra purchase-price cost per share sensitivity:',summary['resolved_gross_profit_after_added_purchase_cost_c_per_share'])
    print('Unverified status breakdown:',summary['not_resolved_by_reason'])
    print('NO executable follower P&L, bid/ask replay, fees or cross-platform equivalence established.')
    print('READ ONLY: public GET for resolutions; writes new output files; no exchange trading actions.')
    print('Report directory:',args.output_dir)

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,AssertionError) as exc:
        print('AUDIT STOPPED (will not substitute guesses):',exc,file=sys.stderr)
        sys.exit(2)
