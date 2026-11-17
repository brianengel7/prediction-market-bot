#!/usr/bin/env python3
"""BreakTheBank V8: proportional post-taker book-price replay (READ ONLY).

Use existing V1 wallet data, V3 settlement cache and V7 batching. Prepare an
immutable intent manifest, then replay actual historical *recorded* top-of-book
snapshots supplied in a CSV. Never downloads or fabricates market data.

USAGE:
  python3 -B src/sports-copy/breakthebank_v8_book_replay_readonly.py --self-test
  python3 -u -B src/sports-copy/breakthebank_v8_book_replay_readonly.py \
    --out ~/sports_copy_research/breakthebank_v8/intent_0001.csv
  python3 -u -B src/sports-copy/breakthebank_v8_book_replay_readonly.py \
    --quotes-csv ~/sports_copy_research/breakthebank_books/bbo.csv \
    --out ~/sports_copy_research/breakthebank_v8/book_replay_0001.json

QUOTES CSV (one observed snapshot per token/time, top level only):
  ts_recv,token_id,bid,bid_size,ask,ask_size
  1792346700.225,1234567,0.48,500,0.50,100

IMPORTANT: ts_recv MUST be when book was actually captured/received, UTC epoch
seconds with fractions allowed. Do NOT use post-hoc trade midpoints. A missing
quote means NO VERIFIED FILL, not a zero price or winning trade. As-of only
(last snapshot <= order time). A recorded resting top-of-book size is a
conditional liquidity proxy, NOT proof of actual queue/execution. V8 consumes
single top-price size, never invents depth; same snapshot cannot be reused by
multiple IOC orders. Real fees and wallet publication delays not known.

NO trading, API requests, network, API keys, GitHub/source edits, or DB writes.
"""
from __future__ import annotations
import argparse
import bisect
from collections import Counter, defaultdict
import csv
import importlib.util
import json
import math
from pathlib import Path
from statistics import median

REPO_PREFIX='/workspaces/'


def load(path, name):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec=importlib.util.spec_from_file_location(name,path)
    if not spec or not spec.loader:
        raise RuntimeError('Cannot load '+str(path))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def source_orders(a):
    base=Path(__file__).resolve().parent
    v1=load(base/'breakthebank_copyability_audit.py','btb_v8_v1')
    v3=load(base/'breakthebank_all_fills_v3_readonly.py','btb_v8_v3')
    v5=load(base/'breakthebank_v5_taker_timing_readonly.py','btb_v8_v5')
    v7=load(base/'breakthebank_v7_proportional_execution_readonly.py','btb_v8_v7')
    prior,records,payouts=v5.require_sources(a.v1_input_dir.expanduser(),a.v3_input_dir.expanduser(),v1,v3)
    selected=v7.eligible(records,a.min_age)
    orders,stat=v7.group_orders(selected,scale=a.scale,min_notional=a.min_notional,
                                min_shares=a.min_shares,max_window_s=a.batch_window)
    if not orders:
        raise RuntimeError('No orders emitted: check inputs')
    if any(r['category']!='sports' for r in orders):
        raise AssertionError('Sports-only filter broken')
    return orders,stat,v1,v3,payouts


def read_quotes(path):
    """Sorted as-received top-of-book observations; reject conflicting duplicates."""
    table=defaultdict(dict)
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        expected={'ts_recv','token_id','bid','bid_size','ask','ask_size'}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError('Quotes need columns '+','.join(sorted(expected)))
        for i,r in enumerate(reader,2):
            try:
                token=str(r['token_id']).strip()
                stamp=float(r['ts_recv'])
                def n(field):
                    x=str(r[field]).strip()
                    return float(x) if x else None
                bid,ask=n('bid'),n('ask')
                bid_size,ask_size=n('bid_size'),n('ask_size')
            except (TypeError,ValueError,KeyError) as e:
                raise ValueError(f'Invalid quote row {i}: {e}') from e
            if not token or not math.isfinite(stamp) or stamp<=0:
                raise ValueError(f'Invalid token/timestamp at row {i}')
            for price,size,side in ((bid,bid_size,'bid'),(ask,ask_size,'ask')):
                if (price is None)!=(size is None):
                    raise ValueError(f'Both {side} and {side}_size required at row {i}')
                if price is not None and (not math.isfinite(price) or not 0<price<1 or
                                           not math.isfinite(size) or size<0):
                    raise ValueError(f'Invalid {side} at row {i}')
            if bid is not None and ask is not None and bid>ask+1e-9:
                raise ValueError(f'Crossed book at row {i}')
            point=(bid,bid_size,ask,ask_size)
            prev=table[token].get(stamp)
            if prev is not None and prev!=point:
                raise ValueError(f'Conflicting same-token/same-time quote at row {i}')
            table[token][stamp]=point
    result={}
    for token,observed in table.items():
        times=sorted(observed)
        result[token]=(times,[observed[t] for t in times])
    return result


def apply_books(orders,quotes,*,scale,delay,max_worse_cents,max_quote_age,fee_coeff,v1,v3,payouts):
    if any(not math.isfinite(q) for q in (scale,delay,max_worse_cents,max_quote_age,fee_coeff)):
        raise ValueError('Nonfinite input')
    if scale<=0 or delay<0 or max_worse_cents<0 or max_quote_age<0 or fee_coeff<0:
        raise ValueError('Invalid replay parameters')
    inventory=defaultdict(float)
    liquidity_used=defaultdict(float)
    attempted=Counter(); fillstats=Counter(); emitted=[]; coverage=Counter()
    # Order and event timestamps are wallet-observed whole seconds; quotes can be sub-second.
    for seq,r in sorted(enumerate(orders),key=lambda x:(x[1]['ts']+delay,x[1]['seq'],x[0])):
        attempt=r['wallet_qty']*scale
        side=r['side'];tok=r['token'];cid=r['market']
        attempted['orders']+=1
        if side=='SELL':
            attempt=min(attempt,max(0.,inventory[(cid,tok)]))
            if attempt<=1e-9:
                fillstats['no_inventory']+=1
                continue
        target=r['ts']+delay
        series=quotes.get(tok)
        if series is None:
            fillstats['missing_token_book']+=1
            continue
        tms,values=series
        ix=bisect.bisect_right(tms,target)-1
        if ix<0:
            fillstats['no_quote_before_order']+=1
            continue
        qt=tms[ix];age=target-qt
        if age>max_quote_age+1e-8:
            fillstats['stale_quote']+=1
            continue
        coverage['fresh_book_orders']+=1
        bid,bids,ask,asks=values[ix]
        price,avail=(ask,asks) if side=='BUY' else (bid,bids)
        if price is None or avail<=1e-9:
            fillstats['empty_side']+=1
            continue
        # A strict aggressive IOC: never cross beyond specified wallet-price drift.
        wall=float(r['wallet_price']);slip=max_worse_cents/100.
        if (side=='BUY' and price>min(1,wall+slip)+1e-9) or (side=='SELL' and price<max(0,wall-slip)-1e-9):
            fillstats['outside_limit']+=1
            continue
        key=(tok,qt,side)
        avail=max(0.,avail-liquidity_used[key])
        q=min(attempt,avail)
        if q<=1e-9:
            fillstats['book_size_exhausted']+=1
            continue
        liquidity_used[key]+=q
        if side=='BUY':inventory[(cid,tok)]+=q
        else:inventory[(cid,tok)]-=q
        # Convert verified simulated copy shares to V3 units for consistent settlement/PnL.
        emitted.append({**r,'wallet_qty':q/scale,'wallet_price':price,
                        'ts':target,'seq':seq,'role':r['role']})
        fillstats['filled_orders']+=1
        if q<attempt-1e-8:
            fillstats['partially_filled_orders']+=1
        fillstats['shares_executed']+=q
        fillstats['wallet_price_difference_share_weighted']+=q*((price-wall) if side=='BUY' else (wall-price))
        if side=='BUY':fillstats['filled_buys']+=1
        else:fillstats['filled_sells']+=1
    markets=v3.replay(emitted,payouts,v1,adverse_cents=0,scale=scale,fee_coeff=fee_coeff,
                       tie_policy='buy-first') if emitted else []
    aggregate=v3.aggregate(markets)
    prices=round(fillstats['wallet_price_difference_share_weighted']/fillstats['shares_executed']*100,4) if fillstats['shares_executed'] else None
    missing=attempted['orders']-fillstats['filled_orders']
    observed_tokens={r['token'] for r in orders}
    coverage_pct=100*fillstats['filled_orders']/attempted['orders'] if attempted['orders'] else None
    # Not a performance bound: omitted orders may have been either profitable or costly.
    return {'delay_after_wallet_print_s':delay,
        'max_worse_cents_per_share':max_worse_cents,
        'max_quote_age_s':max_quote_age,
        'orders_intended':attempted['orders'],
        'orders_with_fresh_book':coverage['fresh_book_orders'],
        'orders_filled_or_partially_filled':fillstats['filled_orders'],
        'orders_unverified_or_rejected':missing,
        'fill_order_rate_pct':round(coverage_pct,2) if coverage_pct is not None else None,
        'partial_fill_orders':fillstats['partially_filled_orders'],
        'tokens_requested':len(observed_tokens),'tokens_in_quote_input':len(observed_tokens & set(quotes)),
        'mean_executed_price_worsening_cents_share_weighted':prices,
        'scored_only':aggregate,
        'unscored_buy_usdc':aggregate['unscored_buy_notional'],
        'hypothetical_fee_coefficient':fee_coeff,
        'rejects':dict(fillstats),
        'interpretation':'Conditional recorded top-of-book IOC proxy; missing quotes/orders do not establish zero real fill rate or profitable true execution. Not account ROI.'}


def write_manifest(path,orders,scale):
    fields=['intent_id','market','token_id','outcome_index','side','wallet_trade_ts_utc',
            'wallet_price','wallet_shares','copy_ratio','target_copy_shares','target_copy_usdc',
            'batch_source_fills','batch_first_ts','batch_wait_s','title']
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,r in enumerate(orders,1):
            w.writerow({'intent_id':i,'market':r['market'],'token_id':r['token'],
            'outcome_index':r['idx'],'side':r['side'],'wallet_trade_ts_utc':r['ts'],
            'wallet_price':r['wallet_price'],'wallet_shares':r['wallet_qty'],
            'copy_ratio':scale,'target_copy_shares':round(r['wallet_qty']*scale,8),
            'target_copy_usdc':round(r['wallet_qty']*scale*r['wallet_price'],6),
            'batch_source_fills':r['batch_source_fills'],
            'batch_first_ts':r['batch_first_ts'],'batch_wait_s':r['batch_wait_s'],
            'title':r['title']})


def self_test():
    from types import SimpleNamespace
    m='0x'+'a'*64; t='token-A'
    def o(ts,side,qty,price,seq):
        return dict(market=m,token=t,idx='0',ts=ts,seq=seq,side=side,
                    wallet_qty=qty,wallet_price=price,category='sports',title='Synthetic',role='maker')
    v1=SimpleNamespace(payout_value=lambda mid,idx,p:p[mid][idx])
    v3=load(Path(__file__).with_name('breakthebank_all_fills_v3_readonly.py'),'btb_v8_test_v3')
    books={t:([99.,101.,102.],[(.30,2.,.40,2.),(.32,2.,.42,3.),(.33,3.,.52,3.)])}
    payoff={m:{'0':1.}}
    orders=[o(100,'BUY',4,.40,1),o(100,'BUY',4,.40,2),o(101,'SELL',5,.50,3)]
    a=apply_books(orders,books,scale=1,delay=1,max_worse_cents=2,
                  max_quote_age=3,fee_coeff=0,v1=v1,v3=v3,payouts=payoff)
    # 1st BUY 3 @0.42; second cannot reuse same snapshot; SELL 3 @.33 denied by 2c limit.
    assert a['orders_filled_or_partially_filled']==1 and a['partial_fill_orders']==1,a
    assert a['rejects']['book_size_exhausted']==1 and a['rejects']['outside_limit']==1,a
    assert a['scored_only']['scored_net_pnl']==1.74,a  # 3 shares worth 1 - .42
    b=apply_books([o(100,'BUY',4,.40,1)],books,scale=1,delay=1,
                  max_worse_cents=0,max_quote_age=3,fee_coeff=0,v1=v1,v3=v3,payouts=payoff)
    assert b['orders_filled_or_partially_filled']==0
    c=apply_books([o(100,'BUY',4,.40,1)],books,scale=1,delay=0,
                  max_worse_cents=0,max_quote_age=.5,fee_coeff=0,v1=v1,v3=v3,payouts=payoff)
    assert c['rejects']['stale_quote']==1
    # no lookahead: a quote at t=101 cannot price an event at t=100.
    d=apply_books([o(100,'BUY',2,.40,1)],{t:([101.],[(.3,10.,.4,10.)])},
                  scale=1,delay=0,max_worse_cents=2,max_quote_age=10,
                  fee_coeff=0,v1=v1,v3=v3,payouts=payoff)
    assert d['rejects']['no_quote_before_order']==1
    print('PASS V8: as-of quotes, no lookahead, limit price, IOC partials, quote-size exhaustion, settlement')
    print('OFFLINE; no network, orders, secrets or repository writes')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--v1-input-dir',type=Path,default=Path.home()/'sports_copy_research'/'breakthebank_v1')
    p.add_argument('--v3-input-dir',type=Path,default=Path.home()/'sports_copy_research'/'breakthebank_v3')
    p.add_argument('--scale',type=float,default=.001)
    p.add_argument('--min-age',type=int,default=1,help='Min seconds after FIRST wallet taker fill')
    p.add_argument('--min-notional',type=float,default=1.)
    p.add_argument('--min-shares',type=float,default=0.)
    p.add_argument('--batch-window',type=int,default=60)
    p.add_argument('--quotes-csv',type=Path,help='Historical BBO with captured as-of timestamps (no interpolation)')
    p.add_argument('--delays',default='5,15,30,60',help='Seconds after wallet print; proxy for feed+route latency, not measured')
    p.add_argument('--limits-cents',default='2,5',help='Maximum adverse cents/sh spent on both buys and sells')
    p.add_argument('--max-book-age',type=float,default=10.,help='Seconds from last captured BBO to order time')
    p.add_argument('--fee-coeff',type=float,default=.04,help='Hypothetical fee coeff, not current actual exchange tariff')
    p.add_argument('--out',type=Path,default=Path.home()/'sports_copy_research'/'breakthebank_v8'/'intents.csv')
    a=p.parse_args()
    if a.self_test:self_test();return
    if any(not math.isfinite(x) for x in (a.scale,a.min_notional,a.min_shares,a.max_book_age,a.fee_coeff)) or a.scale<=0 or a.min_age<1 or a.min_notional<0 or a.min_shares<0 or a.batch_window<0 or a.max_book_age<0 or a.fee_coeff<0:
        p.error('Invalid scale/threshold/age/fee')
    dest=a.out.expanduser().resolve()
    if str(dest).startswith(REPO_PREFIX) or dest.exists():
        p.error('Output must be new and outside repository; refusing overwrite')
    orders,stats,v1,v3,payouts=source_orders(a)
    print('BREAKTHEBANK V8 — HISTORICAL TOP-OF-BOOK REPLAY / INTENT MANIFEST')
    print(f'Sports after first taker +{a.min_age}s; share ratio={a.scale}; min ${a.min_notional}; batch {a.batch_window}s')
    print('Intents:',len(orders),'source fills retained:',stats['copied_source_fills_in_orders'],
          'excluded source fills:',stats['discarded_wallet_fills'])
    print('Distinct markets:',len({x['market'] for x in orders}),
          'outcome tokens:',len({x['token'] for x in orders}))
    dest.parent.mkdir(parents=True,exist_ok=True)
    if a.quotes_csv is None:
        if dest.suffix.lower()!='.csv':p.error('Manifest output --out must end in .csv')
        write_manifest(dest,orders,a.scale)
        print('Order intent manifest:',dest)
        print('Next: obtain book quotes with ts_recv, token_id, bid, bid_size, ask, ask_size; run --quotes-csv')
        print('NO HISTORICAL BOOKS READ: this is a data-request manifest, NOT a backtest.')
    else:
        if dest.suffix.lower()!='.json':p.error('Replay --out must end in .json')
        quote_path=a.quotes_csv.expanduser().resolve()
        quotes=read_quotes(quote_path)
        delays=[float(x) for x in a.delays.split(',')]
        cents=[float(x) for x in a.limits_cents.split(',')]
        if not delays or not cents or any(not math.isfinite(d) or d<0 for d in delays) or any(not math.isfinite(c) or c<0 for c in cents):
            p.error('Invalid delays or price limits')
        runs=[]
        for delay in delays:
            for maxc in cents:
                report=apply_books(orders,quotes,scale=a.scale,delay=delay,
                    max_worse_cents=maxc,max_quote_age=a.max_book_age,
                    fee_coeff=a.fee_coeff,v1=v1,v3=v3,payouts=payouts)
                print(f'{delay:>5.0f}s max+{maxc:.0f}c | book fresh={report["orders_with_fresh_book"]}/{report["orders_intended"]} '
                      f'filled/partial={report["orders_filled_or_partially_filled"]} '
                      f'scored={report["scored_only"]["markets_scored"]} '
                      f'ROI={report["scored_only"]["scored_roi_pct"]}%; '
                      f'no-book/skip={report["orders_unverified_or_rejected"]}')
                runs.append(report)
        dest.write_text(json.dumps({'method':'as-received top-of-book IOC proxy; not guaranteed fillability',
            'quote_input':str(quote_path),'intent_count':len(orders),
            'strategy':{'scale':a.scale,'min_age':a.min_age,'min_notional':a.min_notional,
                        'batch_window':a.batch_window,'fee_coeff_hypothetical':a.fee_coeff},
            'batching':stats,'results':runs},indent=2),encoding='utf-8')
        print('Historical BBO proxy report:',dest)
        print('WARNING: outcomes conditioned on quotes present, completeness must be checked; no proven fills.')
    print('READ ONLY — no external requests, keys, orders or modifications to source repo')

if __name__=='__main__':main()
