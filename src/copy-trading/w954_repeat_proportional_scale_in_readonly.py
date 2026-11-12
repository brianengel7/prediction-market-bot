#!/usr/bin/env python3
"""Read-only, causally timed wallet 954B selective repeat/add-on Kalshi replay.

Place beside these EXISTING local scripts in src/copy-trading/:
  w954_kalshi_crossvenue_backtest_readonly.py
  w954_kalshi_all_fills_backtest_readonly.py
  w954_post_entry_sequence_audit_readonly.py
  w954_followup_copy_window_audit_readonly.py
  w954_binance_signal_audit_readonly.py

No file writes or orders. Public Kalshi GET requests only unless --markets-file
and --trades-file or --book-root are provided. Trade prints are NOT executable
historical asks. Entries use Polymarket wallet observation timestamps, NOT
proven public-detection timestamps. Kalshi settlement determines P&L.

Strategy R: first buy priced 20<=p<40c and true first same-side repeat 4-30s
with no prior opposite buy. Purchase at repeat; optionally add on further
same-side wallet BUYs after repeat, with gap/window/cap controls. Stops adding
once any opposite-side buy appears. Only first 62-ish qualifying markets.

Strategy F: purchase at EVERY qualifying first buy (all ~187). If the repeat
becomes observable, add at repeat and optionally on further same-side buys.
This does NOT pre-select early purchases using future repetition information.

With --sizing proportional: a ten-contract copied starter is the anchor, and
later copied positions target ROUND(10 * wallet cumulative shares / wallet
shares at the moment we opened the copied position). For FIRST+CONFIRM the
anchor is the wallet first fill; for REPEAT+ the anchor is wallet same-side
cumulative shares through the confirmation. Every post-repeat wallet BUY is
counted, even fragmented prints within the four-second aggregation interval.
Only after its time bucket closes do we place one copied add order. A later
opposite-side purchase disables further scale-ins. This is one hypothetical
copy order per observed size change bucket, not every wallet raw fill.

All comparisons match exact Kalshi close timestamp and settled outcome. The
trade-print replay consumes each print's stated quantity only once PER strategy
and latency (not proof an additional trade was available to us).
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import random
import runpy
import sys
from statistics import median
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / 'w954_kalshi_crossvenue_backtest_readonly.py'
ALL_PATH = HERE / 'w954_kalshi_all_fills_backtest_readonly.py'


def refs():
    for p in (CORE_PATH, ALL_PATH):
        if not p.is_file():
            raise RuntimeError(f'Missing required companion script: {p}')
    return runpy.run_path(str(CORE_PATH)), runpy.run_path(str(ALL_PATH))


def repeat_confirmed(s, minimum, maximum):
    r = s.get('first_same')
    if r is None: return False
    age = r['ts'] - s['ts']
    if not minimum <= age <= maximum: return False
    opposite = s.get('first_opp')
    if opposite and (opposite['ts'], opposite['line']) <= (r['ts'], r['line']):
        return False
    return not s.get('prior_opposite', False)


def extra_adds(s, minimum_gap, window, max_adds):
    """Online-filterable same-side add decisions, after confirmation.

    Avoid same-second/multi-fill fragments and stop at first opposing BUY.
    No future events determine whether an earlier entry is made.
    """
    t = s['first_same']
    if t is None or max_adds <= 0: return []
    chosen = []
    last = t['ts']
    stop = s.get('first_opp')
    for x in s['buys']:
        if (x['ts'], x['line']) <= (t['ts'], t['line']): continue
        if x['ts'] - t['ts'] > window: break
        if stop and (stop['ts'], stop['line']) <= (x['ts'], x['line']): break
        if x['ts'] - last < minimum_gap: continue
        chosen.append(x)
        last = x['ts']
        if len(chosen) >= max_adds: break
    return chosen


def index_wallet(events):
    rows = defaultdict(list)
    for e in events: rows[e['start']].append(e)
    return rows


def initial_wallet_size(s, by_start):
    """Find source wallet FIRST purchase for proportional sizing.

    Returns None if joins are unsafe. Default fixed-quantity mode does not
    require sizing join but still enforces the original signal population.
    """
    choices = [x for x in by_start.get(s['start'], []) if
               x['action'] == 'BUY' and x['side'] == s['side'] and
               abs(x['ts'] - s['ts']) <= 3 and
               abs(x['price'] - s['price']) <= .015]
    if not choices: return None
    hit = min(choices, key=lambda x: (abs(x['ts'] - s['ts']),
                                      abs(x['price'] - s['price']), x['line']))
    return hit['size']


def planned_qty(kind, row, args):
    if args.sizing in ('fixed','proportional'):
        return args.base_qty if kind in ('first', 'repeat') else args.add_qty
    size = row if isinstance(row, (int, float)) else row.get('size')
    if size is None: return None
    return int(float(size) * args.scale + .5)




def wallet_first_fill(s, by_start):
    """Match the first wallet fill across independent extracts (<=3s clock drift).

    If the matched source print was observed AFTER signal time, the simulated
    first copied order is delayed until that print's time. No sizing is taken
    from a later record while pretending it was available earlier.
    """
    choices = [r for r in by_start.get(s['start'], [])
               if r['action'] == 'BUY' and r['side'] == s['side']
               and abs(r['ts'] - s['ts']) <= 3
               and abs(r['price'] - s['price']) <= .015
               and math.isfinite(r['size']) and r['size'] > 0]
    if not choices:
        return None
    return min(choices, key=lambda r:(abs(r['ts'] - s['ts']),
                                       abs(r['price'] - s['price']),r['line']))


def wallet_at_repeat(s, first, by_start):
    """Causal known same-side shares at the precise qualifying repeat print."""
    rep = s['first_same']
    low = (first['ts'], first['line'])
    high = (rep['ts'], rep['line'])
    qty = sum(r['size'] for r in by_start.get(s['start'], [])
              if r['action']=='BUY' and r['side']==s['side']
              and low <= (r['ts'],r['line']) <= high)
    if qty < first['size']-1e-8:
        raise RuntimeError('Invalid wallet initial size/confirmation anchor')
    return qty


def proportional_buckets(s, args, max_adds):
    """Aggregate every post-confirmation same-side fill; act at 4s boundaries.

    An opposite-side BUY before a bucket closes cancels that bucket too, since
    no order was placed yet. Grouping avoids treating split fills as separate
    signals, double-rounding, or acting with future information.
    """
    if max_adds < 1:
        return []
    rep = s.get('first_same')
    if rep is None:
        return []
    stop = s.get('first_opp')
    window_end = rep['ts'] + args.add_window
    buckets = {}
    for x in s['buys']:
        if (x['ts'],x['line']) <= (rep['ts'],rep['line']):
            continue
        if x['ts'] > window_end:
            break
        if stop and (stop['ts'],stop['line']) <= (x['ts'],x['line']):
            break
        elapsed = x['ts'] - rep['ts']
        slot = max(1, (elapsed + args.min_add_gap - 1)//args.min_add_gap)
        action_ts = rep['ts'] + slot*args.min_add_gap
        if action_ts > window_end:
            continue
        if stop and stop['ts'] <= action_ts:
            continue
        if action_ts not in buckets:
            buckets[action_ts] = {'ts':action_ts,'wallet_size':0.,'line':x['line'],
                                  'wallet_fills':0}
        buckets[action_ts]['wallet_size'] += x['size']
        buckets[action_ts]['wallet_fills'] += 1
        buckets[action_ts]['line'] = max(buckets[action_ts]['line'],x['line'])
    return [buckets[k] for k in sorted(buckets)[:max_adds]]


def rounded_target(base_qty, cumulative_wallet, anchor_wallet):
    if not (anchor_wallet>0 and math.isfinite(anchor_wallet) and
            math.isfinite(cumulative_wallet) and cumulative_wallet>=anchor_wallet-1e-7):
        raise RuntimeError('Invalid proportional wallet sizing inputs')
    # Consistent round-half-up on TOTAL inventory, not on individual fills.
    return int(math.floor(base_qty*cumulative_wallet/anchor_wallet + .5 + 1e-10))


def build_proportional(sigs, by_start, strategy, max_adds, args):
    """Create causal proportional size changes; base always --base-qty."""
    result=[]
    exclusions=Counter()
    kinds=Counter()
    for s in sigs:
        confirmed=repeat_confirmed(s,args.min_repeat_age,args.max_repeat_age)
        if strategy=='repeat' and not confirmed:
            exclusions['not_a_repeat'] += 1
            continue
        first=wallet_first_fill(s,by_start)
        if first is None:
            exclusions['first_fill_size_join_missing'] += 1
            continue
        if args.base_qty>args.max_market_qty:
            exclusions['base_exceeds_market_cap'] += 1
            continue
        close=s['start']+900
        if strategy=='repeat':
            rep=s['first_same']
            starter_ts=rep['ts']
            starter_line=rep['line']
            # Our copied first position opens only when the second wallet BUY
            # has confirmed; its comparison anchor is the wallet's already-held
            # total at that time, not the very first fill alone.
            anchor=wallet_at_repeat(s,first,by_start)
            starter_kind='repeat'
        else:
            starter_ts=max(s['ts'],first['ts'])
            starter_line=first['line']
            anchor=first['size']
            starter_kind='first'
        if starter_ts>=close:
            exclusions['after_close'] += 1
            continue
        def record(kind,ts,line,qty,wallet_fills=1):
            result.append({'start':s['start'],'close':close,'ts':ts,
                           'side':s['side'],'action':'BUY','line':line,
                           'size':qty,'price':s['price'],'kind':kind,
                           'signal_id':s['start'],'market_first_price':s['price'],
                           'wallet_anchor_shares':anchor,
                           'wallet_fills_aggregated':wallet_fills})
            kinds[kind]+=1
        record(starter_kind,starter_ts,starter_line,args.base_qty)
        planned=args.base_qty
        if not confirmed:
            continue  # No hindsight: all first buys still enter, no later adds.
        rep=s['first_same']
        if strategy=='first':
            # At repeat time we observe all supporting same-side fills so far,
            # including same-second fragments and 0–3-second pre-repeat prints.
            cumulative=wallet_at_repeat(s,first,by_start)
            target=rounded_target(args.base_qty,cumulative,anchor)
            qty=target-planned
            if target>args.max_market_qty:
                exclusions['market_qty_cap_confirmation'] += 1
                continue
            if qty>0 and rep['ts']<close:
                record('repeat',rep['ts'],rep['line'],qty)
                planned=target
            elif qty<0:
                raise RuntimeError('Cumulative wallet shares cannot decrease')
            elif rep['ts']>=close:
                exclusions['after_close'] += 1
                continue
        else:
            cumulative=anchor
        for bucket in proportional_buckets(s,args,max_adds):
            cumulative += bucket['wallet_size']
            target=rounded_target(args.base_qty,cumulative,anchor)
            qty=target-planned
            if target>args.max_market_qty:
                exclusions['market_qty_cap_add'] += 1
                break
            if bucket['ts']>=close:
                exclusions['after_close'] += 1
                break
            if qty<=0:
                exclusions['rounded_add_pending'] += 1
                continue
            record('add',bucket['ts'],bucket['line'],qty,bucket['wallet_fills'])
            planned=target
    return sorted(result,key=lambda x:(x['ts'],x['start'],x['line'])),exclusions,kinds


def build(sigs, by_start, strategy, max_adds, args):
    """Make one independent strategy's proposed orders."""
    if args.sizing == 'proportional':
        return build_proportional(sigs, by_start, strategy, max_adds, args)
    result = []
    exclusions = Counter()
    kinds = Counter()
    for s in sigs:
        has_repeat = repeat_confirmed(s, args.min_repeat_age, args.max_repeat_age)
        if strategy == 'repeat' and not has_repeat:
            exclusions['not_a_repeat'] += 1
            continue
        first_size = initial_wallet_size(s, by_start)
        if first_size is None:
            exclusions['first_fill_size_join_missing'] += 1
            continue
        # A strategy cannot add to a position if its planned base order was
        # impossible even before execution (size cap or rounding).
        if strategy == 'first':
            base_qty = planned_qty('first', first_size, args)
        else:
            base_qty = planned_qty('repeat', s['first_same'], args)
        if base_qty is None or base_qty < 1 or base_qty > args.max_market_qty:
            exclusions['base_qty_unavailable'] += 1
            continue
        close = s['start'] + 900
        pending = []
        if strategy == 'first':
            pending.append(('first', s['ts'], first_size, -1))
        if has_repeat:
            rep = s['first_same']
            pending.append(('repeat', rep['ts'], rep, rep['line']))
            for x in extra_adds(s, args.min_add_gap, args.add_window, max_adds):
                pending.append(('add', x['ts'], x, x['line']))
        # FOR first-only control: build(strategy='first', confirm=False) through
        # separate use of args.extra_confirm in build variants below.
        running = 0
        for kind, ts, wallet_qty, line in pending:
            qty = planned_qty(kind, wallet_qty, args)
            if qty is None or qty < 1:
                exclusions['qty_rounds_to_zero'] += 1
                continue
            if running + qty > args.max_market_qty:
                exclusions['market_qty_cap'] += 1
                continue
            running += qty
            if ts >= close:
                exclusions['after_close'] += 1
                continue
            result.append({'start':s['start'], 'close':close, 'ts':ts,
                           'side':s['side'], 'action':'BUY', 'line':line,
                           'size':qty, 'price': s['price'], 'kind':kind,
                           'signal_id':s['start'], 'market_first_price':s['price']})
            kinds[kind] += 1
    return sorted(result,key=lambda x:(x['ts'],x['start'],x['line'])),exclusions,kinds


def make_variants(first_signals, repeat_signals, by_start, args):
    # Build repeat population and first population from the original matched
    # wallet loader. Follow-ups cannot change the first-entry population.
    variants = {}
    for count in args.add_counts:
        ev, bad, counts = build(repeat_signals, by_start, 'repeat', count, args)
        variants[f'REPEAT+{count}'] = (ev, bad, counts)

    # FIRST: baseline covers all first-price-qualified markets, never just the
    # future repeat subset. Full version adds at repeat if/when seen.
    first_only = []
    miss = Counter()
    for s in first_signals:
        size = initial_wallet_size(s, by_start)
        if size is None:
            miss['first_fill_size_join_missing'] += 1; continue
        qty = args.base_qty if args.sizing=='proportional' else planned_qty('first', size, args)
        if qty is None or qty < 1:
            miss['qty_rounds_to_zero'] += 1; continue
        if qty > args.max_market_qty:
            miss['market_qty_cap'] += 1; continue
        first_only.append({'start':s['start'], 'close':s['start']+900,
                           'ts':s['ts'], 'side':s['side'], 'action':'BUY',
                           'line':-1,'size':qty,'price':s['price'],
                           'kind':'first','signal_id':s['start'],
                           'market_first_price':s['price']})
    variants['FIRST_ONLY'] = (sorted(first_only,key=lambda x:(x['ts'],x['start'])),miss,Counter({'first':len(first_only)}))
    for count in args.add_counts:
        ev,bad,counts = build(first_signals,by_start,'first',count,args)
        variants[f'FIRST+CONFIRM+{count}'] = (ev,bad,counts)
    return variants


def attach_market(events, by_close):
    out = []
    rejected = Counter()
    for e in events:
        matches = by_close.get(e['close'], [])
        if len(matches) != 1:
            rejected['no_market' if not matches else 'ambiguous_market'] += 1
            continue
        m = matches[0]
        outcome = str(m.get('result','')).lower()
        if outcome not in ('yes','no'):
            rejected['invalid_result'] += 1;continue
        side = 'yes' if e['side']=='UP' else 'no'
        out.append({**e,'ticker':m['ticker'],'result':outcome,'kalshi_side':side})
    return out,rejected


def simulate_dependent(events, sim_args, delay, af, pp, pb):
    """Never scale in when required preceding copied entry did not execute.

    A failed first entry suppresses confirm + later adds; a failed repeat
    suppresses later adds. Rerun after pruning so excluded orders never
    reserve/consume scarce historical print volume. Return dropped count.
    """
    remaining = list(events)
    dropped = 0
    for _ in range(4):
        ex, markets, fail = af['simulate'](remaining, sim_args, delay, pp, pb)
        filled_keys={(e['ticker'],e['ts'],e['kind'],e['line']) for e in ex}
        by_ticker=defaultdict(list)
        for e in remaining:by_ticker[e['ticker']].append(e)
        must_drop=set()
        for ticker, rows in by_ticker.items():
            first = next((x for x in rows if x['kind']=='first'),None)
            repeat = next((x for x in rows if x['kind']=='repeat'),None)
            if first is not None:
                first_key=(ticker,first['ts'],first['kind'],first['line'])
                if first_key not in filled_keys:
                    must_drop.update((ticker,x['ts'],x['kind'],x['line'])
                                     for x in rows if x['kind']!='first')
                    continue
            if repeat is not None:
                repeat_key=(ticker,repeat['ts'],repeat['kind'],repeat['line'])
                if repeat_key not in filled_keys:
                    must_drop.update((ticker,x['ts'],x['kind'],x['line'])
                                     for x in rows if x['kind']=='add')
        if not must_drop:
            return ex,markets,fail,dropped
        next_events=[e for e in remaining if
                     (e['ticker'],e['ts'],e['kind'],e['line']) not in must_drop]
        dropped += len(remaining)-len(next_events)
        remaining=next_events
    raise RuntimeError('Execution dependency gating did not stabilize')


def run_price_stress(execs, markets, penalties, fee_func, fee_multiplier):
    """Replay entry penalty from executed fills without changing which fills matched."""
    base_pnl = sum(m['pnl'] for m in markets)
    base_cost = sum(m['buy_cost'] for m in markets)
    result = []
    for cents in penalties:
        delta = 0.
        for r in execs:
            old = r['copy_px']
            new = min(.9999,old+cents/100.)
            f_new = fee_func(new,r['copy_qty'],fee_multiplier)
            delta += (new-old)*r['copy_qty'] + f_new-r['fee']
        pnl = base_pnl - delta
        cost = base_cost + delta
        result.append((cents,pnl,cost,100*pnl/cost if cost>0 else float('nan')))
    return result


def metrics(markets, execs):
    pnl=sum(m['pnl'] for m in markets)
    cost=sum(m['buy_cost'] for m in markets)
    return {'pnl':pnl,'cost':cost,'roi':100*pnl/cost if cost else float('nan'),
            'markets':len(markets),'fills':len(execs),
            'first_fills':sum(x['kind']=='first' for x in execs),
            'repeat_fills':sum(x['kind']=='repeat' for x in execs),
            'add_fills':sum(x['kind']=='add' for x in execs)}


def compare(base,other):
    # Incremental costs and profits are paired on identical market windows.
    b={m['ticker']:m for m in base}
    o={m['ticker']:m for m in other}
    if set(b)!=set(o):
        raise RuntimeError('Compared strategies do not contain identical markets; aborted paired difference')
    dp=sum(o[k]['pnl']-b[k]['pnl'] for k in b)
    dc=sum(o[k]['buy_cost']-b[k]['buy_cost'] for k in b)
    return dp,dc,100*dp/dc if dc>0 else None


def bootstrap_increment(base,other,n,seed):
    if n<1: return None
    b={m['ticker']:m for m in base};o={m['ticker']:m for m in other}
    if set(b)!=set(o):return None
    by_day=defaultdict(list)
    for t,m in b.items():
        day=datetime.fromtimestamp(m['start'], timezone.utc).strftime('%Y-%m-%d')
        by_day[day].append((o[t]['pnl']-m['pnl'],o[t]['buy_cost']-m['buy_cost']))
    days=list(by_day)
    if len(days)<3:return None
    dayvals=[(sum(x for x,_ in by_day[k]),sum(y for _,y in by_day[k])) for k in days]
    rnd=random.Random(seed)
    sims=[]
    for _ in range(n):
        v=sum(dayvals[rnd.randrange(len(days))][0] for _ in days)
        sims.append(v)
    sims.sort()
    return sims[int(.025*n)],sims[min(n-1,int(.975*n))]


def self_test():
    a=SimpleNamespace(min_repeat_age=4,max_repeat_age=30,min_add_gap=4,
                      add_window=120,max_market_qty=50,sizing='fixed',
                      base_qty=10,add_qty=10,scale=.1,add_counts=[0,1,3])
    t=1000
    base={'start':900,'ts':t,'side':'UP','price':.31,
          'first_same':{'ts':1008,'line':3,'side':'UP','action':'BUY','size':75},
          'first_opp':{'ts':1033,'line':8,'side':'DOWN','action':'BUY','size':20},
          'prior_opposite':False,
          'buys':[{'ts':1008,'line':3,'size':75},
                  {'ts':1009,'line':4,'size':20},
                  {'ts':1012,'line':5,'size':40},
                  {'ts':1016,'line':6,'size':40},
                  {'ts':1035,'line':10,'size':70}]}
    fake=[{'start':900,'ts':t,'side':'UP','action':'BUY','price':.31,'size':20,'line':1}]
    assert repeat_confirmed(base,4,30)
    assert [x['ts'] for x in extra_adds(base,4,120,3)]==[1012,1016]
    ev, bad, kinds=build([base],index_wallet(fake),'repeat',3,a)
    assert [x['kind'] for x in ev]==['repeat','add','add'],(ev,bad)
    assert [x['size'] for x in ev]==[10,10,10]
    first=build([base],index_wallet(fake),'first',3,a)[0]
    assert [x['kind'] for x in first]==['first','repeat','add','add']
    no_rep=dict(base,first_same=None,first_opp=None,prior_opposite=False,buys=[])
    # First-all must still buy even when no future confirmation follows.
    vv=make_variants([base,no_rep],[base],index_wallet(fake),a)
    assert len(vv['FIRST_ONLY'][0])==2
    assert len(vv['FIRST+CONFIRM+3'][0])==5
    assert len(vv['REPEAT+3'][0])==3
    wrong=dict(base,first_opp={'ts':1004,'line':2})
    assert not repeat_confirmed(wrong,4,30)
    a.max_market_qty=20
    ev,_,_=build([base],index_wallet(fake),'repeat',3,a)
    assert len(ev)==2,'max market exposure cap must exclude further buys'
    assert run_price_stress([{'copy_px':.4,'copy_qty':10,'fee':.17}],
                           [{'pnl':5.83,'buy_cost':4.17}],
                           [0,1],lambda p,q,f: round(.07*q*p*(1-p),2),1)[1][1] < 5.83
    # Proportional: first 100 wallet shares -> 10 copied contracts.
    # Repeat wallet +50 -> +5 contracts in FIRST strategy; another
    # +20 and +30 in the next 4s bucket -> +5 more contracts.
    b=dict(base,first_same={'ts':1008,'line':3,'side':'UP','action':'BUY','size':50},
           first_opp=None,buys=[
               {'ts':1008,'line':3,'size':50},
               {'ts':1009,'line':4,'size':20},
               {'ts':1010,'line':5,'size':30},
           ])
    fw=[{'start':900,'ts':1000,'side':'UP','action':'BUY','price':.31,'size':100.,'line':1},
        {'start':900,'ts':1008,'side':'UP','action':'BUY','price':.34,'size':50.,'line':3},
        {'start':900,'ts':1009,'side':'UP','action':'BUY','price':.35,'size':20.,'line':4},
        {'start':900,'ts':1010,'side':'UP','action':'BUY','price':.35,'size':30.,'line':5}]
    a.sizing='proportional'; a.max_market_qty=70; a.base_qty=10
    f,bad,_=build([b],index_wallet(fw),'first',3,a)
    r,bad,_=build([b],index_wallet(fw),'repeat',3,a)
    assert [(x['kind'],x['size'],x['ts']) for x in f]==[
        ('first',10,1000),('repeat',5,1008),('add',5,1012)],f
    assert [(x['kind'],x['size'],x['ts']) for x in r]==[
        ('repeat',10,1008),('add',3,1012)],r
    assert rounded_target(10,125,100)==13
    # Two small additions should round on their total, not separately.
    b2=dict(b,first_opp={'ts':1011,'line':9})
    assert not proportional_buckets(b2,a,3), 'Opposite buy before bucket close must cancel add'
    b3=dict(b,buys=[b['first_same'], {'ts':1009,'line':4,'size':20},
                      {'ts':1013,'line':5,'size':30}])
    assert [x['wallet_size'] for x in proportional_buckets(b3,a,3)]==[20,30]
    a.max_market_qty=12
    f,bad,_=build([b],index_wallet(fw),'first',3,a)
    assert [(x['kind'],x['size']) for x in f]==[('first',10)] and bad['market_qty_cap_confirmation']==1
    # End-to-end OFFLINE execution test against independent print buckets.
    # No future print can be borrowed to execute at an earlier time.
    _,all_fills=refs()
    a.max_market_qty=70
    f,_,_=build([b],index_wallet(fw),'first',3,a)
    f=[{**e,'ticker':'KXBTC15M-TEST','kalshi_side':'yes','result':'yes'} for e in f]
    def prints(times):
        return all_fills['prep_prints']({'KXBTC15M-TEST':[
            {'t':float(t),'yes':.35,'taker':'yes','qty':100.,'id':str(t)}
            for t in times]})
    sim_args=SimpleNamespace(quantity_mode='wallet',scale=1.,qty=1,max_qty=None,
                             max_wait_s=0.,max_quote_age_s=2.,fee_multiplier=1.)
    copied,by_market,errors,dropped=simulate_dependent(
        f,sim_args,0,all_fills,prints([1000,1008,1012]),None)
    assert [(x['kind'],x['copy_qty']) for x in copied]==[
        ('first',10),('repeat',5),('add',5)],(copied,errors)
    assert by_market[0]['payout']==20 and dropped==0
    copied,by_market,errors,dropped=simulate_dependent(
        f,sim_args,0,all_fills,prints([1000,1012]),None)
    assert [x['kind'] for x in copied]==['first'] and dropped==1,(copied,errors,dropped)
    print('PASS: causal first/repeat, proportional wallet baseline, cumulative rounding, 4s aggregated adds, opposing stop, caps, fees, offline dependency-aware execution')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--wallet',type=Path)
    p.add_argument('--markets-file',type=Path)
    p.add_argument('--trades-file',type=Path)
    p.add_argument('--price-source',choices=['prints','book'],default='prints')
    p.add_argument('--book-root',type=Path)
    p.add_argument('--book-kind',choices=['bbo','book_snap_1s'],default='bbo')
    p.add_argument('--offline',action='store_true')
    p.add_argument('--latencies',nargs='+',type=int,default=[5,10,20])
    p.add_argument('--add-counts',nargs='+',type=int,default=[0,1,3,5])
    p.add_argument('--min-repeat-age',type=int,default=4)
    p.add_argument('--max-repeat-age',type=int,default=30)
    p.add_argument('--upper-first-price',type=float,default=.40)
    p.add_argument('--min-add-gap',type=int,default=4)
    p.add_argument('--add-window',type=int,default=120)
    p.add_argument('--max-market-qty',type=int,default=250)
    p.add_argument('--sizing',choices=['fixed','wallet','proportional'],default='fixed',
                   help='proportional = scale all later wallet additions relative to copied starter')
    p.add_argument('--base-qty',type=int,default=10)
    p.add_argument('--add-qty',type=int,default=10)
    p.add_argument('--scale',type=float,default=.1)
    p.add_argument('--max-wait-s',type=float,default=5.)
    p.add_argument('--max-quote-age-s',type=float,default=2.)
    p.add_argument('--fee-multiplier',type=float,default=1.)
    p.add_argument('--merge-gap-s',type=int,default=15)
    p.add_argument('--penalties-cents',nargs='+',type=float,default=[0,1,2,5])
    p.add_argument('--bootstrap',type=int,default=1000,
                   help='Day-cluster bootstrap for incremental P&L (not ROI)')
    args=p.parse_args()
    if args.self_test:
        self_test();return
    if (not args.add_counts or min(args.add_counts)<0 or max(args.add_counts)>50 or
        not args.latencies or min(args.latencies)<0 or max(args.latencies)>180 or
        args.base_qty<1 or args.add_qty<1 or args.scale<=0 or
        args.max_market_qty<1 or args.min_add_gap<1 or args.add_window<0 or
        args.min_repeat_age<4 or args.max_repeat_age<args.min_repeat_age or
        args.max_wait_s<0 or args.max_quote_age_s<0 or args.fee_multiplier<0 or
        args.bootstrap<0 or any(x<0 for x in args.penalties_cents)):
        raise SystemExit('Invalid strategy parameters')
    if args.offline and (args.markets_file is None or
        (args.price_source=='prints' and args.trades_file is None)):
        raise SystemExit('--offline needs --markets-file and --trades-file (prints mode)')
    if args.price_source=='book' and not args.book_root:
        raise SystemExit('Historical book mode requires --book-root')
    core,af=refs()
    # Reuse EXACT eligibility logic from prior selective study.
    sig_args=SimpleNamespace(wallet=args.wallet,strategy='first',upper_first_price=args.upper_first_price,
                             min_repeat_age=args.min_repeat_age,max_repeat_age=args.max_repeat_age)
    first_signals=core['extract_signals'](sig_args)
    sig_args.strategy='repeat'
    repeat_signals=core['extract_signals'](sig_args)
    # Verify the repeat markets are truly a subset of the first-buy population.
    by_first={x['start']:x for x in first_signals}
    if len(by_first)!=len(first_signals) or any(r['start'] not in by_first for r in repeat_signals):
        raise RuntimeError('Duplicate signal starts or repeat market absent from first cohort')
    wallet_events,meta=af['prepare_wallet'](args.wallet)
    by_start=index_wallet(wallet_events)
    variants=make_variants(first_signals,repeat_signals,by_start,args)
    all_events=[e for events,_,_ in variants.values() for e in events]
    assert all_events
    print('\n954B SELECTIVE BUY + CONFIRMATION + SAME-SIDE SCALE-IN (READ ONLY)')
    print('='*106)
    print(f'Raw wallet={meta["raw"]:,} across {meta["grouped_markets"]:,} markets; '
          f'first qualifying={len(first_signals)}; qualifying repeats={len(repeat_signals)}')
    print(f'At first buy: eligibility is 20c to <{args.upper_first_price*100:g}c. '
          f'After repeat: adds require {args.min_add_gap}s gap, <= {args.add_window}s after repeat, '
          'stop after opposing buy.')
    print(f'Sizing {args.sizing}, base={args.base_qty} contracts, add={args.add_qty}, '
          f'scale={args.scale}, market cap={args.max_market_qty}; '
          f'latencies={args.latencies}, add limits={args.add_counts}')
    if args.sizing=='proportional':
        print('PROPORTIONAL: subsequent target position = round(baseQty * wallet cumulative / wallet anchor).')
        print('  FIRST anchor = wallet first fill; REPEAT anchor = wallet total at confirmation.')
        print('  Extra buys are summed within causal, closed time buckets; cap overruns are skipped.')
    print('Kalshi metadata: local file' if args.markets_file else 'Loading Kalshi metadata (public GET)...',flush=True)
    unique_starts={e['start'] for e in all_events}
    stub=[{'start':s} for s in sorted(unique_starts)]
    markets=list(core['objrows'](args.markets_file)) if args.markets_file else core['markets_api'](stub)
    by_close=defaultdict(list)
    for m in markets:
        close=core['market_close'](m)
        if close is not None and str(m.get('ticker','')).startswith('KXBTC15M-'):
            by_close[round(close)].append(m)
    paired={}
    for label,(events,pruned,kinds) in variants.items():
        mapped,unmatched=attach_market(events,by_close)
        paired[label]=mapped
        print(f'  {label:<24} intended={len(events):4} matched={len(mapped):4} '
              f'markets={len({e["ticker"] for e in mapped}):3} kinds={dict(kinds)} '
              f'skipped={dict(pruned+unmatched)}')
    all_matched=[e for events in paired.values() for e in events]
    if not all_matched: raise RuntimeError('No Kalshi matches')
    # Download a single superset of print windows for all strategy variants.
    # Use all-fills loader's per-event merged windows and volume-consuming
    # replay, instead of fetching a different history per strategy.
    # The loader accepts these fields on args: latencies/max_wait_s/merge_gap_s,
    # offline/trades_file; the replay accepts sizing/fee/quote-age fields.
    if args.price_source=='prints':
        print('Fetching historical aggressive-buy prints once for all variants (GET only)...',flush=True)
        pp=af['prep_prints'](af['load_trade_prints'](all_matched,args,core));pb=None
    else:
        book=runpy.run_path(str(HERE/'w954_kalshi_historical_bbo_replay_readonly.py'))
        db,files,rows,bad=book['ingest'](args.book_root,args.book_kind,
                                           {e['ticker'] for e in all_matched})
        print(f'Historical book files={files} relevant rows={rows:,} malformed={bad}')
        pb=af['prep_books'](db);pp=None
    # Planned event.size stores exact intended copied contract size; using
    # quantity_mode=wallet, scale=1 prevents simulator from overwriting sizes.
    replay_args=SimpleNamespace(quantity_mode='wallet',scale=1.,qty=1,max_qty=None,
                                max_wait_s=args.max_wait_s,
                                max_quote_age_s=args.max_quote_age_s,
                                fee_multiplier=args.fee_multiplier)
    for delay in args.latencies:
        print(f'\nLATENCY {delay}s | {args.price_source.upper()} (trade prints are NOT buyable asks)')
        outcomes={}
        for name,ev in paired.items():
            ex,ms,reject,dependent_skips=simulate_dependent(ev,replay_args,delay,af,pp,pb)
            outcomes[name]=(ex,ms)
            info=metrics(ms,ex)
            stress=run_price_stress(ex,ms,args.penalties_cents,af['fee'],args.fee_multiplier)
            print(f'  {name:<24} N={info["markets"]:3} fills={info["fills"]:4}/{len(ev):4} '
                  f'first/repeat/add={info["first_fills"]}/{info["repeat_fills"]}/{info["add_fills"]} '
                  f'cost=${info["cost"]:9,.2f} PnL=${info["pnl"]:+9,.2f} ROI={info["roi"]:+7.2f}% '
                  f'reject={dict(reject)} dependent_skips={dependent_skips}')
            print('    Stress net ROI: ' + ' | '.join(f'+{p:g}c {r:+.2f}%' for p,_,_,r in stress))
        print('  PAIRED INCREMENTAL P&L / EXTRA CAPITAL (same underlying markets):')
        for name in list(outcomes):
            base='REPEAT+0' if name.startswith('REPEAT+') else 'FIRST_ONLY'
            if name==base:continue
            b=outcomes[base][1];o=outcomes[name][1]
            try: dp,dc,roi=compare(b,o)
            except RuntimeError as ex:
                print(f'    {name}: not paired: {ex}');continue
            ci=bootstrap_increment(b,o,args.bootstrap,42)
            cint=f' bootstrap day CI on incremental total PnL [${ci[0]:+.2f},${ci[1]:+.2f}]' if ci else ''
            print(f'    {name:<24} versus {base:<12} ΔPnL=${dp:+9.2f} '
                  f'Δcost=${dc:+9.2f} incremental ROI={f"{roi:+.2f}%" if roi is not None else "n/a"}' + cint)
        for base in ('REPEAT+0','FIRST_ONLY'):
            ex,ms=outcomes[base]
            weekly=defaultdict(lambda:[0.,0.,0])
            for m in ms:
                week=datetime.fromtimestamp(m['start'],timezone.utc).strftime('%G-W%V')
                weekly[week][0]+=m['pnl'];weekly[week][1]+=m['buy_cost'];weekly[week][2]+=1
            print('  Weekly',base,':',', '.join(f'{w} N={int(x[2])} ROI={100*x[0]/x[1]:+.1f}%'
                                                 for w,x in sorted(weekly.items()) if x[1]))
    print('\nCAUTIONS: Tests share the same historical wallet data and Kalshi official settlement.')
    print('No live copy/publication latency is known. Trade prints may not be executable.')
    print('Bootstrap is uncertainty of paired total PnL via trading-day resampling, not a guarantee.')
    print('No orders, file writes, or GitHub modifications. Historical public GETs only.')


if __name__ == '__main__':
    try: main()
    except (RuntimeError,ValueError,OSError,AssertionError) as e:
        sys.exit('ABORT: '+str(e))
