#!/usr/bin/env python3
"""READ ONLY: BTC hourly Polymarket 45/45 resting-bid fill feasibility.

Downloads PMData historical BTC-1h daily ZIPs. NO TRADES ARE PLACED.

Three deliberately different observations:
  touch          : outcome's best ask <= bid after the order activates. Not a real fill.
  sell_print     : any aggressor SELL trade at or below bid. Not enough for queue fill.
  queue_proxy    : sell prints at bid exhaust displayed queue ahead plus our size,
                   OR SELL trades through our limit. A trade-tape queue proxy,
                   NOT a verified exchange-side fill. Cancelled queue ahead and
                   missing prints make the proxy imperfect.

No look-ahead into future prices when establishing queue at arrival. Market hours
are parsed from Polymarket's BTC hourly slug using America/New_York. Output
includes eligibility, both/one/none by 15/30/45/60 min, and settlement P&L
when Polymarket resolution is present. Gross P&L excludes fees/rebates.

Setup: python3 -m pip install pandas pyarrow requests
       export PMDATA_API_KEY='...'
       python3 -u -B polymarket_btc1h_4545_backtest.py \\
          --start 2026-10-10 --end 2026-10-16
       python3 -B polymarket_btc1h_4545_backtest.py --self-test
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import os
import re
import sys
import tempfile
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Load project-root .env for Windows/Linux when running scripts directly.
# An already configured process environment takes priority.
try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise RuntimeError("Install .env loader: python -m pip install python-dotenv") from exc
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
UTC = timezone.utc
BASE = 'https://api.pmdata.dev/v1/polymarket'
MONTHS = ('january','february','march','april','may','june','july','august',
          'september','october','november','december')
DATE_PATTERN = re.compile(r'bitcoin-up-or-down-([a-z]+)-(\d{1,2})-(\d{4})-(\d{1,2})(am|pm)-et')


def slug_hour(dt_utc):
    et = dt_utc.astimezone(ET)
    hh = et.hour % 12 or 12
    ap = 'am' if et.hour < 12 else 'pm'
    return f'bitcoin-up-or-down-{MONTHS[et.month-1]}-{et.day}-{et.year}-{hh}{ap}-et'


def hour_from_slug(slug):
    m = DATE_PATTERN.fullmatch(slug)
    if not m:
        return None
    mon, dd, yy, hh, ap = m.groups()
    if mon not in MONTHS:
        return None
    hour = int(hh) % 12 + (12 if ap == 'pm' else 0)
    return datetime(int(yy), MONTHS.index(mon)+1, int(dd), hour, tzinfo=ET).astimezone(UTC)


def dates_between(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def dec_float(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (ValueError, TypeError):
        return None
    return f if math.isfinite(f) else None


def levels(values):
    if values is None or isinstance(values, float):
        return []
    if isinstance(values, (str, bytes)):
        return []
    try:
        return list(values)
    except TypeError:
        return []


def time_seconds(obj):
    # Pandas datetime Timestamp (datetime64[us] from PMData).
    if obj is None:
        return None
    if hasattr(obj, 'timestamp'):
        try:
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=UTC)
            return float(obj.timestamp())
        except (TypeError, ValueError):
            pass
    if isinstance(obj, (int,float)):
        val = float(obj)
        return val / (1e6 if val >= 1e14 else 1e3 if val >= 1e11 else 1)
    try:
        import pandas as pd
        x = pd.to_datetime(obj, utc=True, errors='coerce')
        return float(x.timestamp()) if not pd.isna(x) else None
    except Exception:
        return None


def filename_for(root, day, typ, outcome='Yes'):
    return root / f'{day.isoformat()}_{typ}_{outcome}.zip'


def download_one(session, key, day, typ, outcome, root, offline=False):
    dest = filename_for(root, day, typ, outcome)
    if dest.exists() and zipfile.is_zipfile(dest):
        return dest
    if offline:
        raise RuntimeError(f'Cache missing (offline): {dest}')
    url = f'{BASE}/{typ}/{day:%Y/%m/%d}/btc-1h.zip'
    if typ == 'l2' and outcome == 'No':
        url += '?outcome=No'
    print(f'FETCH {day} {typ} {outcome}', flush=True)
    try:
        with session.get(url, headers={'api_key': key}, timeout=(20, 180), stream=True) as r:
            if r.status_code != 200:
                details = r.text[:260].replace('\n',' ')
                raise RuntimeError(f'PMData HTTP {r.status_code} for {day} {typ} {outcome}: {details}')
            fd, tmp = tempfile.mkstemp(prefix='pmbtc_', suffix='.zip', dir=root)
            try:
                with os.fdopen(fd, 'wb') as f:
                    for part in r.iter_content(chunk_size=1024*1024):
                        if part:
                            f.write(part)
                if not zipfile.is_zipfile(tmp):
                    raise RuntimeError(f'Unexpected non-ZIP response for {day} {typ} {outcome}')
                os.replace(tmp, dest)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
    except Exception as exc:
        raise RuntimeError(f'Could not download {url}: {exc}') from exc
    return dest


class ZipReader:
    def __init__(self, cloud=None):
        self.zip_cache = {}
        self.cloud = cloud
        self.handles = {}

    def close(self):
        for z in self.handles.values():
            z.close()
        self.handles.clear()

    def read(self, zip_path, slug):
        import pandas as pd
        if self.cloud:
            if zip_path not in self.handles:
                self.handles[zip_path] = self.cloud.open_zip(zip_path)
            z = self.handles[zip_path]
        else:
            if not zip_path.exists():
                return None
            z = zipfile.ZipFile(zip_path)
        if zip_path not in self.zip_cache:
            names = [n for n in z.namelist() if n.lower().endswith('.parquet') and not n.startswith('__MACOSX')]
            mapping = {}
            for n in names:
                stem = Path(n).stem
                match = DATE_PATTERN.search(stem)
                if match:
                    mapping[match.group(0)] = n
            self.zip_cache[zip_path] = (mapping, names)
        mapping, names = self.zip_cache[zip_path]
        member = mapping.get(slug)
        if member is None:
            # Different bundle naming? Only scan a file as fallback if names have no slug.
            if not mapping and names:
                raise RuntimeError(f'Unknown ZIP filename format in {zip_path}: {names[:3]}')
            if not self.cloud:
                z.close()
            return None
        try:
            buf = io.BytesIO(z.read(member))
            return pd.read_parquet(buf)
        finally:
            if not self.cloud:
                z.close()


def concatenate_market(reader, root, slug, typ, outcome, dates, cloud=None):
    import pandas as pd
    parts = []
    for d in dates:
        name=filename_for(root,d,typ,outcome)
        part = reader.read(cloud.object_key('btc-1h',name.name) if cloud else name,slug)
        if part is not None and len(part):
            if 'market_slug' in part.columns:
                part = part[part.market_slug.astype(str) == slug]
            if len(part):
                parts.append(part)
    if not parts:
        return None
    # Day partitions and rows are already in replay order (DO NOT re-sort them).
    return pd.concat(parts, ignore_index=True) if len(parts)>1 else parts[0]


def book_audit(l2, start_ts, cutoff_ts, limit, size, max_age=120.0):
    """Arrival queue and first time best ask <= bid. Replays all observed book updates."""
    bid = {}; ask = {}
    state_at_start = None
    valid_book = False
    last_ts = None
    touch = None
    if l2 is None or not len(l2):
        return dict(eligible=False, reason='NO_BOOK', touch_ts=None)
    for r in l2.itertuples(index=False):
        ts = time_seconds(getattr(r,'timestamp',None))
        if ts is None or ts > cutoff_ts:
            continue
        ev = str(getattr(r, 'event_type', ''))
        if ev == 'book':
            bid = {}
            ask = {}
            for p,v in zip(levels(getattr(r,'bid_prices',None)), levels(getattr(r,'bid_sizes',None))):
                pf,vf=dec_float(p),dec_float(v)
                if pf is not None and vf is not None and vf > 0:
                    bid[round(pf,4)] = vf
            for p,v in zip(levels(getattr(r,'ask_prices',None)), levels(getattr(r,'ask_sizes',None))):
                pf,vf=dec_float(p),dec_float(v)
                if pf is not None and vf is not None and vf > 0:
                    ask[round(pf,4)] = vf
            valid_book = True
            last_ts = ts
        elif ev == 'price_change' and valid_book:
            p = dec_float(getattr(r,'pc_price',None))
            v = dec_float(getattr(r,'pc_size',None))
            side = str(getattr(r,'pc_side','')).upper()
            if p is not None and v is not None and side in ('BUY','SELL'):
                levels_map = bid if side=='BUY' else ask
                price = round(p,4)
                if v <= 0:
                    levels_map.pop(price,None)
                else:
                    levels_map[price] = v
                last_ts = ts
        if not valid_book:
            continue
        if ts <= start_ts:
            best_ask = min(ask) if ask else None
            ask_liq = sum(v for p,v in ask.items() if p <= limit+1e-9)
            state_at_start = {
                'entry_best_bid': max(bid) if bid else None,
                'entry_best_ask': best_ask,
                'entry_ask_available': ask_liq,
                'queue_ahead': bid.get(round(limit,4),0.0),
                'state_ts': last_ts,
            }
        elif touch is None and ask and min(ask) <= limit+1e-9:
            # Historical book movement is NOT itself proof our order filled.
            touch = ts

    if state_at_start is None:
        return dict(eligible=False, reason='NO_ASOF_BOOK', touch_ts=None)
    if state_at_start['state_ts'] is None or start_ts-state_at_start['state_ts'] > max_age:
        return dict(eligible=False,reason='STALE_ASOF_BOOK',touch_ts=None,**state_at_start)
    if state_at_start['entry_best_ask'] is None or state_at_start['entry_best_bid'] is None:
        return dict(eligible=False, reason='NO_TWO_SIDED_BOOK', touch_ts=None, **state_at_start)
    is_marketable = state_at_start['entry_best_ask'] <= limit+1e-9
    if is_marketable:
        if state_at_start['entry_ask_available'] >= size-1e-7:
            return dict(eligible=False,reason='MARKETABLE_AT_ENTRY',touch_ts=start_ts,
                        marketable_full=True,**state_at_start)
        return dict(eligible=False,reason='MARKETABLE_PARTIAL_AT_ENTRY',touch_ts=start_ts,
                    marketable_full=False,**state_at_start)
    return dict(eligible=True,reason='RESTING_OK',touch_ts=touch,marketable_full=False,**state_at_start)


def trade_audit(trades, side, start_ts, cutoff_ts, limit, size, queue_ahead):
    """Sell prints at bid and trade-through; does not model cancelled queue-ahead."""
    exact_volume=0.0
    first_print=None
    first_through=None
    queue_filled=None
    aliases = {'yes': {'yes', 'up'}, 'no': {'no', 'down'},
               'up': {'yes', 'up'}, 'down': {'no', 'down'}}
    match_outcomes = aliases[side.lower()]
    if trades is not None:
        for r in trades.itertuples(index=False):
            ts=time_seconds(getattr(r,'timestamp',None))
            if ts is None or ts<start_ts or ts>cutoff_ts:
                continue
            if str(getattr(r,'outcome','')).lower() not in match_outcomes:
                continue
            if str(getattr(r,'side','')).upper() != 'SELL':
                continue
            price=dec_float(getattr(r,'price',None))
            qty=dec_float(getattr(r,'size',None))
            if price is None or qty is None or qty<=0 or price>limit+1e-9:
                continue
            if first_print is None:
                first_print=ts
            if price < limit-1e-8:
                if first_through is None:
                    first_through=ts
            else:
                exact_volume += qty
                if queue_filled is None and exact_volume+1e-8>=queue_ahead+size:
                    queue_filled=ts
    events=[x for x in [queue_filled,first_through] if x is not None]
    return dict(sell_print_ts=first_print, trade_through_ts=first_through,
                queue_proxy_ts=min(events) if events else None,
                sell_volume_at_bid=exact_volume)


def resolution(l2):
    if l2 is None:
        return None
    for r in l2.itertuples(index=False):
        if str(getattr(r,'event_type',''))=='market_resolved':
            winner=str(getattr(r,'winning_outcome','')).lower()
            if winner in ('up','yes'):
                return 'Up'
            if winner in ('down','no'):
                return 'Down'
    return None


def classify_side(side_result, metric, t_deadline):
    if side_result.get('eligible'):
        ts=side_result.get(metric+'_ts')
    elif side_result.get('reason')=='MARKETABLE_AT_ENTRY':
        ts=side_result.get('touch_ts') if metric == 'touch' else None
    else:
        ts=None
    return ts is not None and ts <= t_deadline+1e-6


def summarize(rows, price, size, horizons):
    result={}
    for universe_name, members in (
        ('all_valid_books', [r for r in rows if r['up_book_ok'] and r['down_book_ok']]),
        ('both_resting', [r for r in rows if r['up_eligible'] and r['down_eligible']]),
    ):
        output={}
        for metric in ('touch','sell_print','queue_proxy'):
            out_h=[]
            for mins in horizons:
                both=one=none=0
                both_pnl=one_pnl=none_pnl=0.0
                scored=0
                for r in members:
                    deadline = r['t0'] + 60.0*mins
                    u = r.get(f'up_{metric}_ts')
                    d = r.get(f'down_{metric}_ts')
                    # Entry marketables fill immediately only in the mixed
                    # (non-post-only) universe; post-only maker subset excludes.
                    u_valid=r['up_eligible'] or (universe_name=='all_valid_books' and r.get('up_marketable_full',False))
                    d_valid=r['down_eligible'] or (universe_name=='all_valid_books' and r.get('down_marketable_full',False))
                    uf = u_valid and u is not None and u<=deadline
                    df = d_valid and d is not None and d<=deadline
                    fills=int(uf)+int(df)
                    if fills==2: both+=1
                    elif fills==1: one+=1
                    else: none+=1
                    winner=r.get('winner')
                    if winner in ('Up','Down'):
                        scored+=1
                        profit = (size*(1-price) if winner=='Up' else -size*price) if uf else 0.0
                        profit += (size*(1-price) if winner=='Down' else -size*price) if df else 0.0
                        if fills==2: both_pnl+=profit
                        elif fills==1: one_pnl+=profit
                        else: none_pnl+=profit
                out_h.append(dict(minutes=mins, markets=len(members), both=both, one=one,
                                  none=none, both_pct=round(100*both/len(members),2) if members else None,
                                  resolved=scored, gross_pnl=round(both_pnl+one_pnl+none_pnl,4),
                                  gross_pnl_both=round(both_pnl,4),gross_pnl_one=round(one_pnl,4)))
            output[metric]=out_h
        result[universe_name]=output
    return result


def command(args):
    try:
        import requests
        import pandas
        import pyarrow
    except ImportError as ex:
        raise SystemExit('Install dependencies: python3 -m pip install requests pandas pyarrow') from ex
    start=date.fromisoformat(args.start)
    end=date.fromisoformat(args.end)
    if end<start:
        raise SystemExit('--end must be >= --start')
    if (end-start).days>180:
        raise SystemExit('Please run 180-day or shorter batches for memory/cost control.')
    if args.price<=0 or args.price>=.5 or args.size<=0 or args.latency_ms<0:
        raise SystemExit('Invalid --price, --size, or --latency-ms')
    key=args.pmdata_key or os.getenv('PMDATA_API_KEY') or ''
    if not key and not args.offline:
        raise SystemExit('PMDATA_API_KEY is missing. Set env variable or use --pmdata-key (not recommended for shell history).')
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    cache=out/'cache'
    cloud=None
    if args.storage=='supabase':
        from pmdata_supabase_storage import SupabaseArchive, migrate_cache
        cloud=SupabaseArchive()
        if args.migrate_cache:
            migrate_cache(cloud,'btc-1h',cache)
    else:
        cache.mkdir(parents=True,exist_ok=True)
    # Day -1 warms up the book for midnight markets; +1 catches the last hour
    # and delayed resolution events. These are cached on disk and reused.
    download_days=list(dates_between(start-timedelta(days=1),end+timedelta(days=1)))
    sess=requests.Session()
    for day in download_days:
        for kind,outcome in (('l2','Yes'),('l2','No'),('trades','All')):
            if cloud:
                filename=filename_for(cache,day,kind,outcome).name
                url=f'{BASE}/{kind}/{day:%Y/%m/%d}/btc-1h.zip'
                if kind=='l2' and outcome=='No':
                    url += '?outcome=No'
                cloud.transfer_pmdata(sess,key,url,'btc-1h',filename,offline=args.offline)
            else:
                download_one(sess,key,day,kind,outcome,cache,args.offline)
    reader=ZipReader(cloud=cloud)
    price=round(args.price,4)
    latency=args.latency_ms/1000
    horizons=[15,30,45,60]
    rows=[]
    n=(end-start).days+1
    for ix,day in enumerate(dates_between(start,end),1):
        print(f'PROCESS {day} ({ix}/{n} days)',flush=True)
        for h in range(24):
            t0=datetime(day.year,day.month,day.day,h,tzinfo=UTC)
            slug=slug_hour(t0)
            # Reading previous UTC partition allows reconstructing the as-of
            # book when this market started trading before the selected hour.
            inspect_days=[day-timedelta(days=1),day,day+timedelta(days=1)]
            yes=concatenate_market(reader,cache,slug,'l2','Yes',inspect_days,cloud)
            no=concatenate_market(reader,cache,slug,'l2','No',inspect_days,cloud)
            trades=concatenate_market(reader,cache,slug,'trades','All',inspect_days,cloud)
            if yes is None or no is None:
                print(f'  MISSING L2 {slug} yes={yes is not None} no={no is not None}',flush=True)
                rows.append(dict(slug=slug,t0=t0.timestamp(),missing=True))
                continue
            arrive=t0.timestamp()+latency
            cutoff=t0.timestamp()+3600
            u=book_audit(yes,arrive,cutoff,price,args.size,args.max_book_age_seconds)
            d=book_audit(no,arrive,cutoff,price,args.size,args.max_book_age_seconds)
            winner=resolution(yes) or resolution(no)
            row=dict(slug=slug,t0=t0.timestamp(),missing=False,winner=winner,
                     up_book_ok=u['reason'] not in ('NO_BOOK','NO_ASOF_BOOK','NO_TWO_SIDED_BOOK','STALE_ASOF_BOOK'),
                     down_book_ok=d['reason'] not in ('NO_BOOK','NO_ASOF_BOOK','NO_TWO_SIDED_BOOK','STALE_ASOF_BOOK'),
                     up_eligible=u['eligible'],down_eligible=d['eligible'],
                     up_marketable_full=u.get('marketable_full',False),
                     down_marketable_full=d.get('marketable_full',False),
                     up_reason=u['reason'],down_reason=d['reason'],
                     up_queue_ahead=u.get('queue_ahead'),down_queue_ahead=d.get('queue_ahead'),
                     up_entry_ask=u.get('entry_best_ask'),down_entry_ask=d.get('entry_best_ask'))
            for name,book in (('up',u),('down',d)):
                row[f'{name}_touch_ts'] = book.get('touch_ts') if (book.get('eligible') or book.get('marketable_full')) else None
                if book.get('eligible'):
                    tape=trade_audit(trades,'Yes' if name=='up' else 'No',arrive,cutoff,price,args.size,
                                     book.get('queue_ahead',0.0))
                    for col,val in tape.items():
                        row[f'{name}_{col}']=val
                else:
                    for col in ('sell_print_ts','trade_through_ts','queue_proxy_ts','sell_volume_at_bid'):
                        row[f'{name}_{col}']=arrive if (book.get('marketable_full') and col.endswith('_ts')) else None
            rows.append(row)
    reader.close()
    valid=[r for r in rows if not r['missing']]
    summary=summarize(valid,price,args.size,horizons)
    summary['metadata']=dict(start=args.start,end=args.end,price=price,size=args.size,
                             arrival_latency_ms=args.latency_ms,max_book_age_seconds=args.max_book_age_seconds,
                             markets_expected=n*24,
                             markets_with_two_l2=len(valid),missing=n*24-len(valid),
                             both_resting=sum(r['up_eligible'] and r['down_eligible'] for r in valid),
                             marketable_on_either_side=sum('MARKETABLE' in r['up_reason'] or
                                                          'MARKETABLE' in r['down_reason'] for r in valid),
                             marketable_full_sides=sum(int(r['up_marketable_full'])+
                                                      int(r['down_marketable_full']) for r in valid),
                             book_invalid=sum(not (r['up_book_ok'] and r['down_book_ok']) for r in valid),
                             with_resolution=sum(r['winner'] is not None for r in valid),
                             note='Trade-tape queue proxy != confirmed maker fills; no fees or rebates.')
    with (out/'per_market.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader();writer.writerows(rows)
    with (out/'summary.json').open('w') as f:
        json.dump(summary,f,indent=2,allow_nan=False)
    if cloud:
        tag=f'{args.start}_{args.end}_p{round(price*100):02d}'
        cloud.put_result(out/'summary.json',f'research/polymarket/btc-1h/{tag}_summary.json')
        cloud.put_result(out/'per_market.csv',f'research/polymarket/btc-1h/{tag}_per_market.csv')
    print('\nBTC 1-HOUR 45/45, HISTORICAL, READ-ONLY')
    print(json.dumps(summary['metadata'],indent=2))
    for uni in ('both_resting','all_valid_books'):
        print(f'\n=== {uni.upper()} ===')
        for metric in ('touch','sell_print','queue_proxy'):
            print(f'  {metric}:')
            for row in summary[uni][metric]:
                print(f"    {row['minutes']:>2}m | N={row['markets']:>4} | BOTH={row['both']:>4} | "
                      f"ONE={row['one']:>4} | NONE={row['none']:>4} | "
                      f"both%={row['both_pct']} | settled={row['resolved']} | gross P&L={row['gross_pnl']:+.2f}")
    print('\nCAVEAT: ALL_VALID_BOOKS includes immediate marketable orders (taker, assumed full depth).')
    print('BOTH_RESTING is maker-only. Print/queue signals are not confirmed fills.')
    print(f'Gross P&L assumes {price*100:g}c fill, excludes fees, rebates, partial orders, latency drift.')
    print('\nFiles:',out/'summary.json','and',out/'per_market.csv')


def self_test():
    from types import SimpleNamespace
    from datetime import datetime
    def row(**kw): return SimpleNamespace(**kw)
    t=1731000000.0
    dt=datetime.fromtimestamp(t,UTC)
    assert hour_from_slug(slug_hour(dt))==dt.replace(minute=0,second=0,microsecond=0)
    # Queue 10 ahead and us 1 share. 10.9 shares sold at 45 is not enough,
    # a 0.2 share sale afterwards is enough.
    tapes=[row(timestamp=datetime.fromtimestamp(t+1,UTC),outcome='Yes',side='SELL',price=.45,size=10.9),
           row(timestamp=datetime.fromtimestamp(t+2,UTC),outcome='Yes',side='SELL',price=.45,size=.2)]
    class Frame:
        def __init__(self, rs): self.rs=rs
        def __len__(self):return len(self.rs)
        def itertuples(self,index=False):return iter(self.rs)
    r=trade_audit(Frame(tapes),'Yes',t,t+60,.45,1,10)
    assert r['sell_print_ts']==t+1 and r['queue_proxy_ts']==t+2
    assert trade_audit(Frame(tapes),'No',t,t+60,.45,1,10)['queue_proxy_ts'] is None
    assert trade_audit(Frame([row(timestamp=datetime.fromtimestamp(t+1,UTC), outcome='Up',side='SELL',price=.44,size=5)]),'Yes',t,t+60,.45,1,10)['queue_proxy_ts']==t+1
    book=[row(timestamp=datetime.fromtimestamp(t-2,UTC),event_type='book',
              bid_prices=[.45,.44],bid_sizes=[10,5],ask_prices=[.51,.52],ask_sizes=[9,10],
              pc_price=None,pc_size=None,pc_side=None),
          row(timestamp=datetime.fromtimestamp(t+3,UTC),event_type='price_change',
              bid_prices=None,bid_sizes=None,ask_prices=None,ask_sizes=None,
              pc_price=.44,pc_size=0,pc_side='BUY'),
          row(timestamp=datetime.fromtimestamp(t+4,UTC),event_type='price_change',
              bid_prices=None,bid_sizes=None,ask_prices=None,ask_sizes=None,
              pc_price=.45,pc_size=5,pc_side='SELL')]
    b=book_audit(Frame(book),t,t+60,.45,1)
    assert b['eligible'] and b['queue_ahead']==10 and b['touch_ts']==t+4, b
    assert book_audit(Frame(book),t,t+60,.45,1,max_age=1)['reason']=='STALE_ASOF_BOOK'
    book2=[row(timestamp=datetime.fromtimestamp(t-2,UTC),event_type='book',
               bid_prices=[.43],bid_sizes=[10],ask_prices=[.44],ask_sizes=[5],
               pc_price=None,pc_size=None,pc_side=None)]
    assert book_audit(Frame(book2),t,t+60,.45,1)['reason']=='MARKETABLE_AT_ENTRY'
    test=[dict(t0=t,up_book_ok=True,down_book_ok=True,up_eligible=True,
               down_eligible=True,up_queue_proxy_ts=t+3,down_queue_proxy_ts=t+15,
               up_touch_ts=t+3,down_touch_ts=t+15,up_sell_print_ts=t+3,
               down_sell_print_ts=t+15,winner='Up')]
    s=summarize(test,.45,1,[.1,.25,1.0])
    assert s['both_resting']['queue_proxy'][0]['one']==1
    assert s['both_resting']['queue_proxy'][1]['both']==1
    assert abs(s['both_resting']['queue_proxy'][1]['gross_pnl']-.1)<1e-10
    crossed=dict(test[0],up_eligible=False,up_marketable_full=True,up_queue_proxy_ts=t)
    assert summarize([crossed],.45,1,[1])['all_valid_books']['queue_proxy'][0]['both']==1
    assert summarize([crossed],.45,1,[1])['both_resting']['queue_proxy'][0]['markets']==0
    print('PASS: UTC/ET slug, no future book for queue, bid queue volume, dual leg timing, P&L')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start',default='2026-10-10',help='UTC start date, inclusive')
    p.add_argument('--end',default='2026-10-16',help='UTC end date, inclusive')
    p.add_argument('--price',type=float,default=.45)
    p.add_argument('--size',type=float,default=5,help='Full-share order quantity on each side (default 5)')
    p.add_argument('--latency-ms',type=float,default=250,help='After hourly boundary before order rests')
    p.add_argument('--max-book-age-seconds',type=float,default=120,help='Exclude stale opening book (default 120)')
    p.add_argument('--output-dir',default='data/research/poly_btc1h_4545')
    p.add_argument('--pmdata-key',default=None,help='Prefer PMDATA_API_KEY env variable')
    p.add_argument('--offline',action='store_true',help='Use already cached ZIPs')
    p.add_argument('--storage',choices=('supabase','local'),default='supabase',
                   help='Supabase Storage (default) streams archives without Codespace disk cache')
    p.add_argument('--migrate-cache',action='store_true',
                   help='Upload existing local cache ZIPs to Supabase before replay; never delete files')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:self_test()
    else:command(args)

if __name__=='__main__':
    main()
