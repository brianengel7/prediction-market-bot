#!/usr/bin/env python3
"""READ-ONLY Polymarket BTC 5-minute two-resting-45c-bid research.

Requires polymarket_btc1h_4545_backtest.py in the same directory, so the two
studies use IDENTICAL queue, price-touch, and P&L assumptions.

For a short sample, downloads the selected 5m markets individually (up to 3
Parquet files each). For a larger sample, use --source daily to download the
3 BTC-5m day bundles (Yes L2, No L2, and combined trades). Downloaded files
are cached; interrupted studies can be resumed.

An order is submitted at 5-minute market opening plus --latency-ms. The proxy
models a post-only bid of --price for --size shares on both Up and Down. Its
"queue_proxy" counts aggressor SELL volume at the bid needed to consume the
book's initial displayed queue ahead of our order, or any SELL trade THROUGH
our bid. This is NOT a confirmed fill; no cancellation/priority modeling,
partial fills, maker rebates, or fees. Do not trade based on this proxy alone.

Example:
  python3 -B polymarket_btc5m_4545_backtest.py --self-test
  python3 -u -B polymarket_btc5m_4545_backtest.py \\
      --date 2026-10-12 --start-hour 12 --hours 2 --price .45 \\
      --size 5 --latency-ms 250
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import os
import re
import sys
import tempfile
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Reuse the already-tested hourly queue and P&L machinery: same definitions,
# but 5m market timing, formats, and cached data paths are handled here.
try:
    from polymarket_btc1h_4545_backtest import book_audit, trade_audit, resolution, summarize
except ImportError as ex:
    raise SystemExit(
        'Put polymarket_btc5m_4545_backtest.py alongside '
        'polymarket_btc1h_4545_backtest.py under src/crypto/research/'
    ) from ex

UTC = timezone.utc
BASE = 'https://api.pmdata.dev/v1/polymarket'
SLUG_PATTERN = re.compile(r'btc-updown-5m-\d{10}')


def slots(day: date, start_hour: int, hours: int):
    base = datetime(day.year, day.month, day.day, start_hour, tzinfo=UTC)
    for step in range(hours * 12):
        dt = base + timedelta(minutes=5 * step)
        yield f'btc-updown-5m-{int(dt.timestamp())}', dt.timestamp()


def read_json_array(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            return []
    return x if isinstance(x, list) else []


def gamma_resolution(session, slug):
    """Optional fallback; requires unequivocal 1/0 final settlement prices."""
    try:
        r = session.get(f'https://gamma-api.polymarket.com/markets/slug/{slug}', timeout=15)
        if r.status_code != 200:
            return None
        m = r.json()
        if isinstance(m, list):
            m = m[0] if m else {}
        if not isinstance(m, dict) or m.get('closed') is not True:
            return None
        outcomes = read_json_array(m.get('outcomes'))
        prices = read_json_array(m.get('outcomePrices'))
        if len(outcomes) != 2 or len(prices) != 2:
            return None
        vals = [float(v) for v in prices]
        if not (sorted(vals) == [0.0, 1.0]):
            return None
        return {'up': 'Up', 'yes': 'Up', 'down': 'Down', 'no': 'Down'}.get(
            str(outcomes[vals.index(1.0)]).lower()
        )
    except (ValueError, TypeError, OSError, TimeoutError):
        return None
    except Exception:
        return None


def fetch(session, key: str, url: str, target: Path, offline: bool, allow_404: bool = False):
    """Atomic download with safe reuse. No retries on quota/auth failures."""
    if target.exists() and target.stat().st_size > 0:
        return target
    missing_marker = target.with_name(target.name + '.notfound')
    if missing_marker.exists():
        return None
    if offline:
        raise RuntimeError(f'Offline mode: cache file missing {target}')
    print(f'FETCH {target.name}', flush=True)
    with session.get(url, headers={'api_key': key}, stream=True, timeout=(20, 240)) as response:
        if response.status_code == 404 and allow_404:
            missing_marker.write_text('404\n')
            return None
        if response.status_code != 200:
            detail = response.text[:350].replace('\n', ' ')
            raise RuntimeError(f'PMData HTTP {response.status_code} for {url}: {detail}')
        fd, tmp = tempfile.mkstemp(prefix='.pm5m-', suffix='.part', dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as out:
                for chunk in response.iter_content(1024*1024):
                    if chunk:
                        out.write(chunk)
            if target.suffix == '.zip' and not zipfile.is_zipfile(tmp):
                raise RuntimeError(f'Corrupt ZIP response for {target}')
            if target.suffix == '.parquet':
                with open(tmp, 'rb') as f:
                    if f.read(4) != b'PAR1':
                        raise RuntimeError(f'Non-Parquet response for {target}')
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return target


def get_parquet_market(session, key, cache: Path, slug: str, day: date,
                       kind: str, outcome: str, offline: bool):
    import pandas as pd
    dest = cache / f'{slug}_{kind}_{outcome}.parquet'
    url = f'{BASE}/{kind}/{day:%Y/%m/%d}/{slug}.parquet'
    if kind == 'l2' and outcome == 'No':
        url += '?outcome=No'
    file = fetch(session,key,url,dest,offline,allow_404=True)
    if file is None:
        return None
    df = pd.read_parquet(file)
    if 'market_slug' in df.columns:
        df = df[df.market_slug.astype(str).eq(slug)]
    return df if len(df) else None


class DailyZipReader:
    def __init__(self):
        self.members = {}

    def read(self, zip_path: Path, slug: str):
        import pandas as pd
        if zip_path not in self.members:
            with zipfile.ZipFile(zip_path) as z:
                mapping = {}
                for fn in z.namelist():
                    if not fn.lower().endswith('.parquet') or fn.startswith('__MACOSX'):
                        continue
                    m = SLUG_PATTERN.search(fn)
                    if m:
                        mapping[m.group()] = fn
                self.members[zip_path] = mapping
        filename = self.members[zip_path].get(slug)
        if filename is None:
            return None
        with zipfile.ZipFile(zip_path) as z:
            df = pd.read_parquet(io.BytesIO(z.read(filename)))
        if 'market_slug' in df.columns:
            df = df[df.market_slug.astype(str).eq(slug)]
        return df if len(df) else None


def make_row(slug,t0,up,down,trades,winner,price,size,latency,max_age):
    arrival = t0 + latency
    cutoff = t0 + 300.0
    if up is None or down is None:
        return dict(slug=slug,t0=t0,winner=winner,missing=True)
    u = book_audit(up, arrival, cutoff, price, size, max_age)
    d = book_audit(down, arrival, cutoff, price, size, max_age)
    invalid = ('NO_BOOK','NO_ASOF_BOOK','NO_TWO_SIDED_BOOK','STALE_ASOF_BOOK')
    row=dict(slug=slug,t0=t0,missing=False,winner=winner,
             up_book_ok=u['reason'] not in invalid,
             down_book_ok=d['reason'] not in invalid,
             up_eligible=u['eligible'],down_eligible=d['eligible'],
             up_marketable_full=u.get('marketable_full',False),
             down_marketable_full=d.get('marketable_full',False),
             up_reason=u['reason'],down_reason=d['reason'],
             up_queue_ahead=u.get('queue_ahead'),down_queue_ahead=d.get('queue_ahead'),
             up_entry_bid=u.get('entry_best_bid'),down_entry_bid=d.get('entry_best_bid'),
             up_entry_ask=u.get('entry_best_ask'),down_entry_ask=d.get('entry_best_ask'))
    for label,book in [('up',u),('down',d)]:
        row[f'{label}_touch_ts'] = book.get('touch_ts') if (
            book.get('eligible') or book.get('marketable_full')) else None
        if book.get('eligible'):
            tape = trade_audit(trades,'Yes' if label=='up' else 'No',
                               arrival,cutoff,price,size,book.get('queue_ahead',0.0))
            for key,val in tape.items():
                row[f'{label}_{key}']=val
        else:
            for field in ('sell_print_ts','trade_through_ts','queue_proxy_ts','sell_volume_at_bid'):
                row[f'{label}_{field}'] = arrival if (
                    book.get('marketable_full') and field.endswith('_ts')) else None
    return row


def backtest(args):
    try:
        import requests
        import pandas as pd
        import pyarrow
    except ImportError as exc:
        raise SystemExit('Run python3 -m pip install requests pandas pyarrow') from exc
    day = date.fromisoformat(args.date)
    if not 0 <= args.start_hour <= 23 or args.hours < 1 or args.start_hour + args.hours > 24:
        raise SystemExit('Use --start-hour 0..23 and --hours 1..24 within one UTC day')
    if not (0 < args.price < 0.50) or args.size<=0 or args.latency_ms<0:
        raise SystemExit('Require 0 < price < 0.50, size > 0, latency-ms >= 0')
    key=os.getenv('PMDATA_API_KEY','')
    if not key and not args.offline:
        raise SystemExit('PMDATA_API_KEY missing from your Codespace environment')
    out=Path(args.output_dir)
    cache=out/'cache'
    out.mkdir(parents=True,exist_ok=True)
    cache.mkdir(parents=True,exist_ok=True)
    source=args.source
    if source == 'auto':
        source = 'individual' if args.hours <= 6 else 'daily'
    print(f'BTC 5m sample: {args.date} {args.start_hour:02d}:00–'
          f'{args.start_hour+args.hours:02d}:00 UTC '
          f'({args.hours*12} markets); source={source}; price={args.price:.2f}',flush=True)
    session=requests.Session()
    files={}
    zipreader=DailyZipReader()
    if source=='daily':
        for kind,outcome in (('l2','Yes'),('l2','No'),('trades','All')):
            file=cache/f'{day}_{kind}_{outcome}_btc-5m.zip'
            url=f'{BASE}/{kind}/{day:%Y/%m/%d}/btc-5m.zip'
            if kind=='l2' and outcome=='No':
                url += '?outcome=No'
            files[kind,outcome]=fetch(session,key,url,file,args.offline)
    rows=[]
    for i,(slug,t0) in enumerate(slots(day,args.start_hour,args.hours),1):
        print(f'[{i:>3}/{args.hours*12}] {slug}',flush=True)
        def load(kind,outcome):
            if source=='daily':
                return zipreader.read(files[kind,outcome],slug)
            return get_parquet_market(session,key,cache,slug,day,kind,outcome,args.offline)
        up=load('l2','Yes')
        down=load('l2','No')
        trades=load('trades','All') if up is not None and down is not None else None
        winner=resolution(up) or resolution(down)
        if winner is None and args.gamma_fallback:
            winner=gamma_resolution(session,slug)
        r=make_row(slug,t0,up,down,trades,winner,round(args.price,4),args.size,
                   args.latency_ms/1000,args.max_book_age_seconds)
        rows.append(r)
        if r['missing']:
            print('  Missing Yes or No L2; market excluded',flush=True)
    valid=[r for r in rows if not r['missing']]
    horizons=[1,2,3,4,5]
    result=summarize(valid,round(args.price,4),args.size,horizons)
    result['metadata'] = dict(
        date=args.date,start_hour_utc=args.start_hour,hours=args.hours,source=source,
        price=args.price,size=args.size,latency_ms=args.latency_ms,
        expected_markets=len(rows),valid_l2_markets=len(valid),
        missing_markets=len(rows)-len(valid),
        both_resting=sum(r['up_eligible'] and r['down_eligible'] for r in valid),
        with_resolution=sum(r.get('winner') in ('Up','Down') for r in valid),
        caveat='Prints/queue estimates are not live maker fills; no fees, rebates, or partial fills.'
    )
    # Distinct filenames per test, so independent runs cannot silently overwrite.
    tag=f'{args.date}_{args.start_hour:02d}h_{args.hours:02d}h_p{round(args.price*100):02d}'
    csvfile=out/f'per_market_{tag}.csv'
    jsonfile=out/f'summary_{tag}.json'
    with csvfile.open('w',newline='') as f:
        fields=sorted({k for r in rows for k in r})
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    jsonfile.write_text(json.dumps(result,indent=2,allow_nan=False))
    print('\n=== BTC 5m two resting bids, price %.2f ==='%args.price, flush=True)
    print(json.dumps(result['metadata'],indent=2))
    for universe in ('both_resting','all_valid_books'):
        print(f'\n{universe.upper()}')
        for metric in ('touch','sell_print','queue_proxy'):
            for e in result[universe][metric]:
                print(f"  {metric:11s} {e['minutes']}m: both={e['both']:>3} "
                      f"one={e['one']:>3} none={e['none']:>3} "
                      f"settled={e['resolved']:>3} gross=${e['gross_pnl']:+.2f} "
                      f"(both=${e['gross_pnl_both']:+.2f}, one=${e['gross_pnl_one']:+.2f})")
    if result['metadata']['with_resolution'] < len(valid):
        print('WARNING: Some markets have no recorded resolution; P&L excludes those markets. '
              'Rerun with --gamma-fallback or find missing resolutions before judging ROI.')
    print('READ ONLY; queue_proxy assumes all sell prints at bid consume initial queue '
          'without cancellation or replacement; P&L assumes full fills; no fees/rebates.')
    print('Saved:',csvfile,'and',jsonfile,flush=True)


def self_test():
    import pandas as pd
    from polymarket_btc1h_4545_backtest import self_test as common_self_test
    common_self_test()
    day=date(2026,10,12)
    s=list(slots(day,12,2))
    assert len(s)==24
    assert s[0][0]=='btc-updown-5m-'+str(int(datetime(2026,10,12,12,tzinfo=UTC).timestamp()))
    assert s[-1][1]-s[0][1]==23*300
    def l2(rows):
        return pd.DataFrame(rows)
    t=s[0][1]
    def book(ts):
        return dict(timestamp=datetime.fromtimestamp(ts,UTC),event_type='book',
            bid_prices=[.44,.43],bid_sizes=[5,12],ask_prices=[.55,.56],ask_sizes=[7,13],
            pc_price=None,pc_size=None,pc_side=None,winning_outcome=None)
    def resolved(ts, winner='Up'):
        return dict(timestamp=datetime.fromtimestamp(ts,UTC),event_type='market_resolved',
            winning_outcome=winner,bid_prices=None,bid_sizes=None,
            ask_prices=None,ask_sizes=None,pc_price=None,pc_size=None,pc_side=None)
    u=l2([book(t-2),resolved(t+310)])
    d=l2([book(t-2),resolved(t+310)])
    trades=pd.DataFrame([
        dict(timestamp=datetime.fromtimestamp(t+15,UTC),outcome='Up',side='SELL',price=.44,size=7),
        dict(timestamp=datetime.fromtimestamp(t+30,UTC),outcome='Down',side='SELL',price=.44,size=7),
    ])
    assert resolution(u)=='Up'
    r=make_row(s[0][0],t,u,d,trades,'Up',.45,5,.25,120)
    assert r['up_eligible'] and r['down_eligible']
    assert r['up_queue_proxy_ts']==t+15 and r['down_queue_proxy_ts']==t+30
    a=summarize([r],.45,5,[1])['both_resting']['queue_proxy'][0]
    assert a['both']==1 and abs(a['gross_pnl']-.5)<1e-9, a
    # One filled losing side means negative P&L.
    r2=dict(r,up_queue_proxy_ts=None)
    a=summarize([r2],.45,5,[1])['both_resting']['queue_proxy'][0]
    assert a['one']==1 and abs(a['gross_pnl']+2.25)<1e-9, a
    print('PASS: BTC 5m slug, sampling, real 5m expiry, paired/one-leg P&L')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--date',default='2026-10-12',help='UTC date')
    p.add_argument('--start-hour',type=int,default=12,help='UTC hour (0–23)')
    p.add_argument('--hours',type=int,default=2,help='Hours to test within UTC date (default 2)')
    p.add_argument('--price',type=float,default=.45)
    p.add_argument('--size',type=float,default=5)
    p.add_argument('--latency-ms',type=float,default=250)
    p.add_argument('--max-book-age-seconds',type=float,default=120)
    p.add_argument('--source',choices=('auto','individual','daily'),default='auto',
                   help='Individual market Parquets <=6h; daily ZIP >6h')
    p.add_argument('--gamma-fallback',action='store_true',
                   help='Use Gamma only to resolve markets missing L2 market_resolved event')
    p.add_argument('--offline',action='store_true',help='Rerun from cached files only')
    p.add_argument('--output-dir',default='data/research/poly_btc5m_4545')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    self_test() if args.self_test else backtest(args)


if __name__ == '__main__':
    main()
