#!/usr/bin/env python3
"""Read-only four-wallet Polymarket v2 history and apples-to-apples comparison.

Public HTTP GET only. Writes reports under ~/poly_wallet_comparison_20261105
by default (outside your Git repository). DOES NOT place trades, use credentials,
or modify GitHub/source files.

Usage:
    python3 /tmp/poly_wallet_compare_readonly.py
    python3 /tmp/poly_wallet_compare_readonly.py --start 2026-10-06 --end 2026-11-05T19:00:00Z
    python3 /tmp/poly_wallet_compare_readonly.py --max-pages 2000 --out ~/poly_wallet_comparison_20261105
"""

import argparse
import csv
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = 'https://data-api.polymarket.com/v2'
WALLETS = {
    'tokor': '0xa19cbababc312f9df185e49d7004c249ed1ade6b',
    'asdcaxz': '0x5a06acbaee263540a16a0109d584f6672101ffa2',
    'wallet_954b': '0x954b91dd68eca3aa4ea603ecea83c01e8539c0e0',
    'previous_btc': '0xf705fa045201391d9632b7f3cde06a5e24453ca7',
}
DEFAULT_START = '2026-10-06T00:00:00Z'
DEFAULT_END = '2026-11-05T19:00:00Z'
LOG = logging.getLogger('wallet_compare')

CRYPTO = re.compile(r'\b(bitcoin|btc|ethereum|ether|eth|solana|sol|dogecoin|doge|xrp|ripple|bnb|cardano|ada|litecoin|ltc|avalanche|avax|crypto|cryptocurrency|memecoin|sui|hyperliquid|hype|coinbase|binance)\b', re.I)
SPORTS = re.compile(r'\b(nba|wnba|nfl|nhl|mlb|ufc|f1|formula 1|soccer|football|baseball|basketball|hockey|tennis|premier league|champions league|la liga|serie a|bundesliga|world cup|atp|wta|ncaa|ncaaf|ncaab|pga|golf|cricket|rugby|esports|valorant|counter-strike|cs2|lakers|celtics|knicks|warriors|bulls|heat|mavericks|nuggets|thunder|cavaliers|raptors|hawks|bucks|spurs|sixers|76ers|pistons|nets|clippers|suns|magic|pacers|pelicans|timberwolves|grizzlies|rockets|jazz|trail blazers|hornets|wizards|kings|bruins|rangers|maple leafs|canadiens|flyers|penguins|red wings|blackhawks|oilers|avalanche|lightning|hurricanes|panthers|stars|devils|islanders|sharks|kraken|wild|sabres|senators|capitals|predators|ducks|flames|canucks|blue jackets|golden knights|yankees|mets|dodgers|giants|red sox|astros|phillies|padres|mariners|cowboys|chiefs|eagles|49ers|packers|ravens|bills|bengals|steelers|dolphins|jets|bears|vikings|patriots|commanders|texans|seahawks|saints|buccaneers|falcons|cardinals|browns|broncos|colts|jaguars|titans)\b', re.I)
WEATHER = re.compile(r'\b(temperature|temperatures|weather|rainfall|rain|snowfall|snow|hurricane|tropical storm|wind speed|highest temp|high temp|degrees fahrenheit|degrees celsius)\b', re.I)


def category(row):
    s = ' '.join(str(row.get(k) or '') for k in ('title', 'slug', 'event_slug'))
    # Temperature and precipitation contracts are prioritized over sports names
    # in weather event slugs; deliberately err toward unknown when ambiguous.
    if WEATHER.search(s):
        return 'weather'
    if SPORTS.search(s):
        return 'sports'
    if CRYPTO.search(s):
        return 'crypto'
    return 'other'


def epoch(s):
    if s.isdigit():
        return int(s)
    dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def utc(ts):
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat(timespec='seconds')


def num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


class Client:
    def __init__(self, min_delay=0.3):
        self.min_delay = min_delay
        self.last = 0.0
        self.requests = 0

    def get(self, path, params=None):
        url = API + path + ('?' + urlencode(params) if params else '')
        for attempt in range(8):
            elapsed = time.monotonic() - self.last
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
            self.last = time.monotonic()
            request = Request(url, headers={
                'Accept': 'application/json',
                'User-Agent': 'polymarket-public-wallet-study/1.0',
            })
            try:
                with urlopen(request, timeout=35) as response:
                    self.requests += 1
                    payload = json.load(response)
                if not isinstance(payload, dict) or 'data' not in payload:
                    raise ValueError(f'Malformed v2 API response from {path}: missing data envelope')
                return payload
            except HTTPError as err:
                if err.code not in (429, 500, 502, 503, 504) or attempt == 7:
                    raise RuntimeError(f'API HTTP {err.code} on {path}: {err.reason}') from err
                retry = num(err.headers.get('Retry-After'))
                pause = max(0.5, min(60.0, retry if retry is not None else 1.8**attempt))
                LOG.warning('API HTTP %d; retrying after %.1fs', err.code, pause)
                time.sleep(pause)
            except (URLError, TimeoutError) as err:
                if attempt == 7:
                    raise RuntimeError(f'API network failure on {path}: {err}') from err
                time.sleep(min(20.0, 1.8**attempt))
        raise RuntimeError('Unreachable')

    def pages(self, path, base_params, max_pages):
        """Preserves every filter across cursor pages; fails closed on truncation."""
        cursor = None
        observed_cursors = set()
        for page_no in range(max_pages):
            params = dict(base_params)
            if cursor:
                params['cursor'] = cursor
            payload = self.get(path, params)
            rows = payload['data']
            if not isinstance(rows, list):
                raise ValueError(f'{path}: paginated data is not a list')
            yield rows
            pagination = payload.get('pagination') or {}
            nxt = pagination.get('next_cursor')
            if not nxt:
                return
            if nxt == cursor or nxt in observed_cursors:
                raise ValueError(f'{path}: repeated pagination cursor')
            observed_cursors.add(nxt)
            cursor = nxt
        raise RuntimeError(f'{path}: hit --max-pages {max_pages}; incomplete history, DO NOT RANK.')


def collect(client, path, params, max_pages, jsonl_path, wallet, *, expect_user=True):
    row_count, page_count = 0, 0
    with jsonl_path.open('w', encoding='utf-8') as fout:
        for page in client.pages(path, params, max_pages):
            page_count += 1
            for item in page:
                if not isinstance(item, dict):
                    raise ValueError('Malformed API entry')
                if expect_user and str(item.get('proxy_wallet') or '').lower() != wallet.lower():
                    raise ValueError('Wallet filter broken or lost during pagination!')
                fout.write(json.dumps(item, separators=(',', ':'), ensure_ascii=False) + '\n')
                row_count += 1
            if page_count % 10 == 0:
                LOG.info('%s %s: %d pages / %d rows', wallet[:10], path, page_count, row_count)
    return row_count


def jsonl_iter(path):
    with path.open(encoding='utf-8') as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def pnl_difference(points, start, end):
    """Approx. endpoint cumulative deltas, only if baseline and endpoint exist."""
    valid = [p for p in points if isinstance(p, dict) and num(p.get('timestamp')) is not None]
    valid.sort(key=lambda p: float(p['timestamp']))
    before = [p for p in valid if float(p['timestamp']) <= start]
    until = [p for p in valid if float(p['timestamp']) <= end]
    if not before or not until or until[-1]['timestamp'] < before[-1]['timestamp']:
        return {'available': False, 'reason': 'No baseline at or before window start, or no end point'}
    a, b = before[-1], until[-1]
    out = {'available': True, 'baseline_at': utc(a['timestamp']), 'end_at': utc(b['timestamp'])}
    for field in ('economic_pnl', 'realized_pnl', 'realized_market_pnl', 'realized_combo_pnl',
                  'position_pnl', 'unrealized_pnl', 'volume_usdc', 'trade_count', 'fees_paid'):
        va, vb = num(a.get(field)), num(b.get(field))
        out[field] = vb - va if va is not None and vb is not None else None
    return out


def fnum(x, digits=2):
    return ('N/A' if x is None else f'{x:,.{digits}f}')


def summarize(trades_file, activity_file, wallet, start, end):
    rows = list(jsonl_iter(trades_file))
    rows.sort(key=lambda t: (num(t.get('timestamp')) or 0, str(t.get('transaction_hash') or '')))
    buckets, market_groups = Counter(), defaultdict(lambda: {'title': '', 'category': '', 'fills': 0, 'notional': 0.0})
    lots = defaultdict(deque)
    holds, matched_qty = [], 0.0
    matched_gross_pnl, sell_missing_qty = 0.0, 0.0
    b_notional, s_notional, active_days = 0., 0., set()
    buys, sells, bad = 0, 0, 0
    for t in rows:
        if str(t.get('proxy_wallet') or '').lower() != wallet.lower():
            raise ValueError('Unexpected wallet in trade file')
        ts, size, price = num(t.get('timestamp')), num(t.get('size')), num(t.get('price'))
        action = str(t.get('side') or '').upper()
        token = str(t.get('token_id') or '')
        market = str(t.get('condition_id') or token or 'unknown')
        if ts is None or not (start <= ts <= end) or size is None or price is None or size <= 0 or not (0 <= price <= 1) or action not in ('BUY', 'SELL') or not token:
            bad += 1
            continue
        c = category(t)
        notion = size * price
        buckets[c] += notion
        g = market_groups[market]
        g['category'], g['title'], g['fills'], g['notional'] = c, str(t.get('title') or t.get('slug') or market), g['fills'] + 1, g['notional'] + notion
        active_days.add(utc(ts)[:10])
        if action == 'BUY':
            buys += 1
            b_notional += notion
            lots[token].append([size, price, ts])
        else:
            sells += 1
            s_notional += notion
            remaining = size
            while remaining > 1e-8 and lots[token]:
                lot = lots[token][0]
                matched = min(lot[0], remaining)
                matched_gross_pnl += matched * (price - lot[1])
                matched_qty += matched
                holds.append((ts - lot[2]) / 60.)
                lot[0] -= matched
                remaining -= matched
                if lot[0] <= 1e-8:
                    lots[token].popleft()
            sell_missing_qty += remaining
    outstanding_qty = sum(lot[0] for q in lots.values() for lot in q)
    activities = Counter(str(t.get('type') or '?').upper() for t in jsonl_iter(activity_file))
    tops = sorted(market_groups.items(), key=lambda kv: -kv[1]['notional'])[:20]
    total_notional = b_notional + s_notional
    return {
        'wallet': wallet, 'fills': buys+sells, 'bad_rows': bad, 'active_days': len(active_days),
        'unique_markets': len(market_groups), 'buys': buys, 'sells': sells,
        'notional_usdc': total_notional, 'buy_notional_usdc': b_notional,
        'sell_notional_usdc': s_notional, 'category_notional_usdc': dict(buckets),
        'category_notional_pct': {k:100*v/total_notional if total_notional else 0.0 for k,v in buckets.items()},
        'matched_sell_gross_pnl_usdc': matched_gross_pnl,
        'matched_sell_quantity': matched_qty, 'unmatched_sell_quantity': sell_missing_qty,
        'unclosed_buy_quantity': outstanding_qty,
        'median_fifo_hold_min': statistics.median(holds) if holds else None,
        'activity_types': dict(activities),
        'top_markets': [dict(condition_id=k, **v) for k,v in tops],
        'warning': 'FIFO sell-pairing excludes settlement/merge/redemption and earlier inventory; NOT actual portfolio PnL.'
    }


def make_report(output, start, end, summaries):
    names = list(summaries)
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as fh:
        keys = ['name','wallet','complete','fills','unique_markets','active_days','buys','sells','notional_usdc',
                'crypto_pct','sports_pct','weather_pct','other_pct','median_fifo_hold_min',
                'matched_sell_gross_pnl_usdc','unmatched_sell_quantity',
                'official_window_realized_pnl','official_window_economic_pnl','official_all_time_realized_pnl',
                'official_all_time_economic_pnl','error']
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for name, x in summaries.items():
            if not x.get('complete'):
                writer.writerow(dict(name=name,wallet=WALLETS[name],complete=False,error=x.get('error','unknown')))
                continue
            s, d, p, a = x['trade_summary'], x['pnl_window'], x['user_stats'], x['all_time_pnl']
            writer.writerow(dict(name=name,wallet=WALLETS[name],complete=True,
                fills=s['fills'],unique_markets=s['unique_markets'],active_days=s['active_days'],
                buys=s['buys'],sells=s['sells'],notional_usdc=round(s['notional_usdc'],2),
                crypto_pct=round(s['category_notional_pct'].get('crypto',0),2),
                sports_pct=round(s['category_notional_pct'].get('sports',0),2),
                weather_pct=round(s['category_notional_pct'].get('weather',0),2),
                other_pct=round(s['category_notional_pct'].get('other',0),2),
                median_fifo_hold_min=s['median_fifo_hold_min'],
                matched_sell_gross_pnl_usdc=round(s['matched_sell_gross_pnl_usdc'],2),
                unmatched_sell_quantity=round(s['unmatched_sell_quantity'],6),
                official_window_realized_pnl=d.get('realized_pnl'),
                official_window_economic_pnl=d.get('economic_pnl'),
                official_all_time_realized_pnl=a.get('realized_pnl'),
                official_all_time_economic_pnl=a.get('economic_pnl')))
    lines = [f'# Four-wallet Polymarket comparison',
        '', f'Common window: **{utc(start)} through {utc(end)}** (UTC, inclusive).',
        '', 'Source: Polymarket public Data API v2, retrieved independently per wallet.',
        '', '| Wallet | Fills | Markets | Volume ($) | Crypto % | Sports % | Weather % | Official window realized P&L ($) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, x in summaries.items():
        if not x.get('complete'):
            lines.append(f'| {name} | ERROR | — | — | — | — | — | — |')
            continue
        s,d = x['trade_summary'],x['pnl_window']
        lines.append('| {} | {:,} | {:,} | {:,.2f} | {:.1f}% | {:.1f}% | {:.1f}% | {} |'.format(
            name, s['fills'], s['unique_markets'], s['notional_usdc'],
            s['category_notional_pct'].get('crypto',0),
            s['category_notional_pct'].get('sports',0),
            s['category_notional_pct'].get('weather',0),
            fnum(d.get('realized_pnl'))))
    lines += ['', '## Interpretation rules',
        '- Trades queried with `taker_only=false`, including the maker-side fills.',
        '- Trading volume is the sum of price × shares across wallet fills, **not** the sum of share counts.',
        '- Official P&L window is a *difference in cumulative daily snapshots*, not a fill-level ledger; baseline/end timestamps are in `summary.json`.',
        '- Category classification is heuristic on public market title/slug; check `top_markets` before trusting category shares.',
        '- `matched_sell_gross_pnl` is only a FIFO approximation for exchange SELL exits; it ignores settlements, conversions, splits/merges, and initial inventory. It is NOT realized portfolio profit.',
        '- Neither Polymarket maker fillability nor equivalent Kalshi settlement rules/quotes are established here.',
        '- Comparisons are invalid for any wallet marked ERROR; increase --max-pages or fix the API error.',
        '', '## Top markets (notional)', '']
    for name, x in summaries.items():
        lines.append(f'### {name}')
        if not x.get('complete'):
            lines.extend(['', f'FAILED: {x.get("error")}', ''])
            continue
        s=x['trade_summary']
        lines.extend(['', f'Wallet: `{WALLETS[name]}`',
            f'FIFO matched SELL P&L (non-portfolio measure): ${s["matched_sell_gross_pnl_usdc"]:,.2f}; median FIFO hold: {fnum(s["median_fifo_hold_min"],1)} min.',
            f'Activity event counts: {s["activity_types"]}.',
            f'Official P&L snapshot window: {x["pnl_window"].get("baseline_at", "N/A")} -> {x["pnl_window"].get("end_at", "N/A")}.',
            '', '| Market | Category | Fills | Volume ($) |', '|---|---|---:|---:|'])
        for m in s['top_markets'][:12]:
            title=m['title'].replace('|','/').replace('\n',' ')
            lines.append(f'| {title[:95]} | {m["category"]} | {m["fills"]:,} | {m["notional"]:,.2f} |')
        lines.append('')
    (output / 'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(args):
    start,end=epoch(args.start),epoch(args.end)
    if start >= end:
        raise ValueError('Start must precede end')
    output=Path(args.out).expanduser().resolve()
    if (output/'.git').exists() or '/workspaces/' in str(output):
        raise ValueError('Output must be outside GitHub repository /workspaces')
    output.mkdir(parents=True,exist_ok=True)
    client=Client(args.delay)
    summaries={}
    for name,wallet in WALLETS.items():
        LOG.info('=== %s (%s) ===',name,wallet)
        folder=output/name
        folder.mkdir(exist_ok=True)
        try:
            with (folder/'user_stats.json').open('w') as fh:
                p=client.get('/user-stats',{'user':wallet})
                json.dump(p,fh,indent=2)
            stats=p.get('data') or {}
            p=client.get('/user-pnl',{'user':wallet,'interval':'max','fidelity':'1d'})
            with (folder/'user_pnl.json').open('w') as fh:
                json.dump(p,fh)
            series=(p.get('data') or {}).get('points') or []
            if not isinstance(series,list):
                raise ValueError('Bad PnL series')
            pnl_window=pnl_difference(series,start,end)
            all_time=stats.get('all_time_pnl') or {}
            # Stable date bounds and full wallet filter MUST be resent on every page.
            for endpoint, fname in (('/trades','trades.jsonl'),('/activity','activity.jsonl')):
                params={'user':wallet,'start':start,'end':end,'limit':args.limit}
                if endpoint=='/trades':
                    params['taker_only']='false'
                else:
                    params['type']='REDEEM,MERGE,SPLIT,CONVERSION,MAKER_REBATE,TAKER_REBATE'
                n=collect(client,endpoint,params,args.max_pages,folder/fname,wallet)
                LOG.info('%s %s: %d rows',name,endpoint,n)
            s=summarize(folder/'trades.jsonl',folder/'activity.jsonl',wallet,start,end)
            summaries[name]={'complete':True,'user_stats':stats,'all_time_pnl':all_time,
                             'pnl_window':pnl_window,'trade_summary':s}
        except (RuntimeError,ValueError,KeyError,TypeError) as err:
            LOG.exception('%s failed. Will mark as incomplete; no ranking can be trusted.', name)
            summaries[name]={'complete':False,'error':str(err)}
        (output/'summary.json').write_text(json.dumps({
            'start_utc':utc(start),'end_utc':utc(end),'requested_taker_only':False,
            'requested_wallets':WALLETS,'requests':client.requests,'wallets':summaries,
        },indent=2,ensure_ascii=False))
    make_report(output,start,end,summaries)
    print('\nRead-only collection completed.')
    print('Report:',output/'report.md')
    print('CSV:   ',output/'comparison.csv')
    print('JSON:  ',output/'summary.json')
    for name,x in summaries.items():
        print(' ',name, 'COMPLETE' if x.get('complete') else 'INCOMPLETE: '+x.get('error','?'))
    if not all(x.get('complete') for x in summaries.values()):
        print('WARNING: Incomplete wallet(s); no valid cross-wallet ranking yet.')
        return 2
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start',default=DEFAULT_START,help='inclusive UTC start, YYYY-MM-DD or ISO Z')
    parser.add_argument('--end',default=DEFAULT_END,help='inclusive UTC end, YYYY-MM-DD or ISO Z')
    parser.add_argument('--out',default='~/poly_wallet_comparison_20261105')
    parser.add_argument('--limit',type=int,default=500,help='first-page size on both feeds')
    parser.add_argument('--max-pages',type=int,default=2000,help='hard cap per feed/wallet; mark incomplete on cap')
    parser.add_argument('--delay',type=float,default=0.4,help='minimum spacing between GET requests')
    args=parser.parse_args()
    if not (1<=args.limit<=1000 and args.max_pages>=1 and args.delay>=0.05):
        parser.error('limit must be 1..1000, max-pages >=1, delay >=0.05')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(run(args))

if __name__=='__main__':
    main()
