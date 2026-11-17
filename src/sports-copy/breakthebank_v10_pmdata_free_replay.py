#!/usr/bin/env python3
"""BreakTheBank V10 — free PMData Polymarket historical L2 acquisition + V8 BBO adapter.

Workflow:
  catalog: download free public Polymarket market catalog (no key)
  plan: join V8 intent condition+token to local wallet trades and catalog;
        confirm Polymarket Gamma's conditionId -> clobTokenIds mapping for EVERY
        market with public GETs, then build an exact per-token/day PMData plan.
  download: fetch only 21+ day old L2 market-days with PMDATA_API_KEY and
            a conservative --max-files cap; don't touch restricted dates.
  convert: reconstruct books from original PMData L2 messages, and export
           as-received BBO observations at exact V8 delay targets.

NO orders, wallet signing, private trading APIs, or mutations to repo files.
Missing, recent, nonverified and pre-snapshot data remain MISSING. All user
artifacts reside under home, outside the repository. Do not use unverified
market-token mappings or synthetically fill gaps. This is an execution proxy,
NOT proof of live copyability; wallet-feed visibility lag remains unmeasured.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, date, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

CATALOG_URL='https://public-data.pmdata.dev/v1/polymarket/markets.parquet'
API='https://api.pmdata.dev/v1/polymarket/l2'
GAMMA='https://gamma-api.polymarket.com/markets'
BBO_FIELDS=['ts_recv','token_id','bid','bid_size','ask','ask_size']
PLAN_FIELDS=['date','market','token_id','outcome_index','market_slug','pmdata_outcome','title',
             'target_count','catalog_available','free_eligible','verified_gamma','download_url','local_file']
HEX=re.compile(r'^0x[a-f0-9]{64}$'); TOKEN=re.compile(r'^\d+$'); SLUG=re.compile(r'^[a-zA-Z0-9_-]+$')
SOURCE_COLUMNS=['market_slug','timestamp','local_timestamp','event_type','ask_prices','ask_sizes',
                'bid_prices','bid_sizes','pc_price','pc_size','pc_side','best_ask','best_bid']


def ensure_output(path,*,overwrite=False):
    p=path.expanduser().resolve()
    if str(p).startswith('/workspaces/') or p.exists() and not overwrite:
        raise ValueError('Refusing repository output or overwrite: '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    return p


def utc(ts):
    if isinstance(ts,datetime):
        return ts.replace(tzinfo=timezone.utc).timestamp() if ts.tzinfo is None else ts.timestamp()
    if isinstance(ts,(int,float)):
        if ts>1e14:return ts/1e6
        if ts>1e11:return ts/1e3
        return float(ts)
    if isinstance(ts, str):
        try:
            numeric = float(ts.strip())
        except ValueError:
            pass
        else:
            if not math.isfinite(numeric):
                raise ValueError("Invalid numeric timestamp")
            return utc(numeric)
    if not ts:raise ValueError('Missing timestamp')
    return datetime.fromisoformat(str(ts).replace('Z','+00:00')).replace(tzinfo=timezone.utc).timestamp() if datetime.fromisoformat(str(ts).replace('Z','+00:00')).tzinfo is None else datetime.fromisoformat(str(ts).replace('Z','+00:00')).timestamp()


def read_parquet(path,columns=None,batch_size=50000):
    try:import pyarrow.parquet as pq
    except ImportError as exc:raise RuntimeError('pip install pyarrow') from exc
    pf=pq.ParquetFile(path)
    if columns:
        missing=set(columns)-set(pf.schema_arrow.names)
        if missing:raise ValueError(f'Missing Parquet columns {sorted(missing)} in {path}')
    for batch in pf.iter_batches(batch_size=batch_size,columns=columns):
        o=batch.to_pydict()
        for t in zip(*(o[k] for k in o)):
            yield dict(zip(o,t))


def http_json(url,timeout=30):
    for n in range(5):
        try:
            with urlopen(Request(url,headers={'Accept':'application/json','User-Agent':'readonly-breakthebank-research/1.0'}),timeout=timeout) as f:
                return json.load(f)
        except HTTPError as e:
            if e.code not in (429,500,502,503,504) or n==4:raise
            time.sleep(min(20,2**n+1))
        except (URLError, TimeoutError):
            if n==4:raise
            time.sleep(min(20,2**n+1))
    raise RuntimeError('GET retries exhausted')


def catalog_cmd(args):
    out=ensure_output(args.out)
    print('Downloading PUBLIC PMData market catalog (no key required)',flush=True)
    req=Request(CATALOG_URL,headers={'User-Agent':'readonly-breakthebank-research/1.0'})
    tmp=out.with_suffix(out.suffix+'.partial')
    try:
        with urlopen(req,timeout=80) as response,tmp.open('wb') as f:
            while (c:=response.read(1<<20)):
                f.write(c)
        # Only commit a genuine readable Parquet catalog.
        next(read_parquet(tmp,columns=['market_slug','record_start','record_end','yes_outcome','no_outcome']))
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    print('Catalog:',out)


def load_intents(path,delays):
    targets=defaultdict(set)
    titles={}
    with path.open(newline='',encoding='utf-8') as f:
        dr=csv.DictReader(f)
        needed={'market','token_id','outcome_index','wallet_trade_ts_utc','title'}
        if not needed.issubset(dr.fieldnames or []):
            raise ValueError('V8 manifest missing columns '+str(needed-set(dr.fieldnames or [])))
        for i,r in enumerate(dr,2):
            cid=str(r['market']).lower();tok=str(r['token_id']); idx=str(r['outcome_index'])
            if not HEX.fullmatch(cid) or not TOKEN.fullmatch(tok) or idx not in ('0','1'):
                raise ValueError(f'Invalid condition/token/outcome index at manifest row {i}')
            t=utc(r['wallet_trade_ts_utc'])
            titles[cid]=r['title']
            for delay in delays:
                target=t+delay
                dy=datetime.fromtimestamp(target,timezone.utc).date().isoformat()
                targets[(cid,tok,idx,dy)].add(target)
    if not targets:raise ValueError('Empty V8 manifest')
    market_token=defaultdict(dict)
    for cid,tok,idx,day in targets:
        prev=market_token[cid].get(idx)
        if prev and prev!=tok:raise ValueError(f'Multiple tokens for same outcome index: {cid}')
        market_token[cid][idx]=tok
    return targets,market_token,titles


def slugs_from_trades(path):
    out=defaultdict(set); labels=defaultdict(lambda:defaultdict(set))
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if not line.strip():continue
            j=json.loads(line)
            cid=str(j.get('condition_id') or '').lower();slug=str(j.get('slug') or '')
            if HEX.fullmatch(cid) and slug:out[cid].add(slug)
            idx=j.get('outcome_index')
            outcome=j.get('outcome')
            if HEX.fullmatch(cid) and str(idx) in ('0','1') and outcome:
                labels[cid][str(idx)].add(str(outcome).strip().casefold())
    return out,labels


def parse_json_array(v):
    if isinstance(v,str):
        try:v=json.loads(v)
        except json.JSONDecodeError:return []
    return v if isinstance(v,list) else []



def verify_gamma(cid, slug, known_tokens):
    """Verify exact condition and outcome tokens, including sports events."""

    def confirm(tokens, source):
        tokens = [str(x) for x in tokens]
        if (len(tokens) != 2 or tokens[0] == tokens[1]
                or not all(TOKEN.fullmatch(x) for x in tokens)):
            raise ValueError(f'{source}: invalid token list for {cid}')

        for idx, tok in known_tokens.items():
            if tokens[int(idx)] != str(tok):
                raise ValueError(
                    f'{source}: wrong token for {cid} outcome {idx}'
                )

        print(f'  Verified by {source}', flush=True)
        return True

    sources = [
        ('CLOB', 'https://clob.polymarket.com/markets/' + cid),
        ('Gamma condition', GAMMA + '?' + urlencode({
            'condition_ids': cid, 'limit': 10
        })),
        ('Gamma closed', GAMMA + '?' + urlencode({
            'condition_ids': cid, 'closed': 'true', 'limit': 10
        })),
        ('Gamma event',
         'https://gamma-api.polymarket.com/events/slug/' + quote(slug)),
    ]

    for source, url in sources:
        try:
            data = http_json(url)
        except HTTPError as exc:
            if exc.code in (400, 404):
                continue
            raise

        if source == 'CLOB':
            if (not isinstance(data, dict) or
                str(data.get('condition_id') or '').lower() != cid):
                continue

            raw = data.get('tokens')
            if isinstance(raw, list) and len(raw) == 2:
                return confirm([
                    x.get('token_id')
                    for x in raw if isinstance(x, dict)
                ], source)
            continue

        rows = (
            data if isinstance(data, list)
            else data.get('markets', []) if isinstance(data, dict)
            else []
        )

        matches = [
            r for r in rows
            if isinstance(r, dict)
            and str(r.get('conditionId') or '').lower() == cid
        ]

        if matches:
            mappings = {
                tuple(str(x) for x in parse_json_array(
                    r.get('clobTokenIds')
                ))
                for r in matches
            }

            if len(mappings) != 1:
                raise ValueError(
                    f'Conflicting Gamma token mappings for {cid}'
                )

            return confirm(next(iter(mappings)), source)

    raise RuntimeError(
        f'Cannot independently verify {cid} ({slug}); '
        'no books will be used'
    )


def catalog_rows(path, needed_slugs=None):
    by_slug={}
    for r in read_parquet(path,columns=['market_slug','record_start','record_end','yes_outcome','no_outcome']):
        slug=str(r['market_slug'])
        a=utc(r['record_start']);b=utc(r['record_end'])
        # Duplicate catalog rows (shouldn't occur) are an unresolved ambiguity.
        if needed_slugs is not None and slug not in needed_slugs:
            continue
        if slug in by_slug:raise ValueError('Duplicate PMData catalog slug '+slug)
        by_slug[slug]={'start':datetime.fromtimestamp(a,timezone.utc).date().isoformat(),
                       'end':datetime.fromtimestamp(b,timezone.utc).date().isoformat(),
                       'yes_outcome':r.get('yes_outcome'),'no_outcome':r.get('no_outcome')}
    return by_slug


def build_plan(targets,market_token,titles,slugs,labels,catalog,gamma_verified,today):
    free_before=today-timedelta(days=21)
    result=[]
    for (cid,tok,idx,day),arr in sorted(targets.items()):
        values=slugs.get(cid,set())
        if len(values)!=1:raise ValueError(f'No unique slug for {cid}: {values}')
        slug=next(iter(values))
        if not SLUG.fullmatch(slug):raise ValueError('Unsafe or unusual market_slug: '+slug)
        meta=catalog.get(slug)
        if not meta:availability=False
        else:
            availability=meta['start']<=day<=meta['end']
            observed=labels.get(cid,{}).get(idx,set())
            expected=meta['yes_outcome' if idx=='0' else 'no_outcome']
            if observed and expected and (len(observed)!=1 or str(expected).strip().casefold() not in observed):
                raise ValueError(f'Outcome label index mismatch {cid}: {observed} vs {expected}')
        outcome='Yes' if idx=='0' else 'No'
        url=f'{API}/{day.replace("-","/")}/{quote(slug)}.parquet'+ ('?outcome=No' if idx=='1' else '')
        stem=f'{day}_{cid}_{idx}.parquet'
        result.append({'date':day,'market':cid,'token_id':tok,'outcome_index':idx,
            'market_slug':slug,'pmdata_outcome':outcome,'title':titles[cid],
            'target_count':len(arr),'catalog_available':int(availability),
            'free_eligible':int(date.fromisoformat(day)<free_before),
            'verified_gamma':int(cid in gamma_verified),
            'download_url':url,'local_file':stem})
    return result


def write_plan(path,rows):
    p=ensure_output(path)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=PLAN_FIELDS);w.writeheader();w.writerows(rows)


def gamma_cached_verifications(path, slugs, market_token):
    verified=set()
    if not path.is_file():return verified
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if not line.strip():continue
            j=json.loads(line);cid=str(j.get('market') or '').lower()
            ss=slugs.get(cid,set())
            expected={str(k):str(v) for k,v in market_token.get(cid,{}).items()}
            if len(ss)==1 and j.get('slug')==next(iter(ss)) and j.get('tokens')==expected:
                verified.add(cid)
    return verified


def plan_cmd(args):
    delays=parse_delays(args.delays)
    targets,market_token,titles=load_intents(args.manifest.expanduser(),delays)
    slugs,labels=slugs_from_trades(args.trades.expanduser())
    catalog=catalog_rows(
        args.catalog.expanduser(),
        needed_slugs={
            slug
            for cid in market_token
            for slug in slugs.get(cid, set())
        }
    )
    cache=args.gamma_cache.expanduser().resolve()
    if str(cache).startswith('/workspaces/'):raise ValueError('Gamma cache cannot be in repository')
    verified=gamma_cached_verifications(cache,slugs,market_token)
    if args.verify_gamma:
        today=datetime.now(timezone.utc).date()
        # Prioritize actual catalog-covered FREE request windows, not merely old dates.
        preview=build_plan(targets,market_token,titles,slugs,labels,catalog,set(),today)
        ready=Counter()
        for r in preview:
            if r['free_eligible'] and r['catalog_available']:
                ready[r['market']]+=int(r['target_count'])
        markets=sorted((cid for cid in market_token if cid not in verified),
                       key=lambda cid:(-ready[cid],cid))
        if args.gamma_max_new>0:markets=markets[:args.gamma_max_new]
        cache.parent.mkdir(parents=True,exist_ok=True)
        with cache.open('a',encoding='utf-8') as fd:
            for i,cid in enumerate(markets,1):
                sl=slugs.get(cid,set())
                if len(sl)!=1:raise ValueError(f'Cannot verify ambiguous slug {cid}: {sl}')
                print(f'Verifying Gamma token IDs: {i}/{len(markets)} {cid[:10]}...',flush=True)
                try:
                    verify_gamma(cid,next(iter(sl)),market_token[cid])
                except RuntimeError as exc:
                    print(f'  SKIP unverified: {exc}', flush=True)
                    continue
                fd.write(json.dumps({'market':cid,'slug':next(iter(sl)),
                                      'tokens':{str(k):str(v) for k,v in market_token[cid].items()}})+'\n')
                fd.flush()
                verified.add(cid)
                time.sleep(args.gamma_pause)
    rows=build_plan(targets,market_token,titles,slugs,labels,catalog,verified,
                   datetime.now(timezone.utc).date())
    write_plan(args.out,rows)
    c=Counter((r['catalog_available'],r['free_eligible']) for r in rows)
    print(f'V10 plan: {sum(len(v) for v in targets.values()):,} target timestamps; '
          f'{len(rows)} market/outcome/day files; {len(market_token)} markets')
    print('Free eligible + catalog available:',c[(1,1)],'| free eligible but not listed:',c[(0,1)],
          '| too recent:',sum(1 for r in rows if not r['free_eligible']))
    print('Verified Gamma market-token mappings:',len(verified),'/',len(market_token))
    print('Plan:',args.out)
    if not verified:print('WARNING: Gamma not verified. Download intentionally refuses unverified rows.')


def read_plan(path):
    with path.open(newline='',encoding='utf-8') as f:
        rd=csv.DictReader(f)
        if not set(PLAN_FIELDS).issubset(rd.fieldnames or []):raise ValueError('Invalid V10 plan schema')
        rows=list(rd)
    if len({(r['date'],r['market'],r['outcome_index']) for r in rows})!=len(rows):
        raise ValueError('Duplicate plan keys')
    return rows


def download_cmd(args):
    rows=read_plan(args.plan.expanduser());root=args.root.expanduser().resolve()
    if str(root).startswith('/workspaces/'):raise ValueError('Refusing to write within repository')
    key=os.environ.get('PMDATA_API_KEY','').strip()
    if not key:raise RuntimeError('Set PMDATA_API_KEY in your Codespace environment, do not paste it into chat')
    now=datetime.now(timezone.utc).date()
    eligible=[];recent=missing=unverified=0
    for r in rows:
        if not int(r['verified_gamma']):unverified+=1;continue
        if not int(r['catalog_available']):missing+=1;continue
        # Always enforce cutoff again at download time: never trust plan alone.
        if not date.fromisoformat(r['date'])<now-timedelta(days=21):recent+=1;continue
        eligible.append(r)
    print(f'Free PMData candidates: {len(eligible)} | too-recent: {recent} | '
          f'catalog-unavailable: {missing} | token-unverified: {unverified}',flush=True)
    cap=max(0,args.max_files)
    selected=eligible[:cap]
    if not selected:return
    root.mkdir(parents=True,exist_ok=True)
    existing=downloaded=not_found=0
    for i,r in enumerate(selected,1):
        path=root/r['local_file'];url=r['download_url']
        if path.is_file() and path.stat().st_size>0:existing+=1;continue
        print(f'  [{i}/{len(selected)}] {r["date"]} {r["market_slug"]} {r["pmdata_outcome"]}',flush=True)
        part=path.with_suffix('.parquet.partial')
        req=Request(url,headers={'api_key':key,'User-Agent':'readonly-breakthebank-research/1.0'})
        try:
            for attempt in range(1, 4):
                try:
                    with urlopen(req,timeout=120) as stream,part.open('wb') as f:
                        expected = stream.headers.get('Content-Length')
                        while (block:=stream.read(1<<20)):
                            f.write(block)
                    if expected is not None and part.stat().st_size != int(expected):
                        raise ConnectionError('Incomplete HTTP response body')
                    iterator=read_parquet(part,columns=['market_slug','event_type','local_timestamp'])
                    first=next(iterator,None)
                    if first is None or first['market_slug']!=r['market_slug']:
                        raise ValueError('Downloaded empty or mismatched L2 data; not accepting file')
                    part.replace(path);downloaded+=1
                    break
                except HTTPError:
                    raise
                except (ConnectionError, TimeoutError, URLError, OSError) as exc:
                    part.unlink(missing_ok=True)
                    if attempt == 3:
                        raise RuntimeError(
                            f'PMData download interrupted 3 times for {r["local_file"]}'
                        ) from exc
                    print(f'    Network interruption ({type(exc).__name__}); '
                          f'retrying {attempt+1}/3', flush=True)
                    time.sleep(3 * attempt)
        except HTTPError as e:
            if e.code==404:
                print('    404 missing archive; no generated quotes',flush=True);not_found+=1
            else:
                raise RuntimeError(f'PMData HTTP {e.code}; check plan/key/quota') from e
        finally:part.unlink(missing_ok=True)
    print(f'V10 downloads: new={downloaded}, existing={existing}, unavailable={not_found}. Root: {root}')


class Book:
    def __init__(self):
        self.bids={};self.asks={};self.initialized=False;self.ts=None;self.events=0
    def quote(self,token):
        if not self.initialized or self.ts is None:return None
        bp=max(self.bids) if self.bids else None
        ap=min(self.asks) if self.asks else None
        if bp is not None and ap is not None and bp>ap+1e-8:
            raise ValueError('Crossed reconstructed book; refusing to write invalid quotes')
        return (self.ts,token,bp,self.bids.get(bp) if bp is not None else None,
                ap,self.asks.get(ap) if ap is not None else None)
    def apply(self,row,t):
        typ=str(row['event_type'] or '').lower()
        if typ in ('book','book_snapshot'):
            bp=row.get('bid_prices') or [];bs=row.get('bid_sizes') or []
            ap=row.get('ask_prices') or [];a_s=row.get('ask_sizes') or []
            if len(bp)!=len(bs) or len(ap)!=len(a_s):raise ValueError('Snapshot price-size mismatch')
            self.bids=self.full(bp,bs);self.asks=self.full(ap,a_s)
            self.initialized=True
        elif typ=='price_change':
            if not self.initialized:return
            sd=str(row.get('pc_side') or '').upper()
            if sd not in ('BUY','SELL'):raise ValueError('Unexpected pc_side: '+sd)
            p=float(row['pc_price']);sz=float(row['pc_size'])
            if not math.isfinite(p) or not 0<p<1 or not math.isfinite(sz) or sz<0:
                raise ValueError('Invalid price_change level')
            lv=self.bids if sd=='BUY' else self.asks
            if sz==0:lv.pop(p,None)
            else:lv[p]=sz
        elif typ=='market_resolved':
            self.initialized=False;self.bids.clear();self.asks.clear()
        elif typ!='tick_size_change':
            raise ValueError('Unknown PMData L2 event type '+typ)
        self.ts=t;self.events+=1
    @staticmethod
    def full(prices,sizes):
        result={}
        for p,s in zip(prices,sizes):
            p,s=float(p),float(s)
            if not math.isfinite(p) or not 0<p<1 or not math.isfinite(s) or s<0:
                raise ValueError('Invalid snapshot level')
            if s>0:result[p]=s
        return result



def book_at_targets(rows, targets, token, slug):
    """Conservative PMData source-order, availability-time replay."""
    wanted = sorted(set(targets))
    i = 0
    out = []
    miss = Counter()
    book = Book()
    avail = prev_recv = prev_src = None
    tainted = reset = False

    def flush_before(limit):
        nonlocal i, tainted, reset
        crossed = (
            book.initialized and book.bids and book.asks
            and max(book.bids) > min(book.asks) + 1e-8
        )
        if crossed:
            if not tainted:
                miss['crossed_state_boundaries'] += 1
            tainted = True
        elif reset and book.initialized:
            tainted = False

        reset = False
        while i < len(wanted) and wanted[i] < limit:
            if not book.initialized:
                miss['missing_prior_snapshot'] += 1
            elif tainted:
                miss['quarantined_targets'] += 1
            else:
                q = book.quote(token)
                if q is None:
                    miss['missing_prior_snapshot'] += 1
                else:
                    out.append(q)
            i += 1

    for row in rows:
        if row.get('market_slug') != slug:
            raise ValueError('L2 market slug mismatch')

        recv = utc(row['local_timestamp'])
        src = (
            utc(row['timestamp'])
            if row.get('timestamp') is not None else recv
        )
        if prev_src is not None and src < prev_src - 1e-6:
            raise ValueError('Invalid PMData source timestamp ordering')
        prev_src = src

        if prev_recv is not None and recv < prev_recv - 1e-6:
            miss['receive_timestamp_regressions'] += 1
        prev_recv = recv

        eff = max(
            recv, src,
            avail if avail is not None else -math.inf
        )
        if avail is None or eff > avail + 1e-7:
            flush_before(eff)

        if eff > recv + 1e-6:
            miss['availability_watermark_delays'] += 1
        avail = eff

        typ = str(row.get('event_type') or '').lower()
        book.apply(row, eff)
        if typ in ('book', 'book_snapshot'):
            reset = True
        elif typ == 'market_resolved':
            tainted = True

    if avail is None:
        miss['missing_prior_snapshot'] += len(wanted)
    else:
        flush_before(math.inf)
    return out, miss


def parse_delays(s):
    out=[float(x.strip()) for x in s.split(',')]
    if not out or any(not math.isfinite(x) or x<0 for x in out):raise ValueError('Invalid delays')
    return out


def convert_cmd(args):
    delays=parse_delays(args.delays)
    targets,_,_=load_intents(args.manifest.expanduser(),delays)
    plan=read_plan(args.plan.expanduser())
    root=args.root.expanduser().resolve()
    dest=ensure_output(args.out)
    rows_out=[];stats=Counter();missing=[];failed=[]
    for p in plan:
        k=(p['market'],p['token_id'],p['outcome_index'],p['date'])
        if k not in targets:raise ValueError('Plan/manifest mismatch; recompute plan for identical delays')
        f=root/p['local_file'];stats['needed_files']+=1
        if not f.is_file():missing.append((p['market_slug'],p['date'],p['pmdata_outcome']));continue
        if not int(p['verified_gamma']):raise ValueError('Unverified token mapping file present: '+str(f))
        try:
            out,miss=book_at_targets(read_parquet(f,columns=SOURCE_COLUMNS),targets[k],p['token_id'],p['market_slug'])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            failed.append((str(f),str(exc)))
            continue
        rows_out.extend(out);stats.update(miss);stats['parsed_files']+=1
    observed={}
    for r in rows_out:
        k=(r[1],r[0]);prev=observed.get(k)
        if prev is not None and prev!=r:raise ValueError('Conflicting same-time quotes')
        observed[k]=r
    with dest.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(BBO_FIELDS)
        for r in sorted(observed.values(),key=lambda x:(x[0],x[1])):
            w.writerow([f'{r[0]:.6f}',r[1]]+[f'{v:.9f}' if v is not None else '' for v in r[2:]])
    print(f'V10 PMData BBO: found {stats["parsed_files"]}/{stats["needed_files"]} L2 files; '
          f'{len(observed)} quote observations; missing snapshots: {stats["missing_prior_snapshot"]}')
    print(f'Missing files (not zero fills): {len(missing)}; quote CSV: {dest}')
    print('Replay diagnostics:',
          'receive regressions=', stats['receive_timestamp_regressions'],
          'watermark delays=', stats['availability_watermark_delays'],
          'crossed states=', stats['crossed_state_boundaries'],
          'quarantined targets=', stats['quarantined_targets'])

    if missing:print('First missing files:',missing[:6])
    if failed:print(f'INVALID/REJECTED L2 files (not scored): {len(failed)}; first errors: {failed[:4]}')
    print('V8 book replay should report actual token/order coverage; DO NOT treat incomplete data as overall strategy ROI.')


def self_test():
    cid='0x'+'a'*64;tok='123';date_str='2026-10-16';t0=1792166400.
    targets={(cid,tok,'0',date_str):{t0+1.5,t0+2.5,t0+4}}
    slugs={cid:{'synthetic-game'}};labels={cid:{'0':{'alpha'}}}
    catalog={'synthetic-game':{'start':'2026-10-15','end':'2026-10-17','yes_outcome':'Alpha','no_outcome':'Beta'}}
    plan=build_plan(targets,{cid:{'0':tok}},{cid:'Synthetic'},slugs,labels,catalog,{cid},date(2026,11,16))
    assert len(plan)==1 and plan[0]['free_eligible']==1 and plan[0]['verified_gamma']==1
    def b(t,typ,**kw):
        return dict(market_slug='synthetic-game',local_timestamp=t0+t,event_type=typ,
                    bid_prices=None,bid_sizes=None,ask_prices=None,ask_sizes=None,
                    pc_price=None,pc_size=None,pc_side=None,best_ask=None,best_bid=None,**kw)
    # Remove default keys before override (Python kwargs duplicate illegal).
    def row(t,typ,changes):
        r=b(t,typ);r.update(changes);return r
    rows=[row(1,'price_change',{'pc_side':'BUY','pc_price':.40,'pc_size':10}),
          row(2,'book',{'bid_prices':[.4,.3],'bid_sizes':[8,10],
                        'ask_prices':[.5,.6],'ask_sizes':[5,10]}),
          row(3,'price_change',{'pc_side':'SELL','pc_price':.5,'pc_size':0}),
          row(3.1,'price_change',{'pc_side':'SELL','pc_price':.52,'pc_size':4})]
    out,missing=book_at_targets(rows,[t0+1.5,t0+2.5,t0+3.5,t0+4],tok,'synthetic-game')
    assert missing['missing_prior_snapshot']==1,missing
    assert len(out)==3 and out[0][2]==.4 and out[0][4]==.5 and out[-1][4]==.52,out
    assert all(t<=t0+4 for t,*_ in out)
    try:
        book_at_targets(list(reversed(rows)),[t0+4],tok,'synthetic-game')
    except ValueError:pass
    else:raise AssertionError('receive time regression not detected')
    print('PASS V10: free-age cutoff, token verification gate, full snapshot, absolute deltas,')
    print('          missing initial snapshot, as-received no-lookahead, receive-time regression')
    print('OFFLINE SELF-TEST: no network or trades')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    commands=ap.add_subparsers(dest='cmd',required=True)
    commands.add_parser('self-test')
    q=commands.add_parser('catalog');q.add_argument('--out',type=Path,required=True)
    q=commands.add_parser('plan')
    q.add_argument('--manifest',type=Path,required=True)
    q.add_argument('--trades',type=Path,required=True)
    q.add_argument('--catalog',type=Path,required=True)
    q.add_argument('--out',type=Path,required=True)
    q.add_argument('--delays',default='5,15,30,60')
    q.add_argument('--verify-gamma',action='store_true')
    q.add_argument('--gamma-pause',type=float,default=.15)
    q.add_argument('--gamma-cache',type=Path,default=Path.home()/'sports_copy_research'/'breakthebank_v10'/'gamma_verified.jsonl')
    q.add_argument('--gamma-max-new',type=int,default=10,help='0 means all; default verifies 10 new markets')
    q=commands.add_parser('download')
    q.add_argument('--plan',type=Path,required=True)
    q.add_argument('--root',type=Path,required=True)
    q.add_argument('--max-files',type=int,default=3,help='Conservative pilot download cap; free quota is finite')
    q=commands.add_parser('convert')
    q.add_argument('--manifest',type=Path,required=True)
    q.add_argument('--plan',type=Path,required=True)
    q.add_argument('--root',type=Path,required=True)
    q.add_argument('--out',type=Path,required=True)
    q.add_argument('--delays',default='5,15,30,60')
    a=ap.parse_args()
    if a.cmd=='self-test':self_test()
    elif a.cmd=='catalog':catalog_cmd(a)
    elif a.cmd=='plan':plan_cmd(a)
    elif a.cmd=='download':download_cmd(a)
    elif a.cmd=='convert':convert_cmd(a)

if __name__=='__main__':main()
