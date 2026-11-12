#!/usr/bin/env python3
"""Read-only audit of two same-second 954B trades; no DB changes or orders."""
import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone

import requests

URL = 'https://data-api.polymarket.com/v2/trades'
WALLET = '0x954b91dd68eca3aa4ea603ecea83c01e8539c0e0'
DEFAULT_START = 1794404700
DEFAULT_TS = 1794405094  # 2026-11-11 13:51:34 UTC
OBSERVED = (
    '7f4f3cda52878116a0127d6dc206f3cca1765dab41287d429931b5a97a60bcae',
    '0050fbf887f41f47430235f30278c80ba9a4b6de97c8f8e8d825ae7bb01a2ec2',
)

def field(r, *names):
    return next((r[n] for n in names if n in r and r[n] is not None), None)

def digest(r):
    # Matches parse_trade() in w954_live_copy_v2_sequence.py.
    return hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()

def trade_key(r):
    return (str(field(r, 'slug','market_slug','event_slug','eventSlug') or ''),
            field(r, 'timestamp','ts','time'),
            str(field(r, 'side','action') or '').upper(),
            str(field(r, 'outcome','outcome_name') or '').upper(),
            str(field(r, 'price')),str(field(r, 'size','shares')))

def fetch(start, ts):
    s=requests.Session()
    q={'user':WALLET,'start':start,'end':start+900,'limit':500,'taker_only':'false'}
    out=[]; cursors=set()
    for page in range(12):
        resp=s.get(URL,params=q,timeout=15,headers={'Accept':'application/json'})
        resp.raise_for_status()
        p=resp.json()
        if not isinstance(p,dict) or not isinstance(p.get('data'),list):
            raise RuntimeError('Unrecognized response: '+str(type(p)))
        out.extend(p['data'])
        pg=p.get('pagination') or {}
        cursor=pg.get('next_cursor') or pg.get('nextCursor')
        if not cursor:
            if pg.get('has_more'):
                raise RuntimeError('More pages but no cursor')
            return out
        if cursor in cursors:
            raise RuntimeError('Repeated pagination cursor')
        cursors.add(cursor)
        q['cursor']=cursor
    raise RuntimeError('Exceeded pagination cap: rerun with a narrower window')

def describe(rows, market_start, ts):
    relevant=[r for r in rows
              if ('btc-updown-15m-'+str(market_start)) in str(field(r,'slug','event_slug','eventSlug','market_slug') or '')
              and int(field(r,'timestamp','ts','time') or 0)==ts]
    exact=[]
    for r in relevant:
        try:
            if (str(field(r,'side','action')).upper()=='BUY'
                and str(field(r,'outcome','outcome_name')).upper()=='UP'
                and abs(float(field(r,'price'))-0.83)<0.00001
                and abs(float(field(r,'size','shares'))-51.204817)<0.00001):
                exact.append(r)
        except (ValueError,TypeError):
            pass
    print('954B — SAME-SECOND TRADE ID AUDIT (READ ONLY)')
    print('Market start:',datetime.fromtimestamp(market_start,timezone.utc).isoformat())
    print('Trade second:',datetime.fromtimestamp(ts,timezone.utc).isoformat())
    print('Fetched rows:',len(rows),'| market rows at this second:',len(relevant),
          '| rows matching UP 83c / 51.204817:',len(exact))
    if not exact:
        print('INCONCLUSIVE: historical API returned no exact matches. No dedupe change is justified.')
        return
    by_core=defaultdict(list)
    for i,r in enumerate(exact,1):
        h=digest(r)
        tx=field(r,'transaction_hash','transactionHash','tx_hash','txHash')
        seq=field(r,'log_index','logIndex','sequence_id','sequenceId')
        nat=field(r,'trade_id','tradeId','id')
        print('\nROW',i,'payload_hash=',h,'LOG_MATCH=',h in OBSERVED)
        print(' native_id:',nat,'| transaction_hash:',tx,'| log/sequence_index:',seq)
        print(' asset:',field(r,'token_id','tokenId','asset'),'| condition:',field(r,'condition_id','conditionId'))
        print(' JSON keys:',','.join(sorted(r.keys())))
        by_core[trade_key(r)].append(r)
    if len(exact)>1:
        print('\nPAIRWISE DIFFERENCES')
        for i in range(len(exact)):
            for j in range(i+1,len(exact)):
                a,b=exact[i],exact[j]
                unequal=[k for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)]
                print(' Rows',i+1,j+1,'differing fields:',unequal or '[none]')
                for k in unequal[:8]:
                    print('   ',k,':',str(a.get(k))[:110],'vs',str(b.get(k))[:110])
    if len(exact)==1:
        print('\nRESULT: only ONE matching API row now. The second prior ID might be a changed row or a disappeared fill; cannot prove.')
        return
    txs={str(field(r,'transaction_hash','transactionHash','tx_hash','txHash')) for r in exact}
    txs.discard('None')
    ids={str(field(r,'trade_id','tradeId','id')) for r in exact};ids.discard('None')
    indices={str(field(r,'log_index','logIndex')) for r in exact};indices.discard('None')
    print('\nINTERPRETATION')
    if len(txs)>1:
        print('Different transaction hashes: these are distinct settling transactions; do not dedupe solely by price/size/second.')
    elif len(indices)>1 and len(txs)==1:
        print('Same transaction but distinct log indices: separate events; do not collapse.')
    elif len(ids)>1:
        print('Distinct provider-native IDs: likely distinct records, but check their identity semantics.')
    else:
        print('Same or missing transaction/event identifiers: cannot reliably distinguish fills. Need chain log indices or provider IDs.')
    print('Do not enable real orders based solely on payload hashes.')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--market-start',type=int,default=DEFAULT_START)
    ap.add_argument('--wallet-timestamp',type=int,default=DEFAULT_TS)
    a=ap.parse_args()
    describe(fetch(a.market_start,a.wallet_timestamp),a.market_start,a.wallet_timestamp)
