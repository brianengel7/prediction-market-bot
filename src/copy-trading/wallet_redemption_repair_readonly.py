import json
import math
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path.home() / 'poly_wallet_comparison_20261105'

def num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None

def rows(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

for wallet in ('asdcaxz', 'wallet_954b'):
    markets = defaultdict(lambda: {
        'buy': defaultdict(float), 'cost': 0.0, 'first': None,
        'redeemed': defaultdict(float), 'cash': defaultdict(float),
        'other': 0, 'sells': 0, 'title': ''
    })
    quality = Counter()
    missing_token_payout = 0.0

    for t in rows(ROOT / wallet / 'trades.jsonl'):
        cid = str(t.get('condition_id') or '').lower()
        if not cid:
            continue
        side = str(t.get('side') or '').upper()
        m = markets[cid]
        if side == 'SELL':
            m['sells'] += 1
            continue
        if side != 'BUY':
            continue
        outcome = str(t.get('outcome') or '').strip().lower()
        qty, price, ts = num(t.get('size')), num(t.get('price')), num(t.get('timestamp'))
        if outcome not in ('up', 'down') or qty is None or price is None or ts is None or not (qty > 0 and 0 < price < 1):
            quality['invalid_buy'] += 1
            continue
        m['buy'][outcome] += qty
        m['cost'] += price * qty
        if m['first'] is None or ts < m['first'][0]:
            m['first'] = (ts, outcome)
        m['title'] = str(t.get('title') or m['title'])

    for r in rows(ROOT / wallet / 'activity.jsonl'):
        cid = str(r.get('condition_id') or '').lower()
        if not cid:
            quality['missing_activity_condition'] += 1
            continue
        m = markets[cid]
        typ = str(r.get('type') or '').upper()
        if typ == 'REDEEM':
            outcome = str(r.get('outcome') or '').strip().lower()
            qty, cash = num(r.get('size')), num(r.get('usdc_size'))
            if outcome not in ('up', 'down') or qty is None or cash is None or qty < 0 or cash < 0:
                quality['invalid_redemption'] += 1
                continue
            if not r.get('token_id'):
                quality['missing_token_labeled'] += 1
                if outcome in m['buy']:
                    quality['missing_token_matched_to_purchase_outcome'] += 1
                    missing_token_payout += cash
                else:
                    quality['missing_token_without_observed_buy_outcome'] += 1
            m['redeemed'][outcome] += qty
            m['cash'][outcome] += cash
        elif typ not in ('TRADE', 'MAKER_REBATE', 'TAKER_REBATE'):
            m['other'] += 1

    results = defaultdict(lambda: {'n':0, 'cost':0.0, 'cash':0.0})
    known_first = Counter()
    unresolved_expensive = []
    for cid, m in markets.items():
        if m['first'] is None:
            continue
        buy = m['buy']
        positives = [o for o, cash in m['cash'].items() if cash > 0.001]
        if m['other'] or m['sells']:
            state = 'COMPLEX_OPERATIONS'
        elif len(positives) > 1:
            state = 'AMBIGUOUS_PAYOUT'
        elif len(positives) == 1:
            winner = positives[0]
            actual = m['redeemed'][winner]
            expected = buy.get(winner, 0)
            tolerance = max(0.05, expected * 1e-5)
            state = 'FULL_WINNER_PAYOUT' if expected > 0 and abs(actual-expected) <= tolerance else 'PARTIAL_OR_MISSING_BASIS'
            known_first['known_winner_markets'] += 1
            if m['first'][1] == winner:
                known_first['first_side_won'] += 1
        elif m['redeemed']:
            if len(buy) == 1:
                only = next(iter(buy))
                expected = buy[only]
                actual = m['redeemed'][only]
                state = 'VERIFIED_ZERO_PAYOUT' if abs(expected-actual) <= max(0.05, expected*1e-5) else 'ZERO_PAYOUT_INCOMPLETE'
            else:
                state = 'ZERO_PAYOUT_UNKNOWN'
        else:
            state = 'NO_REDEMPTION_OBSERVED'
        s = results[state]
        s['n'] += 1
        s['cost'] += m['cost']
        s['cash'] += sum(m['cash'].values())
        if state in ('NO_REDEMPTION_OBSERVED','PARTIAL_OR_MISSING_BASIS'):
            unresolved_expensive.append((m['cost'],state,m['title']))

    print('\n'+'='*88)
    print(wallet.upper(), '— OUTCOME-BASED REDEMPTION REPAIR')
    print('Missing-token rows mapped by outcome:', quality['missing_token_matched_to_purchase_outcome'])
    print('Cash on recovered rows: $'+format(missing_token_payout, ',.2f'))
    print('Data quality:',dict(quality))
    print(f"{'Status':<31}{'Markets':>9}{'Buy cost':>15}{'Redeem cash':>16}{'Cash minus cost':>18}")
    for state,s in sorted(results.items()):
        print(f"{state:<31}{s['n']:>9,}{s['cost']:>15,.2f}{s['cash']:>16,.2f}{s['cash']-s['cost']:>18,.2f}")
    n=known_first['known_winner_markets']
    print(f"First purchased side won in {known_first['first_side_won']}/{n} markets with a positive redemption (selection-biased).")
    print('Largest unresolved market costs:')
    for cost,state,title in sorted(unresolved_expensive,reverse=True)[:5]:
        print(f'  ${cost:,.2f} | {state} | {title[:85]}')

print('\nREAD ONLY — no API requests, writes, orders or GitHub changes.')
print('Cash minus cost is NOT P&L for unresolved or incomplete statuses.')
