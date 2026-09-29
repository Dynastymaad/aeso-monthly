"""R11: a $5-wide market. Two ways to trade it:
  CROSS  hit the bid / lift the offer on the next day the mark moved: fill = settle -/+ half-spread (2.50, 3.50)
  POST   leave an order at the settle (the mid) and wait up to N days; filled only if a later settle trades
         through it (market comes to you), else no trade. Zero spread paid, fill risk instead.
Signals as in r9 (outside fair P25-P75), plus a stricter version where the edge must exceed the half-spread."""
import pandas as pd, numpy as np
T = pd.read_csv('research/trading_sim.csv', parse_dates=['d', 'm']).sort_values(['m', 'k', 'd']); T['sig'] = T.sig.fillna('')
T['tm'] = T.d.dt.to_period('M'); T['moved'] = T.groupby(['m', 'k']).settle.diff().fillna(0) != 0
def hrs(m): return 24 * (m + pd.offsets.MonthEnd(0)).day

def signal(row, hs, strict):
    if strict:  # market must sit outside the band by more than the half-spread you will pay
        if row.settle - hs > row.p75: return 'SELL'
        if row.settle + hs < row.p25: return 'BUY'
        return ''
    return row.sig

def run(mode, hs, strict, wait=5):
    out = []
    for (m, k, tm), s in T.groupby(['m', 'k', 'tm']):
        s = s.reset_index(drop=True)
        sigs = [signal(r, hs, strict) for r in s.itertuples()]
        idx = [i for i, v in enumerate(sigs) if v]
        if not idx: continue
        i0 = idx[0]; side = sigs[i0]; sgn = 1 if side == 'SELL' else -1
        nxt = s.iloc[i0 + 1:i0 + 1 + wait]
        if mode == 'CROSS':
            mv = nxt[nxt.moved]
            if not len(mv): continue
            r0 = mv.iloc[0]; px = r0.settle - sgn * hs; fd = r0.d
        else:  # POST at the signal-day settle
            post = s.settle[i0]
            hit = nxt[(nxt.settle >= post) if side == 'SELL' else (nxt.settle <= post)]
            if not len(hit): continue
            px = post; fd = hit.d.iloc[0]
        real = s.real[i0]; pnl = sgn * (px - real)
        out.append({'contract': m.strftime('%b-%y'), 'k': k, 'side': side, 'signal_day': s.d[i0].strftime('%m-%d'), 'fill_day': pd.Timestamp(fd).strftime('%m-%d'),
                    'settle_sig': s.settle[i0], 'fill': px, 'settled': real, 'pnl': pnl, 'pnl_1MW': pnl * hrs(m)})
    return pd.DataFrame(out)

res = []
for mode, hs, strict, wait in [('CROSS', 2.5, False, 5), ('CROSS', 2.5, True, 5), ('CROSS', 3.5, True, 5),
                               ('POST', 2.5, False, 5), ('POST', 2.5, True, 5), ('POST', 2.5, True, 10), ('POST', 2.5, False, 10)]:
    R = run(mode, hs, strict, wait)
    res.append({'mode': mode, 'half_spread': hs, 'strict_signal': strict, 'wait_days': wait, 'trades': len(R), 'wins': int((R.pnl > 0).sum()) if len(R) else 0,
                'mean_pnl': R.pnl.mean() if len(R) else np.nan, 'median': R.pnl.median() if len(R) else np.nan, 'worst': R.pnl.min() if len(R) else np.nan,
                'total_1MW': R.pnl_1MW.sum() if len(R) else 0})
    if mode == 'CROSS' and hs == 2.5 and strict: RC = R
    if mode == 'POST' and strict and wait == 10: RP = R
pd.set_option('display.width', 220)
print(pd.DataFrame(res).round(2).to_string(index=False))
print('\nCROSS at $2.50, strict signal:'); print(RC.round(2).to_string(index=False))
print('\nPOST at the settle, strict signal, wait 10 days:'); print(RP.round(2).to_string(index=False))
# daily version, strict, crossing $2.50 next day
T2 = T.copy(); T2['nx'] = T2.groupby(['m', 'k']).settle.shift(-1)
st = np.where(T2.settle - 2.5 > T2.p75, 'SELL', np.where(T2.settle + 2.5 < T2.p25, 'BUY', '')); T2['st'] = st
sg = T2[(T2.st != '') & T2.nx.notna()]
pnl = np.where(sg.st == 'SELL', sg.nx - 2.5 - sg.real, sg.real - (sg.nx + 2.5))
print(f'\nevery strict-signal day, cross next settle at $2.50: n {len(sg)} of {len(T2)}, mean {pnl.mean():+.2f}, hit {100*(pnl>0).mean():.0f}%   '
      f'always-sell crossing: {(T2.nx.dropna()-2.5-T2[T2.nx.notna()].real).mean():+.2f}')
