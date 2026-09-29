"""R10: same signals as r9, but you cannot trade at the settle you are looking at.
 A  settle on signal day (r9 as reported)                       - optimistic
 B  next settle day, minus a half-spread against you              - you act tomorrow
 C  next day on which the mark MOVED (proxy: someone traded), within 5 days, minus half-spread;
    signal must still be on at that price, else no trade
 D  worst settle of the next 5 days, minus half-spread            - adverse fill
"""
import pandas as pd, numpy as np
T = pd.read_csv('research/trading_sim.csv', parse_dates=['d', 'm']).sort_values(['m', 'k', 'd']); T['sig'] = T.sig.fillna('')
T['tm'] = T.d.dt.to_period('M')
T['moved'] = T.groupby(['m', 'k']).settle.diff().fillna(0) != 0
def hrs(m): return 24 * (m + pd.offsets.MonthEnd(0)).day

def run(rule, half_spread):
    trades = []
    for (m, k, tm), s in T.groupby(['m', 'k', 'tm']):
        s = s.reset_index(drop=True); f = s[s.sig != '']
        if not len(f): continue
        i0 = f.index[0]; side = s.sig[i0]; sgn = 1 if side == 'SELL' else -1
        nxt = s.iloc[i0 + 1:i0 + 6]                       # the next five settle days in this window
        if rule == 'A': px = s.settle[i0]; fd = s.d[i0]
        elif rule == 'B':
            if not len(nxt): continue
            px = nxt.settle.iloc[0] - sgn * half_spread; fd = nxt.d.iloc[0]
        elif rule == 'C':
            mv = nxt[nxt.moved]
            if not len(mv): continue
            r0 = mv.iloc[0]; px = r0.settle - sgn * half_spread; fd = r0.d
            still = (r0.settle > r0.p75) if side == 'SELL' else (r0.settle < r0.p25)
            if not still: continue
        elif rule == 'D':
            if not len(nxt): continue
            j = nxt.settle.idxmin() if side == 'SELL' else nxt.settle.idxmax()
            px = s.settle[j] - sgn * half_spread; fd = s.d[j]
        real = s.real[i0]; pnl = sgn * (px - real)
        trades.append({'contract': m.strftime('%b-%y'), 'k': k, 'side': side, 'signal_day': s.d[i0].strftime('%m-%d'), 'fill_day': pd.Timestamp(fd).strftime('%m-%d'),
                       'settle_sig': s.settle[i0], 'fill': px, 'settled': real, 'pnl': pnl, 'pnl_1MW': pnl * hrs(m)})
    return pd.DataFrame(trades)

res = []
for rule, hs in [('A', 0), ('B', 0.5), ('B', 1.0), ('C', 0.5), ('C', 1.0), ('D', 0.5), ('D', 1.0)]:
    R = run(rule, hs)
    res.append({'rule': rule, 'half_spread': hs, 'trades': len(R), 'wins': int((R.pnl > 0).sum()), 'mean_pnl': R.pnl.mean(), 'median': R.pnl.median(),
                'worst': R.pnl.min(), 'total_1MW': R.pnl_1MW.sum(), 'slip_vs_A': (R.fill - R.settle_sig).mul(np.where(R.side == 'SELL', -1, 1)).mean()})
    if rule == 'C' and hs == 1.0: RC = R
pd.set_option('display.width', 200)
print(pd.DataFrame(res).round(2).to_string(index=False))
print('\nRule C, $1 half-spread, trade by trade:')
print(RC.round(2).to_string(index=False))
# the daily version: every signal day, filled next day minus $1
T2 = T.copy(); T2['nx'] = T2.groupby(['m', 'k']).settle.shift(-1)
sg = T2[(T2.sig != '') & T2.nx.notna()]
pnl = np.where(sg.sig == 'SELL', sg.nx - 1 - sg.real, sg.real - (sg.nx + 1))
print(f'\nevery signal day, filled next settle -$1 against: n {len(sg)}, mean {pnl.mean():+.2f}, hit {100*(pnl>0).mean():.0f}%  (always-sell same fill: {(sg.nx-1-sg.real).mean():+.2f})')
