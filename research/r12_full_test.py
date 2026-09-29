"""R12: the whole thing as it would be traded, last 365 settle days.
Signals: strict (settle clears the model's P25-P75 band by more than half the spread) -> CROSS, or inside the
half-spread but outside the band -> POST at the settle and wait up to 10 days. One position per contract per
calendar month. Then the tracker's guidance applied to the settle path: take profit at the model's bid/offer on
the other side, stop at the historical P75 adverse excursion for the lead. Every trade, every rule."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from model import data as _D, read
import pandas as pd, numpy as np
HS = 2.5
T = pd.read_csv('research/trading_sim.csv', parse_dates=['d', 'm']).sort_values(['m', 'k', 'd']); T['sig'] = T.sig.fillna('')
T['tm'] = T.d.dt.to_period('M'); T['moved'] = T.groupby(['m', 'k']).settle.diff().fillna(0) != 0
pw = _D.power_forward()
exc = read.excursions('2026-09-28'); 
def mae(lead, side, q='p75'):
    e = next((e for e in exc if e['lead_lo'] <= lead <= e['lead_hi']), exc[-1]); return e[f'{"short" if side=="SELL" else "long"}_{q}']
def hrs(m): return 24 * (m + pd.offsets.MonthEnd(0)).day
def path(m, start):
    s = pw[(pw.m == m) & (pw.ed > start)].sort_values('ed'); return s[['ed', 'flat']].values

trades = []
for (m, k, tm), s in T.groupby(['m', 'k', 'tm']):
    s = s.reset_index(drop=True)
    # strict signal first (cross), else soft signal (post)
    strict = [('SELL' if r.settle - HS > r.p75 else 'BUY' if r.settle + HS < r.p25 else '') for r in s.itertuples()]
    soft = list(s.sig)
    i_c = next((i for i, v in enumerate(strict) if v), None); i_p = next((i for i, v in enumerate(soft) if v), None)
    if i_c is None and i_p is None: continue
    if i_c is not None and (i_p is None or i_c <= i_p):
        i0 = i_c; side = strict[i0]; sgn = 1 if side == 'SELL' else -1
        nxt = s.iloc[i0 + 1:i0 + 6]; mv = nxt[nxt.moved]
        if not len(mv): continue
        fill = mv.settle.iloc[0] - sgn * HS; fd = mv.d.iloc[0]; how = 'cross'
    else:
        i0 = i_p; side = soft[i0]; sgn = 1 if side == 'SELL' else -1
        post = s.settle[i0]; nxt = s.iloc[i0 + 1:i0 + 11]
        hit = nxt[(nxt.settle >= post) if side == 'SELL' else (nxt.settle <= post)]
        if not len(hit): continue
        fill = post; fd = hit.d.iloc[0]; how = 'post'
    row = s.iloc[i0]; real = row.real; lead = int((m - pd.Timestamp(fd)).days)
    tp = row.p25 if side == 'SELL' else row.p75                       # model's bid/offer on the other side
    stop = fill + sgn * mae(max(lead, 1), side)                         # historical P75 adverse excursion
    stop90 = fill + sgn * mae(max(lead, 1), side, 'p90')
    # walk the settle path after the fill
    pth = path(m, pd.Timestamp(fd)); hold = sgn * (fill - real)
    def managed(stop_lvl, tp_lvl):
        for ed, px in pth:
            if (side == 'SELL' and px >= stop_lvl) or (side == 'BUY' and px <= stop_lvl): return sgn * (fill - (px + sgn * HS)), 'stopped', pd.Timestamp(ed)
            if tp_lvl is not None and ((side == 'SELL' and px <= tp_lvl) or (side == 'BUY' and px >= tp_lvl)): return sgn * (fill - (px + sgn * HS)), 'took profit', pd.Timestamp(ed)
        return hold, 'held to settle', None
    p_stop, r_stop, _ = managed(stop, None); p_stop90, r_stop90, _ = managed(stop90, None); p_both, r_both, _ = managed(stop, tp)
    adverse = max([sgn * (px - fill) for _, px in pth] + [0])
    trades.append({'contract': m.strftime('%b-%y'), 'k': k, 'side': side, 'how': how, 'signal': s.d[i0].strftime('%m-%d'), 'fill_day': pd.Timestamp(fd).strftime('%m-%d'),
                   'fill': fill, 'fair': row.fair, 'settled': real, 'max_adverse': adverse, 'stop_p75': stop, 'tp': tp,
                   'hold': hold, 'stop75': p_stop, 'stop75_res': r_stop, 'stop90': p_stop90, 'stop90_res': r_stop90, 'stop_tp': p_both, 'stop_tp_res': r_both, 'hrs': hrs(m)})
R = pd.DataFrame(trades)
pd.set_option('display.width', 260)
print(R.drop(columns=['hrs']).round(2).to_string(index=False))
print('\n=== summary, $/MWh per trade (n=%d: %d cross, %d post; %d sells, %d buys) ===' % (len(R), (R.how == 'cross').sum(), (R.how == 'post').sum(), (R.side == 'SELL').sum(), (R.side == 'BUY').sum()))
for c, lab in [('hold', 'hold to settlement'), ('stop75', 'stop at P75 excursion'), ('stop90', 'stop at P90 excursion'), ('stop_tp', 'stop P75 + take profit at model bid/offer')]:
    print(f'{lab:<42} mean {R[c].mean():+6.2f}  median {R[c].median():+6.2f}  wins {int((R[c] > 0).sum())}/{len(R)}  worst {R[c].min():+6.2f}  total on 1 MW ${(R[c] * R.hrs).sum():,.0f}')
for c in ['stop75_res', 'stop90_res', 'stop_tp_res']: print(f'   {c}: ', R[c].value_counts().to_dict())
print(f'\nmax adverse excursion actually seen after fill: median {R.max_adverse.median():.2f}, P75 {R.max_adverse.quantile(.75):.2f}, max {R.max_adverse.max():.2f}')
print('by side:'); print(R.groupby('side')[['hold', 'stop75', 'stop_tp']].mean().round(2))
R.to_csv('research/full_test.csv', index=False)

print('\n=== stop alternatives ===')
Tm = T.set_index(['m', 'k', 'd'])
def thesis(row):
    """exit only when the MODEL changes its mind: fair moves through the fill (sell: fair > fill). Exit at settle -/+ half-spread."""
    m = pd.Timestamp('20' + row.contract[-2:] + '-' + row.contract[:3] + '-01'); sgn = 1 if row.side == 'SELL' else -1
    s = T[(T.m == m) & (T.k == row.k) & (T.d > pd.Timestamp('2026-' + row.fill_day if False else f'{m.year if m.month > int(row.fill_day[:2]) else m.year - 1}-{row.fill_day}'))].sort_values('d')
    for r in s.itertuples():
        if (row.side == 'SELL' and r.fair > row.fill) or (row.side == 'BUY' and r.fair < row.fill): return sgn * (row.fill - (r.settle + sgn * HS)), 'model flipped'
    return row.hold, 'held'
out = R.apply(thesis, axis=1, result_type='expand'); R['thesis'] = out[0]; R['thesis_res'] = out[1]
for mult in [1.5, 2.0]:
    def wide(row, mult=mult):
        m = pd.Timestamp('20' + row.contract[-2:] + '-' + row.contract[:3] + '-01'); sgn = 1 if row.side == 'SELL' else -1
        lvl = row.fill + sgn * mult * (row.stop_p75 - row.fill) * sgn
        fd = pd.Timestamp(f'{m.year if m.month > int(row.fill_day[:2]) else m.year - 1}-{row.fill_day}')
        for ed, px in path(m, fd):
            if (row.side == 'SELL' and px >= lvl) or (row.side == 'BUY' and px <= lvl): return sgn * (row.fill - (px + sgn * HS))
        return row.hold
    R[f'stop_x{mult}'] = R.apply(wide, axis=1)
for c, lab in [('hold', 'hold to settlement'), ('thesis', 'exit only if the model flips through your price'), ('stop_x1.5', 'price stop at 1.5 x P75 excursion'), ('stop_x2.0', 'price stop at 2 x P75 excursion'), ('stop90', 'price stop at P90 excursion')]:
    print(f'{lab:<48} mean {R[c].mean():+6.2f}  wins {int((R[c] > 0).sum())}/{len(R)}  worst {R[c].min():+6.2f}  total 1 MW ${(R[c] * R.hrs).sum():,.0f}')
print('thesis exits:', R.thesis_res.value_counts().to_dict())
print(R[['contract', 'k', 'side', 'fill', 'settled', 'max_adverse', 'hold', 'thesis', 'thesis_res']].round(2).to_string(index=False))
