"""R9: what would the model have bought and sold over the last 365 settle days, and how did it do?
The read on each day uses only: the trailing year before that day's month, the outage plan + seasonal adder,
climatology load/wind, imports regression, and the AECO forward as of that day.  No ensemble, no filed
outages (not archived) - the 'outlook only' read, a lower bound on the live model."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from model import data as _D, panel
from model.curve import Curve
from model.forecast import climatology
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
rng = np.random.default_rng(9)
df = panel.build(); pw = _D.power_forward(); gasf = _D.aeco_forward(); ad = _D.adder_params(); mn = ad['monthly']
real = df.px.resample('MS').mean(); cnt = df.px.resample('MS').size(); real = real[cnt >= 25]
END = pd.Timestamp('2026-09-25'); START = END - pd.Timedelta(days=365)
days = sorted(pw[(pw.ed >= START) & (pw.ed <= END)].ed.unique())
K = 200

def month_sims(m):
    """scenario matrix of daily cushions for month m, from information available before m: returns (K, ndays)"""
    tr = panel.trailing(df, m); me = m + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    if len(tr) < 250: return None, None
    row = mn[mn.month.astype(str) == m.strftime('%Y-%m')]
    hist = mn[mn.month.astype(str) < m.strftime('%Y-%m')]
    if len(row) and len(hist):
        sea = hist[hist.mo == m.month].adder.mean() if (hist.mo == m.month).any() else hist.adder.mean()
        base = float(hist.tail(12).actual.mean()); ref_cap = float(tr.gas_av.mean() + base)
        plan = ref_cap - float(row.plan.iloc[0] + sea)
    else:
        plan = float(tr.gas_av.tail(21).mean())
    t = tr.dropna(subset=['imports', 'cush_int'])
    b = np.linalg.lstsq(np.column_stack([np.ones(len(t)), t.cush_int.values]), t.imports.values, rcond=None)[0]
    zg = rng.standard_normal(K) * 400; zl = rng.standard_normal(K) * 300
    cols = []
    for d in pd.date_range(m, me):
        c = climatology(tr, d)
        load = rng.choice(c.ail.dropna().values, K) + zl; wind = rng.choice(c.wind.dropna().values, K); sol = rng.choice(c.solar.dropna().values, K)
        g = plan + zg + rng.standard_normal(K) * 150
        ci = g + tr.bio.tail(30).mean() + sol + wind - load
        cols.append(ci + b[0] + b[1] * ci)
    return Curve(tr), np.column_stack(cols)

cache = {}
def fair(m, gas):
    if m not in cache: cache[m] = month_sims(m)
    cv, S = cache[m]
    if cv is None: return None
    draws = np.column_stack([cv.draw(S[:, j], gas, rng) for j in range(S.shape[1])])
    mm = draws.mean(1); pt = cv.read(S.ravel(), gas).mean(); mm = mm + (pt - mm.mean())
    return {'fair': pt, 'p25': np.percentile(mm, 25), 'p75': np.percentile(mm, 75), 'p10': np.percentile(mm, 10), 'p90': np.percentile(mm, 90)}

rows = []
for d in days:
    m1 = (pd.Timestamp(d) + pd.offsets.MonthBegin(1)).normalize()
    for k, m in enumerate([m1, m1 + pd.DateOffset(months=1)], start=1):
        if m not in real.index: continue
        s = pw[(pw.ed == d) & (pw.m == m)]
        if not len(s): continue
        gs = gasf[(gasf.m == m) & (gasf.ed <= d)].sort_values('ed')
        gas = float(gs.gas.iloc[-1]) if len(gs) else np.nan
        if not np.isfinite(gas): continue
        fv = fair(m, gas)
        if fv is None: continue
        rows.append({'d': pd.Timestamp(d), 'k': f'M+{k}', 'm': m, 'settle': float(s.flat.iloc[0]), 'gas': gas, 'real': float(real[m]), **fv})
T = pd.DataFrame(rows); T['edge'] = T.fair - T.settle; T['lead'] = (T.m - T.d).dt.days
T['sig'] = np.where(T.settle > T.p75, 'SELL', np.where(T.settle < T.p25, 'BUY', ''))
T['pnl_sell'] = T.settle - T.real; T['pnl_buy'] = -T.pnl_sell
T.to_csv('research/trading_sim.csv', index=False)
print(f'{len(T)} contract-days, {T.d.min():%Y-%m-%d} -> {T.d.max():%Y-%m-%d}, months {T.m.min():%b-%y}..{T.m.max():%b-%y}')

print('\n=== A. every day, every contract: P&L per MWh if you traded the signal that day and held to settlement ===')
for k in ['M+1', 'M+2', 'all']:
    s = T if k == 'all' else T[T.k == k]
    sig = s[s.sig != '']; pnl = np.where(sig.sig == 'SELL', sig.pnl_sell, sig.pnl_buy)
    print(f'{k:<4} signal days {len(sig):4d} of {len(s):4d}  (sell {int((sig.sig=="SELL").sum()):3d} / buy {int((sig.sig=="BUY").sum()):3d})   '
          f'avg P&L {pnl.mean():+6.2f} $/MWh   hit {100*(pnl>0).mean():4.0f}%   '
          f'sells avg {sig[sig.sig=="SELL"].pnl_sell.mean():+6.2f}   buys avg {sig[sig.sig=="BUY"].pnl_buy.mean():+6.2f}')
    print(f'      benchmarks: always sell {s.pnl_sell.mean():+6.2f} (hit {100*(s.pnl_sell>0).mean():.0f}%)   always buy {s.pnl_buy.mean():+6.2f}')

print('\n=== B. one position per contract per calendar month of trading: enter on the first signal day, hold to settle ===')
T['tm'] = T.d.dt.to_period('M')
trades = []
for (m, k, tm), s in T.sort_values('d').groupby(['m', 'k', 'tm']):
    f = s[s.sig != '']
    if not len(f): continue
    e = f.iloc[0]; side = e.sig; pnl = e.pnl_sell if side == 'SELL' else e.pnl_buy
    hrs = 24 * (e.m + pd.offsets.MonthEnd(0)).day
    trades.append({'entered': e.d.strftime('%Y-%m-%d'), 'contract': e.m.strftime('%b-%y'), 'k': k, 'side': side, 'entry': e.settle, 'fair': e.fair,
                   'p25': e.p25, 'p75': e.p75, 'gas': e.gas, 'settled': e.real, 'pnl_mwh': pnl, 'pnl_1MW': pnl * hrs, 'lead_d': int(e.lead),
                   'days_signal_on': int((s.sig == side).sum()), 'days_in_month': len(s)})
TR = pd.DataFrame(trades)
pd.set_option('display.width', 250)
print(TR.round(2).to_string(index=False))
print(f'\ntrades {len(TR)}: sells {int((TR.side=="SELL").sum())}, buys {int((TR.side=="BUY").sum())};  P&L/MWh mean {TR.pnl_mwh.mean():+.2f}, median {TR.pnl_mwh.median():+.2f};  '
      f'hit {100*(TR.pnl_mwh>0).mean():.0f}%;  total on 1 MW per trade ${TR.pnl_1MW.sum():,.0f};  worst {TR.pnl_mwh.min():+.2f}  best {TR.pnl_mwh.max():+.2f}')
print('by side:'); print(TR.groupby('side').pnl_mwh.agg(['count', 'mean', 'median', lambda x: (x > 0).mean()]).round(2))
print('by contract:'); print(TR.groupby('k').pnl_mwh.agg(['count', 'mean', lambda x: (x > 0).mean()]).round(2))
print('\n=== C. stricter: outside P10–P90 only ===')
T['sig2'] = np.where(T.settle > T.p90, 'SELL', np.where(T.settle < T.p10, 'BUY', ''))
s2 = T[T.sig2 != '']; p2 = np.where(s2.sig2 == 'SELL', s2.pnl_sell, s2.pnl_buy)
print(f'signal days {len(s2)} (sell {int((s2.sig2=="SELL").sum())} / buy {int((s2.sig2=="BUY").sum())})  avg P&L {p2.mean():+.2f}  hit {100*(p2>0).mean():.0f}%')
print('\n=== D. by contract month: what the model thought vs what happened (M+1, at ~30 d lead) ===')
q = T[(T.k == 'M+1') & T.lead.between(25, 35)].groupby('m').agg(settle=('settle', 'mean'), fair=('fair', 'mean'), p25=('p25', 'mean'), p75=('p75', 'mean'), real=('real', 'first'), gas=('gas', 'mean'))
q['signal'] = np.where(q.settle > q.p75, 'SELL', np.where(q.settle < q.p25, 'BUY', 'none')); q['market_err'] = q.settle - q.real; q['model_err'] = q.fair - q.real
q.index = q.index.strftime('%b-%y'); print(q.round(1).to_string())
