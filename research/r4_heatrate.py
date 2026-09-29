"""R4: gas in heat-rate terms. Does dividing by AECO make the cushion->price map more stable past 14 days?"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from model import data as _D
_FWD_DAILY = str(_D.CUSHION / 'cache' / 'fwd_aeso_daily.csv'); _GAS = str(_D.CACHE / 'gas_fwd.csv'); _PWR = str(_D.CACHE / 'power_fwd.csv')
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('research/daily.csv', parse_dates=['d'], index_col='d')
df = df[(df.cush_min > -3000) & (df.ail_pk < 15000)].dropna(subset=['cush', 'temp', 'cush_eve'])
g = pd.read_csv(_GAS, parse_dates=['EffectiveDate', 'Strip'])
# daily AECO spot (C$/GJ): XBG, the last mark on or before the gas day
spot = g[(g.ExchangeCode == 'XBG') & (g.EffectiveDate <= g.Strip)].sort_values('EffectiveDate').groupby('Strip').Price.last()
spot = spot.reindex(pd.date_range(spot.index.min(), spot.index.max())).ffill().rename('aeco')
mfwd = g[g.ExchangeCode == 'XAV'].rename(columns={'EffectiveDate': 'ed', 'Strip': 'm', 'Price': 'gasfwd'})
df = df.join(spot, how='left'); df = df.dropna(subset=['aeco'])
df['hr'] = df.px / df.aeco; df['lhr'] = np.log1p(df.hr); df['lpx'] = np.log1p(df.px)
print(f'{len(df)} days, AECO spot {df.aeco.min():.2f}..{df.aeco.max():.2f}, mean {df.aeco.mean():.2f}')
print(df.resample('QS').agg(aeco=('aeco', 'mean'), px=('px', 'mean'), hr=('hr', 'mean'), cush=('cush', 'mean')).round(2).to_string())

print('\n=== A. persistence at horizon: price vs heat rate  (corr of trailing-k mean with next-h mean) ===')
for name, s in [('price', df.px), ('heat rate', df.hr)]:
    out = {}
    for h in [14, 30, 60]:
        fwd = s[::-1].rolling(h).mean()[::-1].shift(-1)
        out[h] = {k: round(fwd.corr(s.rolling(k).mean()), 2) for k in [30, 60, 90, 180]}
    print(name); print(pd.DataFrame(out).to_string())

print('\n=== B. same, but the monthly mean: does next month\'s HR follow this month\'s HR better than price does? ===')
mo = df.resample('MS').mean(numeric_only=True); mo = mo[df.resample('MS').size() >= 20]
for c in ['px', 'hr']:
    print(f'{c}: corr(month, next month) = {mo[c].autocorr(1):.2f}   corr(month, +2) = {mo[c].autocorr(2):.2f}   cv = {mo[c].std()/mo[c].mean():.2f}')

print('\n=== C. cushion curve stability: fit log(px)~cush and log(hr)~cush per quarter; how much do the intercepts wander? ===')
rows = []
for q, s in df.groupby(df.index.to_period('Q')):
    if len(s) < 60: continue
    b1 = np.polyfit(s.cush, s.lpx, 1); b2 = np.polyfit(s.cush, s.lhr, 1)
    rows.append({'q': str(q), 'n': len(s), 'aeco': s.aeco.mean(), 'px@2000': np.expm1(np.polyval(b1, 2000)), 'hr@2000': np.expm1(np.polyval(b2, 2000)),
                 'px@1000': np.expm1(np.polyval(b1, 1000)), 'hr@1000': np.expm1(np.polyval(b2, 1000))})
r = pd.DataFrame(rows).set_index('q'); print(r.round(2).to_string())
print('coefficient of variation across quarters:  px@2000 %.2f  hr@2000 %.2f   px@1000 %.2f  hr@1000 %.2f' %
      (r['px@2000'].std()/r['px@2000'].mean(), r['hr@2000'].std()/r['hr@2000'].mean(), r['px@1000'].std()/r['px@1000'].mean(), r['hr@1000'].std()/r['hr@1000'].mean()))

print('\n=== D. the real test: walk-forward monthly read, perfect-foresight cushion, price curve vs heat-rate curve x forward gas ===')
def kread(tr, ycol, cush_days, bw=250):
    out = []
    for c in cush_days:
        w = np.exp(-0.5 * ((tr.cush - c) / bw) ** 2); out.append((w * tr[ycol]).sum() / w.sum())
    return float(np.expm1(np.mean(out)))
fwdp = pd.read_csv(_PWR, parse_dates=['EffectiveDate', 'Strip']); fwdp = fwdp[fwdp.ExchangeCode == 'XCU']
rows = []
for m in pd.period_range('2025-04', '2026-09', freq='M'):
    t0, t1 = m.start_time, m.end_time
    te = df[(df.index >= t0) & (df.index <= t1)]; tr = df[(df.index < t0) & (df.index >= t0 - pd.Timedelta(days=365))]
    if len(te) < 20 or len(tr) < 250: continue
    gf = mfwd[(mfwd.m == t0) & (mfwd.ed <= t0 - pd.Timedelta(days=25)) & (mfwd.ed >= t0 - pd.Timedelta(days=40))].gasfwd.mean()
    pf = fwdp[(fwdp.Strip == t0) & (fwdp.EffectiveDate <= t0 - pd.Timedelta(days=25)) & (fwdp.EffectiveDate >= t0 - pd.Timedelta(days=40))].Price.mean()
    rows.append({'m': str(m), 'real': te.px.mean(), 'aeco_real': te.aeco.mean(), 'gas_fwd30': gf, 'pwr_fwd30': pf,
                 'px_curve': kread(tr, 'lpx', te.cush.values),
                 'hr_curve_x_gasfwd': kread(tr, 'lhr', te.cush.values) * gf,
                 'hr_curve_x_gasreal': kread(tr, 'lhr', te.cush.values) * te.aeco.mean(),
                 'px_curve_180': kread(tr.tail(180), 'lpx', te.cush.values),
                 'hr_curve_180_x_gasfwd': kread(tr.tail(180), 'lhr', te.cush.values) * gf,
                 'mkt_hr_x_gasfwd': (tr.hr.mean()) * gf})
R = pd.DataFrame(rows).set_index('m'); print(R.round(1).to_string())
sc = []
for c in R.columns:
    if c in ('real', 'aeco_real', 'gas_fwd30'): continue
    e = R[c] - R.real; sc.append({'method': c, 'MAE': e.abs().mean(), 'bias': e.mean(), 'corr': R[c].corr(R.real)})
print(pd.DataFrame(sc).set_index('method').round(2).sort_values('MAE').to_string())
print('\ngas forward at 30d vs realised AECO: MAE %.2f, corr %.2f' % ((R.gas_fwd30 - R.aeco_real).abs().mean(), R.gas_fwd30.corr(R.aeco_real)))
df[['px', 'aeco', 'hr', 'cush']].to_csv('research/daily_hr.csv')
