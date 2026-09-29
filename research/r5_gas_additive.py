"""R5: gas as an additive fuel-cost term rather than a ratio. Alberta gas goes to ~0 in summer, so price/AECO is
undefined exactly when you need it. Test  px = f(cush) + beta*aeco  against  px = f(cush)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from model import data as _D
_FWD_DAILY = str(_D.CUSHION / 'cache' / 'fwd_aeso_daily.csv'); _GAS = str(_D.CACHE / 'gas_fwd.csv'); _PWR = str(_D.CACHE / 'power_fwd.csv')
import pandas as pd, numpy as np, statsmodels.api as sm, warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('research/daily_hr.csv', parse_dates=['d'], index_col='d')
d0 = pd.read_csv('research/daily.csv', parse_dates=['d'], index_col='d')
df = df.join(d0[['px_pk', 'px_op', 'hdd', 'wkend', 'cush_eve', 'ail', 'wind']])
df['lpx'] = np.log1p(df.px)
g = pd.read_csv(_GAS, parse_dates=['EffectiveDate', 'Strip'])
mfwd = g[g.ExchangeCode == 'XAV'].rename(columns={'EffectiveDate': 'ed', 'Strip': 'm', 'Price': 'gasfwd'})

def kcurve(tr, ycol, xs, bw=250):
    return np.array([(np.exp(-0.5 * ((tr.cush - c) / bw) ** 2) * tr[ycol]).sum() / np.exp(-0.5 * ((tr.cush - c) / bw) ** 2).sum() for c in xs])

print('=== A. daily: does AECO carry information once cushion is controlled? ===')
for lab, y in [('log px', df.lpx), ('px', df.px), ('px_op (off-peak, gas-on-margin hours)', df.px_op), ('px_pk', df.px_pk)]:
    X = sm.add_constant(pd.DataFrame({'cush': df.cush, 'cush2': (df.cush / 1000) ** 2, 'aeco': df.aeco, 'wkend': df.wkend}))
    m = sm.OLS(y, X, missing='drop').fit()
    X0 = sm.add_constant(pd.DataFrame({'cush': df.cush, 'cush2': (df.cush / 1000) ** 2, 'wkend': df.wkend}))
    m0 = sm.OLS(y, X0, missing='drop').fit()
    print(f'{lab:<40} R2 without gas {m0.rsquared:.3f} -> with gas {m.rsquared:.3f}   aeco coef {m.params.aeco:+.2f} (t={m.tvalues.aeco:+.1f})')

print('\n=== B. residual of the cushion kernel curve vs AECO, by cushion band (is gas only priced when loose?) ===')
df['fit'] = kcurve(df, 'px', df.cush.values); df['res'] = df.px - df.fit
df['band'] = pd.cut(df.cush, [-1e9, 1200, 1800, 2400, 3000, 1e9], labels=['<1200', '1200-1800', '1800-2400', '2400-3000', '>3000'])
for b, s in df.groupby('band'):
    m = sm.OLS(s.res, sm.add_constant(s.aeco)).fit()
    mo = sm.OLS(s.px_op, sm.add_constant(s.aeco)).fit()
    print(f'{b:<10} n={len(s):3d}  d(price)/d(aeco) = {m.params.aeco:+5.1f} $/MWh per $/GJ (t={m.tvalues.aeco:+.1f})   off-peak px vs aeco slope {mo.params.aeco:+5.1f} (t={mo.tvalues.aeco:+.1f}) intercept {mo.params.const:.1f}')

print('\n=== C. walk-forward monthly read (perfect-foresight cushion): curve only  vs  curve + beta*(gas_fwd - trailing gas) ===')
fwdp = pd.read_csv(_PWR, parse_dates=['EffectiveDate', 'Strip']); fwdp = fwdp[fwdp.ExchangeCode == 'XCU']
rows = []
for m in pd.period_range('2025-04', '2026-09', freq='M'):
    t0, t1 = m.start_time, m.end_time
    te = df[(df.index >= t0) & (df.index <= t1)]; tr = df[(df.index < t0) & (df.index >= t0 - pd.Timedelta(days=365))].copy()
    if len(te) < 20 or len(tr) < 250: continue
    gf = mfwd[(mfwd.m == t0) & (mfwd.ed <= t0 - pd.Timedelta(days=25)) & (mfwd.ed >= t0 - pd.Timedelta(days=40))].gasfwd.mean()
    pf = fwdp[(fwdp.Strip == t0) & (fwdp.EffectiveDate <= t0 - pd.Timedelta(days=25)) & (fwdp.EffectiveDate >= t0 - pd.Timedelta(days=40))].Price.mean()
    # gas-adjusted curve: fit beta on the trailing year with cushion controls, build the curve on px - beta*aeco, add beta*gas_fwd back
    X = sm.add_constant(pd.DataFrame({'cush': tr.cush, 'cush2': (tr.cush / 1000) ** 2, 'aeco': tr.aeco}))
    beta = sm.OLS(tr.px, X).fit().params.aeco
    tr['px_adj'] = tr.px - beta * tr.aeco
    tr['lpx_adj'] = np.log1p(tr.px_adj.clip(lower=0))
    px_curve = float(np.expm1(kcurve(tr, 'lpx', te.cush.values)).mean())
    adj_curve = float(np.expm1(kcurve(tr, 'lpx_adj', te.cush.values)).mean() + beta * gf)
    adj_curve_lin = float(kcurve(tr, 'px_adj', te.cush.values).mean() + beta * gf)
    lin_curve = float(kcurve(tr, 'px', te.cush.values).mean())
    rows.append({'m': str(m), 'real': te.px.mean(), 'gas_fwd': gf, 'gas_real': te.aeco.mean(), 'beta': beta, 'pwr_fwd30': pf,
                 'px_curve': px_curve, 'px_curve_lin': lin_curve, 'gasadj_curve': adj_curve, 'gasadj_curve_lin': adj_curve_lin})
R = pd.DataFrame(rows).set_index('m'); print(R.round(1).to_string())
sc = []
for c in ['pwr_fwd30', 'px_curve', 'px_curve_lin', 'gasadj_curve', 'gasadj_curve_lin']:
    e = R[c] - R.real; sc.append({'method': c, 'MAE': e.abs().mean(), 'bias': e.mean(), 'corr': R[c].corr(R.real)})
print(pd.DataFrame(sc).set_index('method').round(2).to_string())
