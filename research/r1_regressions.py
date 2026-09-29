"""R1: which fundamentals explain daily price, and at what grain. Prints tables."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from model import data as _D
_FWD_DAILY = str(_D.CUSHION / 'cache' / 'fwd_aeso_daily.csv'); _GAS = str(_D.CACHE / 'gas_fwd.csv'); _PWR = str(_D.CACHE / 'power_fwd.csv')
import pandas as pd, numpy as np, statsmodels.api as sm, warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('research/daily.csv', parse_dates=['d'], index_col='d')
df = df[(df.cush_min > -3000) & (df.ail_pk < 15000)]
df['lpx'] = np.log1p(df.px)
df['lpx_pk'] = np.log1p(df.px_pk)
df['gas_out'] = df.gas_av.rolling(30, min_periods=10).max() - df.gas_av   # crude outage proxy
df['net_load'] = df.ail - df.wind - df.solar                              # residual demand
df['pk_net_load'] = df.ail_pk - df.wind
d = df.dropna(subset=['cush', 'temp', 'cush_eve', 'gas_out'])
print(f'{len(d)} days {d.index.min():%Y-%m-%d} -> {d.index.max():%Y-%m-%d}\n')

def r2(y, X):
    X = sm.add_constant(X); m = sm.OLS(y, X).fit()
    return m.rsquared, m

print('=== UNIVARIATE R^2 on log(1+price), daily, whole sample ===')
rows = []
for c in ['cush', 'cush_int', 'cush_eve', 'cush_min', 'net_load', 'pk_net_load', 'ail', 'ail_pk', 'wind', 'solar',
          'gas_av', 'gas_out', 'imports', 'hydro', 'temp', 'hdd', 'cdd', 'tmax', 'tmin', 'wspd', 'rad', 'wkend']:
    rows.append({'x': c, 'R2_lpx': r2(d.lpx, d[[c]])[0], 'R2_px': r2(d.px, d[[c]])[0],
                 'R2_lpx_pk': r2(d.lpx_pk, d[[c]])[0], 'corr': d.lpx.corr(d[c])})
print(pd.DataFrame(rows).set_index('x').round(3).sort_values('R2_lpx', ascending=False).to_string())

print('\n=== MULTIVARIATE, log(1+price) ===')
specs = {
  'weather only':        ['hdd', 'cdd', 'wspd', 'rad', 'wkend'],
  'load only':           ['ail', 'ail_pk', 'wkend'],
  'load+renewables':     ['ail', 'wind', 'solar', 'wkend'],
  'cushion only':        ['cush'],
  'cushion+eve+min':     ['cush', 'cush_eve', 'cush_min'],
  'cushion+interties':   ['cush', 'imports'],
  'cushion+load+wx':     ['cush', 'ail', 'hdd', 'cdd', 'wspd', 'wkend'],
  'everything':          ['cush', 'cush_eve', 'ail', 'wind', 'solar', 'gas_av', 'imports', 'hdd', 'cdd', 'wspd', 'wkend'],
}
for k, cols in specs.items():
    R, m = r2(d.lpx, d[cols])
    print(f'{k:<22} R2={R:.3f}   ' + '  '.join(f'{c}:{m.tvalues[c]:+.1f}' for c in cols))

print('\n=== SAME, but on monthly averages (n months) ===')
mo = d.resample('MS').mean(numeric_only=True); mo = mo[d.resample('MS').size() >= 20]
mo['lpx'] = np.log1p(mo.px)
print(len(mo), 'months')
for k, cols in specs.items():
    try:
        R, m = r2(mo.lpx, mo[cols]); print(f'{k:<22} R2={R:.3f}')
    except Exception as e: print(k, e)

print('\n=== cushion is non-linear: mean price by cushion decile (daily mean cushion) ===')
d['cb'] = pd.qcut(d.cush, 10, labels=False)
print(d.groupby('cb').agg(cush=('cush', 'mean'), px=('px', 'mean'), px_med=('px', 'median'), pk=('px_pk', 'mean'),
                          hrs100=('hrs_gt100', 'mean'), n=('px', 'size')).round(0).to_string())

print('\n=== weather -> load -> cushion chain: how much of load is weather? ===')
R, m = r2(d.ail, d[['hdd', 'cdd', 'wkend', 'rad']]); print(f'ail ~ hdd+cdd+wkend+rad  R2={R:.3f}  hdd coef={m.params.hdd:.0f} MW/deg-day')
R, m = r2(d.cush, d[['hdd', 'cdd', 'wkend', 'wspd', 'rad', 'gas_av']]); print(f'cush ~ wx+gas_av  R2={R:.3f}')
R, m = r2(d.cush, d[['hdd', 'cdd', 'wkend', 'wspd', 'rad']]); print(f'cush ~ wx only    R2={R:.3f}')
R, m = r2(d.wind, d[['wspd']]); print(f'wind ~ wspd R2={R:.3f}')

print('\n=== interties: do imports respond to price or set it? ===')
print('corr(imports, lpx)=%.2f   corr(imports, cush_int)=%.2f' % (d.imports.corr(d.lpx), d.imports.corr(d.cush_int)))
R, m = r2(d.imports, d[['cush_int', 'hdd']]); print(f'imports ~ cush_int+hdd R2={R:.3f} coef cush_int={m.params.cush_int:.3f}')

print('\n=== stability: same regressions fit per half-year ===')
for lab, sub in d.groupby(d.index.to_period('2Q')):
    if len(sub) < 60: continue
    R1, m1 = r2(sub.lpx, sub[['cush']]); R2_, m2 = r2(sub.lpx, sub[['cush', 'ail', 'hdd', 'cdd', 'wspd', 'wkend']])
    print(f'{lab}  n={len(sub):3d}  mean px={sub.px.mean():6.1f}  R2 cush={R1:.2f} slope={m1.params.cush*1000:+.2f}/1000MW   R2 full={R2_:.2f}')
