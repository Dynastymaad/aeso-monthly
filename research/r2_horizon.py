"""R2: horizon. How predictable is the NEXT h-day mean price from what you know today,
and how does the forward contract behave versus what settles?"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from model import data as _D
_FWD_DAILY = str(_D.CUSHION / 'cache' / 'fwd_aeso_daily.csv'); _GAS = str(_D.CACHE / 'gas_fwd.csv'); _PWR = str(_D.CACHE / 'power_fwd.csv')
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('research/daily.csv', parse_dates=['d'], index_col='d')
px = df.px

print('=== A. persistence: corr(trailing k-day mean, forward h-day mean) and MAE of naive forecast ===')
rows = []
for h in [7, 14, 30, 60]:
    fwd = px[::-1].rolling(h).mean()[::-1].shift(-1)          # mean of next h days
    for k in [7, 14, 30, 60, 90]:
        tr = px.rolling(k).mean()
        ok = fwd.notna() & tr.notna()
        rows.append({'h': h, 'k': k, 'corr': fwd[ok].corr(tr[ok]), 'mae': (fwd[ok] - tr[ok]).abs().mean(),
                     'mae_vs_climo': (fwd[ok] - px[ok].mean()).abs().mean(), 'n': int(ok.sum())})
r = pd.DataFrame(rows)
print(r.pivot(index='k', columns='h', values='corr').round(2)); print(r.pivot(index='k', columns='h', values='mae').round(1))

print('\n=== B. seasonal vs recent: does last-year-same-window help? ===')
fwd30 = px[::-1].rolling(30).mean()[::-1].shift(-1)
ly = px.shift(365).rolling(30).mean().shift(-30)            # same 30-day window one year earlier
tr30 = px.rolling(30).mean()
ok = fwd30.notna() & ly.notna() & tr30.notna()
print('corr(next30, last-year same window)=%.2f   corr(next30, trailing30)=%.2f   n=%d' % (fwd30[ok].corr(ly[ok]), fwd30[ok].corr(tr30[ok]), ok.sum()))
X = pd.DataFrame({'tr30': tr30, 'ly': ly})[ok]
b = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X.values]), fwd30[ok].values, rcond=None)[0]
print('blend fit: next30 = %.1f + %.2f*trailing30 + %.2f*lastyear' % tuple(b))

print('\n=== C. the forward contract: XDT flat monthly settle vs realised month ===')
f = pd.read_csv(_FWD_DAILY, parse_dates=['EffectiveDate', 'Strip'])
f = f[(f.ExchangeCode == 'XDT') & (f.Strip.dt.day == 1)].rename(columns={'EffectiveDate': 'ed', 'Strip': 'm', 'Price': 'fwd'})
# keep only true monthly strips: a first-of-month strip that is not a single-day product. Heuristic: the
# strip's month is >= the month after ed (prompt+), or ed is in the same month (balance-of-month product)
real = px.resample('MS').mean().rename('real'); n = px.resample('MS').size()
real = real[n >= 25]
f = f.join(real, on='m').dropna(subset=['real'])
f['lead'] = (f.m - f.ed).dt.days
f = f[f.lead > 0]
f['err'] = f.fwd - f.real
print('months with a settle:', f.m.dt.strftime('%Y-%m').unique().tolist())
for lo, hi in [(1, 15), (16, 45), (46, 75), (76, 120), (121, 200)]:
    s = f[(f.lead >= lo) & (f.lead <= hi)]
    print(f'lead {lo:>3}-{hi:<3}d  n={len(s):4d}  fwd-real mean={s.err.mean():+6.1f}  MAE={s.err.abs().mean():5.1f}  '
          f'fwd>real {100*(s.err>0).mean():4.0f}%   corr(fwd,real)={s.fwd.corr(s.real):.2f}')
print('\nper month, forward at ~30d lead vs realised:')
s = f[(f.lead >= 25) & (f.lead <= 35)].groupby('m').agg(fwd=('fwd', 'mean'), real=('real', 'first'))
s['prem'] = s.fwd - s.real; print(s.round(1).to_string())

print('\n=== D. where does the contract trade, in percentile terms, within its own life? ===')
# for each (month, ed) the percentile of today\'s settle within all settles for that contract so far
f = f.sort_values(['m', 'ed'])
f['pct_sofar'] = f.groupby('m').fwd.transform(lambda s: s.expanding().apply(lambda x: (x[:-1] <= x[-1]).mean() if len(x) > 5 else np.nan, raw=True))
f['final_move'] = f.real - f.fwd
q = f.dropna(subset=['pct_sofar'])
q['pb'] = pd.cut(q.pct_sofar, [0, .2, .4, .6, .8, 1.01], labels=['0-20', '20-40', '40-60', '60-80', '80-100'], include_lowest=True)
print(q[q.lead.between(20, 90)].groupby('pb').agg(n=('final_move', 'size'), real_minus_fwd=('final_move', 'mean'),
                                                  med=('final_move', 'median'), up=('final_move', lambda x: (x > 0).mean())).round(2).to_string())
