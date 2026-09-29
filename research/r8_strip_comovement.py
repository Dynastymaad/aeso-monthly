"""R8: how do the prompt months move together?  Daily changes in the flat settle of M (balance-of-month strip),
M+1, M+2, M+3, regressed on each other.  Also the level relationship (M+1 vs M) for the mean-reversion read."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from model import data as _D
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pw = _D.power_forward(); bom = _D.power_bom()
# balance-of-month = mean of the daily XDT strips from today to month end, per effective date
bom['me'] = bom.d + pd.offsets.MonthEnd(0)
b = bom[(bom.d >= bom.ed) & (bom.d.dt.to_period('M') == bom.ed.dt.to_period('M'))].groupby('ed').flat.mean().rename('M0')
rows = {}
for ed, s in pw.groupby('ed'):
    m1 = (ed + pd.offsets.MonthBegin(1)).normalize()
    r = {}
    for k in (1, 2, 3):
        mk = m1 + pd.DateOffset(months=k - 1); v = s[s.m == mk].flat
        r[f'M{k}'] = float(v.iloc[0]) if len(v) else np.nan
    rows[ed] = r
F = pd.DataFrame(rows).T.join(b, how='left').sort_index()
F = F[F.index >= '2024-08-01']
# keep only days where the contract identity did not roll (same calendar month as the previous row)
same = F.index.to_period('M') == F.index.to_period('M').shift(1) if False else pd.Series(F.index.to_period('M'), index=F.index).eq(pd.Series(F.index.to_period('M'), index=F.index).shift(1))
D = F.diff()[same.values]
D = D.dropna(subset=['M1', 'M2', 'M3'])
print(f'{len(F)} settle days {F.index.min():%Y-%m-%d} -> {F.index.max():%Y-%m-%d};  {len(D)} daily changes within a contract month; {D.M0.notna().sum()} with a balance-of-month strip')
def ols(y, x):
    ok = y.notna() & x.notna(); y = y[ok].values; x = x[ok].values
    A = np.column_stack([np.ones(len(x)), x]); bb, *_ = np.linalg.lstsq(A, y, rcond=None); e = y - A @ bb
    se = np.sqrt((e @ e) / (len(y) - 2) / ((x - x.mean()) ** 2).sum()); r2 = 1 - (e @ e) / ((y - y.mean()) ** 2).sum()
    return bb[1], bb[1] / se, r2, len(y)
print('\n=== A. daily changes: d(M+k) on d(M)  — "if M moves $1, M+k moves $beta" ===')
for base in ['M0', 'M1']:
    print(f'base = {"balance-of-month" if base == "M0" else "prompt month M+1"}')
    for k in ['M1', 'M2', 'M3']:
        if k == base: continue
        bb, t, r2, n = ols(D[k], D[base]); print(f'   d{k} = {bb:+.3f} x d{base}   (t {t:+.1f}, R2 {r2:.2f}, n {n})')
print('\n=== B. same, in percentage terms: dlog(M+k) on dlog(M) ===')
L = np.log(F.clip(lower=1)).diff()[same.values].dropna(subset=['M1', 'M2', 'M3'])
for base in ['M0', 'M1']:
    for k in ['M1', 'M2', 'M3']:
        if k == base: continue
        bb, t, r2, n = ols(L[k], L[base]); print(f'   %d{k} = {bb:+.3f} x %d{base}   (t {t:+.1f}, R2 {r2:.2f}, n {n})')
print('\n=== C. only on days M moved a lot (|dM| >= $2): does the ratio hold? ===')
for base in ['M0', 'M1']:
    big = D[D[base].abs() >= 2]
    for k in ['M1', 'M2', 'M3']:
        if k == base: continue
        bb, t, r2, n = ols(big[k], big[base]); same_sign = float(((big[k] > 0) == (big[base] > 0)).mean())
        print(f'   {base}->{k}: beta {bb:+.3f}  R2 {r2:.2f}  same direction {100*same_sign:.0f}%  n {n}')
print('\n=== D. weekly changes (5 settle days), which is closer to how a position is held ===')
W = F.diff(5)[same.values & pd.Series(F.index.to_period('M'), index=F.index).eq(pd.Series(F.index.to_period('M'), index=F.index).shift(5)).values].dropna(subset=['M1', 'M2', 'M3'])
for base in ['M0', 'M1']:
    for k in ['M1', 'M2', 'M3']:
        if k == base: continue
        bb, t, r2, n = ols(W[k], W[base]); print(f'   {base}->{k}: beta {bb:+.3f} (t {t:+.1f}) R2 {r2:.2f} n {n}')
print('\n=== E. the reverse: does a move in M+2 pull M+1?  and M+1 -> M ===')
for a, bq in [('M2', 'M1'), ('M1', 'M0'), ('M3', 'M2')]:
    bb, t, r2, n = ols(D[bq], D[a]); print(f'   d{bq} = {bb:+.3f} x d{a}  (R2 {r2:.2f})')
print('\n=== F. levels: spread M+1 minus M (balance-of-month), by calendar month ===')
F['spread'] = F.M1 - F.M0; F['mo'] = F.index.month
print(F.groupby('mo').spread.agg(['mean', 'median', 'std', 'count']).round(1).to_string())
print('\ncorr of levels: M0~M1 %.2f  M1~M2 %.2f  M2~M3 %.2f' % (F.M0.corr(F.M1), F.M1.corr(F.M2), F.M2.corr(F.M3)))
F.round(2).to_csv('research/strip_levels.csv')
