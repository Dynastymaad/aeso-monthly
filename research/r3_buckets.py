"""R3: bucketing. Walk-forward over months: with the trailing 365 days of daily history, how should the days be
grouped so that the coming month's mean price is read best?  Benchmarks: naive, last year, the forward market."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from model import data as _D
_FWD_DAILY = str(_D.CUSHION / 'cache' / 'fwd_aeso_daily.csv'); _GAS = str(_D.CACHE / 'gas_fwd.csv'); _PWR = str(_D.CACHE / 'power_fwd.csv')
import pandas as pd, numpy as np, warnings
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')
df = pd.read_csv('research/daily.csv', parse_dates=['d'], index_col='d')
df = df[(df.cush_min > -3000) & (df.ail_pk < 15000)].dropna(subset=['cush', 'temp', 'cush_eve'])
df['lpx'] = np.log1p(df.px); df['net_load'] = df.ail - df.wind - df.solar
fwd = pd.read_csv(_FWD_DAILY, parse_dates=['EffectiveDate', 'Strip'])
fwd = fwd[(fwd.ExchangeCode == 'XDT') & (fwd.Strip.dt.day == 1)]

months = [m for m in pd.period_range('2025-10', '2026-09', freq='M')]
FEATS = ['cush', 'cush_eve', 'net_load', 'hdd', 'wspd', 'gas_av']

def curve_read(tr, target_cush, bw=250):
    """kernel mean of log price around a cushion level, on the training days"""
    w = np.exp(-0.5 * ((tr.cush - target_cush) / bw) ** 2)
    return float(np.expm1((w * tr.lpx).sum() / w.sum()))

def curve_read_dist(tr, cush_days, bw=250):
    """average the curve over a whole distribution of daily cushions (keeps the convexity)"""
    return float(np.mean([curve_read(tr, c, bw) for c in cush_days]))

rows = []
for m in months:
    t0, t1 = m.start_time, m.end_time
    te = df[(df.index >= t0) & (df.index <= t1)]
    tr = df[(df.index < t0) & (df.index >= t0 - pd.Timedelta(days=365))]
    ly = df[(df.index >= t0 - pd.DateOffset(years=1)) & (df.index <= t1 - pd.DateOffset(years=1))]
    if len(te) < 20 or len(tr) < 300 or len(ly) < 20: continue
    real = te.px.mean()
    # forward mark ~30 days before the month
    fm = fwd[(fwd.Strip == t0) & (fwd.EffectiveDate <= t0 - pd.Timedelta(days=25)) & (fwd.EffectiveDate >= t0 - pd.Timedelta(days=40))].Price
    r = {'m': str(m), 'real': real, 'naive365': tr.px.mean(), 'trail30': tr.px.tail(30).mean(),
         'lastyear': ly.px.mean(), 'fwd30d': fm.mean() if len(fm) else np.nan}
    # --- buckets
    # 1 calendar-season bucket: same season (+-45 days of doy) in the trailing year
    doy = te.doy.mean(); sd = np.minimum(np.abs(tr.doy - doy), 365 - np.abs(tr.doy - doy))
    r['season_bucket'] = tr[sd <= 45].px.mean()
    # 2 cushion tercile bucket, perfect foresight of the month\'s mean cushion
    q = pd.qcut(tr.cush, 3, labels=False); tb = int(pd.cut([te.cush.mean()], np.quantile(tr.cush, [0, 1/3, 2/3, 1]), labels=False, include_lowest=True)[0])
    r['cush_tercile_pf'] = tr[q == tb].px.mean()
    # 3 cushion kernel curve, perfect-foresight monthly mean cushion / daily distribution
    r['curve_pf_mean'] = curve_read(tr, te.cush.mean())
    r['curve_pf_dist'] = curve_read_dist(tr, te.cush.values)
    # 4 realistic expected cushion: last year\'s same-month daily cushions shifted by the change in gas availability
    #   (gas_av last 30 days vs same window last year) - a stand-in for the outage plan
    shift = tr.gas_av.tail(30).mean() - df[(df.index >= t0 - pd.DateOffset(years=1) - pd.Timedelta(days=30)) & (df.index < t0 - pd.DateOffset(years=1))].gas_av.mean()
    exp_cush = ly.cush.values + shift
    r['curve_ly_dist'] = curve_read_dist(tr, exp_cush)
    r['curve_ly_dist_noshift'] = curve_read_dist(tr, ly.cush.values)
    # 5 k-means on fundamentals (perfect foresight of the month\'s mean fundamentals)
    X = tr[FEATS]; mu, sd_ = X.mean(), X.std(); Z = (X - mu) / sd_
    km = KMeans(5, n_init=5, random_state=0).fit(Z)
    lab = km.predict(((te[FEATS].mean() - mu) / sd_).values.reshape(1, -1))[0]
    r['kmeans5_pf'] = tr.px[km.labels_ == lab].mean()
    # 6 nearest-days analogue: 60 nearest training days in standardised fundamentals to the month\'s daily profile
    Zt = ((te[FEATS] - mu) / sd_).values; Ztr = Z.values
    dist = ((Zt[:, None, :] - Ztr[None, :, :]) ** 2).sum(-1)
    r['analogue60_pf'] = np.mean([tr.px.values[np.argsort(dist[i])[:60]].mean() for i in range(len(te))])
    rows.append(r)

R = pd.DataFrame(rows).set_index('m')
print(R.round(1).to_string())
print('\n=== scorecard across %d months ===' % len(R))
sc = []
for c in R.columns:
    if c == 'real': continue
    e = R[c] - R.real
    sc.append({'method': c, 'MAE': e.abs().mean(), 'bias': e.mean(), 'corr': R[c].corr(R.real), 'RMSE': np.sqrt((e ** 2).mean())})
print(pd.DataFrame(sc).set_index('method').round(2).sort_values('MAE').to_string())
