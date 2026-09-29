"""R6: the realistic question. With NO weather skill, how well can the month's cushion distribution be built from
(load climatology, wind climatology, persisted gas availability), and what does the price read look like then?"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from model import data as _D
_FWD_DAILY = str(_D.CUSHION / 'cache' / 'fwd_aeso_daily.csv'); _GAS = str(_D.CACHE / 'gas_fwd.csv'); _PWR = str(_D.CACHE / 'power_fwd.csv')
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('research/daily.csv', parse_dates=['d'], index_col='d')
df = df[(df.cush_min > -3000) & (df.ail_pk < 15000)].dropna(subset=['cush', 'temp', 'cush_eve'])
df['lpx'] = np.log1p(df.px); df['nonwind'] = df.gas_av + df.solar   # supply side ex wind (biomass folded into cush anyway)
df['other'] = df.cush - df.wind + df.ail    # = gas_av + bio + solar + imports  (supply ex wind, incl imports)
fwd = pd.read_csv(_FWD_DAILY, parse_dates=['EffectiveDate', 'Strip']); fwd = fwd[(fwd.ExchangeCode == 'XDT') & (fwd.Strip.dt.day == 1)]
rng = np.random.default_rng(3)

def kread(tr, cush_days, bw=250):
    return float(np.mean([np.expm1((np.exp(-0.5 * ((tr.cush - c) / bw) ** 2) * tr.lpx).sum() / np.exp(-0.5 * ((tr.cush - c) / bw) ** 2).sum()) for c in cush_days]))

rows = []
for m in pd.period_range('2025-10', '2026-09', freq='M'):
    t0, t1 = m.start_time, m.end_time
    te = df[(df.index >= t0) & (df.index <= t1)]; tr = df[(df.index < t0) & (df.index >= t0 - pd.Timedelta(days=365))]
    if len(te) < 20 or len(tr) < 300: continue
    doy = te.doy.mean(); sd = np.minimum(np.abs(tr.doy - doy), 365 - np.abs(tr.doy - doy))
    clim = tr[sd <= 30]                                  # same season, trailing year
    # realistic components
    load_c = clim.ail.values                             # load climatology (weather-free)
    wind_c = clim.wind.values                            # wind climatology
    gas_now = tr.gas_av.tail(21).mean()                  # persisted availability (outage plan stand-in)
    other_c = clim.other.values - clim.gas_av.values + gas_now   # swap in current gas availability
    n = 400
    exp_cush = rng.choice(other_c, n) + rng.choice(wind_c, n) - rng.choice(load_c, n)
    # half-realistic: true load (as if the temperature ensemble were perfect) + wind climatology + persisted gas
    exp_cush_tl = rng.choice(other_c, n) + rng.choice(wind_c, n) - rng.choice(te.ail.values, n)
    # true load + true gas availability + wind climatology
    exp_cush_tlg = rng.choice(te.other.values, n) + rng.choice(wind_c, n) - rng.choice(te.ail.values, n)
    fm = fwd[(fwd.Strip == t0) & (fwd.EffectiveDate <= t0 - pd.Timedelta(days=25)) & (fwd.EffectiveDate >= t0 - pd.Timedelta(days=40))].Price
    rows.append({'m': str(m), 'real': te.px.mean(), 'fwd30d': fm.mean(),
                 'cush_real': te.cush.mean(), 'cush_clim': exp_cush.mean(), 'cush_tl': exp_cush_tl.mean(),
                 'read_pf': kread(tr, te.cush.values), 'read_clim': kread(tr, exp_cush), 'read_trueload': kread(tr, exp_cush_tl),
                 'read_trueload_gas': kread(tr, exp_cush_tlg), 'read_seasonmean': clim.px.mean()})
R = pd.DataFrame(rows).set_index('m'); print(R.round(1).to_string())
sc = []
for c in ['fwd30d', 'read_seasonmean', 'read_clim', 'read_trueload', 'read_trueload_gas', 'read_pf']:
    e = R[c] - R.real; sc.append({'method': c, 'MAE': e.abs().mean(), 'bias': e.mean(), 'corr': R[c].corr(R.real)})
print(pd.DataFrame(sc).set_index('method').round(2).to_string())
print('\ncushion forecast skill: corr(clim, real)=%.2f  MAE=%.0f ;  corr(trueload, real)=%.2f MAE=%.0f' % (
    R.cush_clim.corr(R.cush_real), (R.cush_clim - R.cush_real).abs().mean(), R.cush_tl.corr(R.cush_real), (R.cush_tl - R.cush_real).abs().mean()))
print('\nvariance decomposition of monthly mean cushion (12 months): sd of monthly means -> ail %.0f  wind %.0f  gas_av %.0f  imports %.0f  cush %.0f' % tuple(
    df.resample('MS').mean(numeric_only=True).loc['2025-10':'2026-09'][c].std() for c in ['ail', 'wind', 'gas_av', 'imports', 'cush']))
