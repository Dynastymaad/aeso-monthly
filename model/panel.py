"""
panel.py — the daily panel. One row per day: price (flat / peak / off-peak), the
fundamentals, weather, gas. Everything the curve, the buckets, the regressions
and the backtest read from.

Why daily and not hourly: the contract settles against a monthly mean of
hourly prices, and the things you can say something about 30–60 days ahead
(outages, load level, wind climatology) are daily quantities. The hour-of-day
shape is put back at the end through the peak / off-peak split.
"""
import numpy as np, pandas as pd
from . import data

PEAK = (8, 23)          # HE 8..23 is the 7x16 peak block


def build():
    p = data.pool_price().to_frame('price')
    p['he'] = p.index.hour + 1; p['pk'] = p.he.between(*PEAK); p['d'] = p.index.normalize()
    g = p.groupby('d')
    day = pd.DataFrame({'px': g.price.mean(), 'px_pk': p[p.pk].groupby('d').price.mean(),
                        'px_op': p[~p.pk].groupby('d').price.mean(), 'px_max': g.price.max(),
                        'n_hrs': g.price.size(), 'hrs100': g.price.apply(lambda s: int((s > 100).sum())),
                        'hrs300': g.price.apply(lambda s: int((s > 300).sum()))})
    c = data.composition(); c['d'] = c.index.normalize(); c['he'] = c.index.hour + 1
    cg = c.groupby('d')
    cd = pd.DataFrame({'ail': cg.ail.mean(), 'ail_pk': c[c.he.between(*PEAK)].groupby('d').ail.mean(),
                       'wind': cg.wind.mean(), 'solar': cg.solar.mean(), 'gas_av': cg.gas.mean(),
                       'bio': cg.biomass_and_other.mean(), 'hydro': cg.hydro.mean(),
                       'imports': cg.net_imports_actual_scheduled.mean(),
                       'cush': cg.cush.mean(), 'cush_int': cg.cush_int.mean(),
                       'cush_eve': c[c.he.between(17, 21)].groupby('d').cush.mean(),
                       'cush_min': cg.cush.min(), 'n_comp': cg.ail.size()})
    w = data.weather(); w['d'] = w.index.normalize(); wg = w.groupby('d')
    wd = pd.DataFrame({'temp': wg.temp.mean(), 'tmin': wg.temp.min(), 'tmax': wg.temp.max(),
                       'wspd': wg.wind.mean(), 'rad': wg.rad.mean()})
    wd['hdd'] = (18 - wd.temp).clip(lower=0); wd['cdd'] = (wd.temp - 18).clip(lower=0)
    df = day.join(cd, how='left').join(wd, how='left')
    try:
        df = df.join(data.aeco_daily(), how='left')
    except Exception:
        df['aeco'] = np.nan
    df.index.name = 'd'
    df['dow'] = df.index.dayofweek; df['wkend'] = (df.dow >= 5).astype(int)
    df['month'] = df.index.month; df['doy'] = df.index.dayofyear
    # sanity: a partial day or a snapshot glitch is not a data point
    df = df[df.n_hrs >= 20]
    bad = (df.n_comp < 18) | (df.cush_min < -3000) | (df.ail_pk > 14000)
    df.loc[bad, ['ail', 'ail_pk', 'wind', 'solar', 'gas_av', 'bio', 'hydro', 'imports', 'cush', 'cush_int', 'cush_eve', 'cush_min']] = np.nan
    return df


def trailing(df, asof, days=365, exclude_current_month=True):
    """The history window the model is allowed to see: the rolling year before `asof`,
    with the month in progress left out (its days are the thing being read)."""
    end = pd.Timestamp(asof).normalize()
    if exclude_current_month: end = end.replace(day=1)
    start = end - pd.Timedelta(days=days)
    return df[(df.index >= start) & (df.index < end)].dropna(subset=['cush', 'px'])
