"""Build the daily research panel from the staged caches. Writes research/daily.csv"""
import pandas as pd, numpy as np, json, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import data as _D
D = _D.CUSHION / 'cache'          # composition.csv, wx_hourly.csv live in the cushion repo
OUT = Path(__file__).resolve().parent / 'daily.csv'

def pool():
    return _D.pool_price()

def build():
    p = pool()
    h = p.to_frame()
    h['he'] = h.index.hour + 1
    h['peak'] = h.he.between(8, 23)
    h['d'] = h.index.normalize()
    g = h.groupby('d')
    day = pd.DataFrame({'px': g.price.mean(), 'px_pk': h[h.peak].groupby('d').price.mean(),
                        'px_op': h[~h.peak].groupby('d').price.mean(),
                        'px_max': g.price.max(), 'n_hrs': g.price.size(),
                        'hrs_gt100': g.price.apply(lambda s: (s > 100).sum()),
                        'hrs_gt300': g.price.apply(lambda s: (s > 300).sum())})
    c = pd.read_csv(D / 'composition.csv', parse_dates=['datetime_begin'])
    c = c[c.lead_bucket == -1].drop_duplicates('datetime_begin', keep='last').set_index('datetime_begin').sort_index()
    c['gas'] = c[['sc', 'cogen', 'cc', 'gfs']].sum(axis=1)
    c['cush'] = c.gas + c.biomass_and_other + c.wind + c.solar - (c.ail - c.net_imports_actual_scheduled)
    c['cush_int'] = c.gas + c.biomass_and_other + c.wind + c.solar - c.ail   # no help from neighbours
    c['d'] = c.index.normalize(); c['he'] = c.index.hour + 1
    cg = c.groupby('d')
    cd = pd.DataFrame({'ail': cg.ail.mean(), 'ail_pk': cg.ail.max(), 'wind': cg.wind.mean(), 'solar': cg.solar.mean(),
                       'gas_av': cg.gas.mean(), 'hydro': cg.hydro.mean(), 'storage': cg.energy_storage.mean(),
                       'imports': cg.net_imports_actual_scheduled.mean(),
                       'cush': cg.cush.mean(), 'cush_min': cg.cush.min(), 'cush_int': cg.cush_int.mean(),
                       'cush_eve': c[c.he.between(17, 21)].groupby('d').cush.mean(),
                       'n_comp': cg.ail.size()})
    w = pd.read_csv(D / 'wx_hourly.csv', parse_dates=['t']).set_index('t')
    w['d'] = w.index.normalize(); wg = w.groupby('d')
    wd = pd.DataFrame({'temp': wg.temp.mean(), 'tmax': wg.temp.max(), 'tmin': wg.temp.min(),
                       'wspd': wg.wind.mean(), 'rad': wg.rad.mean(), 'cloud': wg.cloud.mean()})
    wd['hdd'] = (18 - wd.temp).clip(lower=0); wd['cdd'] = (wd.temp - 18).clip(lower=0)
    df = day.join(cd, how='left').join(wd, how='left')
    df.index.name = 'd'
    df['dow'] = df.index.dayofweek; df['wkend'] = (df.dow >= 5).astype(int)
    df['month'] = df.index.month; df['doy'] = df.index.dayofyear
    df = df[df.n_hrs >= 20]
    df.to_csv(OUT)
    return df

if __name__ == '__main__':
    df = build()
    print(df.shape, df.index.min(), df.index.max())
    print(df.describe().T[['count', 'mean', 'std', 'min', 'max']].round(1))
