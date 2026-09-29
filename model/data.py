"""
data.py — every input the monthly model reads, and where it comes from.

Nothing here does modelling. Each loader returns a tidy frame and says, in its
docstring, which file it read and why that file is the right source.

Two folders are involved:
  HERE/cache/            what this repo pulls itself (gas, power forwards, long pool price)
  <cushion repo>/cache/  what the 14-day model already pulls every morning
                         (composition, weather, the 44-day cushion outlook)
  <cushion repo>/refresh/  today's AESO feeds (gencap 15d, outage report 90d)
The cushion repo path comes from config.json; the default is the folder next door.
"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent.parent
CACHE = HERE / 'cache'
CFG = {}
for _n in ('config.json', 'config.local.json'):          # config.local.json (gitignored) overrides for testing
    if (HERE / _n).exists(): CFG.update({k: v for k, v in json.loads((HERE / _n).read_text()).items() if v is not None})
CUSHION = Path(CFG.get('cushion_repo') or (HERE.parent / 'aeso-cushion model')).expanduser()
ADDER = Path(CFG.get('adder_dir') or (HERE.parent / 'aeso-outage-adder')).expanduser()
GAS_SUBS = ['sc', 'cogen', 'cc', 'gfs']


def _first(*paths):
    for p in paths:
        if p and Path(p).exists(): return Path(p)
    return None


# ------------------------------------------------------------------ price ---
def pool_price():
    """Hourly settled pool price. cache/pool_long.csv (2021 on, from pull_history.py)
    merged with the cushion repo's poolprice*.json so the last few days are never missing."""
    parts = []
    p = CACHE / 'pool_long.csv'
    if p.exists(): parts.append(pd.read_csv(p, parse_dates=['t']))
    for f in sorted(glob.glob(str(CUSHION / 'refresh' / 'poolprice*.json'))):
        try:
            rows = json.loads(Path(f).read_text(encoding='utf-8-sig'))['return']['Pool Price Report']
            parts.append(pd.DataFrame({'t': [pd.Timestamp(r['begin_datetime_mpt']) for r in rows],
                                       'price': [pd.to_numeric(r.get('pool_price'), errors='coerce') for r in rows]}))
        except Exception:
            continue
    if not parts: raise SystemExit('no pool price anywhere - run pull_history.py first')
    s = pd.concat(parts).dropna().drop_duplicates('t', keep='last').sort_values('t').set_index('t').price
    return s


# ------------------------------------------------------- fundamentals ------
def composition():
    """Hourly actuals from the CANPOWER composition table (the cushion repo's cache/composition.csv).
    lead_bucket -1 is the post-hour snapshot. Gas availability is the four gas sub-fuels."""
    c = pd.read_csv(CUSHION / 'cache' / 'composition.csv', parse_dates=['datetime_begin'])
    if 'lead_bucket' in c: c = c[c.lead_bucket == -1]
    c = c.drop_duplicates('datetime_begin', keep='last').set_index('datetime_begin').sort_index()
    c['gas'] = c[GAS_SUBS].sum(axis=1)
    c['cush'] = c.gas + c.biomass_and_other + c.wind + c.solar - (c.ail - c.net_imports_actual_scheduled)
    c['cush_int'] = c.gas + c.biomass_and_other + c.wind + c.solar - c.ail
    return c


def weather():
    """Hourly province-average weather (Calgary/Edmonton/Pincher Creek), from the cushion repo's wx.py."""
    w = pd.read_csv(CUSHION / 'cache' / 'wx_hourly.csv', parse_dates=['t']).set_index('t').sort_index()
    return w


def aeco_daily():
    """AECO (AB-NIT) daily gas, C$/GJ. Exchange code XBG in ForwardPrices: for each gas day the last mark
    on or before that day. Forward-filled across weekends. NOTE: it goes to ~0 in summer; never divide by it."""
    g = pd.read_csv(CACHE / 'gas_fwd.csv', parse_dates=['EffectiveDate', 'Strip'])
    d = g[(g.ExchangeCode == 'XBG') & (g.EffectiveDate <= g.Strip)].sort_values('EffectiveDate').groupby('Strip').Price.last()
    d = d.reindex(pd.date_range(d.index.min(), d.index.max())).ffill()
    return d.rename('aeco')


def aeco_forward():
    """AECO monthly forward settles, C$/GJ (XAV). Long: EffectiveDate, Strip, Price."""
    g = pd.read_csv(CACHE / 'gas_fwd.csv', parse_dates=['EffectiveDate', 'Strip'])
    return g[g.ExchangeCode == 'XAV'][['EffectiveDate', 'Strip', 'Price']].rename(columns={'EffectiveDate': 'ed', 'Strip': 'm', 'Price': 'gas'})


def power_forward():
    """AESO flat (XCU), 7x16 peak (XCX) and 7x8 off-peak (XCZ) monthly settles. Checked: 16*XCX+8*XCZ = 24*XCU."""
    p = pd.read_csv(CACHE / 'power_fwd.csv', parse_dates=['EffectiveDate', 'Strip'])
    p = p[p.ExchangeCode.isin(['XCU', 'XCX', 'XCZ'])]
    w = p.pivot_table(index=['EffectiveDate', 'Strip'], columns='ExchangeCode', values='Price').reset_index()
    return w.rename(columns={'EffectiveDate': 'ed', 'Strip': 'm', 'XCU': 'flat', 'XCX': 'peak', 'XCZ': 'offpk'})


def power_bom():
    """Balance-of-month flat (XDT) from the cushion repo's fwd_bom.csv if present: daily strips."""
    p = _first(CUSHION / 'cache' / 'fwd_bom.csv', CACHE / 'fwd_bom.csv')
    if p is None: return None
    b = pd.read_csv(p, parse_dates=['EffectiveDate', 'Strip'])
    return b[b.ExchangeCode == 'XDT'][['EffectiveDate', 'Strip', 'Price']].rename(columns={'EffectiveDate': 'ed', 'Strip': 'd', 'Price': 'flat'})


# ----------------------------------------------------- forward supply ------
def gencap():
    """AESO gen-capacity feed, 15 days: per day, gas MC / OP OUT / MBO / AC, and the same for the whole fleet."""
    p = CUSHION / 'refresh' / 'gencap.json'
    if not p.exists(): return None
    j = json.loads(p.read_text(encoding='utf-8-sig'))
    rows = []
    for blk in j.get('return', []):
        for h in blk.get('Hours', []):
            og = h.get('outage_grouping', {})
            rows.append((pd.Timestamp(h['begin_datetime_mpt']).normalize(), blk.get('fuel_type'), blk.get('sub_fuel_type'),
                         float(og.get('MC', 0) or 0), float(og.get('OP OUT', 0) or 0), float(og.get('MBO OUT', 0) or 0), float(og.get('AC', 0) or 0)))
    d = pd.DataFrame(rows, columns=['d', 'f', 's', 'mc', 'op', 'mbo', 'ac'])
    if d.empty: return None
    # hours per day can be < 24 on the last day; normalise by the hour count actually present
    hrs = d[d.f == 'GAS'].groupby('d').size() / d[d.f == 'GAS'].groupby('d').s.nunique()
    g = d[d.f == 'GAS'].groupby('d')[['mc', 'op', 'mbo', 'ac']].sum().div(hrs, axis=0)
    g.columns = ['gas_mc', 'gas_op', 'gas_mbo', 'gas_ac']
    return g


def outage_report():
    """AESO Daily Outage Report (90 days), the cushion repo's refresh/outage_90d.csv: filed MW out by fuel per day."""
    p = _first(CUSHION / 'refresh' / 'outage_90d.csv', CACHE / 'outage_90d.csv')
    if p is None: return None
    txt = p.read_text(errors='ignore').splitlines()
    try:
        hdr = next(i for i, L in enumerate(txt) if 'Date' in L and 'Cogen' in L)
    except StopIteration:
        return None
    o = pd.read_csv(p, skiprows=hdr)
    o.columns = [str(c).strip().strip('"').strip() for c in o.columns]
    o['d'] = pd.to_datetime(o['Date'], format='%d-%b-%Y', errors='coerce')
    o = o.dropna(subset=['d']).set_index('d')
    for c in o.columns:
        if c != 'Date': o[c] = pd.to_numeric(o[c], errors='coerce').fillna(0)
    o['gas_filed'] = o[['SC', 'Cogen', 'CC', 'GFS']].sum(axis=1)
    o['mbo'] = o['MBO'] if 'MBO' in o else 0.0
    o['hydro_out'] = o['Hydro'] if 'Hydro' in o else 0.0
    return o[['gas_filed', 'mbo', 'hydro_out']]


def outlook():
    """The cushion repo's outlook.csv: ECMWF-driven daily load (ail) and wind for 44 days, with the lead."""
    p = CUSHION / 'cache' / 'outlook.csv'
    if not p.exists(): return None
    o = pd.read_csv(p, parse_dates=['date']).set_index('date')
    return o


def adder_params():
    """The outage adder: how much gas outage is not filed yet, by lead and month.
    Read from aeso-outage-adder/monthly_nrgstream.csv (plan vs actual, by month) with the fitted
    lead-time curve. Falls back to the published constants if the folder is not there."""
    mu, sg, base = 2.3746, 1.6152, 1302.0
    seas = pd.Series(base, index=range(1, 13)); yf = 1.0
    p = ADDER / 'monthly_nrgstream.csv'
    m = None
    if p.exists():
        m = pd.read_csv(p); m.columns = [c.strip().lower() for c in m.columns]
        m = m.dropna(subset=['plan', 'actual']).drop_duplicates('month', keep='last').sort_values('month')
        m['adder'] = m.actual - m.plan; m['mo'] = m.month.astype(str).str[5:7].astype(int)
        s = m.groupby('mo').adder.mean()
        for k in range(1, 13):
            if k not in s.index: s.loc[k] = m.adder.mean()
        seas = s.sort_index()
        rec = m.tail(12)
        if len(rec) == 12: yf = float(rec.adder.mean() / seas.reindex(rec.mo).mean())
    return {'mu': mu, 'sg': sg, 'seas': seas, 'yf': yf, 'monthly': m}


def adder_share(lead, mu, sg):
    from math import erf, sqrt
    z = (np.log(np.clip(np.asarray(lead, float), 0.05, None)) - mu) / sg
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
