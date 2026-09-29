"""
pull_history.py -- one-time (and then daily) pull of what the monthly model
needs that the cushion repo does not already cache.

  1. AB-NIT (AECO) gas: daily spot and monthly forward settles  -> cache/gas_fwd.csv
  2. AESO SMP monthly forwards, every exchange code             -> cache/power_fwd.csv
     (XCU is flat; the others are the peak / off-peak / vendor variants,
      pulled so the model can tell them apart instead of guessing)
  3. Settled pool price back to START_YEAR from the AESO API     -> cache/pool_long.csv
     (the cushion repo keeps 760 days; the seasonal buckets want more)

Credentials: reads db.json here, or falls back to the cushion repo's db.json
next door. aeso_key.txt the same way. Nothing is written outside cache/.

    python pull_history.py              # everything
    python pull_history.py --no-api     # SQL only (offline from AESO)
    python pull_history.py --days 1200  # deeper SQL history (default 900)
"""
import argparse, json, sys, time, urllib.request
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'cache'; CACHE.mkdir(exist_ok=True)
NEIGHBOUR = HERE.parent / 'aeso-cushion model'
START_YEAR = 2021


def find(name):
    for p in (HERE / name, NEIGHBOUR / name):
        if p.exists(): return p
    raise SystemExit(f'{name} not found here or in {NEIGHBOUR}')


def conn_str(cfg):
    p = [f"DRIVER={{{cfg['driver']}}}", f"SERVER={cfg['server']}", f"DATABASE={cfg['database']}"]
    if cfg.get('trusted_connection'): p.append('Trusted_Connection=yes')
    else: p += [f"UID={cfg['username']}", f"PWD={cfg['password']}"]
    if cfg.get('trust_server_certificate', True): p.append('TrustServerCertificate=yes')
    return ';'.join(p) + ';'


SQL = """
SELECT   EffectiveDate, Strip, ExchangeCode, CommodityName, MonthlyDaily, NodeName, Price
FROM     Warehouse.dbo.ForwardPrices WITH (NOLOCK)
WHERE    EffectiveDate >= DATEADD(day, -{days}, GETDATE())
  AND    Price IS NOT NULL
  AND    {where}
ORDER BY EffectiveDate, Strip
"""


def pull_sql(days):
    import pyodbc
    cfg = json.loads(find('db.json').read_text())
    cn = pyodbc.connect(conn_str(cfg), timeout=120)
    jobs = [('gas_fwd.csv',   "NodeName = 'AB-NIT' AND CommodityName = 'Natural Gas'"),
            ('power_fwd.csv', "NodeName = 'AESO SMP' AND MonthlyDaily = 'M'")]
    for out, where in jobs:
        t0 = time.time()
        df = pd.read_sql(SQL.format(days=days, where=where), cn)
        df.to_csv(CACHE / out, index=False)
        print(f'  {out:<15} {len(df):>9,} rows  {df.EffectiveDate.min()} -> {df.EffectiveDate.max()}  ({time.time()-t0:.0f}s)', flush=True)
    cn.close()


def pull_pool(start_year):
    key = find('aeso_key.txt').read_text().split(':')[-1].split('=')[-1].strip()
    out = CACHE / 'pool_long.csv'
    old = pd.read_csv(out, parse_dates=['t']) if out.exists() else pd.DataFrame(columns=['t', 'price'])
    today = pd.Timestamp.now().normalize()
    start = pd.Timestamp(f'{start_year}-01-01')
    if len(old): start = max(start, old.t.max().normalize() - pd.Timedelta(days=7))
    parts = [old]
    while start < today + pd.Timedelta(days=1):
        stop = min(start + pd.Timedelta(days=364), today + pd.Timedelta(days=1))
        url = (f'https://apimgw.aeso.ca/public/poolprice-api/v1.1/price/poolPrice'
               f'?startDate={start:%Y-%m-%d}&endDate={stop:%Y-%m-%d}')
        req = urllib.request.Request(url, headers={'accept': 'application/json', 'API-Key': key})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                rows = json.loads(r.read().decode('utf-8-sig'))['return']['Pool Price Report']
            parts.append(pd.DataFrame({'t': [pd.Timestamp(x['begin_datetime_mpt']) for x in rows],
                                       'price': [pd.to_numeric(x.get('pool_price'), errors='coerce') for x in rows]}))
            print(f'  pool {start:%Y-%m-%d} -> {stop:%Y-%m-%d}  {len(rows):,} hours', flush=True)
        except Exception as ex:
            print(f'  pool {start:%Y-%m-%d} -> {stop:%Y-%m-%d}  FAILED: {ex}', flush=True)
        start = stop + pd.Timedelta(days=1)
    s = pd.concat(parts).dropna().drop_duplicates('t', keep='last').sort_values('t')
    s.to_csv(out, index=False)
    print(f'  pool_long.csv    {len(s):>9,} hours  {s.t.min()} -> {s.t.max()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=900)
    ap.add_argument('--no-api', action='store_true')
    ap.add_argument('--start-year', type=int, default=START_YEAR)
    a = ap.parse_args()
    print('[1/2] CANPOWER ForwardPrices: AB-NIT gas + AESO SMP monthly power', flush=True)
    pull_sql(a.days)
    if not a.no_api:
        print(f'[2/2] AESO API: settled pool price from {a.start_year}', flush=True)
        pull_pool(a.start_year)
    print('done.')
