#!/usr/bin/env python3
"""
update.py — one command: pull what is new, rebuild the read, write the page.

    python update.py                # pull (SQL + AESO API), build, page
    python update.py --no-pull      # rebuild from cache/ only (no network)
    python update.py --asof 2026-09-15   # rebuild as of another date (testing)

Stages:
  1 pull      AB-NIT gas + AESO monthly forwards (last 45 days, merged into cache/),
              settled pool price top-up from the AESO API
  2 panel     daily panel from the cushion repo's caches + this repo's
  3 read      curve on the trailing year, outlook, the contracts
  4 verify    walk-forward backtest, regressions, buckets
  5 page      docs/index.html, history/reads.csv

Run it AFTER the cushion repo's morning refresh: it reads that repo's
composition, weather, outlook, gencap and outage report rather than pulling
them twice. config.json says where that repo is.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model import data, panel, read, page          # noqa: E402
import pull_history                                 # noqa: E402

def say(s): print(f'  {s}', flush=True)
def stage(n, s): print(f'\n[{n}/5] {s}', flush=True)


def pull(days=45):
    """Incremental: the last `days` of settles, merged into the existing cache files."""
    import pyodbc
    cfg = json.loads(pull_history.find('db.json').read_text())
    cn = pyodbc.connect(pull_history.conn_str(cfg), timeout=120)
    for out, where in [('gas_fwd.csv', "NodeName = 'AB-NIT' AND CommodityName = 'Natural Gas'"),
                       ('power_fwd.csv', "NodeName = 'AESO SMP' AND MonthlyDaily = 'M'")]:
        t0 = time.time()
        new = pd.read_sql(pull_history.SQL.format(days=days, where=where), cn)
        p = data.CACHE / out
        if p.exists():
            old = pd.read_csv(p)
            key = ['EffectiveDate', 'Strip', 'ExchangeCode']
            for c in ('EffectiveDate', 'Strip'):
                old[c] = pd.to_datetime(old[c]).dt.strftime('%Y-%m-%d'); new[c] = pd.to_datetime(new[c]).dt.strftime('%Y-%m-%d')
            df = pd.concat([old, new]).drop_duplicates(key, keep='last').sort_values(key)
        else:
            df = new
        df.to_csv(p, index=False)
        say(f'{out:<15} +{len(new):,} rows pulled, {len(df):,} on file, latest {df.EffectiveDate.max()}  ({time.time() - t0:.0f}s)')
    cn.close()
    pull_history.pull_pool(2021)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-pull', action='store_true')
    ap.add_argument('--asof', default=None)
    ap.add_argument('--months', type=int, default=None, help='contracts ahead (default from config.json, else 3)')
    a = ap.parse_args()
    cfg = data.CFG
    n_ahead = a.months or int(cfg.get('contracts_ahead', 3))

    stage(1, 'pull — CANPOWER forwards and the AESO price top-up')
    if a.no_pull: say('skipped (--no-pull)')
    else:
        try: pull()
        except Exception as ex: say(f'pull FAILED, continuing from cache/: {ex}')

    stage(2, 'panel')
    df = panel.build()
    asof = pd.Timestamp(a.asof).normalize() if a.asof else (df.index.max() + pd.Timedelta(days=1)).normalize()
    say(f'{len(df):,} days {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}; as of {asof:%Y-%m-%d}')
    say(f"cushion repo: {data.CUSHION}   adder: {data.ADDER}")

    stage(3, 'read — curve, outlook, contracts')
    t0 = time.time()
    contracts, cv, ol, tr, dec = read.read_contracts(df, asof, n=n_ahead)
    for c in contracts:
        s = f"{c['label']:<11} fair {c['fair_mean']:6.2f}  (P10 {c['fair_p10']:5.1f} · P90 {c['fair_p90']:5.1f})  cushion {c['cush_mean'] or 0:5.0f}"
        if 'settle_flat' in c: s += f"   market {c['settle_flat']:6.2f}  edge {c['edge']:+6.2f}  pct(fair) {100 * c['settle_pct_fair']:3.0f}%"
        say(s)
    say(f'{time.time() - t0:.1f}s')

    stage(4, 'verify — backtest, regressions, buckets')
    t0 = time.time()
    R, score, prem = read.backtest(df, asof)
    for k, v in score.items(): say(f"{k:<10} MAE {v['mae']:5.2f}  bias {v['bias']:+5.2f}  corr {v['corr']:.2f}  n={v['n']}")
    rg = read.regressions(tr)
    bk, seas = read.buckets(tr, contracts)
    strip = read.strip_betas(asof)
    exc = read.excursions(asof)
    say(f'{time.time() - t0:.1f}s')

    stage(5, 'page')
    mo = df[(df.index >= asof - pd.Timedelta(days=365))].resample('MS').mean(numeric_only=True)
    ad = data.adder_params()
    fwd_end = pd.read_csv(data.CACHE / 'power_fwd.csv', usecols=['EffectiveDate']).EffectiveDate.max()
    meta = {'asof': asof, 'spread': float(cfg.get('spread', 5.0)), 'price_end': data.pool_price().index.max(), 'fwd_end': str(fwd_end)[:10], 'K': read.K,
            'adder_yf': ad['yf'], 'adder_n': len(ad['monthly']) if ad.get('monthly') is not None else 0,
            'sd_gas': float(mo.gas_av.std()), 'sd_load': float(mo.ail.std()), 'sd_wind': float(mo.wind.std())}
    (ROOT / 'docs').mkdir(exist_ok=True); (ROOT / 'history').mkdir(exist_ok=True)
    page.build(ROOT / 'docs' / 'index.html', contracts, cv, ol, tr, bk, seas, rg, R, score, prem, meta, strip, exc, dec)
    (ROOT / 'docs' / '.nojekyll').touch()
    # the record: one row per contract per run, so the read can be scored against the settle later
    rec = pd.DataFrame([{k: v for k, v in c.items() if k != 'band_w'} for c in contracts]); rec.insert(0, 'asof', asof.strftime('%Y-%m-%d'))
    hp = ROOT / 'history' / 'reads.csv'
    if hp.exists():
        old = pd.read_csv(hp); rec = pd.concat([old, rec]).drop_duplicates(['asof', 'label'], keep='last')
    rec.to_csv(hp, index=False)
    ol.table.round(1).to_csv(ROOT / 'history' / f'outlook_{asof:%Y-%m-%d}.csv')
    say(f'docs/index.html written; {len(rec)} reads on record')
    print('\ndone.', flush=True)


if __name__ == '__main__':
    main()
