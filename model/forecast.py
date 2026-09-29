"""
forecast.py — the daily cushion outlook, one distribution per forward day.

This is where the monthly model lives or dies. Across the last twelve months
the month-to-month swing in gas availability (sd 899 MW) was larger than load
(549) or wind (315); knowing the outage picture took the read from corr 0.32
to 0.64. So each input is used only as far as it has skill, and each says
which source it came from:

  load     ECMWF-ensemble outlook (cushion repo, 44 days) -> climatology + trend
  wind     ensemble to 9 days (measured skill horizon)     -> same-season history
  gas      AESO gencap AC (15 d) -> filed outage report (90 d) -> seasonal outage norm,
           always less the unfiled adder for the lead (aeso-outage-adder)
  imports  regression on the internal cushion, trailing year
  solar / biomass   same-season history / trailing month

Uncertainty is carried as scenarios, not error bars: K scenarios each get a
month-level shock (outages and load level are persistent) plus daily noise, so
the monthly mean keeps the right spread instead of averaging it away.
"""
import numpy as np, pandas as pd
from . import data

K = 400                                   # scenarios
SD_LOAD = {0: 120, 7: 180, 14: 250, 30: 320, 44: 380, 120: 450}    # daily-mean MW, by lead
SD_GAS = {0: 120, 7: 220, 15: 300, 30: 400, 60: 500, 120: 600}
SD_WIND_ENS = {0: 377, 1: 411, 2: 430, 3: 478, 4: 538, 5: 612, 6: 647, 7: 716, 8: 760, 9: 774}
SD_IMP = 220
WIND_BIAS = 165.0                         # the ensemble over-forecasts wind by about this (outlook.py)


def _interp(tab, lead):
    ks = np.array(sorted(tab)); vs = np.array([tab[k] for k in ks])
    return float(np.interp(lead, ks, vs))


def climatology(df, day, half=20):
    """Trailing-year days within ±half days of the same calendar position, same weekday type."""
    doy = pd.Timestamp(day).dayofyear
    dd = np.minimum(np.abs(df.doy - doy), 365 - np.abs(df.doy - doy))
    s = df[(dd <= half) & (df.wkend == int(pd.Timestamp(day).dayofweek >= 5))]
    if len(s) < 8: s = df[dd <= half]
    return s


class Outlook:
    def __init__(self, df, asof, horizon_end, rng=None, tr=None):
        self.rng = rng or np.random.default_rng(7)
        self.asof = pd.Timestamp(asof).normalize()
        self.days = pd.date_range(self.asof, pd.Timestamp(horizon_end))
        self.df = df
        self.tr = tr if tr is not None else df[(df.index < self.asof) & (df.index >= self.asof - pd.Timedelta(days=365))].dropna(subset=['cush'])
        self.ol = data.outlook(); self.gc = data.gencap(); self.orp = data.outage_report(); self.ad = data.adder_params()
        self._imports_fit()
        self.rows = []
        self.cush = np.zeros((K, len(self.days)))
        self._build()

    # --- pieces
    def _imports_fit(self):
        t = self.tr.dropna(subset=['imports', 'cush_int'])
        X = np.column_stack([np.ones(len(t)), t.cush_int.values])
        self.imp_b = np.linalg.lstsq(X, t.imports.values, rcond=None)[0]
        self.imp_sd = float(np.std(t.imports.values - X @ self.imp_b))

    def _gas_mc(self):
        if self.gc is not None and len(self.gc): return float(self.gc.gas_mc.median()), float(self.gc.gas_mbo.median())
        return float(self.tr.gas_av.quantile(0.98) + 2000), 800.0

    def _seasonal_outage(self, month):
        m = self.ad.get('monthly')
        if m is not None and len(m):
            s = m.groupby('mo').actual.mean()
            if month in s.index: return float(s[month] * self.ad['yf'])
            return float(m.actual.mean() * self.ad['yf'])
        return 3000.0

    def _adder(self, lead, month):
        return float(self.ad['seas'][month] * self.ad['yf'] * data.adder_share(lead, self.ad['mu'], self.ad['sg']))

    def _build(self):
        rng = self.rng; tr = self.tr
        mc, mbo = self._gas_mc()
        bio = float(tr.bio.tail(30).mean())
        # month-level (persistent) shocks per scenario
        z_gas = rng.standard_normal(K); z_load = rng.standard_normal(K); z_imp = rng.standard_normal(K)
        # load trend: last 90 days vs the same window a year earlier
        rec = self.df[(self.df.index < self.asof) & (self.df.index >= self.asof - pd.Timedelta(days=90))].ail.mean()
        old = self.df[(self.df.index < self.asof - pd.Timedelta(days=365)) & (self.df.index >= self.asof - pd.Timedelta(days=455))].ail.mean()
        growth = float(np.clip(rec / old, 0.97, 1.05)) if (rec > 0 and old > 0) else 1.0
        for j, day in enumerate(self.days):
            lead = (day - self.asof).days; mo = day.month
            clim = climatology(tr, day)
            # ---- load
            if self.ol is not None and day in self.ol.index and lead <= 44:
                ail = float(self.ol.loc[day, 'ail']); src_l = 'ens'
            else:
                ail = float(clim.ail.mean() * growth); src_l = 'clim'
            sd_l = _interp(SD_LOAD, lead)
            load = ail + 0.7 * sd_l * z_load + 0.7 * sd_l * rng.standard_normal(K)
            # ---- wind
            if self.ol is not None and day in self.ol.index and lead <= 9:
                wm = float(self.ol.loc[day, 'wind']) - WIND_BIAS
                wind = np.clip(wm + SD_WIND_ENS[max(0, min(lead, 9))] * rng.standard_normal(K), 0, None); src_w = 'ens'
            else:
                wind = rng.choice(clim.wind.dropna().values, K) if clim.wind.notna().sum() >= 5 else np.full(K, tr.wind.mean()); src_w = 'clim'
            # ---- solar
            solar = rng.choice(clim.solar.dropna().values, K) if clim.solar.notna().sum() >= 5 else np.full(K, tr.solar.mean())
            # ---- gas availability
            adder = self._adder(max(lead, 1), mo)
            if self.gc is not None and day in self.gc.index and lead <= 15:
                gav = float(self.gc.loc[day, 'gas_ac']) - adder; src_g = 'gencap'
            elif self.orp is not None and day in self.orp.index:
                gav = mc - float(self.orp.loc[day, 'gas_filed']) - float(self.orp.loc[day, 'mbo']) - adder; src_g = 'filed'
            else:
                gav = mc - self._seasonal_outage(mo); src_g = 'seasonal'
            sd_g = _interp(SD_GAS, lead)
            gas = gav + 0.8 * sd_g * z_gas + 0.6 * sd_g * rng.standard_normal(K)
            # ---- imports respond to how tight the province is on its own
            cush_int = gas + bio + solar + wind - load
            imp = self.imp_b[0] + self.imp_b[1] * cush_int + 0.5 * self.imp_sd * z_imp + 0.6 * self.imp_sd * rng.standard_normal(K)
            imp = np.clip(imp, -1100, 900)
            c = cush_int + imp
            self.cush[:, j] = c
            self.rows.append({'d': day, 'lead': lead, 'ail': ail, 'load_src': src_l, 'wind': float(wind.mean()), 'wind_src': src_w,
                              'solar': float(solar.mean()), 'gas_av': gav, 'gas_src': src_g, 'adder': adder, 'imports': float(imp.mean()),
                              'cush': float(c.mean()), 'p10': float(np.percentile(c, 10)), 'p90': float(np.percentile(c, 90))})
        self.table = pd.DataFrame(self.rows).set_index('d')
