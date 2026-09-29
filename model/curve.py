"""
curve.py — the cushion → price map on the trailing year, with gas as a floor.

The findings this encodes (research/, September 2026):

* Daily price against daily mean cushion is convex and stable enough across a
  year that a kernel-smoothed curve on the trailing 365 days, averaged over the
  target month's *distribution* of daily cushions, read the last twelve months
  at MAE $6.7 / corr 0.73 when the cushion was known. Coarse buckets (terciles,
  k-means) lose the convexity: reading at the mean cushion instead of over the
  distribution cost ~$1 of MAE and $6 of bias.

* Gas is a floor, not a multiplier. Price/AECO is undefined when AECO goes to
  zero — which it did in the summer of 2025 while pool averaged $74. Measured
  elasticity of price to gas (research/r7): ~0 when the cushion is under
  1,500 MW, 0.26 in the middle, 0.65 when loose and 0.87 off-peak loose. Tight
  hours are scarcity and gas-blind; loose hours are a heat-rate world. So:
    price = max( scarcity curve(cushion) + β·AECO ,  HR·AECO + VOM )
  The β term (3 peak / 8 off-peak $/MWh per $/GJ) carries the in-sample gas
  sensitivity; the floor (9.5 GJ/MWh + $8 peak, 8.5 GJ/MWh + $3 off-peak — the
  marginal gas unit's cost) is what keeps 1,000 MW at $10 gas from being priced
  like 1,000 MW at $2 gas. Walk-forward, cushion known: floor forms MAE $6.8–7.2,
  additive alone $7.5, pure price/gas ratio $17.9.
"""
import numpy as np, pandas as pd

BW = 250.0                      # kernel bandwidth in MW of cushion
LAM = 0.1                       # median-leaning point read; with the gas floor in place 0.1 sits within $1 of unbiased (research/r7 + backtest)
BETA_OP, BETA_PK = 8.0, 3.0     # $/MWh per $/GJ, off-peak and peak
BETA = (16 * BETA_PK + 8 * BETA_OP) / 24
FLOOR = {'pk': (9.5, 8.0), 'op': (8.5, 3.0)}                 # (GJ/MWh, $/MWh VOM): the marginal gas unit
FLOOR['flat'] = ((16 * FLOOR['pk'][0] + 8 * FLOOR['op'][0]) / 24, (16 * FLOOR['pk'][1] + 8 * FLOOR['op'][1]) / 24)


def gas_floor(gas, k='flat'):
    hr, vom = FLOOR[k]
    return hr * np.maximum(np.asarray(gas, float), 0.0) + vom
BANDS = [-1e9, 1200, 1800, 2400, 3000, 1e9]
BAND_LAB = ['under 1,200', '1,200–1,800', '1,800–2,400', '2,400–3,000', 'over 3,000']


def band_of(c):
    return np.clip(np.digitize(np.asarray(c, float), BANDS) - 1, 0, len(BANDS) - 2)


class Curve:
    def __init__(self, tr, bw=BW):
        tr = tr.dropna(subset=['cush', 'px']).copy()
        self.bw = bw
        self.aeco_ref = float(tr.aeco.mean()) if tr.aeco.notna().any() else 0.0
        a = tr.aeco.fillna(self.aeco_ref)
        self.x = tr.cush.values.astype(float)
        self.y = {'flat': np.log1p((tr.px - BETA * a).clip(lower=0).values),
                  'pk': np.log1p((tr.px_pk - BETA_PK * a).clip(lower=0).values),
                  'op': np.log1p((tr.px_op - BETA_OP * a).clip(lower=0).values)}
        self.beta = {'flat': BETA, 'pk': BETA_PK, 'op': BETA_OP}
        self.n = len(tr)
        # residuals, pooled by cushion band, for the spread around the curve
        self.res = {k: self.y[k] - self._fit(self.x, k) for k in self.y}
        self.bands = band_of(self.x)
        self.dates = tr.index

    def _fit(self, xs, k):
        xs = np.atleast_1d(np.asarray(xs, float))
        w = np.exp(-0.5 * ((self.x[None, :] - xs[:, None]) / self.bw) ** 2)
        return (w * self.y[k][None, :]).sum(1) / np.maximum(w.sum(1), 1e-9)

    def level(self, cush, gas, k='flat'):
        """Expected price at a daily cushion `cush` and AECO `gas`. The mean of the residual
        distribution is put back (a log-space curve reads the median otherwise)."""
        f = self._fit(cush, k); b = band_of(cush)
        mr = np.array([np.expm1(f[i] + self.res[k][self.bands == b[i]]).mean() for i in range(len(f))])
        return np.maximum(mr + self.beta[k] * gas, gas_floor(gas, k))

    def read(self, cush, gas, k='flat'):
        """The point read: median + LAM x (mean - median). Spikes cluster in time, so the trailing year's
        tail at a given cushion over-states what recurs: reading the full conditional mean ran +$8 high
        across twelve walk-forward months, the median -$3 low; LAM = 0.3 sat at -$1 with the same
        correlation. Applied to the monthly average, not to individual days."""
        f = self._fit(cush, k); b = band_of(cush)
        med = np.expm1(f); mean = np.array([np.expm1(f[i] + self.res[k][self.bands == b[i]]).mean() for i in range(len(f))])
        return np.maximum(med + LAM * (mean - med) + self.beta[k] * gas, gas_floor(gas, k))

    def draw(self, cush, gas, rng, k='flat'):
        """One price draw per cushion value: curve + a residual sampled from the same cushion band."""
        f = self._fit(cush, k); b = band_of(cush)
        r = np.array([rng.choice(self.res[k][self.bands == bi]) if (self.bands == bi).any() else rng.choice(self.res[k]) for bi in b])
        return np.maximum(np.expm1(f + r) + self.beta[k] * np.asarray(gas, float), gas_floor(gas, k))

    def implied_hr(self, price, gas):
        """The market's language: price / gas in GJ/MWh. Reported, never fitted (undefined near zero gas)."""
        return float(price / gas) if gas and gas > 0.25 else None

    def table(self, xs=np.arange(400, 4001, 200), gas=None):
        gas = self.aeco_ref if gas is None else gas
        return pd.DataFrame({'cush': xs, 'flat': self.level(xs, gas), 'peak': self.level(xs, gas, 'pk'), 'offpk': self.level(xs, gas, 'op')})
