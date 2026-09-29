"""
read.py — from the curve and the outlook to a number per contract, plus the
three things the page shows alongside it: the buckets, the regressions, and
the model's own walk-forward record.
"""
import numpy as np, pandas as pd, warnings
from . import data, panel
from .curve import Curve, band_of, BAND_LAB, BETA
from .forecast import Outlook, K, climatology
warnings.filterwarnings('ignore')
Q = [10, 25, 50, 75, 90]


def month_range(asof, n=3):
    m0 = pd.Timestamp(asof).normalize().replace(day=1)
    return [m0 + pd.DateOffset(months=i) for i in range(n + 1)]


def latest_settle(fwd, m, asof):
    s = fwd[(fwd.m == m) & (fwd.ed <= asof)].sort_values('ed')
    return s.iloc[-1] if len(s) else None


def own_percentile(fwd, m, asof, col='flat'):
    s = fwd[(fwd.m == m) & (fwd.ed <= asof)].sort_values('ed')[col].dropna()
    if len(s) < 6: return None, len(s)
    return float((s.iloc[:-1] <= s.iloc[-1]).mean()), len(s)


# ------------------------------------------------------- term premium -----
DECAY_B = [0, 15, 30, 60, 90, 120, 180, 240, 300, 365, 450, 550, 2000]

def decay_table(pwr, df, asof):
    """The premium a contract sheds as it approaches. Two layers, both measured on settled months only:
      path(L)  median of settle(L) − the same contract's settle at ~20 d, by lead bucket  (life-range bias)
      prem20   median of settle at 15–30 d − realised month                               (what is left at the front)
      seasonal mean of settle at 60–180 d − realised, by calendar month of the contract
    A contract that starts life 18 months out is structurally near the bottom of its own range by the time it
    is prompt; the own-life percentile is meaningless unless this path is taken out first."""
    asof = pd.Timestamp(asof).normalize().replace(day=1)
    real = df.px.resample('MS').mean(); cnt = df.px.resample('MS').size(); real = real[cnt >= 25]
    f = pwr.join(real.rename('real'), on='m').dropna(subset=['real']); f['lead'] = (f.m - f.ed).dt.days
    f = f[(f.lead > 0) & (f.m < asof)]
    ref = f[f.lead.between(15, 25)].groupby('m').flat.mean().rename('ref20')
    h = f.join(ref, on='m').dropna(subset=['ref20']); h['decay'] = h.flat - h.ref20; h['lb'] = pd.cut(h.lead, DECAY_B)
    path = h.groupby('lb', observed=True).decay.median()
    rows = [{'lead': f'{int(iv.left)}–{int(iv.right) if iv.right < 2000 else "…"} d', 'lo': int(iv.left), 'hi': int(iv.right), 'path': float(v),
             'n': int((h.lb == iv).sum()), 'months': int(h[h.lb == iv].m.nunique())} for iv, v in path.items()]
    f['lb'] = pd.cut(f.lead, DECAY_B); pr = f.groupby('lb', observed=True).apply(lambda s: (s.flat - s.real).median())
    for rrow in rows:
        k = [iv for iv in pr.index if int(iv.left) == rrow['lo']]
        rrow['prem_vs_real'] = float(pr[k[0]]) if k else None
    prem20 = float((f[f.lead.between(15, 30)].flat - f[f.lead.between(15, 30)].real).median())
    s = f[f.lead.between(60, 180)]; seas = s.groupby(s.m.dt.month).apply(lambda x: pd.Series({'prem': (x.flat - x.real).mean(), 'n': x.m.nunique()}))
    return {'rows': rows, 'prem20': prem20, 'seasonal': {int(k): {'prem': float(v.prem), 'n': int(v.n)} for k, v in seas.iterrows()}}

def decay_at(dec, lead):
    for rrow in dec['rows']:
        if rrow['lo'] < lead <= rrow['hi']: return rrow['path']
    return dec['rows'][-1]['path'] if lead > 0 else 0.0


# ------------------------------------------------------------ contracts ---
def read_contracts(df, asof, n=3, rng=None):
    rng = rng or np.random.default_rng(11)
    asof = pd.Timestamp(asof).normalize()
    tr = panel.trailing(df, asof)
    cv = Curve(tr)
    months = month_range(asof, n)
    end = (months[-1] + pd.DateOffset(months=1)) - pd.Timedelta(days=1)
    ol = Outlook(df, asof, end, rng=rng, tr=tr)
    gasf = data.aeco_forward(); pwr = data.power_forward(); bom = data.power_bom()
    dec = decay_table(pwr, df, asof)
    settled = df[(df.index < asof)]
    out = []
    for i, m in enumerate(months):
        m_end = (m + pd.DateOffset(months=1)) - pd.Timedelta(days=1)
        gs = latest_settle(gasf, m, asof); gas = float(gs.gas) if gs is not None else float(tr.aeco.tail(30).mean())
        idx = [j for j, d in enumerate(ol.days) if m <= d <= m_end]
        # settled days of the month in progress
        done = settled[(settled.index >= m) & (settled.index <= m_end)]
        flat = np.zeros(K); pk = np.zeros(K); op = np.zeros(K)
        for j in idx:
            c = ol.cush[:, j]
            flat += cv.draw(c, gas, rng); pk += cv.draw(c, gas, rng, 'pk'); op += cv.draw(c, gas, rng, 'op')
        nf = len(idx); nd = len(done); ntot = nf + nd
        if ntot == 0: continue
        flat = (flat + done.px.sum()) / ntot; pk = (pk + done.px_pk.sum()) / ntot; op = (op + done.px_op.sum()) / ntot
        # centre the scenario distribution on the point read (median-leaning, see Curve.read); the spread stays
        if nf:
            for arr, kk in ((flat, 'flat'), (pk, 'pk'), (op, 'op')):
                pt = (sum(cv.read(ol.cush[:, j], gas, kk).mean() for j in idx) + (done.px.sum() if kk == 'flat' else done.px_pk.sum() if kk == 'pk' else done.px_op.sum())) / ntot
                arr += pt - arr.mean()
        r = {'m': m, 'label': ('BOM ' if i == 0 else '') + m.strftime('%b-%y'), 'days_settled': nd, 'days_forecast': nf,
             'settled_flat': float(done.px.mean()) if nd else None, 'gas_fwd': gas, 'gas_src': 'XAV' if gs is not None else 'trailing spot',
             'fair_mean': float(flat.mean()), 'fair_pk': float(pk.mean()), 'fair_op': float(op.mean()),
             'cush_mean': float(ol.cush[:, idx].mean()) if nf else None,
             'cush_p10': float(np.percentile(ol.cush[:, idx].mean(1), 10)) if nf else None,
             'cush_p90': float(np.percentile(ol.cush[:, idx].mean(1), 90)) if nf else None}
        for q in Q: r[f'fair_p{q}'] = float(np.percentile(flat, q))
        # gas sensitivity: the same month's cushion distribution at other AECO levels, and the implied heat rates
        if nf:
            cd = ol.cush[:, idx].ravel()
            r['fair_at_gas'] = {g: float(cv.read(cd, g).mean() * nf / ntot + (done.px.sum() / ntot if nd else 0)) for g in (1, 2, 4, 6, 10)}
            r['fair_hr'] = cv.implied_hr(r['fair_mean'], gas)
        # bucket weights: where the month's scenario-days fall on the trailing-year cushion bands
        b = band_of(ol.cush[:, idx].ravel()) if nf else np.array([])
        r['band_w'] = [float((b == k).mean()) if nf else 0.0 for k in range(len(BAND_LAB))]
        # the market: the monthly settle, or for the month in progress the balance-of-month strip
        # (daily XDT strips from the latest mark) blended with the days already settled
        ps = latest_settle(pwr, m, asof)
        if i == 0 and bom is not None and nd:
            bb = bom[(bom.d >= asof) & (bom.d <= m_end) & (bom.ed <= asof)]
            if len(bb):
                last_ed = bb.ed.max(); bb = bb[bb.ed == last_ed]
                if len(bb) >= max(1, nf - 2):
                    blend = (bb.flat.mean() * nf + done.px.sum()) / ntot
                    ps = pd.Series({'flat': blend, 'peak': np.nan, 'offpk': np.nan, 'ed': last_ed})
                    r['market_note'] = f'balance-of-month strip {bb.flat.mean():.2f} x {nf} d + settled {done.px.mean():.2f} x {nd} d'
        if ps is not None:
            r.update({'settle_flat': float(ps.flat), 'settle_pk': float(ps.peak) if pd.notna(ps.peak) else None,
                      'settle_op': float(ps.offpk) if pd.notna(ps.offpk) else None, 'settle_date': ps.ed.strftime('%Y-%m-%d')})
            r['edge'] = r['fair_mean'] - r['settle_flat']
            r['settle_hr'] = cv.implied_hr(r['settle_flat'], gas)
            # bid / ask: the model's is where it would buy (P25) and sell (P75); the market's is the settle
            # straddled by the assumed spread from config.json. Net edge is what is left after crossing.
            hs = float(data.CFG.get('spread', 5.0)) / 2
            r['half_spread'] = hs; r['mkt_bid'] = r['settle_flat'] - hs; r['mkt_ask'] = r['settle_flat'] + hs
            r['model_bid'] = r['fair_p25']; r['model_ask'] = r['fair_p75']
            r['net_edge'] = abs(r['edge']) - hs; r['side'] = 'sell' if r['edge'] < 0 else 'buy'   # what is left after crossing, on the side the model leans
            r['call'] = ('SELL' if r['settle_flat'] - hs > r['fair_p75'] else 'BUY' if r['settle_flat'] + hs < r['fair_p25']
                         else 'sell only if posted' if r['settle_flat'] > r['fair_p75'] else 'buy only if posted' if r['settle_flat'] < r['fair_p25'] else 'none')
            r['settle_pct_fair'] = float((flat <= r['settle_flat']).mean())
            pct, nlife = own_percentile(pwr, m, asof); r['settle_pct_life_raw'] = pct; r['settle_life_n'] = nlife
            # premium-adjusted own-life percentile: every past settle of this contract shifted onto today's lead
            life = pwr[(pwr.m == m) & (pwr.ed <= asof)].copy(); life['lead'] = (life.m - life.ed).dt.days
            lead_now = max(int((m - asof).days), 1)
            life['adj'] = life.flat - life.lead.map(lambda L: decay_at(dec, L)) + decay_at(dec, lead_now)
            r['settle_pct_life'] = float((life.adj.iloc[:-1] <= r['settle_flat']).mean()) if len(life) > 5 else None
            r['decay_ahead'] = (decay_at(dec, lead_now) + dec['prem20']) if (m - asof).days > 0 else None   # typical give-back from here to settlement; not defined once the month is running
            sp = dec['seasonal'].get(m.month); r['seasonal_prem'] = sp['prem'] if sp else None; r['seasonal_n'] = sp['n'] if sp else 0
            r['settle_pct_hist'] = float((tr.px <= r['settle_flat']).mean())
            r['lead'] = int((m - asof).days)
            # --- the gap against what is usual at this lead. Fair sits under the market as a rule: the curve
            # carries a premium over what settles (prem_vs_real by lead bucket). Only the part of the gap that
            # exceeds that premium is a view; the rest is the market being the market.
            typ = None
            for rrow in dec['rows']:
                if rrow['lo'] < max(r['lead'], 1) <= rrow['hi'] and rrow.get('prem_vs_real') is not None: typ = rrow['prem_vs_real']
            # a month already in progress has only its unsettled days left to give back
            if typ is not None: typ = typ * (nf / ntot)
            r['prem_typ'] = typ
            r['edge_vs_typ'] = (r['edge'] + typ) if typ is not None else None
            life = pwr[(pwr.m == m) & (pwr.ed <= asof)].flat
            r['settle_hi'] = float(life.max()) if len(life) else None; r['settle_lo'] = float(life.min()) if len(life) else None
        # --- cushion against the same calendar month's normal (settled history, not the trailing year alone)
        hist_m = settled[(settled.index.month == m.month) & settled.cush.notna()]
        r['cush_norm'] = float(hist_m.cush.mean()) if len(hist_m) >= 20 else None
        r['cush_norm_n'] = int(hist_m.index.year.nunique()) if len(hist_m) else 0
        r['cush_vs_norm'] = (r['cush_mean'] - r['cush_norm']) if (r.get('cush_mean') is not None and r['cush_norm'] is not None) else None
        out.append(r)
    out_dec = dec
    return out, cv, ol, tr, out_dec


# -------------------------------------------------------------- buckets ---
def buckets(tr, contracts):
    tr = tr.copy(); tr['band'] = band_of(tr.cush)
    tr['season'] = tr.month.map(lambda m: 'winter' if m in (11, 12, 1, 2) else 'summer' if m in (6, 7, 8) else 'shoulder')
    rows = []
    for k, lab in enumerate(BAND_LAB):
        s = tr[tr.band == k]
        if not len(s): continue
        r = {'band': lab, 'n': int(len(s)), 'px': s.px.mean(), 'p10': s.px.quantile(.1), 'p50': s.px.median(), 'p90': s.px.quantile(.9),
             'px_pk': s.px_pk.mean(), 'px_op': s.px_op.mean(), 'hrs100': s.hrs100.mean(), 'hrs300': s.hrs300.mean(),
             'aeco': s.aeco.mean(), 'wind': s.wind.mean(), 'ail': s.ail.mean(), 'gas_av': s.gas_av.mean(), 'imports': s.imports.mean(),
             'seasons': s.season.value_counts().to_dict(), 'last': s.index.max().strftime('%Y-%m-%d')}
        for c in contracts: r[f'w_{c["label"]}'] = c['band_w'][k]
        rows.append(r)
    # the same split by season, so the reader can see the bands are not just seasons in disguise
    seas = []
    for sname, s in tr.groupby('season'):
        seas.append({'season': sname, 'n': int(len(s)), 'px': s.px.mean(), 'cush': s.cush.mean(),
                     'bands': [float((s.band == k).mean()) for k in range(len(BAND_LAB))]})
    return rows, seas


# ---------------------------------------------------------- regressions ---
class _OLS:
    """Plain OLS with the three things the page shows: R², adjusted R², t-stats. No statsmodels dependency."""
    def __init__(self, y, X):
        y = np.asarray(y, float); X = np.asarray(X, float)
        A = np.column_stack([np.ones(len(X)), X]); n, k = A.shape
        b, *_ = np.linalg.lstsq(A, y, rcond=None); e = y - A @ b
        s2 = (e @ e) / max(n - k, 1); cov = s2 * np.linalg.pinv(A.T @ A)
        self.b = b; self.se = np.sqrt(np.clip(np.diag(cov), 1e-12, None)); self.nobs = n
        sst = ((y - y.mean()) ** 2).sum(); self.rsquared = 1 - (e @ e) / sst if sst > 0 else 0.0
        self.rsquared_adj = 1 - (1 - self.rsquared) * (n - 1) / max(n - k, 1)

def _fit(t, y, cols):
    m = _OLS(t[y].values, t[cols].values)
    m.params = pd.Series(m.b, index=['const'] + list(cols)); m.tvalues = pd.Series(m.b / m.se, index=['const'] + list(cols))
    return m


def regressions(tr):
    t = tr.dropna(subset=['cush', 'hdd', 'ail', 'imports', 'wind', 'aeco', 'cush_eve']).copy()
    t['lpx'] = np.log1p(t.px); t['cush2'] = (t.cush / 1000) ** 2
    def fit(y, cols): return _fit(t, y, cols)
    uni = []
    for lab, c in [('cushion (daily mean)', 'cush'), ('cushion, evening', 'cush_eve'), ('cushion, min hour', 'cush_min'), ('wind generation', 'wind'),
                   ('wind speed', 'wspd'), ('load', 'ail'), ('heating degree days', 'hdd'), ('cooling degree days', 'cdd'),
                   ('net imports', 'imports'), ('gas availability', 'gas_av'), ('AECO', 'aeco'), ('weekend', 'wkend')]:
        if c not in t or t[c].isna().all(): continue
        m = fit('lpx', [c]); mp = fit('px', [c])
        uni.append({'x': lab, 'r2_log': m.rsquared, 'r2_lin': mp.rsquared, 'slope': mp.params[c], 't': m.tvalues[c], 'n': int(m.nobs)})
    multi = []
    for lab, cols in [('weather only', ['hdd', 'cdd', 'wspd', 'wkend']), ('load only', ['ail', 'wkend']), ('load + wind + solar', ['ail', 'wind', 'solar', 'wkend']),
                      ('cushion only', ['cush']), ('cushion + curvature', ['cush', 'cush2']), ('cushion + interties', ['cush', 'cush2', 'imports']),
                      ('cushion + gas', ['cush', 'cush2', 'aeco']), ('cushion + load + weather', ['cush', 'cush2', 'ail', 'hdd', 'cdd', 'wkend']),
                      ('everything', ['cush', 'cush2', 'cush_eve', 'ail', 'wind', 'imports', 'aeco', 'hdd', 'cdd', 'wkend'])]:
        m = fit('lpx', cols)
        multi.append({'spec': lab, 'r2': m.rsquared, 'r2_adj': m.rsquared_adj,
                      'terms': {c: {'coef': float(m.params[c]), 't': float(m.tvalues[c])} for c in cols}})
    # gas by cushion band, off-peak: the "heat rate" that actually exists
    gasrows = []
    t['band'] = band_of(t.cush)
    for k, lab in enumerate(BAND_LAB):
        s = t[t.band == k]
        if len(s) < 25: continue
        mo = _fit(s, 'px_op', ['aeco']); mk = _fit(s, 'px_pk', ['aeco'])
        gasrows.append({'band': lab, 'n': int(len(s)), 'op_slope': mo.params.aeco, 'op_t': mo.tvalues.aeco, 'op_int': mo.params.const,
                        'pk_slope': mk.params.aeco, 'pk_t': mk.tvalues.aeco})
    chain = {}
    m = fit('ail', ['hdd', 'cdd', 'wkend']); chain['load_r2'] = m.rsquared; chain['mw_per_hdd'] = m.params.hdd
    m = fit('wind', ['wspd']); chain['wind_r2'] = m.rsquared
    m = fit('imports', ['cush_int']); chain['imp_r2'] = m.rsquared; chain['imp_slope'] = m.params.cush_int
    return {'uni': uni, 'multi': multi, 'gas': gasrows, 'chain': chain, 'n': int(len(t)),
            'start': t.index.min().strftime('%Y-%m-%d'), 'end': t.index.max().strftime('%Y-%m-%d')}


# ------------------------------------------------------------- backtest ---
def backtest(df, asof, n_months=12, rng=None):
    """Walk-forward: every settled month read from the year before it. Three reads are scored so the
    page can say where the skill comes from:
      perfect   the month's actual daily cushions through the trailing curve  (what the curve is worth)
      realistic climatology load + wind, the filed outage plan + seasonal adder for gas, imports by regression
                (what the outlook is worth with no weather skill)
      forward   the flat contract 30 and 60 days before the month (the benchmark)"""
    rng = rng or np.random.default_rng(5)
    asof = pd.Timestamp(asof).normalize().replace(day=1)
    pwr = data.power_forward(); gasf = data.aeco_forward(); ad = data.adder_params(); mn = ad.get('monthly')
    rows = []
    for i in range(n_months, 0, -1):
        m = asof - pd.DateOffset(months=i); m_end = (m + pd.DateOffset(months=1)) - pd.Timedelta(days=1)
        te = df[(df.index >= m) & (df.index <= m_end)].dropna(subset=['cush', 'px'])
        tr = panel.trailing(df, m)
        if len(te) < 15 or len(tr) < 250: continue
        cv = Curve(tr)
        gs = gasf[(gasf.m == m) & (gasf.ed <= m - pd.Timedelta(days=25)) & (gasf.ed >= m - pd.Timedelta(days=45))].gas
        gas = float(gs.mean()) if len(gs) else float(tr.aeco.tail(30).mean())
        # perfect-foresight cushion
        pf = float(cv.read(te.cush.values, gas).mean())
        # realistic cushion: climatology + outage plan
        clim_days = []; mc = float(tr.gas_av.quantile(0.98)) + 0.0
        plan = None
        if mn is not None and m.strftime('%Y-%m') in set(mn.month.astype(str)):
            row = mn[mn.month.astype(str) == m.strftime('%Y-%m')].iloc[0]
            hist = mn[mn.month.astype(str) < m.strftime('%Y-%m')]
            sea = hist[hist.mo == m.month].adder.mean() if (hist.mo == m.month).any() else hist.adder.mean()
            base = float(mn[(mn.month.astype(str) < m.strftime('%Y-%m'))].tail(12).actual.mean())
            # the level the trailing year ran at, less its own outages, is the reference capability
            ref_cap = float(tr.gas_av.mean() + base)
            plan = ref_cap - float(row.plan + (sea if pd.notna(sea) else 0))
        t = tr.dropna(subset=['imports', 'cush_int'])
        b = np.linalg.lstsq(np.column_stack([np.ones(len(t)), t.cush_int.values]), t.imports.values, rcond=None)[0]
        sims = []
        for d in pd.date_range(m, m_end):
            c = climatology(tr, d)
            n = 40
            load = rng.choice(c.ail.dropna().values, n); wind = rng.choice(c.wind.dropna().values, n); sol = rng.choice(c.solar.dropna().values, n)
            g = np.full(n, plan) if plan is not None else np.full(n, tr.gas_av.tail(21).mean())
            ci = g + tr.bio.tail(30).mean() + sol + wind - load
            sims.append(ci + b[0] + b[1] * ci)
        sims = np.concatenate(sims)
        real_read = float(cv.read(sims, gas).mean())
        f30 = pwr[(pwr.m == m) & (pwr.ed <= m - pd.Timedelta(days=25)) & (pwr.ed >= m - pd.Timedelta(days=40))].flat
        f60 = pwr[(pwr.m == m) & (pwr.ed <= m - pd.Timedelta(days=55)) & (pwr.ed >= m - pd.Timedelta(days=70))].flat
        rows.append({'m': m.strftime('%b-%y'), 'real': float(te.px.mean()), 'perfect': pf, 'realistic': real_read,
                     'fwd30': float(f30.mean()) if len(f30) else np.nan, 'fwd60': float(f60.mean()) if len(f60) else np.nan,
                     'cush_real': float(te.cush.mean()), 'cush_realistic': float(sims.mean()), 'gas_src': 'plan+adder' if plan is not None else 'persisted',
                     'aeco': float(te.aeco.mean()) if te.aeco.notna().any() else np.nan, 'gas_fwd': gas})
    R = pd.DataFrame(rows)
    score = {}
    for c in ['perfect', 'realistic', 'fwd30', 'fwd60']:
        e = (R[c] - R.real).dropna()
        if len(e) < 3: continue
        score[c] = {'mae': float(e.abs().mean()), 'bias': float(e.mean()), 'corr': float(R[c].corr(R.real)), 'n': int(len(e)),
                    'hit': float(((R[c] > R.fwd30) == (R.real > R.fwd30)).mean()) if c not in ('fwd30', 'fwd60') else None}
    # the premium the market charges, by lead
    prem = []
    real = df.px.resample('MS').mean(); cnt = df.px.resample('MS').size(); real = real[cnt >= 25]
    f = pwr.join(real.rename('real'), on='m').dropna(subset=['real']); f['lead'] = (f.m - f.ed).dt.days; f = f[(f.lead > 0) & (f.m < asof)]
    for lo, hi in [(1, 15), (16, 30), (31, 60), (61, 90), (91, 180)]:
        s = f[(f.lead >= lo) & (f.lead <= hi)]
        if len(s) < 20: continue
        prem.append({'lead': f'{lo}–{hi} d', 'n': int(len(s)), 'months': int(s.m.nunique()), 'prem': float((s.flat - s.real).mean()),
                     'mae': float((s.flat - s.real).abs().mean()), 'over': float(((s.flat - s.real) > 0).mean()), 'corr': float(s.flat.corr(s.real))})
    return R, score, prem


# ------------------------------------------------------- strip co-movement ---
def strip_betas(asof, days=365):
    """How the prompt months move together: daily and weekly changes in the flat settle of M+1 regressed on the
    balance-of-month strip (M), and M+2 / M+3 on M+1. Re-measured on the trailing year each run."""
    pw = data.power_forward(); bom = data.power_bom()
    asof = pd.Timestamp(asof)
    rows = {}
    for ed, s in pw[(pw.ed <= asof) & (pw.ed >= asof - pd.Timedelta(days=days + 10))].groupby('ed'):
        m1 = (ed + pd.offsets.MonthBegin(1)).normalize(); r = {}
        for k in (1, 2, 3):
            v = s[s.m == m1 + pd.DateOffset(months=k - 1)].flat; r[f'M{k}'] = float(v.iloc[0]) if len(v) else np.nan
        rows[ed] = r
    F = pd.DataFrame(rows).T.sort_index()
    if bom is not None:
        b = bom[(bom.d >= bom.ed) & (bom.d.dt.to_period('M') == bom.ed.dt.to_period('M'))].groupby('ed').flat.mean().rename('M0')
        F = F.join(b, how='left')
    else:
        F['M0'] = np.nan
    per = pd.Series(F.index.to_period('M'), index=F.index)
    out = []
    for lag, lab in [(1, 'daily'), (5, 'weekly')]:
        same = per.eq(per.shift(lag)).values
        D = F.diff(lag)[same]
        for base, tgt in [('M0', 'M1'), ('M0', 'M2'), ('M1', 'M2'), ('M1', 'M3')]:
            ok = D[base].notna() & D[tgt].notna()
            if ok.sum() < 30: continue
            x = D[base][ok].values; y = D[tgt][ok].values
            A = np.column_stack([np.ones(len(x)), x]); bb, *_ = np.linalg.lstsq(A, y, rcond=None); e = y - A @ bb
            r2 = 1 - (e @ e) / ((y - y.mean()) ** 2).sum(); se = np.sqrt((e @ e) / (len(y) - 2) / ((x - x.mean()) ** 2).sum())
            big = np.abs(x) >= 2
            out.append({'window': lab, 'pair': f'{"BOM" if base == "M0" else "M+1"} → {"M+" + tgt[1]}', 'beta': float(bb[1]), 't': float(bb[1] / se), 'r2': float(r2), 'n': int(len(y)),
                        'same_dir': float(((y[big] > 0) == (x[big] > 0)).mean()) if big.sum() >= 10 else None})
    return out


# ------------------------------------------------------- excursion stats ---
def excursions(asof, months_back=24):
    """For every settled contract and every settle day before it: how far the mark went AGAINST a position opened
    that day before settlement (max adverse excursion), and how far it went for it. By lead bucket and side. This is
    what the trade tracker's stop / take-profit guidance rests on: not a model opinion, the settle paths themselves."""
    pw = data.power_forward(); asof = pd.Timestamp(asof).normalize().replace(day=1)
    rows = []
    for m, s in pw[(pw.m < asof) & (pw.m >= asof - pd.DateOffset(months=months_back))].groupby('m'):
        s = s.sort_values('ed'); path = s.flat.values; eds = s.ed.values
        for i in range(len(path)):
            lead = int((m - pd.Timestamp(eds[i])).days)
            if lead < 1 or lead > 95: continue
            later = path[i:]
            rows.append({'lead': lead, 'short_adverse': float(later.max() - path[i]), 'long_adverse': float(path[i] - later.min())})
    E = pd.DataFrame(rows)
    out = []
    for lo, hi in [(1, 15), (16, 30), (31, 60), (61, 95)]:
        s = E[E.lead.between(lo, hi)]
        if len(s) < 30: continue
        out.append({'lead_lo': lo, 'lead_hi': hi, 'lead': f'{lo}–{hi} d', 'n': int(len(s)),
                    'short_p50': float(s.short_adverse.median()), 'short_p75': float(s.short_adverse.quantile(.75)), 'short_p90': float(s.short_adverse.quantile(.9)),
                    'long_p50': float(s.long_adverse.median()), 'long_p75': float(s.long_adverse.quantile(.75)), 'long_p90': float(s.long_adverse.quantile(.9))})
    return out
