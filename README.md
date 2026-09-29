# AESO monthly

Fair value for the prompt monthly contracts (balance-of-month, M+1, M+2, M+3),
read from the trailing year's cushion→price relationship and a daily cushion
outlook. Sits next to the 14-day cushion model and reuses its data; publishes
its own page.

---

## The daily command

```
.\Refresh.ps1
```

One command, every morning. It refreshes the cushion repo's inputs next door
(unless they are less than six hours old), tops up gas and power forwards and
settled price, rebuilds the read, writes `docs\index.html` and
`docs\artifact.html`, appends today's reads to `history\reads.csv`, and pushes
to GitHub so the Pages site updates. About a minute; the CANPOWER pulls are most
of it.

| flag | what it does |
|---|---|
| `-NoPush` | rebuild and stop, so you can look at `docs\index.html` first |
| `-SkipCushion` | don't touch the cushion repo (you already ran its `Daily.ps1`) |

## The monthly command

```
.\Monthly.ps1
```

Any day in the first week of a month. It walks you through the one input that
cannot be automated (last month's plan vs actual gas outage from NRGStream, one
row in the outage adder's `monthly_nrgstream.csv`), refreshes the adder, does a
deep re-sync of the forward settles, then runs `Refresh.ps1`.

## GitHub, once

```
.\Setup-GitHub.ps1
```

Creates the private repo and turns on Pages from `main:/docs` if the GitHub CLI
(`gh`) is installed and logged in; otherwise it commits locally and prints the
three manual steps. `cache\`, `db.json` and `aeso_key.txt` never leave the
machine. The page is then at `https://<you>.github.io/aeso-monthly/` and the
Claude artifact can be refreshed from `docs\artifact.html` whenever you want a
copy inside Claude.

Underneath all three: `python update.py`, with `--no-pull` and
`--asof YYYY-MM-DD` (rebuild as of a past date, for checking).

## The page

One section: a card per contract (balance-of-month, M+1, M+2, M+3) with fair
value and its P10–P90, peak and off-peak, the market settle with its assumed
bid/ask, the model's own bid/offer (P25/P75), the edge before and after
crossing the spread, fair at other gas prices with implied heat rates, the
expected cushion, and three percentiles spelled out: where the settle sits in
the model's outcomes, in the contract's own history after the usual premium
decay is removed (raw in the text), and against the last year of settled
prices. Everything that used to sit below the cards (outlook chart, buckets,
regressions, backtest tables, strip co-movement, trade tracker) is still
computed on every run and lives in `history\` and the `research\` scripts;
the page shows only the read plus a one-line walk-forward score.

### First time

1. This folder lives at `Documents\aeso-monthly`, next to `aeso-cushion model`
   and `aeso-outage-adder`. `config.json` says where those two are if they move.
2. `pip install pandas numpy pyodbc` (all already there for the cushion repo).
3. `python pull_history.py` once — it fills `cache\` with AB-NIT gas and AESO
   power forwards back to 2024 and settled pool price back to 2021. It reads
   `db.json` and `aeso_key.txt` from the cushion repo, so there is nothing to
   configure. Already done on 28 Sep 2026.
4. `.\Update.ps1 -NoPush`, then open `docs\index.html`.
5. GitHub Pages, same as the cushion repo: empty private repo `aeso-monthly`,
   `git init -b main; git add -A; git commit -m "AESO monthly"; git remote add
   origin ...; git push -u origin main`, then Settings → Pages → branch `main`,
   folder `/docs`. `cache\`, `db.json` and `aeso_key.txt` are gitignored.

---

## What the page says, and why it is built this way

Everything below was measured on your data in September 2026 (the scripts are
in `research\`; each prints its tables). The page re-measures the parts that
can drift on every run and prints its own numbers, so if the model gets worse
the page says so.

### The read

For each contract month, every day gets a distribution of daily-mean cushion
(400 scenarios). Each scenario-day is priced off the trailing year's curve, and
the monthly mean of those daily prices is the fair value; P10–P90 come from the
scenarios. The market settle is then placed as a percentile of that
distribution, of the contract's own life, and of the trailing year's daily
prices.

### Why cushion bands, not months

Walk-forward over the last twelve settled months, reading each month from the
year before it, with the month's actual cushion known:

| grouping of the trailing year | MAE | corr with the settled month |
|---|---|---|
| kernel curve on cushion, averaged over the month's daily cushions | **$6.7** | **0.73** |
| same, read at the month's mean cushion only | $7.4 | 0.75 (bias −$7) |
| cushion terciles | $12.5 | 0.56 |
| k-means on six fundamentals | $14.6 | 0.52 |
| same season (±45 days) last year | $10.7 | 0.21 |
| the 30-day-out forward contract | $13.6 | 0.21 |

Coarse buckets lose the convexity: a 1,000 MW day costs far more than a 3,000
MW day saves, so the read has to be taken over the *distribution* of days. The
page still shows five cushion bands with their price statistics because that
is the readable version of the same idea, and it prints where each contract's
simulated days fall across them.

### Heat rate

1,000 MW of cushion at $4 gas is not 1,000 MW at $10 gas. The question is what
form that takes. Measured (`research\r7_heatrate_floor.py`): the elasticity of
price to gas is ~0 when the cushion is under 1,500 MW, 0.26 in the middle,
0.65 when loose and 0.87 off-peak loose. Tight hours are scarcity and gas-blind;
loose hours are a heat-rate world. So the model is

    price = max( scarcity curve(cushion) + β·AECO ,  HR·AECO + VOM )

with β = 3 peak / 8 off-peak $/MWh per $/GJ for the in-sample sensitivity and a
hard floor at the marginal gas unit (9.5 GJ/MWh + $8 peak, 8.5 GJ/MWh + $3
off-peak). Walk-forward with the cushion known: floor forms MAE $6.8–7.2,
additive alone $7.5, the bare price ÷ AECO ratio $17.9 (it is undefined when
AECO goes to zero, as it did in September 2025 with pool at $73). Each card
shows the read at $1/$2/$4/$6/$10 gas and the implied heat rates of fair and
market.

### The strip moves together

`research\r8_strip_comovement.py`, re-measured on the page each run. M+2 follows
M+1 at about 0.36 per dollar day-to-day and 0.40 over a week; M+3 at 0.29 /
0.30; same direction on ~90% of ≥$2 days. The balance-of-month strip does not
lead the forwards: 0.02 per dollar daily, 0.06 weekly.

### What matters for the cushion 30–60 days out

Standard deviation of the monthly mean over the last year: gas availability
899 MW, load 549, wind 315, imports 220. Load is 70% weather (48 MW per heating
degree-day), which normals and the ECMWF ensemble cover. Wind climatology is
adequate. The swing factor is outages. In the walk-forward, persisting recent
gas availability gave corr 0.32 with the settled month; knowing the outage
picture gave 0.64. Hence the gas-availability ladder: AESO gencap AC for 15
days → the filed outage report for 90 days → the seasonal norm, always less the
unfiled-outage adder for the lead from the outage-adder project.

### The rolling year

365 days ending at the start of the current month. Expanding history was
tested and did worse at this horizon; the price level drifts year to year even
where the shape of the cushion response does not.

### The point read

Median-leaning: median + 0.3 × (mean − median) of the conditional
distribution. Spikes cluster in time, so the trailing year's tail at a given
cushion over-states what recurs: the full conditional mean over-read twelve
months by +$8; the median under-read by −$3; 0.3 sat at −$1 with the same
correlation.

### Bid / ask and the calls

There is no quoted bid/ask feed, so `config.json` carries an assumed spread
(`"spread": 5.0`, $/MWh, the width of the monthly flat). Each card shows the
market bid/ask (settle ∓ half the spread), the model's bid/offer (fair P25 /
P75 — where it would buy and sell), and the edge left after crossing. A card
says SELL or BUY only when the settle clears the model's band by more than half
the spread; "only if posted" means the edge exists but not after crossing.
Change the spread and re-run: everything recomputes.

### Your trades (not on the page any more)

The trade tracker was removed from the page at your request; the logic is in
`model\page.py` history and the excursion statistics in `read.excursions`. Trades live in the browser (export/import JSON
to move them). Each open row shows the mark, P&L, the model's fair and expected
P&L at settle, a take-profit (the model's bid/offer on the other side of your
trade, fair mean beside it), the expected heat (entry moved against you by the
historical P75 / P90 adverse excursion for the lead, measured on every settle
path of the last 24 months, `read.excursions`), the drawdown at the P90 heat
and at the model's tail, and the exit rule: hold while the model's fair stays
on your side of your price, exit when it flips. Price stops were tested on the
last year's trades (`research\r12_full_test.py`) and every one of them lost
money against holding — the heat numbers are for sizing, not stops.

### The full test, last 365 settle days

`research\r12_full_test.py`, on the final model with the $5 spread: strict
signals crossed, soft signals posted, one position per contract per month,
held to settlement. 17 trades (16 sells, 1 buy), 16 winners, mean +$14.71/MWh,
worst −$1.96, $179,900 on 1 MW per trade. Always-sell under the same fills:
+$9.42. The two months where selling lost (Nov-25, May-26) the model bought
one and stood aside on the other. In-sample caveat: the gas β, floor and
median-lean were chosen on this same period; `history\reads.csv` is the
out-of-sample record from 28 Sep 2026 on.

### What the market does

Across 28 settled months the flat contract settled **above** the realised month
57–69% of the time depending on lead, by $7–10 on average, with correlation
0.1–0.15 to the outcome. That premium, and its absence of skill, is the
opportunity; the page's last section re-measures it each run.

---

## Files

| path | what |
|---|---|
| `update.py` | the entry point, five stages |
| `pull_history.py` | the one-time deep pull (also used by `update.py` for the incremental pull) |
| `model\data.py` | every input and where it comes from |
| `model\panel.py` | the daily panel; the rolling-year window |
| `model\curve.py` | cushion→price kernel curve, gas floor, residual pools |
| `model\forecast.py` | the daily cushion outlook (load / wind / gas / imports, by source and lead) |
| `model\read.py` | contracts, buckets, regressions, walk-forward backtest |
| `model\page.py` | → `docs\index.html` |
| `research\` | the scripts behind the numbers above (`panel.py` then `r1`…`r6`) |
| `research\r9_trading_sim.py` | what the model would have bought and sold over the last 365 days (chat report, not on the page) |
| `history\reads.csv` | one row per contract per run — the track record builds here |
| `history\outlook_<date>.csv` | each run's daily outlook, so the outlook itself can be scored later |
| `cache\` | gas and power forwards, pool price since 2021 (gitignored) |

## Inputs it reads from the other two folders

| from | file | used for |
|---|---|---|
| cushion repo `cache\` | `composition.csv` | hourly actuals → daily cushion, load, wind, gas availability, imports |
| | `wx_hourly.csv` | temperature, wind speed → HDD, the regressions |
| | `outlook.csv` | ECMWF-ensemble daily load (44 d) and wind (9 d) |
| | `fwd_bom.csv` | balance-of-month strip, for the month in progress |
| cushion repo `refresh\` | `gencap.json` | gas available capability, 15 days |
| | `outage_90d.csv` | filed gas outages and mothballs, 90 days |
| | `poolprice*.json` | the last days of settled price |
| outage adder | `monthly_nrgstream.csv` | plan vs actual outages by month → the seasonal adder and its level |

If any of these is missing the model falls back (climatology, seasonal norm,
published adder constants) and the page's source strips show it.

## Known limits

- **The backtest cannot see the live inputs' vintages.** Ensemble load, ensemble
  wind and the filed outage report are not archived at the dates the backtest
  needs, so it scores an "outlook only" read (climatology + plan + adder) and a
  "cushion known" read, and the live read sits between them. `history\` starts
  building that archive from the first run.
- **Two years of fundamentals.** Composition starts August 2024; the seasonal
  pieces (wind climatology, seasonal outages) rest on one prior year each.
  Pool price goes back to 2021 but cannot be used without the cushion.
- **October 2025 has twelve days without composition snapshots** and is
  read on the remaining days.
- **The forward files end at the last CANPOWER settle.** If the page's "forwards"
  date is stale, the pull failed; run `python pull_history.py --days 30`.
