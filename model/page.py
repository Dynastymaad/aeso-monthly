"""
page.py — model output -> docs/index.html. Server-side rendered, no JS
dependencies, one file. Charts are inline SVG built here.
"""
import json, html
import numpy as np, pandas as pd
from .curve import BAND_LAB

CSS = """
:root{--bg:#f1f4f7;--surface:#fff;--surface-2:#e7ecf1;--line:#ccd5de;--ink:#101820;--ink-2:#48586a;--ink-3:#78889a;
--accent:#a8570d;--blue:#2a78d6;--red:#c0322a;--green:#16785a;--green-bg:#dff0e8;--red-bg:#fadedb;--amber:#9a6300;--amber-bg:#f8ecd2;--purple:#5b4bb0;
--shadow:0 1px 2px rgba(16,24,32,.06),0 8px 24px -12px rgba(16,24,32,.18)}
@media (prefers-color-scheme:dark){:root{--bg:#0e1319;--surface:#171e26;--surface-2:#1f2831;--line:#2f3b47;--ink:#eef3f8;--ink-2:#aebbc8;--ink-3:#7a8796;
--accent:#e09a4e;--blue:#3987e5;--red:#f07f75;--green:#54c69b;--green-bg:#12312a;--red-bg:#3a1d1a;--amber:#e0ad4a;--amber-bg:#33280f;--purple:#a89ae8;
--shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -14px rgba(0,0,0,.7)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;-webkit-font-smoothing:antialiased}
.num{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
h1,h2,h3{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;margin:0;letter-spacing:-.005em}
h1{font-size:20px}h2{font-size:17px;margin:0 0 10px}h3{font-size:14px;color:var(--ink-2)}
.eyebrow{font-size:10.5px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
header{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--surface) 92%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
.wrap{max-width:1480px;margin:0 auto;padding:12px 20px}
.hrow{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:baseline}.hrow .meta{color:var(--ink-2);font-size:12.5px}
main{max-width:1480px;margin:0 auto;padding:18px 20px 60px;display:grid;gap:20px}
section{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:var(--shadow)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.card{border:1px solid var(--line);border-radius:10px;padding:14px;background:var(--surface-2)}
.card .big{font-size:30px;font-weight:600;line-height:1.1}
.card .row{display:flex;justify-content:space-between;gap:8px;margin:3px 0;font-size:13px}.card .row span:last-child{text-align:right}
.chip{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:600;margin-right:4px}
.buy{background:var(--green-bg);color:var(--green)}.sell{background:var(--red-bg);color:var(--red)}.flat{background:var(--amber-bg);color:var(--amber)}
.neutral{background:var(--surface);color:var(--ink-2);border:1px solid var(--line)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}
th{color:var(--ink-3);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}th:first-child,td:first-child{text-align:left}
td.l,th.l{text-align:left}tr.hi td{background:var(--surface-2)}
p.note{color:var(--ink-2);font-size:13px;margin:8px 0 0;max-width:900px}p.lead{color:var(--ink-2);margin:0 0 12px;max-width:960px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.bar{height:8px;border-radius:4px;background:var(--surface);position:relative;overflow:hidden;border:1px solid var(--line)}
.bar i{position:absolute;top:0;bottom:0;background:var(--blue);opacity:.35}.bar b{position:absolute;top:-2px;width:2px;bottom:-2px;background:var(--accent)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink-2);margin-top:6px}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.wbar{display:flex;height:10px;border-radius:5px;overflow:hidden;border:1px solid var(--line)}.wbar i{display:block}
svg text{font-family:"IBM Plex Sans",system-ui,sans-serif;fill:var(--ink-2);font-size:11px}svg .ax{stroke:var(--line)}svg .grid{stroke:var(--line);stroke-dasharray:2 3}
details{margin-top:10px}tr.closed td{color:var(--ink-3)}input,select,button{font:inherit;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink)}button{cursor:pointer}button.rm{border:none;background:none;color:var(--ink-3)}summary{cursor:pointer;color:var(--ink-2);font-size:13px}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}.kv div{background:var(--surface-2);border-radius:8px;padding:10px 12px}.kv .v{font-size:20px;font-weight:600}
.src{font-size:11.5px;color:var(--ink-3)}
"""
BANDCOL = ['#c0322a', '#eb6834', '#d9a441', '#5aa9d6', '#2a78d6']


def f(v, d=1, pre='', suf=''):
    if v is None or (isinstance(v, float) and not np.isfinite(v)): return '–'
    if v < 0 and pre: return f'−{pre}{abs(v):,.{d}f}{suf}'
    return f'{pre}{v:,.{d}f}{suf}'


def pct(v):
    return '–' if v is None else f'{100 * v:.0f}%'


def verdict(c):
    if 'edge' not in c: return '<span class="chip neutral">no settle</span>'
    if c['days_forecast'] == 0: return '<span class="chip neutral">settled</span>'
    k = c.get('call', 'none')
    if k == 'SELL': return '<span class="chip sell">SELL · clears the spread</span>'
    if k == 'BUY': return '<span class="chip buy">BUY · clears the spread</span>'
    if k.startswith('sell'): return '<span class="chip flat">above P75 · sell only if posted</span>'
    if k.startswith('buy'): return '<span class="chip flat">below P25 · buy only if posted</span>'
    return '<span class="chip flat">inside P25–P75 · no edge</span>'


def ordinal(v):
    if v is None: return '–'
    n = int(round(100 * v)); s = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{s}'


def contract_card(c):
    lo, hi = c['fair_p10'], c['fair_p90']
    span = max(hi - lo, 1); s = c.get('settle_flat')
    def pos(v): return 100 * (v - lo) / span
    bar = f'<div class="bar"><i style="left:{pos(c["fair_p25"]):.0f}%;width:{pos(c["fair_p75"]) - pos(c["fair_p25"]):.0f}%"></i>'
    if s is not None and lo - 0.15 * span <= s <= hi + 0.15 * span: bar += f'<b style="left:{min(max(pos(s), 0), 100):.0f}%"></b>'
    bar += '</div>'
    wb = ''.join(f'<i style="width:{100 * w:.0f}%;background:{BANDCOL[k]}" title="{BAND_LAB[k]}: {100 * w:.0f}%"></i>' for k, w in enumerate(c['band_w']))
    rows = [('Fair value (flat)', f'<b class="num">{f(c["fair_mean"], 2, "$")}</b>'),
            ('P10 / P50 / P90', f'<span class="num">{f(lo, 1, "$")} / {f(c["fair_p50"], 1, "$")} / {f(hi, 1, "$")}</span>'),
            ('Peak 7x16 / off-peak 7x8', f'<span class="num">{f(c["fair_pk"], 1, "$")} / {f(c["fair_op"], 1, "$")}</span>')]
    if s is not None:
        rows += [('Market flat (' + c['settle_date'] + ')', f'<b class="num">{f(s, 2, "$")}</b>'),
                 ('Market bid / ask (settle ∓ ' + f(c["half_spread"], 2, "$") + ')', f'<span class="num">{f(c["mkt_bid"], 2, "$")} / {f(c["mkt_ask"], 2, "$")}</span>'),
                 ('Model bid / offer (fair P25 / P75)', f'<span class="num">{f(c["model_bid"], 2, "$")} / {f(c["model_ask"], 2, "$")}</span>'),
                 ('Market peak / off-peak', f'<span class="num">{f(c.get("settle_pk"), 1, "$")} / {f(c.get("settle_op"), 1, "$")}</span>'),
                 ('Fair − market', f'<b class="num" style="color:var(--{"green" if c["edge"] > 0 else "red"})">{f(c["edge"], 2, "$", "")}</b>'),
                 ('Edge after crossing the spread', f'<b class="num" style="color:var(--{"green" if abs(c["edge"]) > c["half_spread"] else "amber"})">{f(abs(c["edge"]) - c["half_spread"], 2, "$")}</b> <span class="src">{"sell side" if c["edge"] < 0 else "buy side"}</span>'),
                 ('Contract range so far', f'<span class="num">{f(c["settle_lo"], 1, "$")} – {f(c["settle_hi"], 1, "$")}</span>')]
    if c['days_settled']:
        rows.append((f'Settled so far ({c["days_settled"]} d)', f'<span class="num">{f(c["settled_flat"], 2, "$")}</span>'))
    if c.get('market_note'): rows.append(('Market for the month in progress', f'<span class="src">{c["market_note"]}</span>'))
    if c.get('fair_at_gas'):
        rows.append(('Fair if AECO were $1 / $2 / $4 / $6 / $10', '<span class="num">' + ' / '.join(f(v, 0, '$') for v in c['fair_at_gas'].values()) + '</span>'))
        if c.get('fair_hr') and c.get('settle_hr'):
            rows.append(('Implied heat rate, fair · market (GJ/MWh)', f'<span class="num">{f(c["fair_hr"], 1)} · {f(c["settle_hr"], 1)}</span>'))
    rows += [('Expected cushion (P10–P90 of monthly mean)', f'<span class="num">{f(c["cush_mean"], 0)} MW ({f(c["cush_p10"], 0)}–{f(c["cush_p90"], 0)})</span>' if c['cush_mean'] is not None else '–'),
             ('AECO used', f'<span class="num">{f(c["gas_fwd"], 2, "$")}/GJ</span> <span class="src">{c["gas_src"]}</span>')]
    body = ''.join(f'<div class="row"><span>{k}</span><span>{v}</span></div>' for k, v in rows)
    # the percentiles, spelled out
    pcts = ''
    if s is not None:
        pf = c['settle_pct_fair']; pl = c.get('settle_pct_life'); plr = c.get('settle_pct_life_raw'); ph = c['settle_pct_hist']
        items = [(f'{ordinal(pf)} percentile of the model\'s outcomes',
                  f'the market at {f(s, 2, "$")} is above {100 * pf:.0f}% of the {K_SCEN} simulated settlements for this month — {"rich" if pf >= 0.75 else "cheap" if pf <= 0.25 else "inside the middle half"} on the model\'s terms'),
                 (f'{ordinal(pl)} percentile of this contract\'s own history, premium-adjusted' if pl is not None else 'own history: too few settles',
                  (f'every past settle of this contract was shifted onto today\'s lead by the usual decay before ranking; unadjusted it would read {ordinal(plr)} — contracts start life carrying a premium that fades, so the raw number is always low near expiry'
                   + (f'. From this lead the market typically still gives back {f(c["decay_ahead"], 2, "$")} into settlement' if c.get('decay_ahead') else '')
                   + (f'; {["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][c["m"].month - 1]} contracts have carried {f(c["seasonal_prem"], 0, "$")} over the realised month at 60–180 d in the last {c.get("seasonal_n", 0)} years' if c.get('seasonal_prem') is not None else '')) if pl is not None else ''),
                 (f'{ordinal(ph)} percentile of the last year\'s settled prices',
                  f'above {100 * ph:.0f}% of the daily average pool prices of the trailing 365 days — where the contract sits against what has actually been printing')]
        pcts = '<div class="eyebrow" style="margin:10px 0 4px">where the market sits</div>' + ''.join(f'<div class="pct"><b>{a}</b><span>{b}</span></div>' for a, b in items)
    return (f'<div class="card"><div class="eyebrow">{c["label"]}{" · lead " + str(c["lead"]) + " d" if c.get("lead", 0) > 0 else ""}</div>'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin:4px 0 8px"><div class="big num">{f(c["fair_mean"], 2, "$")}</div><div>{verdict(c)}</div></div>'
            f'{bar}<div class="legend"><span><i style="background:var(--blue);opacity:.35"></i>fair P25–P75 (bar spans P10–P90)</span><span><i style="background:var(--accent)"></i>market</span></div>'
            f'<div style="margin:10px 0 8px">{body}</div>{pcts}<div class="eyebrow" style="margin:10px 0 4px">where the month\'s days fall on the cushion bands</div><div class="wbar">{wb}</div></div>')


def svg_outlook(ol, cv, gas_by_day):
    t = ol.table; n = len(t)
    W, H, L, R, T, B = 1100, 300, 50, 60, 16, 40
    x = lambda i: L + (W - L - R) * i / max(n - 1, 1)
    cmin, cmax = min(t.p10.min(), 0), t.p90.max() * 1.05
    y = lambda v: T + (H - T - B) * (1 - (v - cmin) / (cmax - cmin))
    px = cv.read(t.cush.values, np.asarray(gas_by_day))
    pmax = max(px.max() * 1.1, 60)
    y2 = lambda v: T + (H - T - B) * (1 - v / pmax)
    band = 'M' + ' '.join(f'{x(i):.1f},{y(v):.1f}' for i, v in enumerate(t.p90)) + ' L' + ' '.join(f'{x(i):.1f},{y(v):.1f}' for i, v in reversed(list(enumerate(t.p10)))) + 'Z'
    line = 'M' + ' L'.join(f'{x(i):.1f},{y(v):.1f}' for i, v in enumerate(t.cush))
    pline = 'M' + ' L'.join(f'{x(i):.1f},{y2(v):.1f}' for i, v in enumerate(px))
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-height:320px">']
    # source shading for gas
    col = {'gencap': 'var(--green)', 'filed': 'var(--blue)', 'seasonal': 'var(--amber)'}
    for i, r in enumerate(t.itertuples()):
        s.append(f'<rect x="{x(i) - (W - L - R) / n / 2:.1f}" y="{H - B + 6}" width="{(W - L - R) / n + .5:.1f}" height="4" fill="{col.get(r.gas_src, "#999")}"/>')
        s.append(f'<rect x="{x(i) - (W - L - R) / n / 2:.1f}" y="{H - B + 12}" width="{(W - L - R) / n + .5:.1f}" height="4" fill="{"var(--green)" if r.load_src == "ens" else "var(--amber)"}"/>')
    for v in np.arange(0, cmax, 1000):
        s.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/><text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:,.0f}</text>')
    for v in np.arange(0, pmax, 20):
        s.append(f'<text x="{W - R + 6}" y="{y2(v) + 4:.1f}">${v:.0f}</text>')
    s.append(f'<path d="{band}" fill="var(--blue)" opacity=".18"/><path d="{line}" fill="none" stroke="var(--blue)" stroke-width="1.8"/>')
    s.append(f'<path d="{pline}" fill="none" stroke="var(--accent)" stroke-width="1.8" stroke-dasharray="4 3"/>')
    for i, d in enumerate(t.index):
        if d.day == 1 or (i == 0 and (t.index[0] + pd.offsets.MonthBegin(1) - t.index[0]).days > 6):
            s.append(f'<line class="ax" x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{T}" y2="{H - B}"/><text x="{x(i) + 4:.1f}" y="{H - B + 30}">{d:%b %d}</text>')
    s.append(f'<text x="{L}" y="{T - 4}">cushion MW (P10–P90)</text><text x="{W - R}" y="{T - 4}" text-anchor="end">expected daily price, $/MWh (dashed)</text>')
    s.append('</svg>')
    return ''.join(s)


def svg_curve(tab):
    W, H, L, R, T, B = 560, 260, 44, 12, 14, 30
    x = lambda v: L + (W - L - R) * (v - tab.cush.min()) / (tab.cush.max() - tab.cush.min())
    ymax = tab.peak.max() * 1.05
    y = lambda v: T + (H - T - B) * (1 - v / ymax)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%">']
    for v in np.arange(0, ymax, 25): s.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/><text x="{L - 5}" y="{y(v) + 4:.1f}" text-anchor="end">${v:.0f}</text>')
    for v in np.arange(500, 4001, 500): s.append(f'<text x="{x(v):.1f}" y="{H - 8}" text-anchor="middle">{v:,.0f}</text>')
    for col, name in [('var(--red)', 'peak'), ('var(--ink)', 'flat'), ('var(--blue)', 'offpk')]:
        s.append('<path d="M' + ' L'.join(f'{x(a):.1f},{y(b):.1f}' for a, b in zip(tab.cush, tab[name])) + f'" fill="none" stroke="{col}" stroke-width="1.8"/>')
    s.append('</svg>')
    return ''.join(s)


def table(rows, cols, fmt=None, cls=None):
    fmt = fmt or {}
    h = '<table><thead><tr>' + ''.join(f'<th class="{"l" if i == 0 else ""}">{c[1]}</th>' for i, c in enumerate(cols)) + '</tr></thead><tbody>'
    for r in rows:
        h += '<tr>' + ''.join(f'<td class="{"l" if i == 0 else "num"}">{fmt.get(c[0], lambda v: v if isinstance(v, str) else f(v))(r.get(c[0]))}</td>' for i, c in enumerate(cols)) + '</tr>'
    return h + '</tbody></table>'


TRACKER_JS = r"""
(function(){
const D = window.__AESO__; const K='aeso_monthly_trades_v1';
const $=s=>document.querySelector(s); const fmt=(v,d=2)=> (v==null||isNaN(v))?'–':(v<0?'−':'')+'$'+Math.abs(v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
let trades=[]; try{trades=JSON.parse(localStorage.getItem(K)||'[]')}catch(e){trades=[]}
function save(){try{localStorage.setItem(K,JSON.stringify(trades))}catch(e){}}
function hoursIn(m){const d=new Date(m+'-01T00:00:00');return 24*new Date(d.getFullYear(),d.getMonth()+1,0).getDate()}
function leadBucket(lead){return D.excursions.find(e=>lead>=e.lead_lo&&lead<=e.lead_hi)||D.excursions[D.excursions.length-1]}
function contract(m){return D.contracts.find(c=>c.m===m)}
function render(){
  const sel=$('#t_contract'); sel.innerHTML=D.contracts.filter(c=>c.settle_flat!=null).map(c=>`<option value="${c.m}">${c.label}</option>`).join('');
  const tb=$('#t_rows'); let rows='', tot={pnl:0,exp:0,risk:0};
  trades.forEach((t,i)=>{
    const c=contract(t.m); const open=t.exit==null||t.exit==='';
    const mark=c?c.settle_flat:null; const hrs=hoursIn(t.m); const sgn=t.side==='SELL'?-1:1;
    const px = open?mark:parseFloat(t.exit);
    const pnl = px==null?null:sgn*(px-t.price)*t.mw*hrs;
    const exp = (c&&open)?sgn*(c.fair_mean-t.price)*t.mw*hrs:null;
    const lead = c?Math.max(1,Math.round((new Date(c.m+'-01T00:00:00')-new Date(D.asof+'T00:00:00'))/86400000)):null;
    const eb = lead?leadBucket(lead):null;
    const tp = c?(t.side==='SELL'?c.fair_p25:c.fair_p75):null;            // take profit: the model's own bid/offer on the other side
    const tp2= c?c.fair_mean:null;
    const tail = c?(t.side==='SELL'?c.fair_p90:c.fair_p10):null;           // model tail against you
    const mae = eb?(t.side==='SELL'?eb.short_p75:eb.long_p75):null;       // historical adverse excursion at this lead, P75
    const mae90 = eb?(t.side==='SELL'?eb.short_p90:eb.long_p90):null;
    const heat = mae!=null?t.price+ (t.side==='SELL'?mae:-mae):null;        // where the mark typically goes against you before settlement
    const heat90 = mae90!=null?t.price+ (t.side==='SELL'?mae90:-mae90):null;
    const riskAtHeat = mae90!=null?-mae90*t.mw*hrs:null; const riskAtTail = tail!=null?sgn*(tail-t.price)*t.mw*hrs:null;
    const intact = c?((t.side==='SELL'&&c.fair_mean<t.price)||(t.side==='BUY'&&c.fair_mean>t.price)):null;   // the exit rule: the model flipping through your price
    if(open&&pnl!=null){tot.pnl+=pnl;tot.exp+=exp||0;tot.risk+=riskAtHeat||0}
    rows+=`<tr class="${open?'':'closed'}"><td class="l">${t.date}</td><td class="l">${c?c.label:t.m}</td><td class="l">${t.side}</td><td class="num">${t.mw}</td><td class="num">${fmt(t.price)}</td>
      <td class="num">${open?fmt(mark):'closed '+fmt(px)}</td><td class="num" style="color:var(--${pnl>=0?'green':'red'})">${fmt(pnl,0)}</td>
      <td class="num">${open?fmt(c?c.fair_mean:null):'–'}</td><td class="num">${open?fmt(exp,0):'–'}</td>
      <td class="num">${open?fmt(tp)+' <span class=src>· '+fmt(tp2)+'</span>':'–'}</td>
      <td class="num">${open?fmt(heat)+' <span class=src>· '+fmt(heat90)+'</span>':'–'}</td>
      <td class="num" style="color:var(--red)">${open?fmt(riskAtHeat,0)+' <span class=src>· '+fmt(riskAtTail,0)+'</span>':'–'}</td>
      <td class="l">${open?(intact?'<span class="chip buy">thesis intact · hold</span>':'<span class="chip sell">model flipped · exit</span>'):'–'}</td>
      <td class="l">${t.note||''}</td>
      <td class="l">${open?`<button data-i="${i}" class="cl">close</button>`:''} <button data-i="${i}" class="rm">✕</button></td></tr>`;
  });
  tb.innerHTML=rows||'<tr><td colspan="15" class="l src">no trades yet — add one above</td></tr>';
  $('#t_tot').innerHTML=`open P&L <b class="num" style="color:var(--${tot.pnl>=0?'green':'red'})">${fmt(tot.pnl,0)}</b> · expected at settle <b class="num">${fmt(tot.exp,0)}</b> · mark-to-market at the P90 heat <b class="num" style="color:var(--red)">${fmt(tot.risk,0)}</b>`;
  document.querySelectorAll('button.rm').forEach(b=>b.onclick=()=>{trades.splice(+b.dataset.i,1);save();render()});
  document.querySelectorAll('button.cl').forEach(b=>b.onclick=()=>{const v=prompt('exit price ($/MWh flat)');if(v!==null&&v!==''){trades[+b.dataset.i].exit=parseFloat(v);save();render()}});
}
$('#t_add').onclick=()=>{const t={date:$('#t_date').value||D.asof,m:$('#t_contract').value,side:$('#t_side').value,mw:parseFloat($('#t_mw').value)||1,price:parseFloat($('#t_price').value),note:$('#t_note').value,exit:null};
  if(!isFinite(t.price)){alert('price?');return} trades.push(t);save();render();$('#t_price').value='';$('#t_note').value=''};
$('#t_export').onclick=()=>{const s=JSON.stringify(trades);const ta=$('#t_io');ta.value=s;ta.hidden=false;ta.select();
  if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(s).then(()=>{$('#t_msg').textContent='copied — paste it anywhere to keep it'}).catch(()=>{$('#t_msg').textContent='select the text below and copy it'})}else{$('#t_msg').textContent='select the text below and copy it'}};
$('#t_import').onclick=()=>{const ta=$('#t_io');ta.hidden=false;ta.value='';ta.placeholder='paste a saved trades list here, then press load';ta.focus();$('#t_load').hidden=false};
$('#t_load').onclick=()=>{try{const v=JSON.parse($('#t_io').value);if(!Array.isArray(v))throw 0;trades=v;save();render();$('#t_msg').textContent=v.length+' trades loaded';$('#t_load').hidden=true;$('#t_io').hidden=true}catch(x){$('#t_msg').textContent='that is not a trades list'}};
$('#t_date').value=D.asof; render();
})();
"""


K_SCEN = 400


def build(out, contracts, cv, ol, tr, bk, seas, rg, R, score, prem, meta, strip=None, exc=None, dec=None):
    global K_SCEN; K_SCEN = meta.get('K', 400)
    asof = meta['asof']
    cards = ''.join(contract_card(c) for c in contracts)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AESO Monthly Fair Value</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Condensed:wght@600;700&display=swap">
<style>{CSS}
.pct{{display:grid;gap:2px;padding:6px 0;border-top:1px solid var(--line);font-size:12.5px}}.pct b{{font-weight:600}}.pct span{{color:var(--ink-2)}}
</style></head><body>
<header><div class="wrap"><div class="hrow"><h1>AESO monthly · fair value for the prompt months</h1>
<span class="meta">as of <b>{asof:%a %d %b %Y}</b></span><span class="meta">history: rolling 365 d to {tr.index.max():%d %b}, current month excluded ({len(tr)} days)</span>
<span class="meta">settled price to {meta['price_end']:%d %b %H:00}</span><span class="meta">forwards: {meta['fwd_end']}</span><span class="meta">spread assumed {meta['spread']:.2f} wide</span></div></div></header>
<main>
<section><h2>The read</h2>
<p class="lead">Each month's flat price is read off the trailing year's cushion→price curve, averaged over {K_SCEN} scenarios of daily cushions built from the outlook (ensemble load, ensemble then climatology wind, gas availability from AESO gencap → the filed outage report → the seasonal norm, less the unfiled-outage adder). Gas enters as an in-sample sensitivity plus a hard floor at the marginal gas unit's cost. The market's bid/ask is the settle straddled by the assumed spread; the model's is its own P25 / P75. A card says SELL or BUY only when the settle clears the model's band by more than half the spread.</p>
<div class="cards">{cards}</div>
<div class="legend" style="margin-top:10px">bands: {' '.join(f'<span><i style="background:{BANDCOL[k]}"></i>{BAND_LAB[k]} MW</span>' for k in range(5))}</div>
<p class="note">Walk-forward record, re-measured this run: read with the month's cushion known MAE {f(score.get('perfect', {}).get('mae'), 1, '$')} (corr {f(score.get('perfect', {}).get('corr'), 2)}), outlook-only MAE {f(score.get('realistic', {}).get('mae'), 1, '$')} (corr {f(score.get('realistic', {}).get('corr'), 2)}), the 30-day-out market MAE {f(score.get('fwd30', {}).get('mae'), 1, '$')} (corr {f(score.get('fwd30', {}).get('corr'), 2)}, bias {f(score.get('fwd30', {}).get('bias'), 1, '$')}) over {score.get('perfect', {}).get('n', 0)} settled months.</p>
</section>
</main></body></html>"""
    out.write_text(doc, encoding='utf-8')
    body = doc.split('<head>', 1)[1].split('</head>', 1)
    head, rest = body[0], body[1]
    head = head.replace('<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">', '')
    head = head.replace('@media (prefers-color-scheme:dark){:root{', '@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;')
    dark = head.split('@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;', 1)[1].split('}}', 1)[0]
    head = head.replace('*{box-sizing:border-box}', ':root[data-theme="dark"]{color-scheme:dark;' + dark + '}\n*{box-sizing:border-box}', 1)
    head = head.replace('header{position:sticky;top:0;', 'header{position:sticky;top:env(safe-area-inset-top,0px);')
    art = head.strip() + rest.replace('<body>', '').replace('</body></html>', '')
    (out.parent / 'artifact.html').write_text(art, encoding='utf-8')
    return doc
