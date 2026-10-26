"""Small HTML builders for the Streamlit UI (cards, chips, score bars, comparison table).

Everything returned here is a plain string rendered with st.markdown(..., unsafe_allow_html=True), so every
value that could contain user text is escaped with html.escape. No feature engineering or scoring happens here.
"""
from datetime import date
from html import escape

from src.app_helpers import format_inr
from src.recommender import CPU_CLASS, GPU_CLASS, storage_text

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1280px;}
.hero {border-radius: 16px; padding: 1.4rem 1.6rem; margin: 0.2rem 0 1.2rem 0;
       background: linear-gradient(135deg, rgba(42,120,214,0.14), rgba(42,120,214,0.03));
       border: 1px solid rgba(42,120,214,0.22);}
.hero h3 {margin: 0 0 0.35rem 0; font-size: 1.35rem;}
.hero p {margin: 0; opacity: 0.85; font-size: 0.98rem;}
.pill-row {display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.8rem;}
.pill {display: inline-block; font-size: 0.8rem; padding: 0.2rem 0.65rem; border-radius: 999px;
       background: rgba(42,120,214,0.10); border: 1px solid rgba(42,120,214,0.25);}
.chip {display: inline-block; font-size: 0.82rem; padding: 0.22rem 0.6rem; margin: 0 0.35rem 0.4rem 0;
       border-radius: 8px; background: rgba(128,128,128,0.10); border: 1px solid rgba(128,128,128,0.22);}
.chip b {font-weight: 600;}
.note {font-size: 0.88rem; padding: 0.55rem 0.8rem; border-radius: 10px; margin: 0.3rem 0;
       background: rgba(230,160,40,0.10); border-left: 4px solid rgba(230,160,40,0.8);}
.status {font-size: 0.92rem; padding: 0.6rem 0.9rem; border-radius: 10px; margin: 0.6rem 0 0.9rem 0;
         background: rgba(42,120,214,0.08); border-left: 4px solid #2a78d6;}
.status.warn {background: rgba(230,160,40,0.10); border-left-color: rgba(230,160,40,0.9);}
.spec-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.7rem; margin: 0.4rem 0 1rem 0;}
.spec {border: 1px solid rgba(128,128,128,0.22); border-radius: 12px; padding: 0.7rem 0.85rem;}
.spec .k {font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.65;}
.spec .v {font-weight: 600; margin: 0.15rem 0; font-size: 0.95rem;}
.spec .s {font-size: 0.78rem; opacity: 0.7;}
.lap {border: 1px solid rgba(128,128,128,0.25); border-radius: 16px; padding: 1rem 1.1rem 0.6rem 1.1rem;
      height: 100%; background: rgba(128,128,128,0.03);}
.lap.top {border: 2px solid #2a78d6; background: rgba(42,120,214,0.05);}
@media (min-width: 1000px) {.lap {min-height: 720px;}}
.rank {display: inline-block; font-size: 0.75rem; font-weight: 600; padding: 0.15rem 0.55rem; border-radius: 999px;
       background: rgba(42,120,214,0.15); color: #2a78d6; margin-bottom: 0.4rem;}
.lap h4 {margin: 0.1rem 0 0.15rem 0; font-size: 1.05rem; line-height: 1.3;}
.lap .sub {font-size: 0.78rem; opacity: 0.65; margin-bottom: 0.55rem;}
.lap .price {font-size: 1.55rem; font-weight: 700; margin: 0.1rem 0 0 0;}
.lap .src {font-size: 0.75rem; opacity: 0.7; margin-bottom: 0.25rem;}
.lap .budget {font-size: 0.78rem; margin-bottom: 0.55rem;}
.lap ul {margin: 0.2rem 0 0.6rem 0; padding-left: 1.1rem; font-size: 0.86rem;}
.lap li {margin-bottom: 0.15rem;}
.lap .pro li::marker {content: "✓  "; color: #2e9e5b;}
.lap .con li::marker {content: "–  "; color: #c7372f;}
.lap .tradeoff {font-size: 0.8rem; opacity: 0.8; font-style: italic; margin-bottom: 0.5rem;}
.bar-wrap {margin: 0.35rem 0 0.7rem 0;}
.bar-label {display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.15rem;}
.bar {height: 8px; border-radius: 999px; background: rgba(128,128,128,0.18); overflow: hidden;}
.bar > div {height: 100%; border-radius: 999px; background: #2a78d6;}
.cmp-wrap {overflow-x: auto; margin: 0.4rem 0 1rem 0;}
table.cmp {border-collapse: collapse; width: 100%; font-size: 0.87rem; min-width: 560px;}
table.cmp th, table.cmp td {border-bottom: 1px solid rgba(128,128,128,0.2); padding: 0.5rem 0.6rem; text-align: left; vertical-align: top;}
table.cmp th {font-weight: 600;}
table.cmp td.k {opacity: 0.7; white-space: nowrap; width: 150px;}
table.cmp td.best {font-weight: 700;}
.price-card {border: 1px solid rgba(128,128,128,0.25); border-radius: 12px; padding: 1.1rem 1.3rem;
             background: rgba(42,120,214,0.06); margin-bottom: 0.6rem;}
.price-card .label {font-size: 0.9rem; opacity: 0.75; margin: 0;}
.price-card .value {font-size: 2.6rem; font-weight: 700; line-height: 1.15; margin: 0.15rem 0 0.35rem 0;}
.price-card .note {font-size: 0.85rem; opacity: 0.75; margin: 0; background: none; border: none; padding: 0;}
.historic {font-size: 0.88rem; padding: 0.7rem 0.9rem; border-radius: 10px; margin: 0 0 1rem 0;
           background: rgba(128,128,128,0.08); border-left: 4px solid rgba(128,128,128,0.6);}
</style>
"""


def nice_date(iso):
    d = date.fromisoformat(iso)
    return f'{d.day} {d.strftime("%b %Y")}'


def price_text(price):
    return format_inr(round(price))


def chips(evidence):
    if not evidence:
        return ''
    return ''.join(f'<span class="chip">“{escape(phrase)}” → <b>{escape(meaning)}</b></span>' for phrase, meaning in evidence)


def note(text):
    return f'<div class="note">{escape(text)}</div>'


def status_box(text, warn=False):
    return f'<div class="status{" warn" if warn else ""}">{escape(text)}</div>'


def spec_grid(rows):
    cells = ''.join(f'<div class="spec"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div>'
                    f'<div class="s">{escape(s)}</div></div>' for k, v, s in rows)
    return f'<div class="spec-grid">{cells}</div>'


def score_bar(label, value, right=None):
    pct = max(0, min(100, round(value * 100)))
    return (f'<div class="bar-wrap"><div class="bar-label"><span>{escape(label)}</span><span>{escape(right or f"{pct}/100")}</span></div>'
            f'<div class="bar"><div style="width:{pct}%"></div></div></div>')


def laptop_card(rec, rank, budget_set):
    lap = rec.laptop
    rank_label = {1: '#1 Best match', 2: '#2', 3: '#3'}.get(rank, f'#{rank}')
    specs = [lap.cpu, lap.gpu, f'{lap.ram_gb} GB RAM · {lap.storage_label}',
             f'{lap.display_in:g}" {lap.resolution_w}×{lap.resolution_h}' + (f' · {lap.refresh_hz} Hz' if lap.refresh_hz else ''),
             f'{lap.weight_kg:g} kg · {lap.battery_wh:g} Wh battery · {lap.os}']
    if rec.budget_gap is None:
        budget = ''
    elif rec.budget_gap >= 0:
        budget = f'<div class="budget" style="color:#2e9e5b">{price_text(rec.budget_gap)} under your budget</div>'
    else:
        budget = f'<div class="budget" style="color:#c7372f">{price_text(-rec.budget_gap)} over your budget</div>'
    pros = ''.join(f'<li>{escape(p)}</li>' for p in rec.pros)
    cons = ''.join(f'<li>{escape(c)}</li>' for c in (rec.gaps and [f'Below minimum: {g}' for g in rec.gaps] or []) + rec.cons)
    trade = f'<div class="tradeoff">{escape(rec.trade_off)}</div>' if rec.trade_off else ''
    return f"""
<div class="lap{' top' if rank == 1 else ''}">
  <span class="rank">{rank_label}</span>
  <h4>{escape(lap.display_name)}</h4>
  <div class="sub">Part no. {escape(lap.part_number)}</div>
  <div class="price">{price_text(lap.price_inr)}</div>
  <div class="src">Listed on {escape(lap.price_source)} · verified {nice_date(lap.verified_on)}</div>
  {budget}
  {score_bar('Suitability for your needs', rec.score)}
  <ul>{''.join(f'<li>{escape(s)}</li>' for s in specs)}</ul>
  {f'<b style="font-size:0.85rem">Why it fits</b><ul class="pro">{pros}</ul>' if pros else ''}
  {f'<b style="font-size:0.85rem">Trade-offs</b><ul class="con">{cons}</ul>' if cons else ''}
  {trade}
</div>"""


def breakdown_html(rec):
    parts = [score_bar(f"{c['label']} (weight {round(c['weight'] * 100)}%)", c['score']) for c in rec.breakdown]
    return ''.join(parts)


def comparison_table(laptops, scores=None):
    """Side-by-side table. `scores` maps laptop id -> 0-1 suitability (optional). Best value per row in bold."""
    cols = list(laptops)
    rows = [
        ('Price', [price_text(l.price_inr) for l in cols], [-l.price_inr for l in cols]),
        ('Price source', [f'{l.price_source}, verified {nice_date(l.verified_on)}' for l in cols], None),
        ('Processor', [l.cpu for l in cols], None),
        ('Processor class', [CPU_CLASS[l.cpu_tier].split(' (')[0] for l in cols], [l.cpu_tier for l in cols]),
        ('Graphics', [l.gpu for l in cols], None),
        ('Graphics class', [GPU_CLASS[l.gpu_tier].split(' (')[0] for l in cols], [l.gpu_tier for l in cols]),
        ('NVIDIA CUDA', ['Yes' if l.cuda else 'No' for l in cols], [int(l.cuda) for l in cols]),
        ('RAM', [f'{l.ram_gb} GB' for l in cols], [l.ram_gb for l in cols]),
        ('Storage', [l.storage_label for l in cols], [l.storage_gb for l in cols]),
        ('Display', [f'{l.display_in:g}" {l.resolution_w}×{l.resolution_h}' for l in cols], [l.resolution_w * l.resolution_h for l in cols]),
        ('Refresh rate', [f'{l.refresh_hz} Hz' if l.refresh_hz else 'Not stated' for l in cols], [l.refresh_hz or 0 for l in cols]),
        ('Weight', [f'{l.weight_kg:g} kg' for l in cols], [-l.weight_kg for l in cols]),
        ('Battery', [f'{l.battery_wh:g} Wh' for l in cols], [l.battery_wh for l in cols]),
        ('Operating system', [l.os for l in cols], None),
    ]
    if scores:
        rows.insert(1, ('Suitability for your needs', [f'{round(scores[l.id] * 100)}/100' if l.id in scores else '—' for l in cols],
                        [scores.get(l.id, -1) for l in cols]))
    head = ''.join(f'<th>{escape(l.display_name)}</th>' for l in cols)
    body = []
    for label, values, rank in rows:
        best = max(rank) if rank and len(set(rank)) > 1 else None
        tds = ''.join(f'<td class="{"best" if best is not None and r == best else ""}">{escape(v)}</td>'
                      for v, r in zip(values, rank or [None] * len(values)))
        body.append(f'<tr><td class="k">{escape(label)}</td>{tds}</tr>')
    links = ''.join(f'<td><a href="{escape(l.url)}" target="_blank" rel="noopener">Official store page ↗</a></td>' for l in cols)
    body.append(f'<tr><td class="k">Buy / verify</td>{links}</tr>')
    return f'<div class="cmp-wrap"><table class="cmp"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
