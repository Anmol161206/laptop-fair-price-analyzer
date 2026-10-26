"""Visual system for Laptop Advisor: CSS, original device illustrations and HTML builders.

Everything here is presentation only. Strings are rendered with st.markdown(..., unsafe_allow_html=True), so every
value that could contain catalogue or user text is escaped with html.escape. No feature engineering or scoring.

Imagery: product photos are not used (rights unclear). Each laptop gets an original line illustration whose
proportions come from its real data - screen size sets the lid width, resolution sets the screen aspect - and whose
tint comes from its category.
"""
import re
from datetime import date
from html import escape

from src.app_helpers import format_inr
from src.recommender import CPU_CLASS, gpu_class_label, storage_text

# ---------------------------------------------------------------------------------------------- design tokens
PAPER, CARD, INK, MUTED, LINE, ACCENT = '#F6F3EE', '#FFFDF9', '#1C1B19', '#6B665E', '#E4DDD1', '#2B59C3'

CATEGORY_STYLE = {   # body, screen top, screen bottom, label
    'Mac':         ('#B9B6B0', '#3B3F4A', '#7C8496', 'Mac'),
    'Gaming':      ('#2A2833', '#1D2A6B', '#4F6FE0', 'Gaming'),
    'Workstation': ('#2E3B3F', '#1F4A50', '#5E9296', 'Workstation'),
    'Business':    ('#2F3A4C', '#22385E', '#6C86B4', 'Business'),
    'Ultralight':  ('#C9BFAE', '#5B6B86', '#B9C6DA', 'Ultralight'),
    'Budget':      ('#A9A398', '#4E6455', '#9DB5A3', 'Starter'),
    'Everyday':    ('#8E887E', '#46526A', '#9AA8C2', 'Everyday'),
}

ART_BG = {'Mac': '#ECEAE6', 'Gaming': '#E7E9F4', 'Workstation': '#E3ECEA', 'Business': '#E6EAF1',
          'Ultralight': '#F2ECE2', 'Budget': '#E7EEE7', 'Everyday': '#EDE9E2'}


def art_bg(lap):
    tint = ART_BG[category(lap)]
    return f'background: linear-gradient(165deg, {PAPER} 0%, {tint} 100%);'


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp a,
.stApp [data-baseweb="select"] div, .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stMarkdownContainer"] div,
.stApp [data-testid="stMarkdownContainer"] span:not([data-testid="stIconMaterial"]) {{
  font-family: 'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif !important; }}
.stApp h1, .stApp h2, .stApp h3,
.stApp [data-testid="stMarkdownContainer"] :is(h1, h2, h3, .serif, .hero-title, .rec-name, .price, .name, .tprice b, .v, .value h4,
  .story h3, .wordmark em, .tile .size, .cap .big, .hero-list .p),
.stApp table.cmp thead th, .stApp .price-card .value,
.stApp :is(h1, h2, h3, h4) span:not([data-testid="stIconMaterial"]),
.stApp [data-testid="stMarkdownContainer"] :is(h1, h2, h3, h4) span:not([data-testid="stIconMaterial"]) {{
  font-family: 'Fraunces', Georgia, 'Times New Roman', serif !important; }}
.stApp {{ -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; text-rendering: optimizeLegibility; }}
/* Streamlit sets weight 400 on the <p> inside buttons and tabs; restore the intended weights */
.stApp button p {{ font-weight: inherit !important; }}
.stApp button {{ font-weight: 600; }}
/* hide default Streamlit chrome: the app has its own top bar */
header[data-testid="stHeader"], [data-testid="stSidebarCollapsedControl"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stMainMenu"] {{ display:none !important; }}
[data-testid="stAppViewContainer"] {{ background: {PAPER}; }}
.block-container {{ max-width: 1240px; padding: 1.6rem 2rem 5rem 2rem; }}
h1, h2, h3, .serif {{ letter-spacing: -0.01em; }}
a {{ color: {ACCENT}; }}

/* ---------- form controls: one quiet, consistent style ---------- */
.stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"], .stApp [data-baseweb="textarea"] {{
  background:{CARD} !important; border:1px solid {LINE} !important; border-radius:12px !important; }}
.stApp [data-baseweb="input"] input, .stApp [data-baseweb="input"] > div, .stApp [data-baseweb="textarea"] textarea {{ background:transparent !important; }}
.stApp [data-baseweb="select"] > div:focus-within, .stApp [data-baseweb="input"]:focus-within, .stApp [data-baseweb="textarea"]:focus-within {{
  border-color:{ACCENT} !important; box-shadow: 0 0 0 3px rgba(43,89,195,.14) !important; }}
.stApp [data-testid="stMultiSelectTagsContainer"] [role="group"] > span {{ background:#E8EDF8 !important; color:#1E3F8F !important;
  border:1px solid #CCD7F0; border-radius:8px !important; }}
.stApp [data-testid="stMultiSelectTagsContainer"] [role="group"] > span span {{ color:#1E3F8F !important; font-weight:600; }}
.stApp [data-testid="stMultiSelectTagsContainer"] [role="group"] > span button {{ color:#1E3F8F !important; }}
.stApp [data-testid="stWidgetLabel"] p {{ font-weight:600; color:{INK}; font-size:.86rem; }}
.stApp [data-testid="stExpander"] details {{ border:1px solid {LINE}; border-radius:14px; background:{CARD}; }}
.stApp [data-testid="stExpander"] summary p {{ font-size:.86rem; font-weight:600; color:#3E3B36; }}
.stApp [data-testid="stNumberInputContainer"] button {{ background:transparent !important; }}
.st-key-ex_collection button {{ border-radius:999px !important; font-weight:600; background:{CARD}; }}
.st-key-ex_collection button[aria-checked="true"] {{ background:{INK} !important; color:{CARD} !important; border-color:{INK} !important; }}
.st-key-ex_collection button[aria-checked="true"] p {{ color:{CARD} !important; }}
.stApp [data-testid="stBaseButton-secondary"] {{ border-radius:999px; border-color:#CFC6B7; background:{CARD}; }}
.stApp [data-testid^="stBaseButton-primary"] {{ border-radius:999px; font-weight:700; }}
.stApp [data-testid="stAlert"] > div {{ background:{CARD} !important; border:1px solid {LINE}; border-radius:16px; color:{INK} !important; }}
.stApp [data-testid="stAlert"] p {{ color:#3E3B36 !important; }}
[class*="st-key-lab_"] {{ background:{CARD}; border-color:{LINE} !important; border-radius:20px !important; padding: 1.1rem 1.2rem !important; }}
[class*="st-key-lab_"] h4 {{ font-weight:500 !important; font-size:1.3rem !important; padding-top:0 !important; }}

/* ---------- top bar ---------- */
.topbar {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; padding: .2rem 0 .9rem 0; }}
.wordmark {{ display:flex; align-items:center; gap:.55rem; font-weight:800; font-size:1.05rem; letter-spacing:-.01em; color:{INK}; }}
.wordmark .mark {{ width:28px; height:28px; border-radius:8px; background:{INK}; display:grid; place-items:center; }}
.wordmark em {{ font-family:'Fraunces',serif; font-style:italic; font-weight:500; color:{ACCENT}; }}
.topbar .stamp {{ font-size:.78rem; color:{MUTED}; border:1px solid {LINE}; border-radius:999px; padding:.25rem .7rem; background:{CARD}; white-space:nowrap; }}

/* ---------- navigation (st.tabs; Streamlit 1.64 renders role=tablist/tab) ---------- */
.stTabs [role="tablist"] {{ gap: 1.3rem; scrollbar-width:none; }}
.stTabs [role="tablist"]::-webkit-scrollbar {{ display:none; }}
.stTabs [role="tab"] {{ color:{MUTED}; white-space:nowrap; }}
.stTabs [role="tab"] p {{ font-size: .93rem; font-weight:600 !important; }}
.stTabs [role="tab"]:hover {{ color:{INK}; }}
.stTabs [role="tab"][aria-selected="true"] {{ color:{INK}; }}
.stTabs [role="tab"][aria-selected="true"] p {{ font-weight:800 !important; }}

/* ---------- hero ---------- */
.st-key-hero {{ background: radial-gradient(1200px 500px at 85% -10%, rgba(43,89,195,.10), transparent 60%), {CARD};
  border:1px solid {LINE}; border-radius: 28px; padding: 2.4rem 2.4rem 2rem 2.4rem; margin-top: 1rem;
  box-shadow: 0 30px 60px -40px rgba(28,27,25,.35); }}
.eyebrow {{ text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; color:{MUTED}; }}
.hero-title {{ font-family:'Fraunces',serif; font-weight:500; font-size: clamp(2.4rem, 5.2vw, 4.3rem); line-height:1.02;
  letter-spacing:-.025em; color:{INK}; margin:.55rem 0 1rem 0; }}
.hero-title em {{ font-style:italic; color:{ACCENT}; }}
.hero-sub {{ font-size: clamp(1rem, 1.35vw, 1.13rem); line-height:1.6; color:#3E3B36; max-width: 34rem; margin:0 0 1.2rem 0; }}
.st-key-hero textarea {{ font-size:1rem !important; line-height:1.55 !important; border-radius:16px !important; background:{PAPER} !important; }}
.st-key-hero [data-testid="stTextArea"] label p {{ font-weight:700; color:{INK}; }}
.st-key-fm_find button {{ height: 3.15rem; border-radius: 999px; font-weight: 800; font-size: 1rem; letter-spacing:.01em;
  box-shadow: 0 14px 24px -14px rgba(43,89,195,.75); }}
.st-key-examples {{ gap:.45rem !important; }}
.st-key-examples button {{ border-radius:999px !important; padding:.2rem .9rem !important; min-height:2.1rem !important;
  background:transparent !important; border:1px solid {LINE} !important; color:{INK} !important; font-weight:600 !important; }}
.st-key-examples button:hover {{ border-color:{ACCENT} !important; color:{ACCENT} !important; }}
.hint {{ font-size:.86rem; color:{MUTED}; margin-top:.3rem; }}
.hint b {{ color:{INK}; }}

.hero-art {{ border-radius:22px; padding: 1.4rem 1.4rem 1rem 1.4rem; margin-top:.4rem;
  background: radial-gradient(420px 220px at 50% 38%, rgba(255,255,255,.95), transparent 70%), linear-gradient(170deg, #EDF0F7 0%, #F1ECE3 100%);
  border:1px solid {LINE}; }}
.hero-art .stage {{ position:relative; height: 250px; }}
.hero-art .stage::after {{ content:""; position:absolute; left:6%; right:6%; bottom:4px; height:16px; border-radius:50%;
  background: radial-gradient(closest-side, rgba(28,27,25,.16), transparent); }}
.hero-art .dev {{ position:absolute; bottom:12px; filter: drop-shadow(0 18px 16px rgba(28,27,25,.20)); }}
.hero-art .d2 {{ width:60%; left:20%; bottom:58px; z-index:1; }}
.hero-art .d1 {{ width:43%; right:0; z-index:3; }}
.hero-art .d1 .pin {{ left:auto; right:-6px; }}
.hero-art .d3 {{ width:41%; left:0; z-index:2; }}
.hero-art .pin {{ position:absolute; top:-9px; left:-6px; width:22px; height:22px; border-radius:50%; background:{INK}; color:{CARD};
  font-size:.7rem; font-weight:800; display:grid; place-items:center; box-shadow:0 0 0 3px rgba(255,253,249,.9); }}
.hero-list {{ list-style:none; margin:.9rem 0 0 0; padding:0; border-top:1px solid rgba(28,27,25,.10); }}
.hero-list li {{ display:grid; grid-template-columns: 22px minmax(0,1fr) auto; gap:.7rem; align-items:center; padding:.55rem 0;
  border-bottom:1px solid rgba(28,27,25,.07); font-size:.8rem; }}
.hero-list li:last-child {{ border-bottom:none; }}
.hero-list .n {{ width:20px; height:20px; border-radius:50%; border:1px solid {INK}; font-size:.66rem; font-weight:800; display:grid; place-items:center; }}
.hero-list b {{ display:block; font-size:.86rem; color:{INK}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.hero-list .meta {{ color:{MUTED}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; display:block; }}
.hero-list .p {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.02rem; color:{INK}; text-align:right; white-space:nowrap; }}
.hero-list .p small {{ display:block; font-family:'Manrope',sans-serif; font-weight:500; font-size:.68rem; color:{MUTED}; }}

.values {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 1rem; margin: 1.4rem 0 .4rem 0; }}
.value {{ border-top: 1px solid {LINE}; padding-top: .9rem; }}
.value h4 {{ font-family:'Fraunces',serif; font-weight:500; font-size:1.12rem; margin:0 0 .25rem 0; color:{INK}; }}
.value p {{ margin:0; color:{MUTED}; font-size:.9rem; line-height:1.5; }}

/* ---------- section headings ---------- */
.section {{ margin: 2.4rem 0 .9rem 0; }}
.section h2 {{ font-weight:500; font-size: clamp(1.6rem, 2.6vw, 2.25rem); line-height:1.1; margin:.25rem 0 .35rem 0; color:{INK}; }}
.section p {{ color:{MUTED}; margin:0; max-width: 46rem; line-height:1.55; }}

/* ---------- brief (what we heard / fine-tune) ---------- */
.st-key-brief, .st-key-tune {{ background:{CARD}; border:1px solid {LINE}; border-radius:20px; padding:1.2rem 1.3rem; }}
.chip {{ display:inline-flex; gap:.35rem; align-items:baseline; font-size:.84rem; padding:.32rem .7rem; margin:0 .4rem .45rem 0;
  border-radius:999px; background:{PAPER}; border:1px solid {LINE}; }}
.chip q {{ color:{MUTED}; quotes:"“" "”"; }}
.chip b {{ font-weight:700; color:{INK}; }}
.note {{ font-size:.88rem; padding:.65rem .85rem; border-radius:12px; margin:.4rem 0; background:#FBF3E4; border:1px solid #EFD9B4; color:#5A4520; }}
.status {{ font-size:.95rem; padding:.75rem 1rem; border-radius:14px; margin:.4rem 0 1rem 0; background:{CARD}; border:1px solid {LINE}; }}
.status.warn {{ background:#FBF3E4; border-color:#EFD9B4; color:#5A4520; }}

.spec-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:.75rem; margin:.2rem 0 .6rem 0; }}
.spec {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px; padding:.85rem .95rem; }}
.spec .k {{ display:flex; align-items:center; gap:.4rem; font-size:.7rem; text-transform:uppercase; letter-spacing:.12em; color:{MUTED}; font-weight:700; }}
.spec .v {{ font-family:'Fraunces',serif; font-size:1.08rem; margin:.35rem 0 .15rem 0; color:{INK}; line-height:1.25; }}
.spec .s {{ font-size:.78rem; color:{MUTED}; }}

/* ---------- recommendations ---------- */
.rec1 {{ display:grid; grid-template-columns: minmax(0, 5fr) minmax(0, 7fr); gap: 1.8rem; background:{CARD};
  border:1px solid {LINE}; border-radius:26px; padding:1.6rem; box-shadow: 0 30px 60px -45px rgba(28,27,25,.5); }}
.rec1 .art {{ position:relative; border-radius:20px; display:flex; flex-direction:column; justify-content:center; padding:3rem 1.6rem 1.4rem 1.6rem; min-height: 300px; }}
.rec1 .art .dev {{ width: 92%; margin: auto; filter: drop-shadow(0 26px 22px rgba(28,27,25,.22)); }}
.rec1 .art .tag {{ position:absolute; top:1rem; left:1rem; font-size:.66rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase;
  background:rgba(255,253,249,.85); border:1px solid {LINE}; border-radius:999px; padding:.2rem .6rem; }}
.rec1 .cap {{ display:flex; align-items:flex-end; gap:.7rem; margin-top:1.2rem; padding-top:.9rem; border-top:1px solid rgba(28,27,25,.1);
  font-size:.78rem; color:#4A463F; line-height:1.35; }}
.rec1 .cap .big {{ font-family:'Fraunces',serif !important; font-size:2.1rem; line-height:1; color:{INK}; }}
.badge {{ display:inline-block; font-size:.7rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; padding:.3rem .65rem;
  border-radius:999px; background:{INK}; color:{CARD}; }}
.badge.soft {{ background:{PAPER}; color:{INK}; border:1px solid {LINE}; }}
.rec-name {{ font-family:'Fraunces',serif; font-weight:500; font-size: clamp(1.45rem, 2.4vw, 2rem); line-height:1.12; margin:.55rem 0 .2rem 0; color:{INK}; }}
.rec-meta {{ font-size:.8rem; color:{MUTED}; }}
.price-row {{ display:flex; align-items:baseline; flex-wrap:wrap; gap:.4rem .8rem; margin:.9rem 0 .2rem 0; }}
.price {{ font-family:'Fraunces',serif; font-size: clamp(1.7rem, 2.6vw, 2.2rem); font-weight:600; color:{INK}; }}
.under {{ font-size:.82rem; font-weight:700; color:#2E7D4F; }}
.over {{ font-size:.82rem; font-weight:700; color:#B4432F; }}
.src {{ font-size:.78rem; color:{MUTED}; }}
.fit {{ display:flex; align-items:center; gap:.7rem; margin:.9rem 0 .2rem 0; font-size:.84rem; color:{MUTED}; }}
.fit .bar {{ flex:1; height:6px; border-radius:999px; background:{LINE}; overflow:hidden; max-width: 240px; }}
.fit .bar i {{ display:block; height:100%; background:{ACCENT}; border-radius:999px; }}
.fit b {{ color:{INK}; }}
.specline {{ display:flex; flex-wrap:wrap; gap:.4rem; margin:.8rem 0; padding:0; list-style:none; }}
.stApp :is(.specline, .tile ul, .hero-list) {{ padding-left:0 !important; }}
.stApp :is(.specline, .tile ul, .hero-list) li {{ margin:0 !important; }}
.specline li {{ font-size:.8rem; padding:.28rem .6rem; border-radius:8px; background:{PAPER}; border:1px solid {LINE}; }}
.cols2 {{ display:grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin-top:.6rem; }}
.list-title {{ font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; font-weight:800; color:{MUTED}; margin-bottom:.35rem; }}
ul.pros, ul.cons {{ list-style:none; padding:0; margin:0; font-size:.88rem; line-height:1.45; }}
ul.pros li, ul.cons li {{ position:relative; padding-left:1.25rem; margin-bottom:.4rem; }}
ul.pros li::before {{ content:""; position:absolute; left:0; top:.42rem; width:.55rem; height:.55rem; border-radius:50%; background:#2E7D4F; }}
ul.cons li::before {{ content:""; position:absolute; left:.05rem; top:.62rem; width:.5rem; height:2px; background:#B4432F; }}
.cta {{ display:inline-flex; align-items:center; gap:.45rem; margin-top:1rem; padding:.7rem 1.15rem; border-radius:999px; background:{INK};
  color:{CARD} !important; font-weight:800; font-size:.9rem; text-decoration:none !important; transition: transform .15s ease, background .15s ease; }}
.cta:hover {{ background:{ACCENT}; transform: translateY(-1px); }}
.cta.ghost {{ background:transparent; color:{INK} !important; border:1px solid {INK}; }}
.cta.ghost:hover {{ background:{INK}; color:{CARD} !important; }}
.tradeoff {{ font-size:.82rem; color:{MUTED}; font-style:italic; margin-top:.6rem; }}

.rec2 {{ background:{CARD}; border:1px solid {LINE}; border-radius:22px; padding:1.25rem; height:100%; }}
.rec2 .head {{ display:grid; grid-template-columns: 34% minmax(0,1fr); gap:1rem; align-items:center; margin-bottom:.6rem; }}
.rec2 .art {{ border-radius:16px; padding:.9rem .6rem; align-self:start; aspect-ratio: 5 / 4; display:grid; place-items:center; }}
.rec2 .art svg {{ width:100%; }}
.rec2 .rec-name {{ font-size:1.18rem; margin-top:.45rem; }}
.rec2 .price-row {{ margin-top:.45rem; }}
.rec2 .price {{ font-size:1.45rem; }}
.rec2 .specline li {{ font-size:.75rem; padding:.2rem .5rem; }}
.rec2 ul.pros, .rec2 ul.cons {{ font-size:.84rem; }}

/* ---------- discovery board ---------- */
.board {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(232px, 1fr)); gap: 1.1rem; align-items:start; margin-top: .6rem; }}
.board p:empty, .tile > p:empty {{ display:none; }}
.tile {{ display:block; background:{CARD}; border:1px solid {LINE}; border-radius:20px; overflow:hidden; text-decoration:none !important;
  color:{INK} !important; transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; }}
.tile:hover, .tile:focus-visible {{ transform: translateY(-3px); box-shadow: 0 24px 40px -28px rgba(28,27,25,.55); border-color:#CFC6B7; outline:none; }}
.tile .art {{ position:relative; padding: 2.4rem 1.6rem .7rem 1.6rem; }}
.tile .art svg {{ max-width: 82%; margin: 0 auto; transition: transform .25s ease; }}
.tile:hover .art svg {{ transform: translateY(-3px) scale(1.02); }}
.tile .tag {{ position:absolute; top:.7rem; left:.7rem; font-size:.62rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase;
  background:rgba(255,253,249,.85); border:1px solid {LINE}; border-radius:999px; padding:.16rem .5rem; color:{INK}; }}
.tile .size {{ position:absolute; top:.72rem; right:.8rem; font-family:'Fraunces',serif !important; font-size:.82rem; color:#4A463F; }}
.tile .body {{ padding: .85rem 1rem 1rem 1rem; border-top:1px solid {LINE}; }}
.tile .brand {{ font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; font-weight:800; color:{MUTED}; }}
.tile .name {{ font-family:'Fraunces',serif; font-size:1.08rem; line-height:1.22; margin:.2rem 0 .1rem 0; }}
.tile .part {{ font-size:.7rem; color:{MUTED}; font-family: ui-monospace, 'SF Mono', Menlo, monospace; }}
.tile .tprice {{ margin:.5rem 0 .55rem 0; }}
.tile .tprice b {{ font-family:'Fraunces',serif; font-size:1.35rem; font-weight:600; }}
.tile ul {{ list-style:none; margin:0; padding:0; display:flex; flex-wrap:wrap; gap:.28rem; }}
.tile li {{ font-size:.72rem; padding:.16rem .45rem; border-radius:6px; background:{PAPER}; color:#3E3B36; }}
.tile .foot {{ display:flex; justify-content:space-between; align-items:center; gap:.4rem; margin-top:.75rem; padding-top:.6rem;
  border-top:1px dashed {LINE}; font-size:.72rem; color:{MUTED}; }}
.tile .go {{ font-weight:800; color:{ACCENT}; opacity:.0; transform: translateX(-4px); transition: all .18s ease; }}
.tile:hover .go, .tile:focus-visible .go {{ opacity:1; transform:none; }}
.story {{ grid-column: span 2; align-self: stretch; border-radius:20px; padding:1.7rem 1.8rem 0 1.8rem; color:{CARD}; min-height: 200px;
  display:flex; flex-direction:column; gap:.1rem; overflow:hidden;
  background: radial-gradient(520px 260px at 85% 100%, rgba(255,255,255,.18), transparent 65%), var(--bg, {INK}); }}
.story .sart {{ order:2; display:flex; align-items:flex-end; justify-content:center; gap:3%; margin-top:auto; padding-top:1.4rem; }}
.story .sdev {{ flex: 0 0 31%; filter: drop-shadow(0 18px 16px rgba(0,0,0,.32)); margin-bottom:-4px; }}
.story .sdev:nth-child(2) {{ flex-basis: 38%; margin-bottom: 0; }}
.story .sart {{ gap: 0; }}
.story .k {{ font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; font-weight:800; opacity:.75; }}
.story h3 {{ font-weight:500; font-size:1.6rem; line-height:1.12; margin:.35rem 0 .45rem 0; color:{CARD}; }}
.story p {{ margin:0; font-size:.9rem; line-height:1.5; opacity:.85; max-width: 30rem; }}
.empty {{ text-align:center; padding: 3rem 1rem; border:1px dashed {LINE}; border-radius:20px; color:{MUTED}; background:{CARD}; }}
.empty h3 {{ font-weight:500; color:{INK}; margin:0 0 .3rem 0; }}
.st-key-board_filters {{ background:{CARD}; border:1px solid {LINE}; border-radius:20px; padding: 1rem 1.2rem .4rem 1.2rem; }}
.count {{ font-size:.85rem; color:{MUTED}; margin:.6rem 0 0 0; }}

/* ---------- comparison ---------- */
.cmp-wrap {{ overflow-x:auto; margin:.6rem 0 1rem 0; background:{CARD}; border:1px solid {LINE}; border-radius:20px; }}
table.cmp {{ border-collapse:separate; border-spacing:0; width:100%; font-size:.87rem; min-width: 620px; }}
table.cmp th, table.cmp td {{ border-bottom:1px solid {LINE}; padding:.65rem .8rem; text-align:left; vertical-align:top; }}
table.cmp tr:last-child td {{ border-bottom:none; }}
table.cmp thead th {{ background:{CARD}; font-family:'Fraunces',serif; font-weight:500; font-size:1rem; line-height:1.2; }}
table.cmp thead .art {{ width: 150px; margin-bottom:.5rem; }}
table.cmp td.k {{ color:{MUTED}; white-space:nowrap; width:160px; font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; font-weight:700; }}
table.cmp td.best {{ font-weight:800; color:{INK}; }}
.cmp-hint {{ display:none; }}
.cmp-wrap:focus-visible {{ outline:2px solid {ACCENT}; outline-offset:2px; }}
table.cmp td.best::after {{ content:" ●"; color:{ACCENT}; font-size:.6rem; vertical-align:middle; }}

/* ---------- price lab ---------- */
.st-key-labtitle h1 {{ font-weight:500; font-size: clamp(1.8rem, 3vw, 2.5rem); margin-bottom:.2rem; }}
.historic {{ font-size:.9rem; padding:.85rem 1rem; border-radius:14px; margin:.2rem 0 1rem 0; background:{CARD}; border:1px solid {LINE}; line-height:1.55; }}
.price-card {{ border:1px solid {LINE}; border-radius:20px; padding:1.2rem 1.4rem; background:{CARD}; margin-bottom:.7rem; }}
.price-card .label {{ font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; font-weight:800; color:{MUTED}; margin:0; }}
.price-card .value {{ font-family:'Fraunces',serif; font-size:2.6rem; font-weight:600; line-height:1.1; margin:.3rem 0 .4rem 0; }}
.price-card .note {{ font-size:.85rem; color:{MUTED}; margin:0; background:none; border:none; padding:0; }}

.footer {{ margin-top:3.5rem; padding-top:1.2rem; border-top:1px solid {LINE}; display:flex; flex-wrap:wrap; justify-content:space-between; gap:.6rem; font-size:.8rem; color:{MUTED}; }}

/* ---------- responsive ---------- */
@media (max-width: 1000px) {{
  .rec1 {{ grid-template-columns: 1fr; }}
  .rec1 .art {{ min-height: 0; padding-top:2.6rem; }}
  .rec1 .art .dev {{ width: min(78%, 420px); }}
  .hero-art .stage {{ height: 210px; }}
}}
@media (min-width: 641px) and (max-width: 1000px) {{
  .st-key-hero [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
  .st-key-hero [data-testid="stColumn"] {{ flex: 1 1 100% !important; width: 100% !important; min-width: 100% !important; }}
  .hero-art {{ display:grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1.4rem; align-items:center; margin-top:.6rem; }}
  .hero-art .hero-list {{ margin-top:0; border-top:none; }}
  .hero-list b, .hero-list .meta {{ white-space:normal; line-height:1.3; }}
}}
@media (max-width: 760px) {{
  .block-container {{ padding: .6rem 1rem 4rem 1rem; }}
  .st-key-hero {{ padding: 1.5rem 1.2rem 1.4rem 1.2rem; border-radius: 22px; }}
  .values {{ grid-template-columns: 1fr; gap:.6rem; }}
  .cols2 {{ grid-template-columns: 1fr; }}
  .story {{ grid-column: 1 / -1; min-height: 0; padding: 1.3rem 1.3rem; }}
  .story h3 {{ font-size: 1.35rem; }}
  .rec1 {{ padding: 1rem; border-radius:22px; gap:1.1rem; }}
  .rec2 {{ padding: 1rem; }}
  .rec2 .head {{ grid-template-columns: 38% minmax(0,1fr); gap:.8rem; }}
  .hero-art.compact {{ display:none; }}
  .cmp-hint {{ display:flex; justify-content:space-between; align-items:center; margin:.6rem 0 -.2rem 0; font-size:.8rem; font-weight:700; color:{ACCENT}; }}
  .cmp-hint span {{ font-weight:600; color:{MUTED}; }}
  table.cmp {{ min-width:0; width:max-content; font-size:.82rem; }}
  table.cmp th:not(:first-child), table.cmp td:not(.k) {{ width:190px; min-width:190px; max-width:190px; }}
  table.cmp thead th {{ font-size:.92rem; }}
  table.cmp thead .art {{ width:120px; }}
  table.cmp td.k, table.cmp thead th:first-child {{ position:sticky; left:0; z-index:2; background:{CARD}; width:104px; min-width:104px; max-width:104px;
    white-space:normal; overflow-wrap:normal; word-break:normal; hyphens:none; text-transform:none; letter-spacing:0; font-size:.74rem;
    line-height:1.3; padding-left:.75rem; box-shadow: 1px 0 0 {LINE}; }}
  .hero-art {{ padding: 1rem 1rem .6rem 1rem; }}
  .hero-art .stage {{ height: 175px; }}
  .hero-art .d2 {{ bottom: 40px; }}
  .hero-list li {{ font-size:.76rem; gap:.55rem; padding:.45rem 0; }}
  .hero-list b {{ font-size:.82rem; }}
  .section {{ margin: 2rem 0 .7rem 0; }}
  .st-key-board_filters {{ padding: .8rem .9rem .3rem .9rem; }}
  .stTabs [role="tablist"] {{ gap: .9rem; }}
  .stTabs [role="tab"] p {{ font-size: .88rem; }}
  .board {{ grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap:.75rem; }}
  .tile .body {{ padding:.7rem .75rem .8rem .75rem; }}
  .tile .name {{ font-size:.98rem; }}
  .tile .go {{ opacity:1; transform:none; }}
  .topbar .stamp {{ display:none; }}
  .tile .art {{ padding: 2.1rem .9rem .5rem .9rem; }}
  .tile .art svg {{ max-width: 90%; }}
  .tile .foot {{ font-size:.66rem; }}
}}
@media (max-width: 400px) {{
  .board {{ grid-template-columns: 1fr 1fr; gap:.6rem; }}
  .tile .size {{ font-size:.72rem; right:.55rem; }}
  .tile .tag {{ left:.5rem; top:.55rem; }}
  .tile .go {{ display:none; }}
  .hero-art .stage {{ height: 150px; }}
  .hero-art .d2 {{ bottom: 34px; }}
  .hero-list b {{ white-space:normal; line-height:1.25; }}
  .tile .foot .w {{ display:none; }}
  .stTabs [role="tablist"] {{ gap: .62rem; }}
  .stTabs [role="tab"] p {{ font-size: .84rem; }}
}}
@media (prefers-reduced-motion: reduce) {{ .tile, .cta, .tile .go {{ transition:none; }} }}
</style>
"""

# ---------------------------------------------------------------------------------------------- formatting
def nice_date(iso):
    d = date.fromisoformat(iso)
    return f'{d.day} {d.strftime("%b %Y")}'


def short_date(iso):
    d = date.fromisoformat(iso)
    return f'{d.day} {d.strftime("%b")}'


def price_text(price):
    return 'Price not verified' if price is None else format_inr(round(price))


def weight_text(lap):
    return 'weight not stated' if lap.weight_kg is None else f'{lap.weight_kg:g} kg'


def battery_text(lap):
    return 'battery not stated' if lap.battery_wh is None else f'{lap.battery_wh:g} Wh battery'


def seller(lap):
    return f'{lap.brand} store' if lap.source_type == 'manufacturer' else 'Vijay Sales'


def _rank(value, sign=1):
    return float('-inf') if value is None else sign * value


def category(lap):
    """Display category for colour and collections (presentation only)."""
    m = lap.model.lower()
    if lap.os == 'macOS':
        return 'Mac'
    if 'zbook' in m:
        return 'Workstation'
    if lap.gpu_dedicated:
        return 'Gaming'
    if 'probook' in m or 'elitebook' in m:
        return 'Business'
    if lap.weight_kg and lap.weight_kg <= 1.45:
        return 'Ultralight'
    if lap.price_inr and lap.price_inr < 60_000:
        return 'Budget'
    return 'Everyday'


def short_model(lap):
    """Readable model name without brand prefix, part number or retail filler words."""
    m = lap.model
    m = re.sub(re.escape(lap.part_number), '', m, flags=re.I)
    m = re.sub(rf'^{re.escape(lap.brand)}\s+', '', m, flags=re.I)
    m = re.sub(r'\b(Thin\s*(?:and|&)\s*Light|Laptop|Notebook PC|Notebook|AI-Powered|PC|Next Gen AI|AI)\b', ' ', m, flags=re.I)
    m = re.sub(r'\s+inch\b', '"', m)
    m = re.sub(r'\s{2,}', ' ', m).strip(' -&')
    return m if len(m) >= 3 else f'Laptop {lap.part_number}'


def cpu_short(lap):
    c = re.sub(r'\(exact model not stated\)', '', lap.cpu)
    c = re.sub(r'\b(Intel|AMD|\d{2}th Gen|Apple)\b', '', c)
    c = re.sub(r'\s{2,}', ' ', c).strip()
    return c


def gpu_short(lap):
    g = lap.gpu
    m = re.search(r'RTX\s*(?:PRO\s*)?\d{3,4}(?:\s?Ti)?(?:\s*Ada|\s*Blackwell)?', g)
    if m:
        return m.group(0).replace('  ', ' ')
    if 'Arc' in g:
        return re.sub(r'.*?(Arc[^,(]*).*', r'\1', g).replace('GPU', '').replace('Graphics', '').strip() or 'Intel Arc'
    m = re.search(r'Radeon\s+\d{3}M', g)
    if m:
        return m.group(0)
    if lap.os == 'macOS':
        return re.sub(r'.*?(\d+-core GPU).*', r'M-series \1', g)
    return 'Integrated graphics'


# ---------------------------------------------------------------------------------------------- illustration
BRAND_SCREEN = {   # screen tint by brand (body colour and tag carry the category)
    'HP': ('#28477A', '#7EA3DA'), 'Lenovo': ('#33343F', '#8D90A6'), 'ASUS': ('#1E4A63', '#62A0C4'),
    'Dell': ('#1D5160', '#70ADB9'), 'Acer': ('#2C5A44', '#8BBB9E'), 'Apple': ('#3B3F4A', '#7C8496'),
}


# Silhouette per category: (bezel, base depth, base overhang, chin). Gaming bases are deep with rear vents; Macs and
# ultralights are thin with slim bezels; starter machines have thicker bezels and a chin.
SILHOUETTE = {'Mac': (4, 6, 10, 0), 'Ultralight': (5, 6, 11, 0), 'Business': (6, 8, 12, 0), 'Everyday': (6, 9, 13, 0),
              'Budget': (8, 9, 13, 7), 'Gaming': (6, 13, 15, 0), 'Workstation': (6, 11, 14, 0)}


def _wallpaper(lap, sx, sy, w, h, seed):
    """Brand-specific screen pattern (recognisable at a glance) with a per-laptop offset from `seed`."""
    k = (seed % 37) / 37
    if category(lap) in ('Gaming', 'Workstation'):                  # angular ridge + accent horizon
        p1 = f'{sx:.1f},{sy + h * .78:.1f} {sx + w * (.22 + k * .1):.1f},{sy + h * .42:.1f} {sx + w * .45:.1f},{sy + h * .66:.1f} {sx + w * (.68 + k * .08):.1f},{sy + h * .3:.1f} {sx + w:.1f},{sy + h * .62:.1f} {sx + w:.1f},{sy + h:.1f} {sx:.1f},{sy + h:.1f}'
        accent = '#7CF2D2' if category(lap) == 'Workstation' else ('#FF6B6B' if seed % 2 else '#7AA2FF')
        return (f'<polygon points="{p1}" fill="#000" opacity=".22"/>'
                f'<line x1="{sx:.1f}" y1="{sy + h * .78:.1f}" x2="{sx + w:.1f}" y2="{sy + h * .78:.1f}" stroke="{accent}" stroke-width="1.2" opacity=".75"/>'
                f'<circle cx="{sx + w * (.7 + k * .15):.1f}" cy="{sy + h * .24:.1f}" r="{h * .1:.1f}" fill="{accent}" opacity=".55"/>')
    if lap.brand == 'HP':                                           # concentric arcs from a corner
        cx, cy = sx + w * (.05 + k * .2), sy + h * 1.05
        return ''.join(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{h * r:.1f}" fill="none" stroke="#fff" stroke-width="{h * .07:.1f}" opacity="{o}"/>'
                       for r, o in ((.45, .16), (.8, .11), (1.15, .07)))
    if lap.brand == 'Lenovo':                                       # diagonal bands
        return ''.join(f'<polygon points="{sx + w * (a + k * .2):.1f},{sy:.1f} {sx + w * (a + .18 + k * .2):.1f},{sy:.1f} {sx + w * (a - .12 + k * .2):.1f},{sy + h:.1f} {sx + w * (a - .3 + k * .2):.1f},{sy + h:.1f}" fill="#fff" opacity="{o}"/>'
                       for a, o in ((.35, .12), (.65, .07), (.95, .1)))
    if lap.brand == 'ASUS':                                         # spun-metal rings
        cx, cy = sx + w * (.62 + k * .2), sy + h * .45
        return ''.join(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{h * r:.1f}" fill="none" stroke="#fff" stroke-width="1" opacity="{o}"/>'
                       for r, o in ((.12, .35), (.24, .25), (.36, .18), (.5, .12), (.66, .08))) + f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{h * .06:.1f}" fill="#fff" opacity=".35"/>'
    if lap.brand == 'Dell':                                         # horizontal waves
        out = []
        for j, o in enumerate((.16, .11, .07)):
            y = sy + h * (.45 + j * .17 + k * .05)
            out.append(f'<path d="M{sx:.1f},{y:.1f} C{sx + w * .3:.1f},{y - h * .18:.1f} {sx + w * .6:.1f},{y + h * .18:.1f} {sx + w:.1f},{y - h * .04:.1f} L{sx + w:.1f},{sy + h:.1f} L{sx:.1f},{sy + h:.1f}Z" fill="#fff" opacity="{o}"/>')
        return ''.join(out)
    if lap.brand == 'Apple':                                        # soft flowing ribbon
        return (f'<path d="M{sx:.1f},{sy + h * .7:.1f} C{sx + w * .35:.1f},{sy + h * (.2 + k * .2):.1f} {sx + w * .6:.1f},{sy + h * 1.0:.1f} {sx + w:.1f},{sy + h * .35:.1f} L{sx + w:.1f},{sy + h:.1f} L{sx:.1f},{sy + h:.1f}Z" fill="#fff" opacity=".14"/>'
                f'<circle cx="{sx + w * (.3 + k * .3):.1f}" cy="{sy + h * .3:.1f}" r="{h * .28:.1f}" fill="#fff" opacity=".08"/>')
    cx1, cy1 = sx + w * (.55 + k * .3), sy + h * (.2 + k * .2)                 # Acer and others: soft blobs
    return (f'<circle cx="{cx1:.1f}" cy="{cy1:.1f}" r="{h * .6:.1f}" fill="#fff" opacity=".13"/>'
            f'<circle cx="{sx + w * (.15 + k * .2):.1f}" cy="{sy + h * .85:.1f}" r="{h * .38:.1f}" fill="#fff" opacity=".08"/>')


def device_svg(lap, uid=''):
    """Original illustration of the laptop: lid sized by its real screen size, screen by its real aspect ratio,
    silhouette by category (deep vented base for gaming, slim bezels and notch for Mac, chin for starter) and
    wallpaper pattern by brand."""
    cat = category(lap)
    body, s1, s2, _ = CATEGORY_STYLE[cat]
    if cat not in ('Gaming', 'Workstation'):                        # gaming/workstation keep their darker screens
        s1, s2 = BRAND_SCREEN.get(lap.brand, (s1, s2))
    bezel, depth, over, chin = SILHOUETTE[cat]
    gid = re.sub(r'[^a-z0-9]', '', (lap.id + uid).lower())
    scale = 0.80 + (min(max(lap.display_in, 13.0), 16.2) - 13.0) / 3.2 * 0.20
    lid_w = 210 * scale
    scr_w = lid_w - 2 * bezel
    scr_h = scr_w * lap.resolution_h / lap.resolution_w
    lid_h = scr_h + 2 * bezel + chin
    x0, y0 = (240 - lid_w) / 2, 8
    sx, sy = x0 + bezel, y0 + bezel
    yb = y0 + lid_h
    seed = sum(ord(ch) * (i + 1) for i, ch in enumerate(lap.id))
    base = (f'{x0 - over:.1f},{yb:.1f} {x0 + lid_w + over:.1f},{yb:.1f} {x0 + lid_w + over - 6:.1f},{yb + depth:.1f} '
            f'{x0 - over + 6:.1f},{yb + depth:.1f}')
    extras = []
    if cat == 'Mac':
        extras.append(f'<rect x="{120 - scr_w * .07:.1f}" y="{y0:.1f}" width="{scr_w * .14:.1f}" height="{bezel + 3.2:.1f}" rx="2.4" fill="{body}"/>')
    if cat in ('Gaming', 'Workstation'):
        extras += [f'<line x1="{x0 + lid_w * (.16 + i * .045):.1f}" y1="{yb + depth * .3:.1f}" x2="{x0 + lid_w * (.16 + i * .045):.1f}" y2="{yb + depth * .78:.1f}" '
                   f'stroke="#000" stroke-opacity=".35" stroke-width="1.4"/>' for i in range(4)]
        extras += [f'<line x1="{x0 + lid_w * (.84 - i * .045):.1f}" y1="{yb + depth * .3:.1f}" x2="{x0 + lid_w * (.84 - i * .045):.1f}" y2="{yb + depth * .78:.1f}" '
                   f'stroke="#000" stroke-opacity=".35" stroke-width="1.4"/>' for i in range(4)]
    if lap.gpu_dedicated:
        extras.append(f'<rect x="{x0 + lid_w * .34:.1f}" y="{yb + depth * .42:.1f}" width="{lid_w * .32:.1f}" height="1.8" rx=".9" fill="#6E8BFF" opacity=".95"/>')
    if chin:
        extras.append(f'<rect x="{120 - 9:.1f}" y="{sy + scr_h + chin * .45:.1f}" width="18" height="2" rx="1" fill="#000" opacity=".18"/>')
    if cat == 'Business':
        extras += [f'<line x1="{sx:.1f}" y1="{sy + scr_h * f:.1f}" x2="{sx + scr_w:.1f}" y2="{sy + scr_h * f:.1f}" stroke="#fff" stroke-opacity=".07"/>'
                   for f in (.25, .5, .75)]
    label = escape(f'Illustration of {lap.display_name}, {lap.display_in:g}-inch screen')
    return f"""<svg viewBox="0 0 240 {yb + depth + 12:.0f}" role="img" aria-label="{label}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block">
<defs><linearGradient id="s{gid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{s1}"/><stop offset="1" stop-color="{s2}"/></linearGradient>
<linearGradient id="b{gid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{body}"/><stop offset="1" stop-color="{body}" stop-opacity=".82"/></linearGradient>
<clipPath id="c{gid}"><rect x="{sx:.1f}" y="{sy:.1f}" width="{scr_w:.1f}" height="{scr_h:.1f}" rx="2.5"/></clipPath></defs>
<ellipse cx="120" cy="{yb + depth + 5:.1f}" rx="{lid_w * .56:.1f}" ry="4" fill="#1C1B19" opacity=".10"/>
<rect x="{x0:.1f}" y="{y0}" width="{lid_w:.1f}" height="{lid_h:.1f}" rx="{7 if bezel > 4 else 6}" fill="url(#b{gid})"/>
<rect x="{sx:.1f}" y="{sy:.1f}" width="{scr_w:.1f}" height="{scr_h:.1f}" rx="2.5" fill="url(#s{gid})"/>
<g clip-path="url(#c{gid})">{_wallpaper(lap, sx, sy, scr_w, scr_h, seed)}
<polygon points="{sx:.1f},{sy:.1f} {sx + scr_w * .42:.1f},{sy:.1f} {sx + scr_w * .18:.1f},{sy + scr_h:.1f} {sx:.1f},{sy + scr_h:.1f}" fill="#fff" opacity=".07"/></g>
<polygon points="{base}" fill="url(#b{gid})"/>
<rect x="{x0 - over:.1f}" y="{yb:.1f}" width="{lid_w + 2 * over:.1f}" height="1.4" fill="#fff" opacity=".25"/>
<rect x="{120 - lid_w * .08:.1f}" y="{yb:.1f}" width="{lid_w * .16:.1f}" height="2.4" rx="1.2" fill="#000" opacity=".18"/>
{''.join(extras)}
</svg>"""


# ---------------------------------------------------------------------------------------------- chrome
def topbar(verified_on):
    mark = ('<span class="mark"><svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><rect x="2" y="3" width="12" height="8" rx="1.5" '
            'fill="none" stroke="#F6F3EE" stroke-width="1.6"/><path d="M1 13h14" stroke="#F6F3EE" stroke-width="1.6" stroke-linecap="round"/></svg></span>')
    return (f'<div class="topbar"><div class="wordmark">{mark}<span>Laptop <em>Advisor</em></span></div>'
            f'<div class="stamp">Prices checked {escape(nice_date(verified_on))}</div></div>')


def hero_copy(count):
    return f"""
<div class="eyebrow">Laptop advice for India · {count} configurations, hand-checked</div>
<div class="hero-title">Your next laptop<br><em>starts with you.</em></div>
<p class="hero-sub">Describe how you study, work and play. We'll turn it into the specs that matter and a shortlist of
three laptops worth a closer look, with prices checked at Indian stores.</p>"""


def hero_art(laps, compact=False):
    """Three real catalogue laptops as an editorial still life, captioned like a magazine spread (numbered pins).
    `compact` (a search is showing) hides the still life on phones so results come sooner."""
    def item(n, lap, line):
        name = re.sub(r'\s*\(.*?\)', '', short_model(lap))
        return (f'<li><span class="n">{n}</span><div><b>{escape(lap.brand)} {escape(name)}</b><span class="meta">{escape(line)}</span></div>'
                f'<span class="p">{escape(price_text(lap.price_inr))}<small>{escape(seller(lap))} · {escape(short_date(lap.verified_on))}</small></span></li>')
    a, b, c = laps
    lines = (f'{weight_text(a)} · {gpu_short(a)}', f'{gpu_short(b)} · {b.ram_gb} GB RAM', f'{c.ram_gb} GB · {storage_text(c.storage_gb)} SSD')
    devs = ''.join(f'<div class="dev d{n}">{device_svg(lap, f"h{n}")}<span class="pin">{n}</span></div>' for n, lap in enumerate(laps, start=1))
    items = ''.join(item(n, lap, line) for n, (lap, line) in enumerate(zip(laps, lines), start=1))
    return (f'<div class="hero-art{" compact" if compact else ""}"><div class="stage" aria-hidden="true">{devs}</div>'
            f'<ol class="hero-list" aria-label="From the catalogue">{items}</ol></div>')


VALUES_HTML = """<div class="values">
<div class="value"><h4>Write it like you'd say it</h4><p>"Mechanical student, SolidWorks, under 1.2 lakh" is enough. We pick out the course, software, games and budget.</p></div>
<div class="value"><h4>Prices checked at the source</h4><p>Every laptop is an exact configuration with its part number, price and the date we checked it, from a brand store or an authorised retailer.</p></div>
<div class="value"><h4>Trade-offs, out loud</h4><p>Each pick says why it fits and what it gives up, whether that's weight, graphics, storage or your budget.</p></div>
</div>"""


def section(kicker, title, text=''):
    return (f'<div class="section"><div class="eyebrow">{escape(kicker)}</div><h2>{escape(title)}</h2>'
            + (f'<p>{escape(text)}</p>' if text else '') + '</div>')


def footer(count, verified_on):
    return (f'<div class="footer"><span>Laptop Advisor · {count} verified configurations · prices checked {escape(nice_date(verified_on))}</span>'
            '<span>Prices and stock change; always confirm on the seller page. Applied AI &amp; Data Science project.</span></div>')


# ---------------------------------------------------------------------------------------------- brief
def chips(evidence):
    return ''.join(f'<span class="chip"><q>{escape(p)}</q>→ <b>{escape(m)}</b></span>' for p, m in evidence)


def note(text):
    return f'<div class="note">{escape(text)}</div>'


def status_box(text, warn=False):
    return f'<div class="status{" warn" if warn else ""}">{escape(text)}</div>'


ICONS = {
    'Memory (RAM)': '<path d="M2 5h12v6H2z M4 11v2 M7 11v2 M10 11v2 M12 11v2" />',
    'Processor': '<path d="M4 4h8v8H4z M6 1v3 M10 1v3 M6 12v3 M10 12v3 M1 6h3 M1 10h3 M12 6h3 M12 10h3" />',
    'Graphics': '<path d="M1 4h14v8H1z M4 8h.01 M8 6a2 2 0 1 0 0 4a2 2 0 1 0 0-4" />',
    'Storage': '<path d="M2 3h12v10H2z M2 9h12 M11 11h.01" />',
    'Display': '<path d="M2 3h12v8H2z M6 14h4 M8 11v3" />',
    'Operating system': '<path d="M2 2h5v5H2z M9 2h5v5H9z M2 9h5v5H2z M9 9h5v5H9z" />',
}


def spec_grid(rows):
    cells = []
    for k, v, s in rows:
        icon = ICONS.get(k, '')
        svg = (f'<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" '
               f'stroke-linecap="round" aria-hidden="true">{icon}</svg>') if icon else ''
        cells.append(f'<div class="spec"><div class="k">{svg}{escape(k)}</div><div class="v">{escape(v)}</div><div class="s">{escape(s)}</div></div>')
    return f'<div class="spec-grid">{"".join(cells)}</div>'


# ---------------------------------------------------------------------------------------------- recommendations
def fit_label(score):
    return 'Strong fit' if score >= .75 else 'Good fit' if score >= .6 else 'Partial fit'


def _budget_line(rec):
    if rec.budget_gap is None:
        return ''
    if rec.budget_gap >= 0:
        return f'<span class="under">{price_text(rec.budget_gap)} under budget</span>'
    return f'<span class="over">{price_text(-rec.budget_gap)} over budget</span>'


def _source_line(lap):
    stock = ' · in stock when checked' if lap.availability == 'in_stock' else ''
    return f'<div class="src">Listed on {escape(lap.price_source)} · verified {escape(nice_date(lap.verified_on))}{stock}</div>'


def _lists(rec):
    pros = ''.join(f'<li>{escape(p)}</li>' for p in rec.pros)
    cons_items = [f'Below minimum: {g}' for g in rec.gaps] + rec.cons
    cons = ''.join(f'<li>{escape(c)}</li>' for c in cons_items)
    left = f'<div><div class="list-title">Why it fits</div><ul class="pros">{pros}</ul></div>' if pros else ''
    right = f'<div><div class="list-title">What you give up</div><ul class="cons">{cons}</ul></div>' if cons else ''
    return f'<div class="cols2">{left}{right}</div>' if (left or right) else ''


def _specline(lap):
    items = [cpu_short(lap), gpu_short(lap), f'{lap.ram_gb} GB RAM', lap.storage_label,
             f'{lap.display_in:g}" {lap.resolution_w}×{lap.resolution_h}' + (f' · {lap.refresh_hz} Hz' if lap.refresh_hz else ''),
             weight_text(lap), battery_text(lap), lap.os]
    return '<ul class="specline">' + ''.join(f'<li>{escape(i)}</li>' for i in items) + '</ul>'


def _fit(rec):
    pct = round(rec.score * 100)
    return (f'<div class="fit"><b>{fit_label(rec.score)}</b><div class="bar"><i style="width:{pct}%"></i></div>'
            f'<span>{pct}/100</span></div>')


def _art_caption(lap):
    hz = f' · {lap.refresh_hz} Hz' if lap.refresh_hz else ''
    return (f'<div class="cap"><span class="big">{lap.display_in:g}″</span>'
            f'<span>{lap.resolution_w}×{lap.resolution_h}{hz}<br>{escape(weight_text(lap))}</span></div>')


def rec_primary(rec):
    lap = rec.laptop
    cat = CATEGORY_STYLE[category(lap)][3]
    return f"""<div class="rec1 lap">
<div class="art" style="{art_bg(lap)}"><span class="tag">{escape(cat)}</span><div class="dev">{device_svg(lap, 'r1')}</div>{_art_caption(lap)}</div>
<div>
  <span class="badge">Best match</span>
  <div class="rec-name">{escape(lap.display_name)}</div>
  <div class="rec-meta">Part no. {escape(lap.part_number)}</div>
  <div class="price-row"><span class="price">{price_text(lap.price_inr)}</span>{_budget_line(rec)}</div>
  {_source_line(lap)}
  {_fit(rec)}
  {_specline(lap)}
  {_lists(rec)}
  <a class="cta" href="{escape(lap.url)}" target="_blank" rel="noopener">View at {escape(seller(lap))} ↗</a>
</div></div>"""


def rec_secondary(rec, rank):
    lap = rec.laptop
    trade = f'<div class="tradeoff">{escape(rec.trade_off)}</div>' if rec.trade_off else ''
    return f"""<div class="rec2 lap">
<div class="head">
<div class="art" style="{art_bg(lap)}">{device_svg(lap, f'r{rank}')}</div>
<div><span class="badge soft">Option {rank}</span>
<div class="rec-name">{escape(lap.display_name)}</div>
<div class="rec-meta">Part no. {escape(lap.part_number)}</div>
<div class="price-row"><span class="price">{price_text(lap.price_inr)}</span>{_budget_line(rec)}</div></div>
</div>
{_source_line(lap)}
{_fit(rec)}
{_specline(lap)}
{_lists(rec)}
{trade}
<a class="cta ghost" href="{escape(lap.url)}" target="_blank" rel="noopener">View at {escape(seller(lap))} ↗</a>
</div>"""


def score_bar(label, value, right=None):
    pct = max(0, min(100, round(value * 100)))
    return (f'<div class="fit"><span style="min-width:12rem">{escape(label)}</span><div class="bar"><i style="width:{pct}%"></i></div>'
            f'<span>{escape(right or f"{pct}/100")}</span></div>')


def breakdown_html(rec):
    return ''.join(score_bar(f"{c['label']} · weight {round(c['weight'] * 100)}%", c['score']) for c in rec.breakdown)


# ---------------------------------------------------------------------------------------------- discovery board
def tile(lap):
    cat = CATEGORY_STYLE[category(lap)][3]
    chips_ = [cpu_short(lap), f'{lap.ram_gb} GB · {storage_text(lap.storage_gb)}', gpu_short(lap)]
    if lap.weight_kg is not None:
        chips_.append(weight_text(lap))
    screen = f'{lap.display_in:g}″' + (f' · {lap.refresh_hz} Hz' if lap.refresh_hz and lap.refresh_hz > 60 else '')
    label = escape(f'{lap.display_name}, {price_text(lap.price_inr)}, opens {seller(lap)} in a new tab')
    return f"""<a class="tile" href="{escape(lap.url)}" target="_blank" rel="noopener" aria-label="{label}">
<div class="art" style="{art_bg(lap)}"><span class="tag">{escape(cat)}</span><span class="size">{escape(screen)}</span>{device_svg(lap, 't')}</div>
<div class="body">
  <div class="brand">{escape(lap.brand)}</div>
  <div class="name">{escape(short_model(lap))}</div>
  <div class="part">{escape(lap.part_number)}</div>
  <div class="tprice"><b>{price_text(lap.price_inr)}</b></div>
  <ul>{''.join(f'<li>{escape(c)}</li>' for c in chips_)}</ul>
  <div class="foot"><span>{escape(seller(lap))} · <span class="w">checked </span>{escape(short_date(lap.verified_on))}</span><span class="go">View ↗</span></div>
</div></a>"""


def story(kicker, title, text, bg, laps=()):
    uid = 's' + re.sub(r'[^a-z]', '', kicker.lower())
    art = ''.join(f'<div class="sdev">{device_svg(l, uid)}</div>' for l in list(laps)[:3])
    return (f'<div class="story" style="--bg:{bg}">' + (f'<div class="sart" aria-hidden="true">{art}</div>' if art else '')
            + f'<div class="k">{escape(kicker)}</div><h3>{escape(title)}</h3><p>{escape(text)}</p></div>')


def board(laptops, stories=None):
    """Grid of product tiles; `stories` = {position: html} editorial cards inserted between tiles."""
    stories = stories or {}
    cells = []
    for i, lap in enumerate(laptops):
        if i in stories:
            cells.append(stories[i])
        cells.append(tile(lap))
    return f'<div class="board">{"".join(cells)}</div>'


def empty_state(title, text):
    return f'<div class="empty"><h3 class="serif">{escape(title)}</h3><p>{escape(text)}</p></div>'


# ---------------------------------------------------------------------------------------------- comparison
def comparison_table(laptops, scores=None, uid='c'):
    """Side-by-side table with illustrations. `scores` maps laptop id -> 0-1 suitability. Best value per row marked."""
    cols = list(laptops)
    rows = [
        ('Price', [price_text(l.price_inr) for l in cols], [_rank(l.price_inr, -1) for l in cols]),
        ('Price source', [f'{l.price_source}, verified {nice_date(l.verified_on)}' for l in cols], None),
        ('Processor', [l.cpu for l in cols], None),
        ('Processor class', [CPU_CLASS[l.cpu_tier].split(' (')[0] for l in cols], [l.cpu_tier for l in cols]),
        ('Graphics', [l.gpu for l in cols], None),
        ('Graphics class', [gpu_class_label(l) for l in cols], [l.gpu_tier for l in cols]),
        ('NVIDIA CUDA', ['Yes' if l.cuda else 'No' for l in cols], [int(l.cuda) for l in cols]),
        ('RAM', [f'{l.ram_gb} GB' for l in cols], [l.ram_gb for l in cols]),
        ('Storage', [l.storage_label for l in cols], [l.storage_gb for l in cols]),
        ('Display', [f'{l.display_in:g}" {l.resolution_w}×{l.resolution_h}' for l in cols], [l.resolution_w * l.resolution_h for l in cols]),
        ('Refresh rate', [f'{l.refresh_hz} Hz' if l.refresh_hz else 'Not stated' for l in cols], [l.refresh_hz or 0 for l in cols]),
        ('Weight', [weight_text(l).replace('weight not stated', 'Not stated') for l in cols], [_rank(l.weight_kg, -1) for l in cols]),
        ('Battery', [f'{l.battery_wh:g} Wh' if l.battery_wh else 'Not stated' for l in cols], [_rank(l.battery_wh) for l in cols]),
        ('Operating system', [l.os for l in cols], None),
    ]
    if scores:
        rows.insert(1, ('Meets your recommended specs', [f'{round(scores[l.id] * 100)}/100' if l.id in scores else '—' for l in cols],
                        [scores.get(l.id, -1) for l in cols]))
    head = ''.join(f'<th><div class="art">{device_svg(l, uid)}</div>{escape(l.display_name)}</th>' for l in cols)   # unique ids per table
    body = []
    for label, values, rank in rows:
        best = max(rank) if rank and len(set(rank)) > 1 else None
        tds = ''.join(f'<td class="{"best" if best is not None and r == best else ""}">{escape(v)}</td>'
                      for v, r in zip(values, rank or [None] * len(values)))
        body.append(f'<tr><td class="k">{escape(label)}</td>{tds}</tr>')
    links = ''.join(f'<td><a href="{escape(l.url)}" target="_blank" rel="noopener">'
                    f'{"Official store page" if l.source_type == "manufacturer" else "Retailer page (Vijay Sales)"} ↗</a></td>' for l in cols)
    body.append(f'<tr><td class="k">Buy / verify</td>{links}</tr>')
    hint = f'<div class="cmp-hint" aria-hidden="true">Swipe to compare other laptops →<span>{len(cols)} laptops</span></div>'
    return (f'{hint}<div class="cmp-wrap" role="region" tabindex="0" aria-label="Comparison of {len(cols)} laptops; scroll sideways to see every column">'
            f'<table class="cmp"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
