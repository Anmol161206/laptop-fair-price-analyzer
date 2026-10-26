"""Find My Laptop: rule-based requirement extraction, spec recommendation and explainable ranking.

Pipeline (all deterministic, no external services):
    free text --extract_requirements()--> Requirements (editable in the UI)
    Requirements --spec_profile()--> SpecProfile (minimum + recommended hardware, with reasons)
    SpecProfile + catalogue --recommend()--> ranked laptops with a score breakdown, pros, cons and trade-offs

The historical XGBoost price model is NOT used here: it was trained on 2016-18 European prices and cannot
represent modern hardware or current Indian prices.
"""
import re
from dataclasses import dataclass, field

from src.catalogue import load_catalogue

# --------------------------------------------------------------------------------------------------
# Workload profiles (minimum / recommended hardware). Tiers follow catalogue/laptops.json "tier_rules".
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Workload:
    key: str
    label: str
    min_ram: int
    rec_ram: int
    min_storage: int
    rec_storage: int
    min_cpu: int
    rec_cpu: int
    min_gpu: int
    rec_gpu: int
    reason: str
    prefer_cuda: bool = False
    prefer_refresh: int | None = None
    requires_os: str | None = None


WORKLOADS = {w.key: w for w in [
    Workload('everyday', 'Everyday use & study', 8, 16, 256, 512, 1, 2, 1, 1,
             'Notes, browsing, video calls and office apps run well on entry hardware; 16 GB keeps many tabs smooth.'),
    Workload('office', 'Business & office work', 8, 16, 256, 512, 1, 2, 1, 1,
             'Excel, PowerPoint and video meetings need little graphics power; RAM matters most for multitasking.'),
    Workload('programming', 'Programming & software development', 16, 16, 512, 1024, 2, 3, 1, 1,
             'IDEs, browsers, local servers and emulators use a lot of memory, so 16 GB is the practical minimum.'),
    Workload('data_science', 'Data science & machine learning', 16, 32, 512, 1024, 3, 4, 1, 3,
             'Large datasets need RAM, and GPU training in PyTorch/TensorFlow is most widely supported on NVIDIA CUDA GPUs.',
             prefer_cuda=True),
    Workload('engineering', 'Engineering & CAD', 16, 32, 512, 1024, 3, 4, 2, 3,
             'CAD and simulation tools (AutoCAD, SolidWorks, MATLAB, ANSYS) benefit from a fast processor and capable graphics.'),
    Workload('design', 'Graphic & UI design', 16, 16, 512, 1024, 2, 3, 1, 2,
             'Photoshop, Illustrator and Figma need RAM for large files; a sharp display helps.'),
    Workload('video', 'Video editing', 16, 32, 512, 1024, 3, 4, 2, 3,
             'Editing and exporting video (Premiere Pro, DaVinci Resolve, Final Cut Pro) is processor- and GPU-heavy, and footage fills storage fast.'),
    Workload('3d', '3D & game development', 16, 32, 512, 1024, 3, 4, 3, 4,
             'Blender, Unity and Unreal Engine rely heavily on the GPU for viewport and rendering performance.'),
    Workload('gaming_esports', 'Esports & casual gaming', 16, 16, 512, 1024, 2, 3, 3, 3,
             'Titles like Valorant, CS2 and Minecraft need a dedicated GPU and a high-refresh display; most PC games target Windows.',
             prefer_refresh=144, requires_os='Windows'),
    Workload('gaming_aaa', 'AAA gaming', 16, 16, 512, 1024, 3, 4, 3, 4,
             'Recent AAA games are GPU-bound and large to install; most PC games and anti-cheat systems target Windows.',
             prefer_refresh=144, requires_os='Windows'),
]}

# Keyword evidence: (regex, workload key). Order matters only for the evidence list shown to the user.
WORKLOAD_PATTERNS = [
    (r'\b(computer science|cse|\bcs\b|information technology|\bit\b student|software engineer\w*|developer|coding|programm\w+|python|java\b|javascript|c\+\+|web dev\w*|app dev\w*|vs ?code|android studio|leetcode|dsa)\b', 'programming'),
    (r'\b(data scien\w+|data analy\w+|machine learning|\bml\b|deep learning|\bai\b|artificial intelligence|tensorflow|pytorch|cuda|neural network\w*|kaggle|llm\w*)\b', 'data_science'),
    (r'\b(mechanical|civil|electrical|aerospace|automobile|architecture|architect|autocad|solidworks|catia|ansys|matlab|simulink|revit|fusion ?360|creo|cad\b|simulation\w*)\b', 'engineering'),
    (r'\b(graphic design\w*|designer|ui/?ux|ux|photoshop|illustrator|figma|lightroom|canva|indesign)\b', 'design'),
    (r'\b(video edit\w*|editing videos|premiere|after effects|davinci|resolve|final cut|youtube\w*|content creat\w+|vlog\w*)\b', 'video'),
    (r'\b(blender|unity|unreal|3d model\w*|3d render\w*|maya|3ds ?max|game dev\w*|animation)\b', '3d'),
    (r'\b(valorant|cs ?2|counter[- ]strike|fortnite|minecraft|league of legends|dota|apex|esports?|casual gaming|light gaming|some gaming|occasional gaming)\b', 'gaming_esports'),
    (r'\b(gta|cyberpunk|elden ring|call of duty|cod\b|red dead|aaa|hardcore gaming|heavy gaming|high[- ]end gaming|latest games|ray tracing)\b', 'gaming_aaa'),
    (r'\b(mba|bba|b\.?com|commerce|accountan\w+|\bca\b|finance|business|excel|powerpoint|ms office|office work|marketing|hr)\b', 'office'),
    (r'\b(student|college|school|notes|browsing|netflix|online classes|assignments)\b', 'everyday'),
]
GENERIC_GAMING = r'\b(gaming|games?|gamer)\b'

# Software that only runs on one OS (hard requirement).
OS_REQUIRED_SOFTWARE = [
    (r'\b(solidworks|catia|ansys)\b', 'Windows', 'is not available for macOS'),
    (r'\b(final cut|logic pro|xcode|ios dev\w*|iphone app\w*)\b', 'macOS', 'only runs on macOS'),
]

LEVELS = ('Low', 'Medium', 'High')
OS_CHOICES = ('Any', 'Windows', 'macOS')
SCREEN_CHOICES = ('Any', 'Compact (14" or smaller)', 'Large (15.3" or bigger)')


@dataclass
class Requirements:
    workloads: list = field(default_factory=lambda: ['everyday'])
    budget_max: float | None = None
    os_preference: str = 'Any'
    portability: str = 'Medium'
    battery: str = 'Medium'
    screen: str = 'Any'
    required_os: str | None = None          # from software that only runs on one OS
    evidence: list = field(default_factory=list)   # (phrase found, what it means)
    notes: list = field(default_factory=list)      # ambiguity / assumption messages


# --------------------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------------------
_UNITS = {'k': 1_000, 'thousand': 1_000, 'lakh': 100_000, 'lakhs': 100_000, 'lac': 100_000, 'lacs': 100_000, 'l': 100_000}
_NOT_MONEY_AFTER = re.compile(r'^\s*(display|video|monitor|screen|resolution|gaming|footage|hz|fps|gb|tb|mah|wh|nits)\b')


def parse_budget(text):
    """Return the maximum budget in rupees mentioned in `text`, or None.

    Understands '₹70,000', 'Rs 70000', '70k', '1.2 lakh', '60-80k', 'between 70k and 90k', 'under 1 lakh'.
    Numbers followed by words like 'display' or 'video' (e.g. '4k display') are ignored.
    """
    t = text.lower().replace(',', '')
    amounts = []
    # Comma-grouped numbers (Indian '1,00,000' or '80,000') are treated as amounts even without a currency sign.
    for m in re.finditer(r'(?<![\d,])(\d{1,2},\d{2},\d{3}|\d{2,3},\d{3})(?![\d,])', text):
        amounts.append(float(m.group(1).replace(',', '')))
    unit = r'(k|thousand|lakhs?|lacs?|l)\b'
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(k|thousand|lakhs?|lacs?|l)?\s*(?:-|–|to|and)\s*(\d+(?:\.\d+)?)\s*' + unit, t):
        if _NOT_MONEY_AFTER.match(t[m.end():]):
            continue
        hi = float(m.group(3)) * _UNITS[m.group(4)]
        amounts.append(hi)
    for m in re.finditer(r'(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*' + unit, t):
        if not _NOT_MONEY_AFTER.match(t[m.end():]):
            amounts.append(float(m.group(1)) * _UNITS[m.group(2)])
    for m in re.finditer(r'(?:₹|rs\.?\s*|inr\s*)(\d{4,7}(?:\.\d+)?)\b', t):
        amounts.append(float(m.group(1)))
    for m in re.finditer(r'\b(?:budget|under|below|within|upto|up to|max(?:imum)?|less than|around|about)\D{0,12}?(\d{5,7})\b', t):
        amounts.append(float(m.group(1)))
    plausible = [a for a in amounts if 15_000 <= a <= 1_000_000]
    if not plausible:
        return None
    best = max(plausible)
    if re.search(r'\b(around|about|approx\w*|roughly)\b', t):
        best *= 1.10        # "around 80k" -> allow up to ~88k
    return round(best)


def extract_requirements(text):
    """Turn a free-text description into editable Requirements, recording evidence and assumptions."""
    t = (text or '').lower()
    req = Requirements(workloads=[])
    for pattern, key in WORKLOAD_PATTERNS:
        for m in re.finditer(pattern, t):
            if key not in req.workloads:
                req.workloads.append(key)
            req.evidence.append((m.group(0).strip(), WORKLOADS[key].label))
            break
    if re.search(GENERIC_GAMING, t) and not ({'gaming_esports', 'gaming_aaa'} & set(req.workloads)):
        req.workloads.append('gaming_esports')
        req.evidence.append(('gaming', WORKLOADS['gaming_esports'].label + ' (no specific games named)'))
    if not req.workloads:
        req.workloads = ['everyday']
        req.notes.append('No specific course, job or software was recognised, so general study/everyday use is assumed. '
                         'Edit the workloads below if this is wrong.')
    if 'everyday' in req.workloads and len(req.workloads) > 1:
        req.workloads.remove('everyday')     # a specific workload already covers everyday needs
        req.evidence = [e for e in req.evidence if e[1] != WORKLOADS['everyday'].label]

    req.budget_max = parse_budget(t)
    if req.budget_max is None:
        req.notes.append('No budget was mentioned, so laptops at every price are considered. Add a budget to narrow the results.')

    for pattern, os_name, why in OS_REQUIRED_SOFTWARE:
        m = re.search(pattern, t)
        if m:
            req.required_os = os_name
            req.evidence.append((m.group(0), f'{os_name} required ({m.group(0)} {why})'))
    if re.search(r"\b(no|not|don'?t want|avoid|hate)\s+(a\s+)?(mac|macs|macbook\w*|macos|apple)\b", t):
        req.os_preference = 'Windows'
        req.evidence.append(('no Mac', 'Windows preferred'))
    elif re.search(r'\b(mac|macs|macbook\w*|macos|apple)\b', t):
        req.os_preference = 'macOS'
        req.evidence.append(('Mac', 'macOS preferred'))
    elif re.search(r'\bwindows\b', t):
        req.os_preference = 'Windows'
        req.evidence.append(('Windows', 'Windows preferred'))

    if re.search(r"\b(don'?t care about weight|weight (doesn'?t|does not) matter|desktop replacement|stays? on (my )?desk)\b", t):
        req.portability = 'Low'
    elif re.search(r'\b(light\w*|portable|carry\w*|travel\w*|commut\w+|thin|slim|backpack|on the go|hostel to class)\b', t):
        req.portability = 'High'
        req.evidence.append(('portable / light', 'Portability is important'))
    if re.search(r'\b(battery|unplugged|all[- ]day|without (a )?charger|long hours)\b', t):
        req.battery = 'High'
        req.evidence.append(('battery', 'Battery life is important'))

    if re.search(r'\b(big|large|bigger)\s+(screen|display)\b|\b(15\.6|16|17)\s*(?:"|inch|in\b|”)', t):
        req.screen = SCREEN_CHOICES[2]
    elif re.search(r'\b(small|compact)\s*(screen|display|laptop)?\b|\b(13|14)\s*(?:"|inch|in\b|”)', t):
        req.screen = SCREEN_CHOICES[1]

    if req.required_os and req.os_preference not in ('Any', req.required_os):
        req.notes.append(f'You mentioned {req.os_preference}, but your software needs {req.required_os}; '
                         f'{req.required_os} laptops are shown.')
    return req


# --------------------------------------------------------------------------------------------------
# Spec recommendation
# --------------------------------------------------------------------------------------------------
CPU_CLASS = {1: 'Entry (Core 3 class)', 2: 'Mainstream (Core Ultra 5 / Ryzen 5 class)',
             3: 'Performance (H/HX-class 5-7 series or Apple M5)', 4: 'High performance (7/9-series HX/H or Apple M5 Pro)',
             5: 'Top tier (9-series HX)'}
GPU_CLASS = {1: 'Integrated graphics is enough', 2: 'Strong integrated (Intel Arc or Apple M-series)',
             3: 'Dedicated GPU (RTX 3050/4050/5050 class) or Apple Pro GPU', 4: 'Mid-range dedicated (RTX 5060 class)',
             5: 'Upper-mid dedicated (RTX 5070 class)'}


@dataclass(frozen=True)
class SpecProfile:
    min_ram: int
    rec_ram: int
    min_storage: int
    rec_storage: int
    min_cpu: int
    rec_cpu: int
    min_gpu: int
    rec_gpu: int
    prefer_cuda: bool
    prefer_refresh: int | None
    required_os: str | None
    primary_label: str
    reasons: tuple


def spec_profile(req):
    """Combine the selected workloads into one hardware profile (the most demanding value of each spec)."""
    ws = [WORKLOADS[k] for k in req.workloads if k in WORKLOADS] or [WORKLOADS['everyday']]
    required_os = req.required_os or next((w.requires_os for w in ws if w.requires_os), None)
    primary = max(ws, key=lambda w: (w.rec_gpu + w.rec_cpu + w.rec_ram / 8, w.key))
    return SpecProfile(
        min_ram=max(w.min_ram for w in ws), rec_ram=max(w.rec_ram for w in ws),
        min_storage=max(w.min_storage for w in ws), rec_storage=max(w.rec_storage for w in ws),
        min_cpu=max(w.min_cpu for w in ws), rec_cpu=max(w.rec_cpu for w in ws),
        min_gpu=max(w.min_gpu for w in ws), rec_gpu=max(w.rec_gpu for w in ws),
        prefer_cuda=any(w.prefer_cuda for w in ws),
        prefer_refresh=max((w.prefer_refresh for w in ws if w.prefer_refresh), default=None),
        required_os=required_os, primary_label=primary.label,
        reasons=tuple(f'{w.label}: {w.reason}' for w in ws))


def storage_text(gb):
    return f'{gb // 1024} TB' if gb >= 1024 else f'{gb} GB'


def profile_summary(p):
    """Human-readable recommended specs for the UI."""
    rows = [
        ('Memory (RAM)', f'{p.rec_ram} GB recommended', f'minimum {p.min_ram} GB'),
        ('Processor', CPU_CLASS[p.rec_cpu], f'minimum: {CPU_CLASS[p.min_cpu]}'),
        ('Graphics', GPU_CLASS[p.rec_gpu], 'NVIDIA (CUDA) preferred for ML' if p.prefer_cuda else f'minimum: {GPU_CLASS[p.min_gpu]}'),
        ('Storage', f'{storage_text(p.rec_storage)} SSD recommended', f'minimum {storage_text(p.min_storage)}'),
    ]
    if p.prefer_refresh:
        rows.append(('Display', f'{p.prefer_refresh} Hz or faster', 'smoother motion in games'))
    if p.required_os:
        rows.append(('Operating system', f'{p.required_os} required', 'needed by your software or games'))
    return rows


# --------------------------------------------------------------------------------------------------
# Scoring and ranking
# --------------------------------------------------------------------------------------------------
BASE_WEIGHTS = {'workload_fit': 0.50, 'value': 0.15, 'portability': 0.15, 'battery': 0.10, 'screen': 0.10}
COMPONENT_LABELS = {'workload_fit': 'Workload fit', 'value': 'Value for money', 'portability': 'Portability',
                    'battery': 'Battery friendliness', 'screen': 'Screen size match'}
POWER_FACTOR = {'efficient': 1.0, 'performance': 0.65, 'gaming': 0.3}


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def weights_for(req):
    """Weights from the user's priorities, renormalised to sum to 1. With a budget set, price is already
    constrained by the filter, so value for money counts for less."""
    w = dict(BASE_WEIGHTS)
    if req.budget_max is not None:
        w['value'] = 0.10
    w['portability'] = {'Low': 0.05, 'Medium': 0.15, 'High': 0.30}[req.portability]
    w['battery'] = {'Low': 0.03, 'Medium': 0.10, 'High': 0.25}[req.battery]
    if req.screen == 'Any':
        w['screen'] = 0.0
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def workload_fit(lap, p):
    """0-1: how far the laptop reaches the recommended (not just minimum) specs."""
    parts = [clamp(lap.ram_gb / p.rec_ram), clamp(lap.cpu_tier / p.rec_cpu), clamp(lap.gpu_tier / p.rec_gpu),
             clamp(lap.storage_gb / p.rec_storage)]
    if p.prefer_cuda:
        parts.append(1.0 if lap.cuda else 0.0)
    if p.prefer_refresh:
        parts.append(clamp((lap.refresh_hz or 60) / p.prefer_refresh))
    return sum(parts) / len(parts)


def portability_score(lap):
    return clamp((2.6 - lap.weight_kg) / (2.6 - 1.2))


def battery_score(lap):
    """Battery-friendliness estimate from capacity and power class - not a measured runtime."""
    return 0.4 * clamp((lap.battery_wh - 40) / 45) + 0.6 * POWER_FACTOR[lap.power_class]


def screen_score(lap, req):
    if req.screen == SCREEN_CHOICES[1]:
        return 1.0 if lap.display_in <= 14.2 else 0.3
    if req.screen == SCREEN_CHOICES[2]:
        return 1.0 if lap.display_in >= 15.3 else 0.3
    return 1.0


def meets_minimum(lap, p):
    gaps = []
    if lap.ram_gb < p.min_ram:
        gaps.append(f'{lap.ram_gb} GB RAM (minimum {p.min_ram} GB)')
    if lap.cpu_tier < p.min_cpu:
        gaps.append('processor below the minimum class')
    if lap.gpu_tier < p.min_gpu:
        gaps.append('graphics below the minimum level')
    if lap.storage_gb < p.min_storage:
        gaps.append(f'{storage_text(lap.storage_gb)} storage (minimum {storage_text(p.min_storage)})')
    return gaps


@dataclass
class Recommendation:
    laptop: object
    score: float                 # 0-1 total
    breakdown: list              # [{'key','label','score','weight'}]
    pros: list
    cons: list
    gaps: list                   # unmet minimum specs (only in fallback results)
    budget_gap: float | None     # rupees under (+) or over (-) budget
    trade_off: str = ''


@dataclass
class RecommendationResult:
    status: str                  # 'ok' | 'below_minimum' | 'over_budget'
    message: str
    profile: SpecProfile
    recommendations: list
    excluded: dict               # laptop id -> reason


def _explain(lap, p, req):
    pros, cons = [], []
    label = p.primary_label.lower()
    if p.required_os and lap.os == p.required_os:
        pros.append(f'Runs {lap.os}, which your software or games require')
    if lap.ram_gb >= p.rec_ram:
        pros.append(f'{lap.ram_gb} GB RAM meets the {p.rec_ram} GB recommended for {label}')
    elif lap.ram_gb >= p.min_ram:
        cons.append(f'{lap.ram_gb} GB RAM meets the minimum, but {p.rec_ram} GB is recommended for {label}')
    if p.prefer_cuda:
        if lap.cuda:
            pros.append(f'{lap.gpu} supports NVIDIA CUDA, which PyTorch and TensorFlow use for GPU training')
        else:
            cons.append('No NVIDIA CUDA GPU: GPU-accelerated deep learning is limited; cloud GPUs (e.g. Colab) are an alternative')
    if p.rec_gpu >= 2:
        if lap.gpu_tier >= p.rec_gpu:
            pros.append(f'{lap.gpu} meets the recommended graphics level for {label}')
        elif lap.gpu_tier >= p.min_gpu:
            cons.append(f'{lap.gpu} is below the recommended graphics level for {label}')
    if p.rec_cpu >= 3 and lap.cpu_tier >= p.rec_cpu:
        pros.append(f'{lap.cpu} is in the recommended processor class')
    if p.prefer_refresh and lap.refresh_hz and lap.refresh_hz >= p.prefer_refresh:
        pros.append(f'{lap.refresh_hz} Hz display for smoother motion in games')
    elif p.prefer_refresh and not lap.refresh_hz:
        cons.append('Display refresh rate is not stated as high-refresh')
    if lap.storage_gb < p.rec_storage:
        cons.append(f'{storage_text(lap.storage_gb)} SSD may fill quickly; {storage_text(p.rec_storage)} is recommended')
    if lap.weight_kg <= 1.45:
        pros.append(f'Light at {lap.weight_kg:g} kg, easy to carry every day')
    elif lap.weight_kg >= 2.2:
        cons.append(f'Heavy at {lap.weight_kg:g} kg, less convenient to carry daily')
    if lap.power_class == 'gaming':
        cons.append('High-power gaming hardware: expect to stay near a charger for demanding work')
    elif lap.power_class == 'efficient' and lap.battery_wh >= 50:
        pros.append(f'Efficient processor with a {lap.battery_wh:g} Wh battery')
    if lap.resolution_w >= 2560 and ({'design', 'video', 'engineering'} & set(req.workloads)):
        pros.append(f'High-resolution {lap.resolution_w}×{lap.resolution_h} display')
    if lap.ram_upgradeable:
        pros.append(f'{lap.brand} lists the RAM as upgradeable')
    if lap.os == 'macOS' and 'engineering' in req.workloads:
        cons.append('Some engineering software (e.g. SolidWorks, ANSYS) does not run on macOS')
    return pros[:4], cons[:3]


def _trade_off(lap, top):
    bits = []
    diff = top.price_inr - lap.price_inr
    if abs(diff) >= 1000:
        bits.append(f'₹{abs(diff):,.0f} {"cheaper" if diff > 0 else "more expensive"}')
    dw = top.weight_kg - lap.weight_kg
    if abs(dw) >= 0.1:
        bits.append(f'{abs(dw):.2f} kg {"lighter" if dw > 0 else "heavier"}')
    if lap.gpu_tier != top.gpu_tier:
        bits.append('stronger graphics' if lap.gpu_tier > top.gpu_tier else 'weaker graphics')
    if lap.ram_gb != top.ram_gb:
        bits.append(f'{"more" if lap.ram_gb > top.ram_gb else "less"} RAM ({lap.ram_gb} GB)')
    return ('Compared with #1: ' + ', '.join(bits)) if bits else 'Very similar to #1'


def score_laptop(lap, p, req, value_norm):
    w = weights_for(req)
    fit = workload_fit(lap, p)
    comps = {'workload_fit': fit, 'value': value_norm, 'portability': portability_score(lap),
             'battery': battery_score(lap), 'screen': screen_score(lap, req)}
    breakdown = [{'key': k, 'label': COMPONENT_LABELS[k], 'score': comps[k], 'weight': w[k]} for k in BASE_WEIGHTS if w[k] > 0]
    return sum(c['score'] * c['weight'] for c in breakdown), breakdown


def recommend(req, laptops=None, top_n=3):
    """Rank catalogue laptops for `req`. Never returns a laptop that violates the OS requirement."""
    if laptops is None:
        laptops = load_catalogue()[1]
    p = spec_profile(req)
    os_needed = p.required_os or (req.os_preference if req.os_preference != 'Any' else None)
    excluded, os_ok, in_budget, suitable = {}, [], [], []
    for lap in laptops:
        if os_needed and lap.os != os_needed:
            excluded[lap.id] = f'needs {os_needed}'
            continue
        os_ok.append(lap)
        if req.budget_max is not None and lap.price_inr > req.budget_max:
            excluded[lap.id] = 'over budget'
            continue
        in_budget.append(lap)
        gaps = meets_minimum(lap, p)
        if gaps:
            excluded[lap.id] = 'below minimum: ' + '; '.join(gaps)
            continue
        suitable.append(lap)

    override = ''
    if req.os_preference != 'Any' and p.required_os and req.os_preference != p.required_os:
        override = (f'You chose {req.os_preference}, but {p.primary_label.lower()} needs {p.required_os}, '
                    f'so only {p.required_os} laptops are considered. ')
    if suitable:
        pool, status = suitable, 'ok'
        message = f'{len(suitable)} of {len(laptops)} verified laptops meet your requirements.'
    elif in_budget:
        pool, status = in_budget, 'below_minimum'
        message = ('No verified laptop within your budget meets all minimum specs. These are the closest options; '
                   'their shortcomings are listed.')
    else:
        pool, status = os_ok, 'over_budget'
        cheapest = min((l.price_inr for l in os_ok), default=None)
        message = ('No verified laptop in the catalogue fits your budget'
                   + (f' (the cheapest verified option for your OS needs is ₹{cheapest:,.0f})' if cheapest else '')
                   + '. The recommended specifications above still apply.')
    if not pool:
        return RecommendationResult('over_budget', 'No laptop in the catalogue matches the operating-system requirement.',
                                    p, [], excluded)

    raw_value = {l.id: workload_fit(l, p) / l.price_inr for l in pool}
    best_value = max(raw_value.values())
    recs = []
    for lap in pool:
        total, breakdown = score_laptop(lap, p, req, raw_value[lap.id] / best_value)
        pros, cons = _explain(lap, p, req)
        gap = None if req.budget_max is None else req.budget_max - lap.price_inr
        recs.append(Recommendation(lap, total, breakdown, pros, cons, meets_minimum(lap, p), gap))
    recs.sort(key=lambda r: (-r.score, r.laptop.price_inr, r.laptop.id))
    recs = recs[:top_n]
    for r in recs[1:]:
        r.trade_off = _trade_off(r.laptop, recs[0].laptop)
    message = override + message
    if req.portability == 'High' and recs and all(r.laptop.weight_kg > 2.0 for r in recs):
        message += (' Portability matters to you, but every laptop that meets these performance needs weighs over '
                    '2 kg; the lighter models in the catalogue fall below the minimum specs.')
    elif req.portability == 'High' and recs and recs[0].laptop.weight_kg > 2.0:
        message += (f' Note: the top match weighs {recs[0].laptop.weight_kg:g} kg; it wins on performance, '
                    'so check the lighter alternatives below if carrying it daily matters more.')
    return RecommendationResult(status, message, p, recs, excluded)


EXAMPLE_QUERIES = {
    'CS student': "I'm a B.Tech computer science student. I code in Python and Java, and want to start learning machine learning. Budget around 1.2 lakh, should be light enough to carry to college.",
    'Gamer': 'I want a laptop for gaming - mostly Valorant and GTA V - plus some video editing for YouTube. Budget up to 2 lakh, weight does not matter.',
    'Mechanical engineer': 'Mechanical engineering student, I need to run SolidWorks, AutoCAD and MATLAB. Budget under 1.6 lakh.',
    'Business user': 'MBA student. Mostly Excel, PowerPoint and online classes. I travel a lot, so battery life and a light laptop matter. I prefer a Mac.',
}
