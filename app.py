"""Laptop Advisor - Streamlit front end (Laptop Fair-Price Analyzer project).

    conda activate laptop_price
    streamlit run app.py

Sections (st.tabs):
- Find: hero brief -> rule-based requirement extraction + explainable ranking over the verified catalogue
  (src/recommender.py, src/catalogue.py, catalogue/laptops.json).
- Explore: a browsable board of every verified configuration, with collections and filters.
- Compare: side-by-side view (reuses the recommendation data).
- Price Lab: the original historical XGBoost estimator via `predict_laptop_price()` (logic unchanged).
- How it works: methodology and limitations.

Architecture: no feature engineering or scoring here. Presentation lives in src/ui_components.py.
"""
import streamlit as st

from src.app_helpers import (build_specs, compare_listed_price, format_inr, format_resolution, model_summary,
                             resolution_label, run_prediction)
from src.catalogue import load_catalogue
from src.predict import get_input_options, load_artifacts
from src.recommender import (BASE_WEIGHTS, EXAMPLE_QUERIES, LEVELS, OS_CHOICES, SCREEN_CHOICES, WORKLOADS,
                             Requirements, extract_requirements, profile_summary, recommend, workload_fit)
from src import ui_components as ui

st.set_page_config(page_title='Laptop Advisor · Find your next laptop', page_icon='💻', layout='wide',
                   initial_sidebar_state='collapsed')
st.markdown(ui.CSS, unsafe_allow_html=True)

# Model + metadata come from the production loader (cached once per process with functools.lru_cache).
_, METADATA = load_artifacts()
OPTIONS = get_input_options()
SUMMARY = model_summary(METADATA)
CATALOGUE_META, LAPTOPS = load_catalogue()
LAPTOP_BY_NAME = {l.display_name: l for l in LAPTOPS}
BRANDS = sorted({l.brand for l in LAPTOPS})
VERIFIED_ON = CATALOGUE_META['catalogue_version']
LABEL_TO_KEY = {w.label: w.key for w in WORKLOADS.values()}


def default_index(options, preferred):
    return options.index(preferred) if preferred in options else 0


def _by_id(lid):
    return next((l for l in LAPTOPS if l.id == lid), None)


# ---------------------------------------------------------------------------------------------- sidebar (secondary)
with st.sidebar:
    st.markdown('### Laptop Advisor')
    st.write(f'{len(LAPTOPS)} verified laptop configurations from {len(BRANDS)} brands, checked on '
             f'{ui.nice_date(VERIFIED_ON)} at official brand stores and an authorised retailer.')
    st.markdown('### Price Lab model')
    st.write(f"{SUMMARY['model_label']} · trained on {SUMMARY['training_rows']:,} laptops (2016–18 listings)  \n"
             f"Log-price R² ≈ {SUMMARY['r2_log_price']:.3f} · MAE ≈ {format_inr(round(SUMMARY['mae_rupees'], -2))}  \n"
             f"Median percentage error ≈ {SUMMARY['median_ape_percent']:.0f}%")
    st.caption(f"Measured on {SUMMARY['holdout_rows']} laptops held out from training. Model version {SUMMARY['model_version']}.")
    st.divider()
    st.markdown('**Applied AI & Data Science Project**')

st.markdown(ui.topbar(VERIFIED_ON), unsafe_allow_html=True)
tab_find, tab_explore, tab_compare, tab_price, tab_about = st.tabs(['Find', 'Explore', 'Compare', 'Price Lab', 'How it works'])


# ============================================================================================== FIND
def _set_example(text):
    st.session_state['fm_query'] = text


def _apply_requirements(req):
    """Copy extracted requirements into the editable widgets' session state."""
    st.session_state['fm_workloads'] = [WORKLOADS[k].label for k in req.workloads]
    st.session_state['fm_budget'] = float(req.budget_max) if req.budget_max else None
    st.session_state['fm_os'] = req.required_os or req.os_preference
    st.session_state['fm_portability'] = req.portability
    st.session_state['fm_battery'] = req.battery
    st.session_state['fm_screen'] = req.screen
    st.session_state['fm_extracted'] = req
    st.session_state['fm_active'] = True


HERO_PICKS = [l for l in (_by_id('apple-macbook-air-13-m5-16-512'), _by_id('lenovo-83s000fyin'),
                          min(LAPTOPS, key=lambda l: l.price_inr or 1e12)) if l]

with tab_find:
    with st.container(key='hero'):
        left, right = st.columns([11, 9], gap='large')
        with left:
            st.markdown(ui.hero_copy(len(LAPTOPS)), unsafe_allow_html=True)
            st.text_area('Tell us about you', key='fm_query', height=118,
                         placeholder="e.g. I'm a second-year mechanical engineering student. I use SolidWorks and MATLAB, "
                                     "play a little Valorant, and want to stay under ₹1.2 lakh.")
            if st.button('Find my laptop', key='fm_find', type='primary', width='stretch'):
                _apply_requirements(extract_requirements(st.session_state.get('fm_query', '').strip()))
                st.session_state['fm_new_search'] = True
            st.markdown('<div class="hint">Not sure what to write? Start from a profile:</div>', unsafe_allow_html=True)
            with st.container(key='examples', horizontal=True):
                for name, text in EXAMPLE_QUERIES.items():
                    st.button(name, key=f'fm_ex_{name}', on_click=_set_example, args=(text,))
            st.markdown(f'<div class="hint">Prefer to browse? Open <b>Explore</b> to see all {len(LAPTOPS)} laptops.</div>',
                        unsafe_allow_html=True)
        with right:
            if len(HERO_PICKS) == 3:
                st.markdown(ui.hero_art(HERO_PICKS, compact=bool(st.session_state.get('fm_active'))), unsafe_allow_html=True)

    if not st.session_state.get('fm_active'):
        st.markdown(ui.VALUES_HTML, unsafe_allow_html=True)
    else:
        extracted = st.session_state['fm_extracted']
        st.markdown(ui.section('Your brief', 'Here is what we heard.',
                               'Everything below is editable. Change a detail and the shortlist updates instantly.'),
                    unsafe_allow_html=True)
        heard, tune = st.columns([5, 7], gap='medium')
        with heard:
            with st.container(key='brief'):
                st.markdown('<div class="list-title">From your description</div>', unsafe_allow_html=True)
                st.markdown(ui.chips(extracted.evidence) if extracted.evidence else
                            '<span class="chip"><b>No specific needs detected</b></span>', unsafe_allow_html=True)
                for n in extracted.notes:
                    st.markdown(ui.note(n), unsafe_allow_html=True)
        with tune:
            with st.container(key='tune'):
                st.markdown('<div class="list-title">Fine-tune</div>', unsafe_allow_html=True)
                st.multiselect('What you will use it for', [w.label for w in WORKLOADS.values()], key='fm_workloads')
                c1, c2 = st.columns(2)
                c1.number_input('Maximum budget (₹)', key='fm_budget', min_value=0.0, step=5000.0, format='%.0f',
                                placeholder='No limit')
                c2.selectbox('Operating system', OS_CHOICES, key='fm_os',
                             help='Some software needs a specific OS (e.g. SolidWorks needs Windows, Final Cut Pro needs macOS).')
                c3, c4 = st.columns(2)
                c3.select_slider('Portability matters', LEVELS, key='fm_portability')
                c4.select_slider('Battery life matters', LEVELS, key='fm_battery')
                c5, c6 = st.columns(2)
                c5.selectbox('Screen size', SCREEN_CHOICES, key='fm_screen')
                c6.multiselect('Brands', BRANDS, key='fm_brands', placeholder='All brands')

        workloads = [LABEL_TO_KEY[l] for l in st.session_state.get('fm_workloads', [])] or ['everyday']
        os_choice = st.session_state.get('fm_os', 'Any')
        budget = st.session_state.get('fm_budget')
        req = Requirements(workloads=workloads, budget_max=budget if budget else None,
                           os_preference=os_choice, portability=st.session_state.get('fm_portability', 'Medium'),
                           battery=st.session_state.get('fm_battery', 'Medium'),
                           screen=st.session_state.get('fm_screen', 'Any'),
                           required_os=extracted.required_os if os_choice in ('Any', extracted.required_os) else None)
        brands = st.session_state.get('fm_brands') or BRANDS
        result = recommend(req, laptops=tuple(l for l in LAPTOPS if l.brand in brands))
        st.session_state['fm_result'] = result
        if st.session_state.pop('fm_new_search', False):
            st.session_state['cmp_select'] = [r.laptop.display_name for r in result.recommendations]

        st.markdown(ui.section('What to look for', 'The specs that matter for you.'), unsafe_allow_html=True)
        st.markdown(ui.spec_grid(profile_summary(result.profile)), unsafe_allow_html=True)
        with st.expander('Why these specs?'):
            for reason in result.profile.reasons:
                st.markdown(f'- {reason}')

        st.markdown(ui.section('Your shortlist', 'Three laptops worth a closer look.'), unsafe_allow_html=True)
        st.markdown(ui.status_box(result.message, warn=result.status != 'ok'), unsafe_allow_html=True)
        if result.status == 'over_budget':
            # Never present over-budget laptops as recommendations; keep them out of the way and clearly labelled.
            st.markdown(ui.empty_state('Nothing fits this budget yet',
                                       'Look for a laptop with the specs above, or raise your budget. We only list '
                                       'configurations we could verify at official stores or an authorised retailer.'),
                        unsafe_allow_html=True)
            if result.recommendations:
                with st.expander('See the closest verified options above your budget'):
                    for rec in result.recommendations:
                        lap = rec.laptop
                        st.markdown(f'- **{lap.display_name}**: {ui.price_text(lap.price_inr)} '
                                    f'({ui.price_text(-rec.budget_gap)} over budget), verified '
                                    f'{ui.nice_date(lap.verified_on)}, [seller page]({lap.url})'
                                    if rec.budget_gap is not None else f'- **{lap.display_name}**')
        elif result.recommendations:
            first, rest = result.recommendations[0], result.recommendations[1:]
            st.markdown(ui.rec_primary(first), unsafe_allow_html=True)
            with st.expander('How this score was built'):
                st.markdown(ui.breakdown_html(first), unsafe_allow_html=True)
                st.caption('Fit = a weighted mix of workload fit, value, portability, battery and screen match. '
                           'Weights follow your priorities above.')
            if rest:
                cols = st.columns(len(rest), gap='medium')
                for i, (col, rec) in enumerate(zip(cols, rest), start=2):
                    with col:
                        st.markdown(ui.rec_secondary(rec, i), unsafe_allow_html=True)
                        with st.expander('How this score was built'):
                            st.markdown(ui.breakdown_html(rec), unsafe_allow_html=True)
            st.markdown(ui.section('Side by side', 'How your shortlist compares.'), unsafe_allow_html=True)
            st.markdown(ui.comparison_table([r.laptop for r in result.recommendations],
                                            {r.laptop.id: r.score for r in result.recommendations}, uid='shortlist'),
                        unsafe_allow_html=True)
            st.caption('Prices are the official store or authorised retailer listing on the verification date and may '
                       'have changed. Always confirm the price, configuration and stock on the seller page before buying.')

# ============================================================================================== EXPLORE
COLLECTIONS = {
    'All laptops': lambda l: True,
    'Under ₹60k': lambda l: l.price_inr is not None and l.price_inr < 60_000,
    'For coding': lambda l: l.ram_gb >= 16 and l.cpu_tier >= 2,
    'Gaming': lambda l: l.gpu_dedicated and 'zbook' not in l.model.lower(),
    'Light & long-lasting': lambda l: l.weight_kg is not None and l.weight_kg <= 1.5 and l.power_class == 'efficient',
    'Creative work': lambda l: l.ram_gb >= 16 and (l.gpu_tier >= 2 or l.resolution_w >= 2560),
    'Business & pro': lambda l: any(k in l.model.lower() for k in ('probook', 'elitebook', 'zbook')),
    'Mac': lambda l: l.os == 'macOS',
}
SORTS = {'Price: low to high': lambda l: (l.price_inr or 1e12), 'Price: high to low': lambda l: -(l.price_inr or 0),
         'Lightest first': lambda l: (l.weight_kg or 99), 'Most memory': lambda l: (-l.ram_gb, l.price_inr or 1e12)}

with tab_explore:
    st.markdown(ui.section('The board', f'{len(LAPTOPS)} laptops, each one checked by hand.',
                           'Every tile is an exact configuration with its part number, price and the date we checked it. '
                           'Open any tile to see it at the seller.'), unsafe_allow_html=True)
    with st.container(key='board_filters'):
        st.pills('Collection', list(COLLECTIONS), key='ex_collection', default='All laptops', label_visibility='collapsed')
        f1, f2, f3 = st.columns([4, 4, 3])
        prices = [l.price_inr for l in LAPTOPS if l.price_inr]
        lo, hi = int(min(prices) // 5000 * 5000), int(-(-max(prices) // 5000) * 5000)
        f1.multiselect('Brand', BRANDS, key='ex_brands', placeholder='All brands')
        f2.slider('Budget (₹)', lo, hi, (lo, hi), step=5000, key='ex_budget', format='₹%d')
        f3.selectbox('Sort', list(SORTS), key='ex_sort')
    coll = st.session_state.get('ex_collection') or 'All laptops'
    brands_sel = st.session_state.get('ex_brands') or BRANDS
    bmin, bmax = st.session_state.get('ex_budget', (lo, hi))
    shown = [l for l in LAPTOPS if COLLECTIONS[coll](l) and l.brand in brands_sel
             and l.price_inr is not None and bmin <= l.price_inr <= bmax]
    shown.sort(key=SORTS[st.session_state.get('ex_sort', 'Price: low to high')])
    st.markdown(f'<p class="count">{len(shown)} of {len(LAPTOPS)} laptops match</p>', unsafe_allow_html=True)
    if not shown:
        st.markdown(ui.empty_state('No laptops match these filters', 'Try a wider budget or another collection.'),
                    unsafe_allow_html=True)
    else:
        stories = {}
        if coll == 'All laptops' and st.session_state.get('ex_sort', 'Price: low to high') == 'Price: low to high' and len(shown) > 12:
            starter = [l for l in LAPTOPS if COLLECTIONS['Under ₹60k'](l)]
            gaming = [l for l in LAPTOPS if COLLECTIONS['Gaming'](l)]
            light = [l for l in LAPTOPS if COLLECTIONS['Light & long-lasting'](l)]
            stories[0] = ui.story('Starter picks', f'{len(starter)} honest laptops under ₹60,000.',
                                  'Expect 8 GB of memory and integrated graphics: right for notes, browsing, Office and '
                                  'online classes.', '#3F5847', starter[:3])
            if gaming:
                stories[min(11, len(shown) - 1)] = ui.story('Built for play', f'{len(gaming)} laptops with NVIDIA RTX graphics.',
                                      f'From {ui.price_text(min(l.price_inr for l in gaming))}. The usual trade-off is '
                                      'weight and battery life.', '#232A5C', gaming[-3:])
            if light:
                stories[min(23, len(shown) - 1)] = ui.story('Carry it everywhere', f'{len(light)} laptops at 1.5 kg or less.',
                                      'Efficient chips and slim chassis for long days between classes, meetings and trains.',
                                      '#6B5A43', light[:3])
        limit = st.session_state.get('ex_limit', 24)
        st.markdown(ui.board(shown[:limit], {k: v for k, v in stories.items() if k < limit}), unsafe_allow_html=True)
        if len(shown) > limit:
            def _more():
                st.session_state['ex_limit'] = limit + 24
            st.button(f'Show more laptops ({len(shown) - limit} left)', key='ex_more', on_click=_more, width='stretch')

# ============================================================================================== COMPARE
with tab_compare:
    st.markdown(ui.section('Compare', 'Put laptops side by side.',
                           'Pick two to four laptops. Your latest shortlist is selected for you.'), unsafe_allow_html=True)
    if 'cmp_select' not in st.session_state:
        st.session_state['cmp_select'] = [l.display_name for l in LAPTOPS[:3]]
    chosen = st.multiselect('Laptops', list(LAPTOP_BY_NAME), key='cmp_select', max_selections=4)
    if len(chosen) >= 2:
        fm_result = st.session_state.get('fm_result')
        scores = None
        if fm_result is not None and st.session_state.get('fm_active'):
            p = fm_result.profile
            scores = {LAPTOP_BY_NAME[n].id: workload_fit(LAPTOP_BY_NAME[n], p) for n in chosen}
        st.markdown(ui.comparison_table([LAPTOP_BY_NAME[n] for n in chosen], scores, uid='compare'), unsafe_allow_html=True)
        if scores:
            st.caption(f'"Meets your recommended specs" uses your current brief ({fm_result.profile.primary_label}): the share '
                       'of the recommended RAM, processor, graphics and storage each laptop reaches, so 100 means it meets '
                       'or exceeds all of them. It ignores budget, so laptops outside your budget can be compared too.')
    else:
        st.markdown(ui.empty_state('Pick at least two laptops', 'Search by brand or model name in the box above.'),
                    unsafe_allow_html=True)

# ============================================================================================== PRICE LAB
with tab_price:
    with st.container(key='labtitle'):
        st.title('Laptop Fair-Price Analyzer')
    st.markdown("""
<div class="historic"><b>Historical reference model, not a current price.</b> The Price Lab is where this project
began: an XGBoost model trained on 1,273 laptops listed around 2016–2018 in Europe, with euro prices converted to
rupees at a fixed rate (× 53.28). Use it to explore how specifications related to price in that data. It does not
reflect today's Indian market, and Laptop Advisor's recommendations do not use it.</div>""", unsafe_allow_html=True)
    st.markdown('**Estimate the expected price of a laptop from its specifications using a machine-learning model '
                'trained on historical laptop data.**')
    st.caption('This model estimates a reference/fair price based on hardware specifications. It does not fetch '
               'current prices from Amazon, Flipkart, or other retailers.')

    form_col, result_col = st.columns([3, 2], gap='large')

    observed = OPTIONS['observed']
    ranges = OPTIONS['numeric_ranges']
    ram_values = [d['value'] for d in observed['ram_gb']]
    ssd_values = [d['value'] for d in observed['ssd_gb']]
    hdd_values = [d['value'] for d in observed['hdd_gb']]
    resolutions = [format_resolution(d['width'], d['height']) for d in observed['resolutions']]   # most common first

    with form_col:
        with st.form('laptop_form', border=False):
            c1, c2 = st.columns(2)
            with c1.container(border=True, key='lab_basics'):
                st.markdown('#### Laptop basics')
                company = st.selectbox('Company', OPTIONS['company'], key='company',
                                       index=default_index(OPTIONS['company'], 'Dell'))
                type_name = st.selectbox('Laptop type', OPTIONS['type_name'], key='type_name',
                                         index=default_index(OPTIONS['type_name'], 'Notebook'),
                                         help='Ultrabook = thin and light premium laptop; Workstation = professional '
                                              'machine for engineering/design work; 2 in 1 Convertible = laptop that '
                                              'folds into a tablet.')
                ram_gb = st.selectbox('RAM', ram_values, key='ram_gb', index=default_index(ram_values, 8),
                                      format_func=lambda v: f'{v} GB')
                weight_kg = st.number_input('Weight (kg)', key='weight_kg', min_value=ranges['weight_kg'][0],
                                            max_value=ranges['weight_kg'][1], value=2.0, step=0.05, format='%.2f',
                                            help='Limited to the weight range of laptops the model was trained on.')
                os_name = st.selectbox('Operating system', OPTIONS['os'], key='os',
                                       index=default_index(OPTIONS['os'], 'Windows 10'))

            with c2.container(border=True, key='lab_cpu'):
                st.markdown('#### Processor and graphics')
                cpu = st.selectbox('CPU', OPTIONS['cpu'], key='cpu',
                                   index=default_index(OPTIONS['cpu'], 'Intel Core i5 7200U 2.5GHz'),
                                   help='Processor model and base clock speed. Type to search. The model reads the '
                                        'processor family, generation, power class (e.g. U = low-power, '
                                        'HQ = high-performance) and GHz from this name.')
                gpu = st.selectbox('GPU', OPTIONS['gpu'], key='gpu',
                                   index=default_index(OPTIONS['gpu'], 'Intel HD Graphics 620'),
                                   help='Graphics processor. "Intel HD/UHD/Iris" = integrated graphics; '
                                        'Nvidia GeForce / Quadro and AMD Radeon = dedicated graphics. Type to search.')
                st.caption(f"{len(OPTIONS['cpu'])} CPUs and {len(OPTIONS['gpu'])} GPUs are available - "
                           'the models present in the training data.')

            c3, c4 = st.columns(2)
            with c3.container(border=True, key='lab_storage'):
                st.markdown('#### Storage')
                ssd_gb = st.selectbox('SSD capacity', ssd_values, key='ssd_gb', index=default_index(ssd_values, 256),
                                      format_func=lambda v: 'None' if v == 0 else f'{v} GB',
                                      help='Solid-state drive: fast storage. Choose None if the laptop has no SSD.')
                hdd_gb = st.selectbox('HDD capacity', hdd_values, key='hdd_gb', index=default_index(hdd_values, 0),
                                      format_func=lambda v: 'None' if v == 0 else f'{v} GB',
                                      help='Hard-disk drive: slower, cheaper storage. 1000 GB = 1 TB.')

            with c4.container(border=True, key='lab_display'):
                st.markdown('#### Display')
                inches = st.number_input('Screen size (inches)', key='inches', min_value=ranges['inches'][0],
                                         max_value=ranges['inches'][1], value=15.6, step=0.1, format='%.1f',
                                         help='Diagonal screen size. Limited to the sizes in the training data.')
                resolution = st.selectbox('Resolution', resolutions, key='resolution',
                                          index=default_index(resolutions, '1920 × 1080'), format_func=resolution_label,
                                          help='Width × height in pixels. Listed from most to least common.')
                touchscreen = st.toggle('Touchscreen', key='touchscreen')
                ips = st.toggle('IPS panel', key='ips', help='IPS panels have better colours and viewing angles.')

            with st.container(border=True, key='lab_listed'):
                st.markdown('#### Compare with a listed price (optional)')
                listed_price = st.number_input('Seller / Listed Price (₹)', key='listed_price', min_value=0.0,
                                               value=None, step=1000.0, format='%.0f',
                                               placeholder='Leave blank to skip',
                                               help="Enter a seller's asking price to see how it compares with "
                                                    "the model's estimate.")

            submitted = st.form_submit_button('Estimate Price', key='submit', type='primary', width='stretch')

    if submitted:
        specs = build_specs(company=company, type_name=type_name, ram_gb=ram_gb, weight_kg=weight_kg, os=os_name,
                            cpu=cpu, gpu=gpu, ssd_gb=ssd_gb, hdd_gb=hdd_gb, inches=inches, resolution=resolution,
                            touchscreen=touchscreen, ips=ips)
        with st.spinner('Estimating price...'):
            result, error = run_prediction(specs)
        st.session_state['last'] = {'specs': specs, 'result': result, 'error': error, 'listed_price': listed_price}

    with result_col:
        last = st.session_state.get('last')
        if last is None:
            st.info('Choose the laptop specifications and click **Estimate Price**.')
        elif last['error']:
            st.error(f"Could not estimate a price: {last['error']}")
        else:
            result = last['result']
            st.markdown(f"""
<div class="price-card">
  <p class="label">Estimated Price (historical reference)</p>
  <p class="value">{format_inr(result['predicted_price_rounded'])}</p>
  <p class="note">This is the model's estimated reference price based on laptops with similar specifications
  in the 2016–18 training data (converted from euros). It is not a current retail price.</p>
</div>""", unsafe_allow_html=True)

            for warning in result['warnings']:
                st.warning(warning)

            if last['listed_price'] is not None:
                st.markdown('##### Listed price comparison')
                try:
                    cmp = compare_listed_price(result['predicted_price'], last['listed_price'])
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    m1, m2 = st.columns(2)
                    m1.metric('Estimated model price', format_inr(result['predicted_price_rounded']))
                    m2.metric('Listed price', format_inr(last['listed_price']))
                    st.info(f"{cmp['sentence']}  \n{cmp['amount_text']}.")
                    st.caption('The model does not know about discounts, condition, warranty, bundled extras or current '
                               'market conditions, so a difference is not by itself a judgement about the listing.')

            st.markdown('##### Model performance on unseen evaluation data')
            st.markdown(
                f"- Typical absolute error: about {format_inr(round(SUMMARY['mae_rupees'], -2))}\n"
                f"- Median percentage error: about {SUMMARY['median_ape_percent']:.0f}%. Historically, half of "
                f"predictions had a percentage error of roughly {SUMMARY['median_ape_percent']:.0f}% or less.\n"
                '- Performance is generally weaker for unusual premium laptops and workstations.')

            with st.expander('What the model derived from these specifications'):
                f = result['features']
                st.markdown(
                    f"- Processor family: **{f['Cpu brand']}** · class: **{f['cpu_type']}** · "
                    f"generation: **{f['cpu_gen']}** · {f['cpu_ghz']} GHz\n"
                    f"- Graphics tier: **{f['gpu_tier']}**\n"
                    f"- Pixel density: **{f['ppi']:.0f} PPI** ({f['X_res']} × {f['Y_res']} on {f['Inches']}\")\n"
                    f"- Operating-system group: **{f['os']}** · company group: **{f['Company_grouped']}**")
                st.caption('Computed by the production feature pipeline (src/features.py).')

# ============================================================================================== HOW IT WORKS
with tab_about:
    st.markdown(ui.section('How it works', 'Simple rules, real data, honest trade-offs.'), unsafe_allow_html=True)
    a1, a2 = st.columns(2, gap='large')
    with a1:
        st.markdown('#### How Find works')
        st.markdown(f"""
1. **Understanding your needs.** Your description is matched against a rule set of courses, professions, software
   and games (e.g. "SolidWorks" means Engineering & CAD and requires Windows). Budget phrases like "1.2 lakh",
   "70k" or "under ₹90,000" are recognised. Everything detected is shown and can be edited. No paid AI is used.
2. **Recommended specs.** Each workload has a minimum and a recommended level for RAM, processor class, graphics
   class and storage. Combining workloads takes the most demanding value of each.
3. **Filtering.** Laptops that need the wrong operating system, cost more than your budget, or fall below the
   minimum specs are removed. Your budget is never relaxed. If nothing within budget meets the minimum specs, the
   closest options within budget are shown with their shortcomings. If nothing fits the budget at all, the app says
   so, and over-budget options are only listed separately.
4. **Scoring.** The remaining laptops get a transparent weighted score:
   workload fit {round(BASE_WEIGHTS['workload_fit'] * 100)}%, value for money {round(BASE_WEIGHTS['value'] * 100)}%,
   portability {round(BASE_WEIGHTS['portability'] * 100)}%, battery friendliness {round(BASE_WEIGHTS['battery'] * 100)}%
   and screen match {round(BASE_WEIGHTS['screen'] * 100)}%. These are the default weights; they shift with your
   priorities, and value counts for less once you set a budget. Every part is shown under "How this score was built".
5. **Explanations.** Reasons and trade-offs are generated only from the laptop's listed specifications.
""")
        st.markdown('#### The verified catalogue')
        st.markdown(f"""
- {len(LAPTOPS)} configurations from {', '.join(BRANDS)}, each checked on {ui.nice_date(VERIFIED_ON)} on the manufacturer's official
  Indian online store (HP, Apple) or on an authorised Indian retailer (Vijay Sales): price, part number and
  specifications, plus stock status for retailer listings.
- Prices are the listing on that date and can change. Details a page did not state (often weight, battery or the
  exact processor model) are shown as "not stated" and are never guessed.
- Processor and graphics "classes" follow a documented rule based on the chip family, not on benchmark scores; when a
  processor's power suffix isn't stated, the lower class is assumed.
- Recommendations are the best matches **within this catalogue**, not across the whole market.
- The laptop drawings are original illustrations generated from each configuration's screen size and resolution;
  they are not product photos.
""")
    with a2:
        st.markdown('#### The Price Lab model')
        st.markdown(f"""
- An XGBoost regression model trained on {SUMMARY['training_rows']:,} laptops from a public Kaggle dataset
  (2016–18 European listings; prices converted from euros at a fixed × 53.28).
- On held-out laptops it reached log-price R² ≈ {SUMMARY['r2_log_price']:.3f} and an average error of about
  {format_inr(round(SUMMARY['mae_rupees'], -2))} (median error ≈ {SUMMARY['median_ape_percent']:.0f}%).
- It is kept as a separate, clearly labelled research tool. It is **not** used to rank or price modern laptops:
  its inputs describe 2016–18 hardware and its prices are converted European prices.
""")
        st.markdown('#### Limitations')
        st.markdown("""
- Prices are not live and do not include bank offers, exchange deals or discounts.
- Stock changes; retailer listings were in stock when checked, but always confirm on the seller page.
- Specification classes are simplified and are not a substitute for reviews or benchmarks.
- Recommendations are only as broad as the catalogue.
- This is an academic project and does not give financial or purchasing advice.
""")

st.markdown(ui.footer(len(LAPTOPS), VERIFIED_ON), unsafe_allow_html=True)
