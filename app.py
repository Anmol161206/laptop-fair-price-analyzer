"""Laptop Fair-Price Analyzer - Streamlit front end.

    conda activate laptop_price
    streamlit run app.py

Tabs:
- Find My Laptop: rule-based requirement extraction + explainable ranking over a small verified catalogue
  (src/recommender.py, src/catalogue.py, catalogue/laptops.json).
- Compare: side-by-side view of catalogue laptops (reuses the recommendation data).
- Historical Price Estimator: the original XGBoost model via `predict_laptop_price()` (unchanged).
- About: methodology and limitations.

Architecture: this UI contains no feature engineering or scoring logic. The historical model's features come from
src/features.py; recommendations come from src/recommender.py. The two never mix.
"""
import streamlit as st

from src.app_helpers import (build_specs, compare_listed_price, format_inr, format_resolution, model_summary,
                             resolution_label, run_prediction)
from src.catalogue import load_catalogue
from src.predict import get_input_options, load_artifacts
from src.recommender import (BASE_WEIGHTS, EXAMPLE_QUERIES, LEVELS, OS_CHOICES, SCREEN_CHOICES, WORKLOADS,
                             Requirements, extract_requirements, profile_summary, recommend, workload_fit)
from src import ui_components as ui

st.set_page_config(page_title='Laptop Fair-Price Analyzer', page_icon='💻', layout='wide')
st.markdown(ui.CSS, unsafe_allow_html=True)

# Model + metadata come from the production loader (cached once per process with functools.lru_cache).
_, METADATA = load_artifacts()
OPTIONS = get_input_options()
SUMMARY = model_summary(METADATA)
CATALOGUE_META, LAPTOPS = load_catalogue()
LAPTOP_BY_NAME = {l.display_name: l for l in LAPTOPS}
VERIFIED_ON = ui.nice_date(CATALOGUE_META['catalogue_version'])


def default_index(options, preferred):
    return options.index(preferred) if preferred in options else 0


# ---------------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('### Laptop Advisor')
    st.write('Find a laptop that fits your course, work and budget, and compare it with others.')
    st.markdown('### Verified catalogue')
    st.write(f'{len(LAPTOPS)} laptop configurations from official Indian brand stores, checked on {VERIFIED_ON}.')
    st.markdown('### Historical price model')
    st.write(SUMMARY['model_label'])
    st.write(f"Trained on {SUMMARY['training_rows']:,} laptops (2016–18 listings)  \n"
             f"Log-price R² ≈ {SUMMARY['r2_log_price']:.3f}  \n"
             f"MAE ≈ {format_inr(round(SUMMARY['mae_rupees'], -2))}  \n"
             f"Median percentage error ≈ {SUMMARY['median_ape_percent']:.0f}%")
    st.caption(f"Measured on {SUMMARY['holdout_rows']} laptops held out from training. "
               f"Model version {SUMMARY['model_version']}.")
    st.divider()
    st.markdown('**Applied AI & Data Science Project**')

# ---------------------------------------------------------------------------------------------- header
st.title('Laptop Fair-Price Analyzer')
st.markdown(f"""
<div class="hero">
  <h3>Your laptop-buying assistant</h3>
  <p>Describe what you study, what software you use, whether you game and what you can spend. You get recommended
  specs, the three best-matching laptops from a verified catalogue, and a clear explanation of the trade-offs.</p>
  <div class="pill-row"><span class="pill">No sign-up</span><span class="pill">No paid AI</span>
  <span class="pill">{len(LAPTOPS)} verified configurations</span><span class="pill">Prices checked {VERIFIED_ON}</span></div>
</div>""", unsafe_allow_html=True)

tab_find, tab_compare, tab_price, tab_about = st.tabs(
    ['🔎 Find My Laptop', '⚖️ Compare', '📈 Historical Price Estimator', 'ℹ️ About'])


# ============================================================================================== FIND MY LAPTOP
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


LABEL_TO_KEY = {w.label: w.key for w in WORKLOADS.values()}

with tab_find:
    st.markdown('#### Tell us what you need')
    st.text_area('Describe your course or job, software, games, budget and preferences', key='fm_query', height=110,
                 placeholder='e.g. "Second-year mechanical engineering student. I use SolidWorks and MATLAB, play some '
                             'Valorant, budget around 1.3 lakh, and I carry it to college every day."')
    st.caption('Try an example:')
    ex_cols = st.columns(len(EXAMPLE_QUERIES))
    for col, (name, text) in zip(ex_cols, EXAMPLE_QUERIES.items()):
        col.button(name, key=f'fm_ex_{name}', on_click=_set_example, args=(text,), width='stretch')

    if st.button('Find my laptop', key='fm_find', type='primary', width='stretch'):
        query = st.session_state.get('fm_query', '').strip()
        _apply_requirements(extract_requirements(query))
        st.session_state['fm_new_search'] = True

    if st.session_state.get('fm_active'):
        extracted = st.session_state['fm_extracted']
        st.divider()
        left, right = st.columns([2, 3], gap='large')
        with left:
            st.markdown('#### What we understood')
            if extracted.evidence:
                st.markdown(ui.chips(extracted.evidence), unsafe_allow_html=True)
            for n in extracted.notes:
                st.markdown(ui.note(n), unsafe_allow_html=True)
            st.caption('Adjust anything below; results update instantly.')
            st.multiselect('Workloads', [w.label for w in WORKLOADS.values()], key='fm_workloads')
            st.number_input('Maximum budget (₹)', key='fm_budget', min_value=0.0, step=5000.0, format='%.0f',
                            placeholder='No limit')
            st.selectbox('Operating system', OS_CHOICES, key='fm_os',
                         help='Some software needs a specific OS (e.g. SolidWorks needs Windows, Final Cut Pro needs macOS).')
            c1, c2 = st.columns(2)
            c1.select_slider('Portability importance', LEVELS, key='fm_portability')
            c2.select_slider('Battery importance', LEVELS, key='fm_battery')
            st.selectbox('Screen size', SCREEN_CHOICES, key='fm_screen')

        workloads = [LABEL_TO_KEY[l] for l in st.session_state.get('fm_workloads', [])] or ['everyday']
        os_choice = st.session_state.get('fm_os', 'Any')
        budget = st.session_state.get('fm_budget')
        req = Requirements(workloads=workloads, budget_max=budget if budget else None,
                           os_preference=os_choice, portability=st.session_state.get('fm_portability', 'Medium'),
                           battery=st.session_state.get('fm_battery', 'Medium'),
                           screen=st.session_state.get('fm_screen', 'Any'),
                           required_os=extracted.required_os if os_choice in ('Any', extracted.required_os) else None)
        result = recommend(req)
        st.session_state['fm_result'] = result
        if st.session_state.pop('fm_new_search', False):
            st.session_state['cmp_select'] = [r.laptop.display_name for r in result.recommendations]

        with right:
            st.markdown('#### Recommended specifications')
            st.markdown(ui.spec_grid(profile_summary(result.profile)), unsafe_allow_html=True)
            with st.expander('Why these specs?'):
                for reason in result.profile.reasons:
                    st.markdown(f'- {reason}')

        st.markdown('#### Best matches from the verified catalogue')
        st.markdown(ui.status_box(result.message, warn=result.status != 'ok'), unsafe_allow_html=True)
        if result.status == 'over_budget':
            # Never present over-budget laptops as recommendations; keep them out of the way and clearly labelled.
            st.markdown(ui.note('Look for a laptop that matches the recommended specifications above, or raise your '
                                'budget. The catalogue only lists configurations verified on official stores.'),
                        unsafe_allow_html=True)
            if result.recommendations:
                with st.expander('See the closest verified options above your budget'):
                    for rec in result.recommendations:
                        lap = rec.laptop
                        st.markdown(f'- **{lap.display_name}**: {ui.price_text(lap.price_inr)} '
                                    f'({ui.price_text(-rec.budget_gap)} over budget), verified '
                                    f'{ui.nice_date(lap.verified_on)}, [official store page]({lap.url})'
                                    if rec.budget_gap is not None else f'- **{lap.display_name}**')
        elif result.recommendations:
            cols = st.columns(len(result.recommendations), gap='medium')
            for i, (col, rec) in enumerate(zip(cols, result.recommendations), start=1):
                with col:
                    st.markdown(ui.laptop_card(rec, i, req.budget_max is not None), unsafe_allow_html=True)
                    st.link_button(f'View on {rec.laptop.brand} store ↗', rec.laptop.url, width='stretch')
                    with st.expander('Score breakdown'):
                        st.markdown(ui.breakdown_html(rec), unsafe_allow_html=True)
                        st.caption('Suitability = weighted sum of the parts above. Weights change with your '
                                   'portability and battery priorities.')
            st.markdown('#### Side-by-side')
            st.markdown(ui.comparison_table([r.laptop for r in result.recommendations],
                                            {r.laptop.id: r.score for r in result.recommendations}),
                        unsafe_allow_html=True)
            st.caption('Prices are the official store listing on the verification date and may have changed. '
                       'Always confirm the price and configuration on the store page before buying.')

# ============================================================================================== COMPARE
with tab_compare:
    st.markdown('#### Compare laptops side by side')
    st.caption('Pick 2–4 laptops from the verified catalogue. Your latest recommendations are pre-selected.')
    if 'cmp_select' not in st.session_state:
        st.session_state['cmp_select'] = [l.display_name for l in LAPTOPS[:3]]
    chosen = st.multiselect('Laptops', list(LAPTOP_BY_NAME), key='cmp_select', max_selections=4)
    if len(chosen) >= 2:
        fm_result = st.session_state.get('fm_result')
        scores = None
        if fm_result is not None and st.session_state.get('fm_active'):
            p = fm_result.profile
            scores = {LAPTOP_BY_NAME[n].id: workload_fit(LAPTOP_BY_NAME[n], p) for n in chosen}
        st.markdown(ui.comparison_table([LAPTOP_BY_NAME[n] for n in chosen], scores), unsafe_allow_html=True)
        if scores:
            st.caption('"Suitability for your needs" here is the workload-fit part of the score (how close each laptop '
                       'is to your recommended specs), so laptops outside your budget can be compared too.')
    else:
        st.caption('Select at least two laptops to compare.')

# ============================================================================================== HISTORICAL ESTIMATOR
with tab_price:
    st.markdown("""
<div class="historic"><b>Historical reference model, not a current price.</b> This XGBoost model was trained on
1,273 laptops listed around 2016–2018 in Europe, with euro prices converted to rupees at a fixed rate (× 53.28).
Use it to explore how specifications related to price in that data. It does not reflect today's Indian market, and
it is not used by Find My Laptop.</div>""", unsafe_allow_html=True)
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
            with c1.container(border=True):
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

            with c2.container(border=True):
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
            with c3.container(border=True):
                st.markdown('#### Storage')
                ssd_gb = st.selectbox('SSD capacity', ssd_values, key='ssd_gb', index=default_index(ssd_values, 256),
                                      format_func=lambda v: 'None' if v == 0 else f'{v} GB',
                                      help='Solid-state drive: fast storage. Choose None if the laptop has no SSD.')
                hdd_gb = st.selectbox('HDD capacity', hdd_values, key='hdd_gb', index=default_index(hdd_values, 0),
                                      format_func=lambda v: 'None' if v == 0 else f'{v} GB',
                                      help='Hard-disk drive: slower, cheaper storage. 1000 GB = 1 TB.')

            with c4.container(border=True):
                st.markdown('#### Display')
                inches = st.number_input('Screen size (inches)', key='inches', min_value=ranges['inches'][0],
                                         max_value=ranges['inches'][1], value=15.6, step=0.1, format='%.1f',
                                         help='Diagonal screen size. Limited to the sizes in the training data.')
                resolution = st.selectbox('Resolution', resolutions, key='resolution',
                                          index=default_index(resolutions, '1920 × 1080'), format_func=resolution_label,
                                          help='Width × height in pixels. Listed from most to least common.')
                touchscreen = st.toggle('Touchscreen', key='touchscreen')
                ips = st.toggle('IPS panel', key='ips', help='IPS panels have better colours and viewing angles.')

            with st.container(border=True):
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

# ============================================================================================== ABOUT
with tab_about:
    a1, a2 = st.columns(2, gap='large')
    with a1:
        st.markdown('#### How Find My Laptop works')
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
   priorities, and value counts for less once you set a budget. Every part is shown in the score breakdown.
5. **Explanations.** Reasons and trade-offs are generated only from the laptop's listed specifications.
""")
        st.markdown('#### The verified catalogue')
        st.markdown(f"""
- {len(LAPTOPS)} configurations, each checked on the manufacturer's official Indian online store on {VERIFIED_ON}
  (price, part number, processor, memory, storage, display, graphics, weight and battery).
- Prices are the store's listing on that date and can change. Fields a store page did not state are left blank.
- Processor and graphics "classes" follow a documented rule based on the chip family, not on benchmark scores.
- The catalogue is small (HP and Apple only so far), so recommendations are the best matches **within it**, not
  across the whole market.
""")
    with a2:
        st.markdown('#### The historical price model')
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
- Stock and availability are not tracked; check the store page.
- Specification classes are simplified and are not a substitute for reviews or benchmarks.
- Recommendations are only as broad as the catalogue.
- This is an academic project and does not give financial or purchasing advice.
""")
