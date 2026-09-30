"""Laptop Fair-Price Analyzer - Streamlit front end.

    conda activate laptop_price
    streamlit run app.py

Architecture: this UI only collects specifications and calls `predict_laptop_price()`.
All feature engineering (PPI, CPU brand/type/generation, GPU tier, company grouping) happens in
src/features.py inside the production pipeline. Do not add feature logic here.
"""
import streamlit as st

from src.app_helpers import (build_specs, compare_listed_price, format_inr, format_resolution, model_summary,
                             resolution_label, run_prediction)
from src.predict import get_input_options, load_artifacts

st.set_page_config(page_title='Laptop Fair-Price Analyzer', page_icon='💻', layout='wide')

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1250px;}
.price-card {border: 1px solid rgba(128,128,128,0.25); border-radius: 12px; padding: 1.1rem 1.3rem;
             background: rgba(42,120,214,0.06); margin-bottom: 0.6rem;}
.price-card .label {font-size: 0.9rem; opacity: 0.75; margin: 0;}
.price-card .value {font-size: 2.6rem; font-weight: 700; line-height: 1.15; margin: 0.15rem 0 0.35rem 0;}
.price-card .note {font-size: 0.85rem; opacity: 0.75; margin: 0;}
</style>
""", unsafe_allow_html=True)

# Model + metadata come from the production loader (cached once per process with functools.lru_cache).
_, METADATA = load_artifacts()
OPTIONS = get_input_options()
SUMMARY = model_summary(METADATA)


def default_index(options, preferred):
    return options.index(preferred) if preferred in options else 0


# ---------------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('### Model')
    st.write(SUMMARY['model_label'])
    st.markdown('### Training laptops')
    st.write(f"{SUMMARY['training_rows']:,}")
    st.markdown('### Evaluation')
    st.write(f"Log-price R² ≈ {SUMMARY['r2_log_price']:.3f}  \n"
             f"MAE ≈ {format_inr(round(SUMMARY['mae_rupees'], -2))}  \n"
             f"Median percentage error ≈ {SUMMARY['median_ape_percent']:.0f}%")
    st.caption(f"Measured on {SUMMARY['holdout_rows']} laptops held out from training. "
               f"Model version {SUMMARY['model_version']}.")
    st.divider()
    st.markdown('**Applied AI & Data Science Project**')

# ---------------------------------------------------------------------------------------------- header
st.title('Laptop Fair-Price Analyzer')
st.markdown('**Estimate the expected price of a laptop from its specifications using a machine-learning model '
            'trained on historical laptop data.**')
st.caption('This model estimates a reference/fair price based on hardware specifications. It does not fetch '
           'current prices from Amazon, Flipkart, or other retailers.')

form_col, result_col = st.columns([3, 2], gap='large')

# ---------------------------------------------------------------------------------------------- form
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

# ---------------------------------------------------------------------------------------------- results
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
  <p class="label">Estimated Price</p>
  <p class="value">{format_inr(result['predicted_price_rounded'])}</p>
  <p class="note">This is the model's estimated reference price based on laptops with similar specifications
  in the training data. It is not a current retail price.</p>
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

# ---------------------------------------------------------------------------------------------- explanations
st.divider()
e1, e2 = st.columns(2)
with e1.expander('How does this work?'):
    st.markdown("""
The estimate comes from a machine-learning model (XGBoost) that learned how laptop prices relate to
hardware specifications, using a historical dataset of laptops and their listed prices.

It uses:
- company and laptop type
- RAM and weight
- CPU characteristics
- GPU class
- SSD and HDD capacity
- screen size, resolution and pixel density
- touchscreen and IPS panel
- operating system

Some useful information is not typed in directly but **derived automatically** from what you choose.
For example, the CPU name gives its generation, power class and clock speed. The GPU name gives its
performance tier. Screen size and resolution give the pixel density (PPI).
""")
with e2.expander('Limitations'):
    st.markdown("""
- The dataset is **historical**, not live retailer data.
- Current discounts and promotions are not considered.
- Laptop age and condition are not modelled.
- The exact model and its build quality may carry information that is not in the dataset.
- Prediction errors are larger for very expensive and unusual laptops.
- This is an **estimated reference price**, not a guaranteed market price.
""")
