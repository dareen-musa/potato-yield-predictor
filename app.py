import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Potato Yield Predictor", page_icon="🥔", layout="wide")

FEATURES = [
    'Heat_Stress_Days (HSD)', 'Temp_Shock (TS) ', 'Soil_Depth_cm',
    'cloudcover%', 'Wet_Days (WD)',
    'Nitrogen_%', 'Eff_Solar_Radiation (ESR)',
    'Phosphorus_mg_kg', 'Potassium_mg_kg',
    'Organic_Matter_%', 'Drainage_Quality',
    'pH', 'Soil_Temperature_C',
]

# Human-facing label, help text, units, and widget type for each feature.
# HSD/TS/WD are daily 0/1 flags aggregated to "fraction of the year" in training data.
FEATURE_INFO = {
    'Heat_Stress_Days (HSD)': dict(
        label="Heat Stress Days", unit="fraction of year", fmt="%.2f",
        help="Share of days in the year where temperatures were high enough to stress "
             "potato plants (heat stress slows tuber growth). 0 = no stressful days, "
             "1 = every day was a heat-stress day."),
    'Temp_Shock (TS) ': dict(
        label="Temperature Shock", unit="fraction of year", fmt="%.2f",
        help="Share of days with a sudden, sharp temperature swing. Rapid swings can "
             "shock plants and reduce yield even if the average temperature looks fine."),
    'Soil_Depth_cm': dict(
        label="Soil Depth", unit="cm", fmt="%.1f",
        help="Depth of usable topsoil available to plant roots. Deeper soil generally "
             "gives roots and tubers more room to grow."),
    'cloudcover%': dict(
        label="Cloud Cover", unit="%", fmt="%.1f",
        help="Average percentage of the sky covered by clouds. More cloud cover means "
             "less direct sunlight reaching the crop."),
    'Wet_Days (WD)': dict(
        label="Wet Days", unit="fraction of year", fmt="%.2f",
        help="Share of days in the year with significant rainfall. Too many wet days "
             "can waterlog soil; too few can stress the crop with drought."),
    'Nitrogen_%': dict(
        label="Nitrogen", unit="%", fmt="%.3f",
        help="Percentage of nitrogen in the soil. Nitrogen is a key nutrient for leafy "
             "growth, which fuels the plant's ability to produce tubers."),
    'Eff_Solar_Radiation (ESR)': dict(
        label="Effective Solar Radiation", unit="index", fmt="%.1f",
        help="A combined measure of how much usable sunlight energy reached the crop "
             "during the growing season. More effective radiation generally supports "
             "more plant growth."),
    'Phosphorus_mg_kg': dict(
        label="Phosphorus", unit="mg/kg", fmt="%.1f",
        help="Amount of phosphorus in the soil. Phosphorus supports root and tuber "
             "development in potatoes."),
    'Potassium_mg_kg': dict(
        label="Potassium", unit="mg/kg", fmt="%.1f",
        help="Amount of potassium in the soil. Potassium is strongly linked to tuber "
             "size and starch content — this is the single most influential input for "
             "this model's predictions."),
    'Organic_Matter_%': dict(
        label="Organic Matter", unit="%", fmt="%.2f",
        help="Percentage of decomposed organic material in the soil. Higher organic "
             "matter usually improves soil fertility and water retention."),
    'pH': dict(
        label="Soil pH", unit="pH", fmt="%.2f",
        help="Acidity/alkalinity of the soil. Potatoes generally prefer slightly acidic "
             "soil (roughly pH 5-6.5)."),
    'Soil_Temperature_C': dict(
        label="Soil Temperature", unit="°C", fmt="%.1f",
        help="Average soil temperature. Potato tubers develop best within a moderate "
             "soil temperature range — too hot or too cold slows tuber formation."),
}

DRAINAGE_MAP = {'Poor': 0, 'Fair': 1, 'Good': 2}
FEATURE_INFO['Drainage_Quality'] = dict(
    label="Drainage Quality", unit="", fmt="%.0f",
    help="How well the soil drains excess water. Poor drainage can waterlog "
         "roots; good drainage helps prevent tuber rot.")


@st.cache_resource
def load_artifacts():
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    with open('feature_ranges.json') as f:
        ranges = json.load(f)
    with open('feature_importance.json') as f:
        importance = json.load(f)
    return model, scaler, ranges, importance


model, scaler, ranges, importance = load_artifacts()
hist_avg = ranges['__target__']['mean']

st.title("🥔 Potato Yield Predictor")
st.caption(
    "ASCEND REU research project — predicting annual potato yield for Bogura, Bangladesh "
    "(2015–2023) from weather, soil, and crop-stress indicators using XGBoost. "
    "Model performance (walk-forward evaluation, 2019–2023): **RMSE ≈ 21,860 metric tons, "
    "MAPE ≈ 1.84%**."
)

st.divider()

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Input features")
    st.caption("Adjust the sliders to match a growing season, then predict.")

    values = {}
    cols = st.columns(2)
    feature_order = [f for f in FEATURES if f != 'Drainage_Quality']

    for i, feat in enumerate(feature_order):
        info = FEATURE_INFO[feat]
        r = ranges[feat]
        col = cols[i % 2]
        with col:
            values[feat] = st.slider(
                f"{info['label']} ({info['unit']})",
                min_value=float(r['min']),
                max_value=float(r['max']),
                value=float(r['mean']),
                step=(r['max'] - r['min']) / 200 if r['max'] > r['min'] else 0.01,
                format=info['fmt'],
                help=info['help'],
                key=feat,
            )

    with cols[len(feature_order) % 2]:
        drainage_label = st.select_slider(
            "Drainage Quality",
            options=['Poor', 'Fair', 'Good'],
            value='Good',
            help="How well the soil drains excess water. Poor drainage can waterlog "
                 "roots; good drainage helps prevent tuber rot.",
        )
        values['Drainage_Quality'] = DRAINAGE_MAP[drainage_label]

    predict_clicked = st.button("🔮 Predict Yield", type="primary", use_container_width=True)

with right:
    st.subheader("Prediction")

    if predict_clicked:
        row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
        scaled_row = row.copy()
        scaled_row[FEATURES] = scaler.transform(row[FEATURES])
        prediction = float(model.predict(scaled_row[FEATURES])[0])

        delta = prediction - hist_avg
        pct = delta / hist_avg * 100

        st.metric(
            "Predicted Annual Yield",
            f"{prediction:,.0f} metric tons",
            delta=f"{delta:+,.0f} vs. historical average ({pct:+.1f}%)",
        )
        st.caption(f"Historical average (2015–2023): **{hist_avg:,.0f} metric tons**")

        if abs(pct) > 5:
            st.info(
                "This is a notable swing from the historical average — mainly driven "
                "by Potassium, which dominates this model's predictions (see chart)."
            )
    else:
        st.info("Set your inputs on the left and click **Predict Yield**.")

    st.divider()
    st.subheader("Feature Importance")
    st.caption("How much each feature drives this model's predictions.")

    imp_series = pd.Series(importance).sort_values(ascending=True)
    imp_labels = [FEATURE_INFO.get(f, {}).get('label', f) for f in imp_series.index]

    fig = go.Figure(go.Bar(
        x=imp_series.values,
        y=imp_labels,
        orientation='h',
        marker_color='#c2410c',
    ))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Importance",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "This model relies heavily on just a few features — Potassium alone accounts "
        "for the large majority of predictive power, with Soil Temperature, Nitrogen, "
        "and Soil Depth contributing modestly. Most other features have little "
        "measurable effect in this model."
    )

st.divider()

with st.expander("What do these features mean? (plain-language guide)"):
    for feat in FEATURES:
        info = FEATURE_INFO[feat]
        st.markdown(f"**{info['label']}** — {info['help']}")

with st.expander("About this model"):
    st.markdown(
        """
This app is built on an **XGBoost** regression model
(`n_estimators=100, max_depth=5, learning_rate=0.3`) trained on daily weather,
soil, and crop-stress data from Bogura, Bangladesh (2015–2023), aggregated to
one row per year.

**Important context:**
- The reported accuracy (RMSE ≈ 21,860 metric tons, MAPE ≈ 1.84%) comes from a
  walk-forward evaluation: training on prior years and testing on each held-out
  year from 2019–2023.
- The underlying dataset has only **9 years** of aggregated data, so this model
  should be read as a research prototype demonstrating feature relationships,
  not a production forecasting tool.
- Feature values are standardized (z-scored) internally to match how the model
  was trained; the sliders show realistic raw values from the original dataset.
        """
    )
