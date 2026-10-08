"""
Reproduces the ASCEND REU notebook's final XGBoost pipeline (REU_Colab.ipynb)
and saves the artifacts the Streamlit app needs: the fitted scaler, the
trained model, and the yearly feature stats used to build the UI.

Pipeline (matches the notebook):
1. Load daily-level dataset.
2. Label-encode Drainage_Quality (Poor=0, Fair=1, Good=2).
3. Fit a StandardScaler on the 13 final features at DAILY granularity
   (the notebook scales before aggregating; since StandardScaler is a
   per-column linear transform, scaling then averaging == averaging then
   scaling with the same mean_/scale_, so this is equivalent and lets us
   fit the scaler on just these 13 columns).
4. Aggregate to one row per year (mean of scaled features, mean of target).
5. Train XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.3,
   random_state=42) on all years (2015-2023) for the deployed model.
6. Save model.joblib, scaler.joblib, and feature_ranges.json (built from
   RAW, unscaled daily values so the app can show realistic slider ranges).
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

FEATURES = [
    'Heat_Stress_Days (HSD)', 'Temp_Shock (TS) ', 'Soil_Depth_cm',
    'cloudcover%', 'Wet_Days (WD)',
    'Nitrogen_%', 'Eff_Solar_Radiation (ESR)',
    'Phosphorus_mg_kg', 'Potassium_mg_kg',
    'Organic_Matter_%', 'Drainage_Quality',
    'pH', 'Soil_Temperature_C',
]
TARGET = 'Yield (Production-Hector M. Ton)'

df = pd.read_csv("combined dataset(latest).csv")
df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Drainage_Quality'] = df['Drainage_Quality'].replace({'Poor': 0, 'Fair': 1, 'Good': 2}).astype(int)

# --- realistic ranges for the UI, from RAW (unscaled) daily values ---
raw_stats = df[FEATURES].agg(['min', 'max', 'mean', 'std']).T
feature_ranges = {
    feat: {stat: float(raw_stats.loc[feat, stat]) for stat in ['min', 'max', 'mean', 'std']}
    for feat in FEATURES
}
feature_ranges['__target__'] = {
    'min': float(df[TARGET].min()),
    'max': float(df[TARGET].max()),
    'mean': float(df[TARGET].mean()),
    'std': float(df[TARGET].std()),
}
with open('feature_ranges.json', 'w') as f:
    json.dump(feature_ranges, f, indent=2)

# --- fit scaler on daily values (matches notebook's pre-aggregation scaling) ---
scaler = StandardScaler()
scaler.fit(df[FEATURES])
joblib.dump(scaler, 'scaler.joblib')

# --- aggregate to yearly, using the SAME scaled values the notebook trained on ---
df_scaled = df.copy()
df_scaled[FEATURES] = scaler.transform(df[FEATURES])
df_yearly = df_scaled.groupby('Year')[FEATURES + [TARGET]].mean()

X = df_yearly[FEATURES]
y = df_yearly[TARGET]

model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.3, random_state=42, verbosity=0)
model.fit(X, y)
joblib.dump(model, 'model.joblib')

importances = dict(zip(FEATURES, model.feature_importances_.tolist()))
with open('feature_importance.json', 'w') as f:
    json.dump(importances, f, indent=2)

print("Saved model.joblib, scaler.joblib, feature_ranges.json, feature_importance.json\n")
print("Feature importances (this deployed model, trained on all 9 years):")
for feat, imp in sorted(importances.items(), key=lambda kv: -kv[1]):
    print(f"  {feat:35s} {imp:.4f}")

print(f"\nHistorical average yield: {df[TARGET].mean():,.0f} metric tons")
print(f"Yearly-mean yield range: {df_yearly[TARGET].min():,.0f} - {df_yearly[TARGET].max():,.0f}")

# sanity check: predict on training years, report in-sample RMSE/MAPE
preds = model.predict(X)
rmse = float(np.sqrt(np.mean((preds - y.values) ** 2)))
mape = float(np.mean(np.abs(preds - y.values) / y.values) * 100)
print(f"\nIn-sample (all years) RMSE: {rmse:,.0f}  MAPE: {mape:.2f}%")
print("(Note: this is a full-data fit for deployment, not the walk-forward")
print(" 2019-2023 holdout eval that produced the reported 21,860 / 1.84% figures.)")
