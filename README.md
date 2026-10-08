# Potato Yield Predictor

A Streamlit app that predicts annual potato yield for Bogura, Bangladesh from
weather, soil, and crop-stress indicators, built on an XGBoost model trained
during an ASCEND REU (Research Experience for Undergraduates) project.

## Project summary

Potato yield in Bogura, Bangladesh was modeled from nine years (2015–2023) of
daily weather, soil, and crop-stress data, aggregated to one row per year.
Several regression models were compared; XGBoost gave the lowest error and is
the model this app deploys. The app lets you set realistic values for the 13
most relevant features and see a predicted annual yield, how it compares to
the historical average, and which features the model actually relies on.

## Dataset

Source: [Heterogeneous Data Set for Meta-Model Prediction (Kaggle)](https://www.kaggle.com/datasets/tahjibahmed/heterogeneous-data-set-for-meta-model-prediction)

The raw dataset is not included in this repository — download it from the
link above if you want to retrain the model yourself (see **Retraining**
below). It contains 1,698 daily records (2015–2023) with weather, soil
chemistry, and crop-stress columns; this project aggregates those to 9 yearly
records for modeling.

## Method

- **Features (13):** Heat Stress Days, Temperature Shock, Soil Depth, Cloud
  Cover, Wet Days, Nitrogen, Effective Solar Radiation, Phosphorus, Potassium,
  Organic Matter, Drainage Quality, pH, and Soil Temperature. (Heat Stress
  Days, Temperature Shock, and Wet Days are daily 0/1 indicators in the raw
  data, aggregated to a "fraction of the year" when averaged per year.)
- **Preprocessing:** features are standardized (`StandardScaler`) before
  aggregating to yearly means; `Drainage_Quality` is label-encoded
  (Poor=0, Fair=1, Good=2).
- **Model:** `XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.3,
  random_state=42)`.
- **Evaluation:** walk-forward (expanding-window) validation — for each test
  year in 2019–2023, the model trains on all prior years (starting 2015) and
  predicts the held-out year.

## Results

Walk-forward evaluation, 2019–2023 (lower RMSE/MAE/MAPE is better; R² here is
computed once, pooled across all 5 held-out years' predictions):

| Model | Avg RMSE (metric tons) | Avg MAPE | R² (pooled, cross-year) |
|---|---:|---:|---:|
| **XGBoost** | **21,860** | **1.84%** | **-0.257** |
| RF (tuned) | 23,785 | 2.00% | -0.500 |
| Gradient Boosting | 29,661 | 2.49% | -1.503 |
| KNN (K=2) | 25,767 | 2.17% | -0.923 |
| Ridge (α=10000) | 31,278 | 2.63% | -1.585 |
| Lasso (α=10000) | 33,266 | 2.80% | -1.943 |

**Honest caveat on R²:** every model tested, including the best-performing
XGBoost model, had a **negative R²** in this walk-forward evaluation. A
negative R² means the model's held-out predictions were less accurate than
simply guessing the training-period mean every time. With only 9 years of
yearly-aggregated data and 5 held-out test points, this isn't surprising — the
low RMSE/MAPE numbers above look good in absolute terms, but R² reveals the
model isn't capturing much real year-to-year signal beyond the target's own
narrow historical range (the yearly mean yield only varies from about
1.10M to 1.23M metric tons across the whole dataset). Treat this app as a
research prototype demonstrating feature relationships, not a reliable
forecasting tool.

### Deployed model vs. validation model: feature importance

The model shipped in this repo (`model.joblib`) is refit on **all 9 years**
of data for deployment, rather than the 2015–2022 training window used during
validation. Its feature importances shift slightly as a result:

| Feature | Validation model (train 2015–2022) | Deployed model (train 2015–2023) |
|---|---:|---:|
| Potassium | 0.827 | 0.839 |
| Soil Depth | 0.088 | 0.040 |
| Soil Temperature | 0.070 | 0.062 |
| Nitrogen | 0.013 | 0.053 |
| Heat Stress Days | 0.003 | 0.002 |
| Organic Matter | ~0.000 | 0.004 |
| All others (Phosphorus, Cloud Cover, Temp Shock, Wet Days, Drainage Quality, pH, Eff. Solar Radiation) | 0.000 | 0.000 |

In both versions, **Potassium dominates** the model's predictions by a wide
margin; everything below the top 3–4 features contributes almost nothing.

## Running the app

```bash
git clone https://github.com/dareen-musa/potato-yield-predictor.git
cd potato-yield-predictor
pip install -r requirements.txt
streamlit run app.py
```

The repo already includes a trained `model.joblib` and `scaler.joblib`, so
this is all you need to run the app.

## Retraining

To retrain from scratch (e.g. after changing the model or getting an updated
dataset):

1. Download the dataset from the [Kaggle link above](#dataset) and save it in
   the project root as `combined dataset(latest).csv`.
2. Run:
   ```bash
   pip install -r requirements.txt
   python train_and_save_model.py
   ```
   This regenerates `model.joblib`, `scaler.joblib`, `feature_ranges.json`,
   and `feature_importance.json`.

## Project structure

```
app.py                      # Streamlit app
train_and_save_model.py     # Reproduces the notebook's pipeline; saves model + scaler
REU_Colab.ipynb             # Original research notebook (model selection, evaluation, plots)
model.joblib                 # Trained XGBoost model (committed)
scaler.joblib                # Fitted StandardScaler (committed)
feature_ranges.json         # Real min/max/mean per feature, used for UI slider ranges
feature_importance.json     # Deployed model's feature_importances_
requirements.txt
```
