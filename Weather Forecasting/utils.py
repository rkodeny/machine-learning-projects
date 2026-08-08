"""
Weather Forecasting — Utility Functions
Time-series forecasting: engineer calendar + lag/rolling features, then forecast
with classic regressors against naive / seasonal-naive baselines, using a
chronological train/test split (never shuffle time series).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Per-dataset configuration (the only thing that changes between TS projects):
DATE_COL = "datetime"          # datetime column name
VALUE_COL = "temperature"      # raw target column name
FREQ = "h"                     # "D" daily, "h" hourly
SEASONAL_PERIOD = 24           # 7 for daily (weekly), 24 for hourly (daily)
LAGS = [1, 2, 3, 24, 48, 168]  # lag features to use as features (in hours)
ROLL_WINDOWS = [24, 168]       # rolling windows to use as features (in hours)

# Data Loading
def load_data(filepath="weather.csv"):
    df = pd.read_csv(filepath)
    df = df.rename(columns={DATE_COL: "date", VALUE_COL: "y"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "y"]]

# Feature Engineering
def add_calender_features(df):
    # Make a copy so the original DataFrame is not modified
    df = df.copy()
    d = df["date"].dt              # Shortcut: access datetime properties from the date column
    df["month"] = d.month          # Month number: 1-12
    df["day"] = d.day              # Day of month: 1-31
    df["dayofweek"] = d.dayofweek  # Day of week: Monday=0, Sunday=6
    df["dayofyear"] = d.dayofyear  # Day number within the year: 1-365/366
    # Weekend flag: 1 for Saturday/Sunday, else 0
    df["is_weekend"] = (d.dayofweek >= 5).astype(int) 
    # If the data frequency is hourly, add hour of day
    if FREQ == "h":
        df["hour"] = d.hour
    return df

# Lag Features
def add_lag_features(df, lags=LAGS, windows=ROLL_WINDOWS):
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for window in windows:
        df[f"roll_mean_{window}"] = df["y"].shift(1).rolling(window=window).mean()
        df[f"roll_std_{window}"] = df["y"].shift(1).rolling(window=window).std()
    return df

def build_features(df):
    df = add_calender_features(df)
    df = add_lag_features(df)
    return df.dropna().reset_index(drop=True)

def feature_columns(df):
    return [col for col in df.columns if col not in ("date", "y")]

# Split Data into Train/Test Sets
def chronological_split(df, test_frac=0.2):
    """Last `test_frac` of the timeline is the test set (no shuffling)."""
    n = len(df)
    cut = int(n * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

# baseline
def naive_forecast(train, test):
    """Naive forecast: predict the last observed value from the training set."""
    last_value = train["y"].iloc[-1]
    return np.full(len(test), last_value)

def seasonal_naive_forecast(full_df, test, period=SEASONAL_PERIOD):
    """Predict y[t] = y[t-period] (e.g. same weekday last week / same hour yesterday)."""
    y = full_df["y"].values
    idx = test.index.values
    return np.array([y[i - period] if i - period >= 0 else y[i] for i in idx])

# Models & Evaluation
def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),   
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=7),
    }

def evaluate(y_true, y_pred):
    """MAE / RMSE / MAPE(%) / R²."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "MAPE": round(float(mape), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def run_all(feat_df, test_frac=0.2):
    """Fit baselines + all regressors on a chronological split; return a results DataFrame."""
    train, test = chronological_split(feat_df, test_frac)
    cols = feature_columns(feat_df)
    Xtr, ytr = train[cols], train["y"]
    Xte, yte = test[cols], test["y"]
    rows = []
    rows.append({"model": "Naive", **evaluate(yte, naive_forecast(train, test))})
    rows.append({"model": "Seasonal Naive", **evaluate(yte, seasonal_naive_forecast(feat_df, test))})
    preds = {}
    for name, model in get_models().items():
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        preds[name] = p
        rows.append({"model": name, **evaluate(yte, p)})
    res = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return res, train, test, preds