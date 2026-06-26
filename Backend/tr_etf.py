"""
tr_etf.py
---------
Trains BOTH XGBoost and Gaussian Process Regressor models on LIVE ETF
data pulled via yfinance — replacing the old justetf.csv-based training
(tr.py / tr2.py), since the CSV's ISIN/country-exposure features can't
be derived from live data.

Same feature engineering as tr_stocks.py (features.py), just trained on
a default list of well-known, liquid ETFs instead of individual stocks.

Output: xgb_etf_model.joblib, gpr_etf_model.joblib, etf_scaler.joblib
"""

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

from features import FEATURE_COLS, TARGET_COL, engineer_features
from news_fetcher import DEFAULT_ETF_TICKERS

TRAINING_PERIOD = "2y"
TICKERS = DEFAULT_ETF_TICKERS


def build_dataset(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        print(f"📥 Fetching history for {ticker}...")
        hist = yf.Ticker(ticker).history(period=TRAINING_PERIOD)
        if hist.empty or len(hist) < 60:
            print(f"⚠️ Not enough history for {ticker}, skipping.")
            continue
        feat = engineer_features(hist)
        feat["ticker"] = ticker
        frames.append(feat)

    if not frames:
        raise RuntimeError("No data fetched for any ETF ticker — check network/tickers.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"✅ Built dataset: {len(combined)} rows across {len(frames)} ETFs.")
    return combined


def main():
    data = build_dataset(TICKERS)
    X = data[FEATURE_COLS].values
    y = data[TARGET_COL].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    print("\n🚀 Training XGBoost...")
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)
    print(f"XGBoost Test MSE: {mean_squared_error(y_test, xgb_model.predict(X_test)):.6f}")

    print("\n🚀 Training Gaussian Process Regressor...")
    kernel = C(1.0, (1e-3, 1e3)) * RBF(2.0, (1e-2, 1e2))
    gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
    if len(X_train) > 2000:
        idx = np.random.RandomState(42).choice(len(X_train), 2000, replace=False)
        gpr_model.fit(X_train[idx], y_train[idx])
    else:
        gpr_model.fit(X_train, y_train)
    gpr_pred_mean, _ = gpr_model.predict(X_test, return_std=True)
    print(f"GPR Test MSE: {mean_squared_error(y_test, gpr_pred_mean):.6f}")

    joblib.dump(xgb_model, "xgb_etf_model.joblib")
    joblib.dump(gpr_model, "gpr_etf_model.joblib")
    joblib.dump(scaler, "etf_scaler.joblib")
    print("\n✅ Saved: xgb_etf_model.joblib, gpr_etf_model.joblib, etf_scaler.joblib")


if __name__ == "__main__":
    main()
