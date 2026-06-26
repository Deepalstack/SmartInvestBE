"""
features.py
-----------
Shared technical-indicator feature engineering, used identically for both
stocks and ETFs since both are now tracked via tickers (yfinance) instead
of CSV-derived fields. Having one shared implementation means the stock
model and ETF model are trained/predicted on the exact same feature
definitions, just on different ticker universes.
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "ma_5", "ma_20", "volatility_20",
    "volume_change", "rsi_14",
    "pct_from_52wk_high", "pct_from_52wk_low",
]
TARGET_COL = "fwd_return"
FORWARD_RETURN_DAYS = 5


def engineer_features(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a feature row per trading day from raw OHLCV history.

        ma_5, ma_20            - short/medium moving averages (trend)
        volatility_20          - 20-day rolling std of daily returns (risk)
        volume_change          - day-over-day volume change (momentum)
        rsi_14                 - 14-day Relative Strength Index (momentum)
        pct_from_52wk_high      - distance below the 52-week high
        pct_from_52wk_low       - distance above the 52-week low

    Target:
        fwd_return              - % return FORWARD_RETURN_DAYS ahead
    """
    df = hist.copy()
    df["daily_return"] = df["Close"].pct_change()

    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["volatility_20"] = df["daily_return"].rolling(20).std()
    df["volume_change"] = df["Volume"].pct_change()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    rolling_high_252 = df["Close"].rolling(252, min_periods=20).max()
    rolling_low_252 = df["Close"].rolling(252, min_periods=20).min()
    df["pct_from_52wk_high"] = (df["Close"] - rolling_high_252) / rolling_high_252
    df["pct_from_52wk_low"] = (df["Close"] - rolling_low_252) / rolling_low_252

    df["fwd_return"] = df["Close"].shift(-FORWARD_RETURN_DAYS) / df["Close"] - 1

    return df


def compute_live_features(ticker: str) -> np.ndarray | None:
    """
    Computes the SAME feature set from a fresh yfinance pull, for live
    prediction. Returns None if there isn't enough history yet.
    """
    import yfinance as yf

    hist = yf.Ticker(ticker).history(period="1y")
    if hist.empty or len(hist) < 30:
        return None

    daily_return = hist["Close"].pct_change()
    ma_5 = hist["Close"].rolling(5).mean().iloc[-1]
    ma_20 = hist["Close"].rolling(20).mean().iloc[-1]
    volatility_20 = daily_return.rolling(20).std().iloc[-1]
    volume_change = hist["Volume"].pct_change().iloc[-1]

    delta = hist["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_14 = (100 - (100 / (1 + rs))).iloc[-1]

    rolling_high = hist["Close"].rolling(len(hist), min_periods=20).max().iloc[-1]
    rolling_low = hist["Close"].rolling(len(hist), min_periods=20).min().iloc[-1]
    last_close = hist["Close"].iloc[-1]
    pct_from_52wk_high = (last_close - rolling_high) / rolling_high
    pct_from_52wk_low = (last_close - rolling_low) / rolling_low

    values = [ma_5, ma_20, volatility_20, volume_change, rsi_14, pct_from_52wk_high, pct_from_52wk_low]
    if any(pd.isna(v) for v in values):
        return None
    return np.array(values).reshape(1, -1)
