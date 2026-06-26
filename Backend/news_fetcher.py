"""
news_fetcher.py
-----------------
Fetches live price data (via yfinance) and recent news headlines
(via NewsAPI) for any tracked ticker — used for both stocks and ETFs,
since both are tracked the same way now (tickers, not ISINs/CSV rows).
"""

import datetime
import os

import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
NEWSAPI_URL = "https://newsapi.org/v2/everything"

DEFAULT_STOCK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "JNJ",
]

# Well-known, liquid ETFs across regions/asset classes — used as the
# default tracked list now that ETF data comes live from yfinance
# instead of justetf.csv.
DEFAULT_ETF_TICKERS = [
    "SPY", "VOO", "QQQ", "VTI", "IWDA.L", "VWCE.DE",
]


def fetch_live_quote(ticker: str) -> dict | None:
    """
    Fetches a current snapshot for one ticker via yfinance.
    Works identically for stocks and ETFs — yfinance doesn't distinguish.
    Returns None if data is unavailable (bad ticker, network issue, etc.)
    """
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        hist = t.history(period="5d")
        if hist.empty:
            return None

        last_close = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_close
        change_pct = ((last_close - prev_close) / prev_close * 100) if prev_close else 0.0

        return {
            "ticker": ticker,
            "price": round(last_close, 2),
            "change_pct": round(change_pct, 2),
            "day_high": round(float(info.get("day_high", last_close)), 2),
            "day_low": round(float(info.get("day_low", last_close)), 2),
            "fifty_two_week_high": round(float(info.get("year_high", 0) or 0), 2),
            "fifty_two_week_low": round(float(info.get("year_low", 0) or 0), 2),
            "market_cap": info.get("market_cap"),
            "fetched_at": datetime.datetime.utcnow().isoformat(),
        }
    except Exception as e:
        print(f"⚠️ yfinance fetch failed for {ticker}: {e}")
        return None


def fetch_company_name(ticker: str) -> str:
    """Best-effort lookup of a readable name for a ticker (works for ETFs too)."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker


def fetch_news_for_ticker(ticker: str, company_name: str = None, page_size: int = 5) -> list[dict]:
    """Fetches recent news articles mentioning the ticker/fund via NewsAPI."""
    if not NEWSAPI_KEY:
        print("⚠️ NEWSAPI_KEY not set — skipping news fetch.")
        return []

    query = company_name or ticker
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY,
    }

    try:
        resp = requests.get(NEWSAPI_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for a in data.get("articles", []):
            articles.append({
                "ticker": ticker,
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "source": (a.get("source") or {}).get("name", "unknown"),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            })
        return articles
    except requests.exceptions.RequestException as e:
        print(f"⚠️ NewsAPI fetch failed for {ticker}: {e}")
        return []
