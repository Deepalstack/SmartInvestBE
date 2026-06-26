"""
app.py
------
SmartInvest backend — pure JSON REST API (no HTML templates).
Frontend is a separate React app that calls these endpoints.

Auth: JWT (flask-jwt-extended). Login/register return a token; all other
endpoints require it in the Authorization: Bearer <token> header.

Recommendations are intentionally generic in the API response — fields
are named "recommendation_1" (XGBoost-based) and "recommendation_2"
(GPR-based) rather than naming the algorithm, so the frontend can label
them neutrally (e.g. "Recommendation 1" / "Recommendation 2") without
ever surfacing which model produced which.

Run:
    pip install -r requirements.txt
    python tr_stocks.py   # one-time: train stock models
    python tr_etf.py       # one-time: train ETF models
    python app.py
"""

import datetime
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, get_jwt_identity, jwt_required,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

from features import compute_live_features
from news_fetcher import DEFAULT_ETF_TICKERS, DEFAULT_STOCK_TICKERS, fetch_live_quote
from scheduler import start_scheduler

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-this-in-production")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = datetime.timedelta(days=7)

db = SQLAlchemy(app)
jwt = JWTManager(app)
CORS(app)  # allow the React dev server (different origin) to call this API


# ---------------- Database Models ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class ActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    asset_type = db.Column(db.String(10))  # 'etf' or 'stock'
    ticker = db.Column(db.String(20))
    prediction = db.Column(db.String(50))
    certainty = db.Column(db.String(50))
    recommendation = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)


# --- Stock tables ---
class StockTracked(db.Model):
    __tablename__ = "stock_tracked"
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), unique=True, nullable=False)
    added_by = db.Column(db.String(80))
    added_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)


class StockQuote(db.Model):
    __tablename__ = "stock_quote"
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), nullable=False)
    price = db.Column(db.Float)
    change_pct = db.Column(db.Float)
    day_high = db.Column(db.Float)
    day_low = db.Column(db.Float)
    fifty_two_week_high = db.Column(db.Float)
    fifty_two_week_low = db.Column(db.Float)
    fetched_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)


class StockNews(db.Model):
    __tablename__ = "stock_news"
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), nullable=False)
    title = db.Column(db.String(500))
    description = db.Column(db.Text)
    source = db.Column(db.String(120))
    url = db.Column(db.String(500))
    published_at = db.Column(db.String(50))
    sentiment_label = db.Column(db.String(20))
    sentiment_score = db.Column(db.Float)
    finbert_label = db.Column(db.String(20))
    vader_label = db.Column(db.String(20))
    explanation = db.Column(db.Text)
    fetched_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)


# --- ETF tables (identical shape, separate tables) ---
class EtfTracked(db.Model):
    __tablename__ = "etf_tracked"
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), unique=True, nullable=False)
    added_by = db.Column(db.String(80))
    added_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)


class EtfQuote(db.Model):
    __tablename__ = "etf_quote"
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), nullable=False)
    price = db.Column(db.Float)
    change_pct = db.Column(db.Float)
    day_high = db.Column(db.Float)
    day_low = db.Column(db.Float)
    fifty_two_week_high = db.Column(db.Float)
    fifty_two_week_low = db.Column(db.Float)
    fetched_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)


class EtfNews(db.Model):
    __tablename__ = "etf_news"
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), nullable=False)
    title = db.Column(db.String(500))
    description = db.Column(db.Text)
    source = db.Column(db.String(120))
    url = db.Column(db.String(500))
    published_at = db.Column(db.String(50))
    sentiment_label = db.Column(db.String(20))
    sentiment_score = db.Column(db.Float)
    finbert_label = db.Column(db.String(20))
    vader_label = db.Column(db.String(20))
    explanation = db.Column(db.Text)
    fetched_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

with app.app_context():
    db.create_all()

MODELS = {
    "stock": {"tracked": StockTracked, "quote": StockQuote, "news": StockNews},
    "etf": {"tracked": EtfTracked, "quote": EtfQuote, "news": EtfNews},
}
DEFAULTS = {"stock": DEFAULT_STOCK_TICKERS, "etf": DEFAULT_ETF_TICKERS}


# ---------------- Load ML Models ----------------
def _load_model_set(prefix: str):
    try:
        xgb_model = joblib.load(f"xgb_{prefix}_model.joblib")
        gpr_model = joblib.load(f"gpr_{prefix}_model.joblib")
        scaler = joblib.load(f"{prefix}_scaler.joblib")
        print(f"✅ {prefix} models loaded.")
        return {"xgb": xgb_model, "gpr": gpr_model, "scaler": scaler, "loaded": True}
    except FileNotFoundError:
        print(f"⚠️ {prefix} model files not found — run tr_{prefix}.py first.")
        return {"xgb": None, "gpr": None, "scaler": None, "loaded": False}


ML_MODELS = {"stock": _load_model_set("stock"), "etf": _load_model_set("etf")}


def predict_recommendation(ticker: str, asset_type: str) -> dict | None:
    """
    Runs both trained models (XGBoost + GPR) on live features for `ticker`.
    Returns generically-named fields (recommendation_1 / recommendation_2)
    so the API never exposes which algorithm produced which number.
    """
    model_set = ML_MODELS[asset_type]
    if not model_set["loaded"]:
        return None

    features = compute_live_features(ticker)
    if features is None:
        return None

    features_scaled = model_set["scaler"].transform(features)

    xgb_pred = float(model_set["xgb"].predict(features_scaled)[0])
    gpr_pred_mean, gpr_pred_std = model_set["gpr"].predict(features_scaled, return_std=True)
    gpr_pred_mean = float(gpr_pred_mean[0])
    gpr_pred_std = float(gpr_pred_std[0])

    def verdict(predicted_return, uncertainty=None):
        if uncertainty is not None:
            certainty = float(np.clip(1 - uncertainty * 5, 0, 1))
        else:
            certainty = float(np.clip(0.5 + predicted_return * 5, 0, 1))
        if predicted_return > 0.03 and certainty > 0.6:
            label = "Strong Buy"
        elif predicted_return > 0.01:
            label = "Buy"
        elif predicted_return > -0.01:
            label = "Hold"
        else:
            label = "Sell"
        return {"predicted_return": round(predicted_return, 4), "certainty_pct": int(certainty * 100), "label": label}

    rec_1 = verdict(xgb_pred)                       # XGBoost-based, generically named
    rec_2 = verdict(gpr_pred_mean, gpr_pred_std)     # GPR-based, generically named

    return {
        "recommendation_1": rec_1,
        "recommendation_2": rec_2,
    }


# ---------------- Auth Routes ----------------
@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 409

    user = User(username=username, password=generate_password_hash(password))
    db.session.add(user)
    db.session.commit()

    token = create_access_token(identity=username)
    return jsonify({"token": token, "username": username}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({"error": "Invalid username or password"}), 401

    token = create_access_token(identity=username)
    return jsonify({"token": token, "username": username}), 200


# ---------------- Generic asset endpoints (ETF + Stock share logic) ----------------
def _list_quotes(asset_type: str):
    model = MODELS[asset_type]
    tickers = [t.ticker for t in model["tracked"].query.all()] or DEFAULTS[asset_type]
    quotes = []
    for tk in tickers:
        latest = (model["quote"].query
                  .filter_by(ticker=tk)
                  .order_by(model["quote"].fetched_at.desc())
                  .first())
        if latest:
            quotes.append({
                "ticker": latest.ticker,
                "price": latest.price,
                "change_pct": latest.change_pct,
                "day_high": latest.day_high,
                "day_low": latest.day_low,
                "fetched_at": latest.fetched_at.isoformat() if latest.fetched_at else None,
            })
    return tickers, quotes


@app.route("/api/etf/quotes", methods=["GET"])
@jwt_required()
def etf_quotes():
    tickers, quotes = _list_quotes("etf")
    return jsonify({"tickers": tickers, "quotes": quotes})


@app.route("/api/stocks/quotes", methods=["GET"])
@jwt_required()
def stock_quotes():
    tickers, quotes = _list_quotes("stock")
    return jsonify({"tickers": tickers, "quotes": quotes})


def _recommend(asset_type: str):
    data = request.get_json(silent=True) or {}
    ticker = (data.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400

    model = MODELS[asset_type]
    latest = (model["quote"].query
              .filter_by(ticker=ticker)
              .order_by(model["quote"].fetched_at.desc())
              .first())
    price = latest.price if latest else None
    change_pct = latest.change_pct if latest else None

    if price is None:
        live = fetch_live_quote(ticker)
        if live:
            price = live["price"]
            change_pct = live["change_pct"]

    prediction = predict_recommendation(ticker, asset_type)

    username = get_jwt_identity()

    if prediction:
        result = {
            "ticker": ticker,
            "price": price,
            "change_pct": change_pct,
            **prediction,
            "model_status": "trained",
        }
        log_rec = f"{result['recommendation_1']['label']} / {result['recommendation_2']['label']}"
    elif price is not None:
        result = {
            "ticker": ticker, "price": price, "change_pct": change_pct,
            "recommendation_1": None, "recommendation_2": None,
            "model_status": "not_trained",
            "message": f"Run tr_{asset_type}.py to enable recommendations for this asset class.",
        }
        log_rec = "model not trained"
    else:
        return jsonify({"error": f"No data available for ticker '{ticker}'"}), 404

    db.session.add(ActivityLog(
        username=username, asset_type=asset_type, ticker=ticker,
        prediction=str(price), certainty="-", recommendation=log_rec,
    ))
    db.session.commit()

    return jsonify(result)


@app.route("/api/etf/recommend", methods=["POST"])
@jwt_required()
def etf_recommend():
    return _recommend("etf")


@app.route("/api/stocks/recommend", methods=["POST"])
@jwt_required()
def stock_recommend():
    return _recommend("stock")


def _add_tracked(asset_type: str):
    data = request.get_json(silent=True) or {}
    new_ticker = (data.get("ticker") or "").strip().upper()
    if not new_ticker:
        return jsonify({"error": "ticker is required"}), 400

    model = MODELS[asset_type]
    username = get_jwt_identity()

    if not model["tracked"].query.filter_by(ticker=new_ticker).first():
        db.session.add(model["tracked"](ticker=new_ticker, added_by=username))
        db.session.commit()
        live = fetch_live_quote(new_ticker)
        if live:
            db.session.add(model["quote"](
                ticker=new_ticker, price=live["price"], change_pct=live["change_pct"],
                day_high=live["day_high"], day_low=live["day_low"],
                fifty_two_week_high=live["fifty_two_week_high"],
                fifty_two_week_low=live["fifty_two_week_low"],
                fetched_at=datetime.datetime.utcnow(),
            ))
            db.session.commit()

    tickers = [t.ticker for t in model["tracked"].query.all()]
    return jsonify({"tickers": tickers}), 201


@app.route("/api/etf/track", methods=["POST"])
@jwt_required()
def etf_track():
    return _add_tracked("etf")


@app.route("/api/stocks/track", methods=["POST"])
@jwt_required()
def stock_track():
    return _add_tracked("stock")


# ---------------- News & Sentiment ----------------
def _list_news(asset_type: str):
    model = MODELS[asset_type]
    tickers = [t.ticker for t in model["tracked"].query.all()] or DEFAULTS[asset_type]
    selected = (request.args.get("ticker") or "").strip().upper()

    query = model["news"].query
    if selected:
        query = query.filter_by(ticker=selected)
    articles = query.order_by(model["news"].fetched_at.desc()).limit(50).all()

    return jsonify({
        "tickers": tickers,
        "selected_ticker": selected,
        "has_news_key": bool(os.environ.get("NEWSAPI_KEY")),
        "articles": [
            {
                "ticker": a.ticker,
                "title": a.title,
                "description": a.description,
                "source": a.source,
                "url": a.url,
                "published_at": a.published_at,
                "sentiment_label": a.sentiment_label,
                "explanation": a.explanation,
            }
            for a in articles
        ],
    })


@app.route("/api/etf/news", methods=["GET"])
@jwt_required()
def etf_news():
    return _list_news("etf")


@app.route("/api/stocks/news", methods=["GET"])
@jwt_required()
def stock_news():
    return _list_news("stock")


# ---------------- Run Server ----------------
if __name__ == "__main__":
    print("🚀 SmartInvest API running on http://127.0.0.1:5000/")
    start_scheduler(app, db, MODELS, DEFAULTS)
    app.run(debug=True, use_reloader=False)
