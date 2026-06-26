"""
news_sentiment.py
------------------
Combines FinBERT (transformer-based financial sentiment) with VADER
(lightweight rule-based sentiment) into a single verdict per headline,
plus a short human-readable "why" explanation.

FinBERT is loaded ONCE (lazily) and reused — loading the model is the
expensive part, not running inference. Intended to be called from the
hourly background job, not per-request.
"""

from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()


def vader_sentiment(text: str) -> dict:
    scores = _vader.polarity_scores(text or "")
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "score": compound}


@lru_cache(maxsize=1)
def _get_finbert_pipeline():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")


def finbert_sentiment(text: str) -> dict:
    if not text:
        return {"label": "neutral", "score": 0.0}
    clf = _get_finbert_pipeline()
    result = clf(text[:512])[0]
    label = result["label"].lower()
    score = float(result["score"])
    signed = score if label == "positive" else (-score if label == "negative" else 0.0)
    return {"label": label, "score": signed, "raw_confidence": score}


def combined_sentiment(text: str) -> dict:
    fb = finbert_sentiment(text)
    vd = vader_sentiment(text)
    avg_score = (fb["score"] + vd["score"]) / 2.0

    agree = (
        (fb["label"] == "positive" and vd["label"] == "positive")
        or (fb["label"] == "negative" and vd["label"] == "negative")
        or (fb["label"] == "neutral" and vd["label"] == "neutral")
    )

    if not agree and fb["label"] != vd["label"]:
        final_label = "mixed"
    elif avg_score >= 0.05:
        final_label = "positive"
    elif avg_score <= -0.05:
        final_label = "negative"
    else:
        final_label = "neutral"

    return {
        "final_label": final_label,
        "final_score": round(avg_score, 4),
        "finbert_label": fb["label"],
        "finbert_score": round(fb["score"], 4),
        "vader_label": vd["label"],
        "vader_score": round(vd["score"], 4),
        "agree": agree,
    }


def explain_impact(headline: str, ticker: str, sentiment: dict) -> str:
    label = sentiment["final_label"]
    if label == "positive":
        verb = "may support a rise in"
    elif label == "negative":
        verb = "may put pressure on"
    elif label == "mixed":
        verb = "shows mixed signals for"
    else:
        verb = "appears neutral for"

    confidence_note = "" if sentiment["agree"] else " (models disagreed — treat with caution)"
    return f'Headline "{headline.strip()}" {verb} {ticker}{confidence_note}.'
