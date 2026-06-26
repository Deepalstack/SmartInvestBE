"""
scheduler.py
------------
Hourly background job that refreshes live quotes + news + sentiment for
BOTH tracked stocks and tracked ETFs (same mechanism for each now that
both use tickers via yfinance).
"""

import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from news_fetcher import fetch_company_name, fetch_live_quote, fetch_news_for_ticker
from news_sentiment import combined_sentiment, explain_impact

REFRESH_INTERVAL_MINUTES = 60


def _refresh_asset_class(db, QuoteModel, NewsModel, TrackedModel, default_tickers, asset_label):
    tickers = [t.ticker for t in TrackedModel.query.all()]
    if not tickers:
        tickers = default_tickers
        for tk in tickers:
            db.session.add(TrackedModel(ticker=tk))
        db.session.commit()

    print(f"🔄 [{datetime.datetime.utcnow().isoformat()}] Refreshing {len(tickers)} {asset_label}...")

    for ticker in tickers:
        quote = fetch_live_quote(ticker)
        if quote:
            db.session.add(QuoteModel(
                ticker=ticker,
                price=quote["price"],
                change_pct=quote["change_pct"],
                day_high=quote["day_high"],
                day_low=quote["day_low"],
                fifty_two_week_high=quote["fifty_two_week_high"],
                fifty_two_week_low=quote["fifty_two_week_low"],
                fetched_at=datetime.datetime.utcnow(),
            ))

        company_name = fetch_company_name(ticker)
        articles = fetch_news_for_ticker(ticker, company_name=company_name)
        for art in articles:
            text = f"{art['title']}. {art.get('description') or ''}"
            sentiment = combined_sentiment(text)
            explanation = explain_impact(art["title"], ticker, sentiment)

            db.session.add(NewsModel(
                ticker=ticker,
                title=art["title"],
                description=art.get("description", ""),
                source=art["source"],
                url=art["url"],
                published_at=art["published_at"],
                sentiment_label=sentiment["final_label"],
                sentiment_score=sentiment["final_score"],
                finbert_label=sentiment["finbert_label"],
                vader_label=sentiment["vader_label"],
                explanation=explanation,
                fetched_at=datetime.datetime.utcnow(),
            ))

    db.session.commit()


def run_refresh_job(app, db, models, defaults):
    """
    models / defaults are dicts keyed by 'stock' and 'etf', each containing
    the relevant SQLAlchemy model classes / default ticker list, so this one
    job refreshes both asset classes without duplicating logic.
    """
    with app.app_context():
        _refresh_asset_class(
            db, models["stock"]["quote"], models["stock"]["news"], models["stock"]["tracked"],
            defaults["stock"], "stocks",
        )
        _refresh_asset_class(
            db, models["etf"]["quote"], models["etf"]["news"], models["etf"]["tracked"],
            defaults["etf"], "ETFs",
        )
        print("✅ Refresh complete (stocks + ETFs).")


def start_scheduler(app, db, models, defaults):
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        func=lambda: run_refresh_job(app, db, models, defaults),
        trigger="interval",
        minutes=REFRESH_INTERVAL_MINUTES,
        id="refresh_job",
        next_run_time=datetime.datetime.now(),
    )
    scheduler.start()
    print(f"📅 Scheduler started — refreshing every {REFRESH_INTERVAL_MINUTES} minutes.")
    return scheduler
