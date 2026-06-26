import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

export default function News({ assetType }) {
  const [tickers, setTickers] = useState([]);
  const [selected, setSelected] = useState("");
  const [articles, setArticles] = useState([]);
  const [hasNewsKey, setHasNewsKey] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const fetchNews = assetType === "etf" ? api.etfNews : api.stockNews;

  async function load(ticker) {
    setLoading(true);
    setError("");
    try {
      const data = await fetchNews(ticker);
      setTickers(data.tickers);
      setArticles(data.articles);
      setHasNewsKey(data.has_news_key);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load(selected);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [assetType, selected]);

  return (
    <div>
      <div className="page-header">
        <div className="page-eyebrow">FinBERT + VADER · combined sentiment</div>
        <h1 className="page-title">News — {assetType === "etf" ? "ETFs" : "Stocks"}</h1>
        <p className="page-note">
          Headlines refreshed hourly, each scored by two independent sentiment models.
          {!hasNewsKey && (
            <span style={{ color: "var(--signal-sell)", display: "block", marginTop: 6 }}>
              No news API key configured on the backend — this feed will stay empty until one is added.
            </span>
          )}
        </p>
      </div>

      <div className="news-filter">
        <select value={selected} onChange={(e) => setSelected(e.target.value)}>
          <option value="">All tracked tickers</option>
          {tickers.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>

      {error && <div className="error-text" style={{ marginBottom: 16 }}>{error}</div>}

      {loading ? (
        <div className="loading-text">Loading headlines…</div>
      ) : articles.length === 0 ? (
        <div className="empty-state">
          No news yet for this selection. The hourly refresh job will populate this automatically.
        </div>
      ) : (
        articles.map((a, i) => (
          <div className={`news-card ${a.sentiment_label}`} key={i}>
            <div className="news-card-head">
              <a className="news-title" href={a.url} target="_blank" rel="noreferrer">{a.title}</a>
              <span className="news-tag">{a.ticker} · {a.sentiment_label}</span>
            </div>
            <p className="news-desc">{a.description}</p>
            <div className="news-meta">{a.source} — {a.published_at}</div>
            <p className="news-explanation">{a.explanation}</p>
          </div>
        ))
      )}
    </div>
  );
}
