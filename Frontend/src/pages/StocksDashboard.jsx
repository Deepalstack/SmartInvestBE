import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";
import LedgerTable from "../components/LedgerTable.jsx";
import RecommendationPanel from "../components/RecommendationPanel.jsx";

export default function StocksDashboard() {
  const [quotes, setQuotes] = useState(null);
  const [result, setResult] = useState(null);
  const [newTicker, setNewTicker] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function loadQuotes() {
    try {
      const data = await api.stockQuotes();
      setQuotes(data.quotes);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadQuotes();
  }, []);

  async function handleRecommend(ticker) {
    setError("");
    try {
      const data = await api.stockRecommend(ticker);
      setResult(data);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleTrack(e) {
    e.preventDefault();
    if (!newTicker.trim()) return;
    try {
      await api.stockTrack(newTicker.trim());
      setNewTicker("");
      loadQuotes();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div>
      <div className="page-header">
        <div className="page-eyebrow">Live data · yfinance</div>
        <h1 className="page-title">Stock Ledger</h1>
        <p className="page-note">
          Live stock prices, refreshed hourly. Select a ticker to generate recommendations
          from two independent models.
        </p>
      </div>

      {error && <div className="error-text" style={{ marginBottom: 16 }}>{error}</div>}
      {loading ? (
        <div className="loading-text">Loading ledger…</div>
      ) : (
        <LedgerTable quotes={quotes} onSelect={handleRecommend} />
      )}

      <RecommendationPanel result={result} />

      <form className="inline-form" onSubmit={handleTrack}>
        <input
          placeholder="ADD STOCK TICKER (e.g. NFLX)"
          value={newTicker}
          onChange={(e) => setNewTicker(e.target.value)}
        />
        <button type="submit">Track</button>
      </form>
    </div>
  );
}
