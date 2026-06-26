import React from "react";

function stampClass(label) {
  if (!label) return "hold";
  const l = label.toLowerCase();
  if (l.includes("buy")) return "buy";
  if (l.includes("sell")) return "sell";
  return "hold";
}

function RecCard({ title, rec }) {
  if (!rec) {
    return (
      <div className="rec-card">
        <div className="rec-label">{title}</div>
        <div className="rec-sub">No data</div>
      </div>
    );
  }

  const pct = (rec.predicted_return * 100).toFixed(2);
  return (
    <div className="rec-card">
      <span className={`stamp ${stampClass(rec.label)}`}>{rec.label}</span>
      <div className="rec-label">{title}</div>
      <div className="rec-metric">{pct > 0 ? "+" : ""}{pct}%</div>
      <div className="rec-sub">expected move · {rec.certainty_pct}% certainty</div>
    </div>
  );
}

export default function RecommendationPanel({ result }) {
  if (!result) return null;

  if (result.model_status === "not_trained") {
    return (
      <div className="rec-panel">
        <div className="rec-label">{result.ticker}</div>
        <p className="page-note" style={{ marginTop: 8 }}>{result.message}</p>
      </div>
    );
  }

  return (
    <div className="rec-panel">
      <div className="rec-label">{result.ticker} — current price {result.price}</div>
      <div className="rec-grid">
        <RecCard title="Recommendation 1" rec={result.recommendation_1} />
        <RecCard title="Recommendation 2" rec={result.recommendation_2} />
      </div>
    </div>
  );
}
