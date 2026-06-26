import React from "react";

export default function LedgerTable({ quotes, onSelect }) {
  if (!quotes || quotes.length === 0) {
    return <div className="empty-state">No quotes yet. The hourly refresh job populates this — check back shortly.</div>;
  }

  return (
    <div className="ledger">
      <div className="ledger-row header">
        <div>Ticker</div>
        <div></div>
        <div className="ledger-cell">Price</div>
        <div className="ledger-cell">Change</div>
        <div className="ledger-cell">Day high</div>
        <div></div>
      </div>
      {quotes.map((q) => (
        <div className="ledger-row" key={q.ticker}>
          <div className="ledger-ticker">{q.ticker}</div>
          <div></div>
          <div className="ledger-cell">{q.price}</div>
          <div className={`ledger-cell ledger-change ${q.change_pct >= 0 ? "up" : "down"}`}>
            {q.change_pct >= 0 ? "+" : ""}{q.change_pct}%
          </div>
          <div className="ledger-cell">{q.day_high}</div>
          <div className="ledger-action">
            <button onClick={() => onSelect(q.ticker)}>Recommend</button>
          </div>
        </div>
      ))}
    </div>
  );
}
