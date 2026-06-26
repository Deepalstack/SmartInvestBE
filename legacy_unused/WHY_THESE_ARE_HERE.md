# Legacy / unused files

These files are **not used by the current app** (per the project's own
README). They're kept here for reference instead of being deleted —
delete this whole folder if you don't need it.

| File | Why it's legacy |
|---|---|
| `main.py`, `main2.py` | Earlier standalone Flask/script prototypes, superseded by `backend/app.py` |
| `gr.py`, `tr.py`, `tr2.py`, `col.py` | Early CSV-based model-training/exploration scripts, superseded by `backend/tr_stocks.py` + `backend/tr_etf.py` |
| `scatter.py` | Old Flask+Jinja (server-rendered HTML) version of the app, superseded by the JSON-API `backend/app.py` + React frontend |
| `db.py` | Standalone DB-inspection script; points at old table names (`users`, `activity_log`) that no longer match the current schema |
| `justetf.csv` | Old ETF dataset; ETFs are now fetched live via yfinance instead |
| `gpr_model.joblib`, `xgb_model.joblib`, `scaler.joblib` | Models trained on `justetf.csv`, no longer loaded by `app.py` |

Active equivalents live in `../backend/`:
`xgb_stock_model.joblib`, `gpr_stock_model.joblib`, `stock_scaler.joblib`.

Note: `backend/tr_etf.py` is meant to *produce* `xgb_etf_model.joblib`,
`gpr_etf_model.joblib`, and `etf_scaler.joblib` — those weren't present
in the uploaded zip, so you'll need to run `python tr_etf.py` once to
generate them before the ETF recommendation endpoint will work.
