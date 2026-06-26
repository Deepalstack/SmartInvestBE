from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
import plotly.express as px
import plotly
import json
import joblib

# ------------------- Flask Setup -------------------
app = Flask(__name__)

# ------------------- Load Model & Data -------------------
DATA_FILE = "justetf.csv"
MODEL_FILE = "gpr_model.joblib"
SCALER_FILE = "scaler.joblib"

print("Loading data and model...")
df = pd.read_csv(DATA_FILE)
gpr = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

feature_cols = [
    'quote52Low', 'quote52High',
    'exposureCountry_Slovakia',
    'exposureCountry_Slovenia',
    'exposureCountry_Bulgaria'
]
target_col = 'ytdReturnCUR'

print(f"✅ Using all {len(df)} ETFs for plotting — no cleaning applied")

# ------------------- 1. Plot Scatter Chart (No Cleaning) -------------------
fig = px.scatter(
    df,
    x='fundSizeMillions',
    y='ytdReturnCUR',
    color='ter',
    hover_name='name' if 'name' in df.columns else None,
    hover_data={
        'isin': True,
        'fundProvider': True if 'fundProvider' in df.columns else False,
        'fundSizeMillions': True,
        'ytdReturnCUR': True,
        'ter': True
    },
    color_continuous_scale='Viridis',
    log_x=True,
    title='ETF Market Analysis: YTD Return vs Fund Size (All Data)',
    labels={
        'fundSizeMillions': 'Fund Size (Millions USD - Log Scale)',
        'ytdReturnCUR': 'Year-to-Date Return',
        'ter': 'Total Expense Ratio (TER)'
    }
)

fig.update_traces(marker=dict(size=6, opacity=0.85, line=dict(width=0.4, color='DarkSlateGrey')))
fig.update_layout(height=700, plot_bgcolor='white')
scatter_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

available_isins = sorted(df['isin'].dropna().unique())

# ------------------- 2. HTML Template -------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ETF Scatter Plot + Model Recommendation (Full Raw)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 font-sans">
    <div class="max-w-6xl mx-auto bg-white shadow-lg rounded-lg p-6 mt-10">
        <h1 class="text-3xl font-bold text-blue-800 mb-4 text-center">
            ETF Scatter Plot + Investment Recommendation (Full Raw Data)
        </h1>

        <div id="plot"></div>

        <form method="POST" class="text-center mt-10">
            <label class="text-lg font-semibold text-gray-700">
                Select ETF ISIN:
            </label><br>
            <select name="isin" id="isin"
                class="mt-3 p-2 w-80 border rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none" required>
                <option value="" disabled selected>Select an ISIN</option>
                {% for isin in isins %}
                <option value="{{ isin }}">{{ isin }}</option>
                {% endfor %}
            </select>
            <button type="submit"
                class="ml-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-semibold">
                Get Recommendation
            </button>
        </form>

        {% if result %}
        <div class="mt-8 p-4 bg-gray-50 border rounded-lg shadow-sm text-center">
            <h2 class="text-xl font-semibold text-gray-800 mb-2">
                Recommendation for {{ result['isin'] }}
            </h2>
            <p><strong>Predicted YTD Return:</strong> {{ result['predicted'] }} ± {{ result['ci'] }} (95% CI)</p>
            <p><strong>Certainty:</strong> {{ result['certainty'] }}%</p>
            <p><strong>Uncertainty:</strong> {{ result['uncertainty'] }}%</p>
            <p><strong>Recommendation:</strong>
                <span class="font-bold text-blue-700">{{ result['recommendation'] }}</span>
            </p>
        </div>
        {% endif %}
    </div>

    <script>
        const scatterData = {{ scatter_json | safe }};
        Plotly.newPlot('plot', scatterData.data, scatterData.layout);
    </script>
</body>
</html>
"""

# ------------------- 3. Flask Route -------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        user_isin = request.form.get("isin", "").strip()

        if user_isin not in available_isins:
            result = {
                "isin": user_isin,
                "predicted": "N/A",
                "ci": "N/A",
                "certainty": 0,
                "uncertainty": 0,
                "recommendation": "Invalid ISIN"
            }
        else:
            etf_data = df[df['isin'] == user_isin]

            for col in feature_cols:
                if col not in etf_data.columns:
                    etf_data[col] = 0.0

            latest_features = etf_data[feature_cols].ffill().tail(1).fillna(0).values
            latest_scaled = scaler.transform(latest_features)

            pred_mean, pred_std = gpr.predict(latest_scaled, return_std=True)
            y_all = df[target_col].ffill().values
            y_min, y_max = np.min(y_all), np.max(y_all)
            mean_norm = np.clip((pred_mean[0] - y_min) / (y_max - y_min), 0, 1)
            std_norm = np.clip(pred_std[0] / (y_max - y_min), 0, 1)

            certainty = mean_norm * (1 - std_norm)
            uncertainty = 1 - certainty

            certainty_pct = int(certainty * 100)
            uncertainty_pct = int(uncertainty * 100)

            if certainty > 0.7:
                rec = "Strong Buy"
            elif certainty > 0.4:
                rec = "Buy"
            elif certainty > 0.2:
                rec = "Hold"
            else:
                rec = "Sell"

            result = {
                "isin": user_isin,
                "predicted": f"{pred_mean[0]:.4f}",
                "ci": f"{1.96 * pred_std[0]:.4f}",
                "certainty": certainty_pct,
                "uncertainty": uncertainty_pct,
                "recommendation": rec
            }

    return render_template_string(HTML_TEMPLATE, scatter_json=scatter_json, result=result, isins=available_isins)

# ------------------- 4. Run Server -------------------
if __name__ == "__main__":
    print("🚀 Running ETF Scatter Plot + Dropdown ISIN Recommendation (Full Raw) → http://127.0.0.1:5000/")
    app.run(debug=True)
