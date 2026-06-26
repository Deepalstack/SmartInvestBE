from flask import Flask, render_template_string, request, redirect, url_for, session, g
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import plotly.express as px
import plotly
import json
import joblib
import datetime

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
db = SQLAlchemy(app)

# ---------------- Database Models ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class ActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    isin = db.Column(db.String(50))
    prediction = db.Column(db.String(50))
    certainty = db.Column(db.String(50))
    recommendation = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Initialize database
with app.app_context():
    db.create_all()

# ---------------- Load Model & Data ----------------
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

# ---------------- Scatter Plot (Raw) ----------------
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
)
fig.update_traces(marker=dict(size=6, opacity=0.85, line=dict(width=0.4, color='DarkSlateGrey')))
fig.update_layout(height=700, plot_bgcolor='white')
scatter_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

available_isins = sorted(df['isin'].dropna().unique())

# ---------------- HTML Templates ----------------
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Login - SmartInvest</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 flex items-center justify-center h-screen">
    <div class="bg-white p-8 rounded-lg shadow-md w-96">
        <h2 class="text-2xl font-bold mb-6 text-center text-blue-700">Login to SmartInvest</h2>
        <form method="POST">
            <input name="username" placeholder="Username" required class="w-full p-2 border rounded mb-4">
            <input name="password" type="password" placeholder="Password" required class="w-full p-2 border rounded mb-4">
            <button class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Login</button>
        </form>
        {% if error %}<p class="text-red-600 mt-3 text-center">{{ error }}</p>{% endif %}
        <p class="text-sm text-gray-600 mt-4 text-center">New here? <a href="/register" class="text-blue-600 font-semibold">Register</a></p>
    </div>
</body>
</html>
"""

REGISTER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Register - SmartInvest</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 flex items-center justify-center h-screen">
    <div class="bg-white p-8 rounded-lg shadow-md w-96">
        <h2 class="text-2xl font-bold mb-6 text-center text-blue-700">Create Account</h2>
        <form method="POST">
            <input name="username" placeholder="Username" required class="w-full p-2 border rounded mb-4">
            <input name="password" type="password" placeholder="Password" required class="w-full p-2 border rounded mb-4">
            <button class="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">Register</button>
        </form>
        {% if error %}<p class="text-red-600 mt-3 text-center">{{ error }}</p>{% endif %}
        <p class="text-sm text-gray-600 mt-4 text-center">Already have an account? <a href="/login" class="text-blue-600 font-semibold">Login</a></p>
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SmartInvest Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 font-sans">
    <div class="max-w-6xl mx-auto bg-white shadow-lg rounded-lg p-6 mt-10">
        <div class="flex justify-between items-center mb-4">
            <h1 class="text-3xl font-bold text-blue-800">SmartInvest Dashboard</h1>
            <a href="/logout" class="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded">Logout</a>
        </div>
        <div id="plot"></div>

        <form method="POST" class="text-center mt-10">
            <label class="text-lg font-semibold text-gray-700">Select ETF ISIN:</label><br>
            <select name="isin" class="mt-3 p-2 w-80 border rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none" required>
                <option value="" disabled selected>Select an ISIN</option>
                {% for isin in isins %}
                <option value="{{ isin }}">{{ isin }}</option>
                {% endfor %}
            </select>
            <button type="submit" class="ml-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-semibold">
                Get Recommendation
            </button>
        </form>

        {% if result %}
        <div class="mt-8 p-4 bg-gray-50 border rounded-lg shadow-sm text-center">
            <h2 class="text-xl font-semibold text-gray-800 mb-2">Recommendation for {{ result['isin'] }}</h2>
            <p><strong>Predicted YTD Return:</strong> {{ result['predicted'] }} ± {{ result['ci'] }} (95% CI)</p>
            <p><strong>Certainty:</strong> {{ result['certainty'] }}%</p>
            <p><strong>Recommendation:</strong> {{ result['recommendation'] }}</p>
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

# ---------------- Routes ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        if User.query.filter_by(username=username).first():
            return render_template_string(REGISTER_TEMPLATE, error="Username already exists")

        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template_string(REGISTER_TEMPLATE)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session["username"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template_string(LOGIN_TEMPLATE, error="Invalid username or password")

    return render_template_string(LOGIN_TEMPLATE)

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

    result = None
    if request.method == "POST":
        user_isin = request.form["isin"]
        etf_data = df[df['isin'] == user_isin]
        latest_features = etf_data[feature_cols].ffill().tail(1).fillna(0).values
        latest_scaled = scaler.transform(latest_features)
        pred_mean, pred_std = gpr.predict(latest_scaled, return_std=True)
        y_all = df[target_col].ffill().values
        y_min, y_max = np.min(y_all), np.max(y_all)
        mean_norm = np.clip((pred_mean[0] - y_min) / (y_max - y_min), 0, 1)
        std_norm = np.clip(pred_std[0] / (y_max - y_min), 0, 1)
        certainty = mean_norm * (1 - std_norm)
        certainty_pct = int(certainty * 100)
        rec = "Strong Buy" if certainty > 0.7 else "Buy" if certainty > 0.4 else "Hold" if certainty > 0.2 else "Sell"
        result = {"isin": user_isin, "predicted": f"{pred_mean[0]:.4f}", "ci": f"{1.96 * pred_std[0]:.4f}", "certainty": certainty_pct, "recommendation": rec}
        log = ActivityLog(username=session["username"], isin=user_isin, prediction=result["predicted"], certainty=str(certainty_pct), recommendation=rec)
        db.session.add(log)
        db.session.commit()

    return render_template_string(DASHBOARD_TEMPLATE, scatter_json=scatter_json, isins=available_isins, result=result)

# ---------------- Run Server ----------------
if __name__ == "__main__":
    print("🚀 SmartInvest running on http://127.0.0.1:5000/")
    app.run(debug=True)
