import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import xgboost as xgb

# Load dataset
data = pd.read_csv('justetf.csv')  # Replace with actual CSV filename

# Features and target columns
feature_cols = ['quote52Low', 'quote52High', 
                'exposureCountry_Slovakia', 
                'exposureCountry_Slovenia', 
                'exposureCountry_Bulgaria']
target_col = 'ytdReturnCUR'

# Handle missing values
X = data[feature_cols].ffill().values
y = data[target_col].ffill().values

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Prepare and train XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_scaled, y)

# Save the trained model and scaler
joblib.dump(model, 'xgb_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Load saved model and scaler for reuse
model = joblib.load('xgb_model.joblib')
scaler = joblib.load('scaler.joblib')

# Display available ETF ISIN codes for user
available_symbols = sorted(data['isin'].unique())
print("Available ETF ISIN codes:")
print(", ".join(available_symbols))

# Loop to ask for valid ISIN input
while True:
    user_etf = input("\nEnter the ETF ISIN you want to invest in (from the above list): ").strip()
    if user_etf in available_symbols:
        break
    else:
        print("Invalid ISIN entered. Please try again.")

# Filter ETF data for user's choice
etf_data = data[data['isin'] == user_etf]

# Get latest features for this ETF
latest_features = etf_data[feature_cols].ffill().tail(1).values
latest_scaled = scaler.transform(latest_features)

# Predict expected return (no uncertainty from XGB)
pred_mean = model.predict(latest_scaled)[0]

# Normalize mean relative to historical returns
y_min, y_max = np.min(y), np.max(y)
mean_norm = (pred_mean - y_min) / (y_max - y_min)
mean_norm = np.clip(mean_norm, 0, 1)

certainty = mean_norm
uncertainty = 1 - certainty

certainty_pct = int(certainty * 100)
uncertainty_pct = int(uncertainty * 100)

# Display results
print(f"\nFor ETF '{user_etf}':")
print(f"Certainty (recommended investment %): {certainty_pct}%")
print(f"Uncertainty (recommended NOT to invest %): {uncertainty_pct}%")
print(f"Predicted YTD Return: {pred_mean:.4f}")
