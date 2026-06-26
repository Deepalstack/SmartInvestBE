import numpy as np
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load('xgb_model.joblib')  # Adjust filepath
scaler = joblib.load('scaler.joblib')

# Load dataset
data = pd.read_csv('justetf.csv')  # Adjust filepath

feature_cols = ['quote52Low', 'quote52High', 'exposureCountry_Slovakia', 'exposureCountry_Slovenia', 'exposureCountry_Bulgaria']
target_col = 'ytdReturnCUR'

available_symbols = sorted(data['isin'].unique())
print("Available ETF ISINs:")
print(", ".join(available_symbols))

while True:
    user_etf = input("\nEnter the ETF ISIN (or type 'exit' to quit): ").strip()
    if user_etf.lower() == 'exit':
        print("Exiting program. Goodbye!")
        break
    if user_etf not in available_symbols:
        print("Invalid ISIN entered. Please try again.")
        continue

    etf_data = data[data['isin'] == user_etf]
    latest_features = etf_data[feature_cols].ffill().tail(1).values
    latest_scaled = scaler.transform(latest_features)

    pred_mean = model.predict(latest_scaled)[0]

    y_all = data[target_col].ffill().values
    y_min, y_max = np.min(y_all), np.max(y_all)
    mean_norm = np.clip((pred_mean - y_min) / (y_max - y_min), 0, 1)
    
    certainty = mean_norm
    uncertainty = 1 - certainty

    certainty_pct = int(certainty * 100)
    uncertainty_pct = int(uncertainty * 100)

    print(f"\nETF '{user_etf}' Investment Certainty: {certainty_pct}%")
    
    print(f"Predicted YTD Return: {pred_mean:.4f}")

    if certainty > 0.7:
        print("Recommendation: Strong Buy")
    elif certainty > 0.4:
        print("Recommendation: Buy")
    elif certainty > 0.2:
        print("Recommendation: Hold")
    else:
        print("Recommendation: Sell")
