import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('justetf.csv')
print(data.head())
print(data.info())
print(data.describe())

# Feature columns
feature_cols = ['quote52Low', 'quote52High', 'exposureCountry_Slovakia', 'exposureCountry_Slovenia', 'exposureCountry_Bulgaria']

# Select features and target
X = data[feature_cols].ffill()
y = data['ytdReturnCUR'].ffill()

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data (no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Initialize and train XGBoost regressor
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Predict on test set
preds = model.predict(X_test)

# Calculate and print MSE for evaluation
mse = mean_squared_error(y_test, preds)
print(f'Mean Squared Error: {mse}')

# Plot actual vs predicted
plt.plot(y_test.values, label='Actual')
plt.plot(preds, label='Predicted')
plt.legend()
plt.show()

# Save model and scaler
joblib.dump(model, 'xgb_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
