import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
data = pd.read_csv('justetf.csv')
print(data.head())
print(data.info())
print(data.describe())
# Feature columns (example)
feature_cols = ['quote52Low', 'quote52High', 'someOtherPriceFeature', 'exposureCountry_Slovakia', 'exposureCountry_Slovenia', 'exposureCountry_Bulgaria']
# Select feature columns from your data
X = data[['quote52Low', 'quote52High', 'exposureCountry_Slovakia', 'exposureCountry_Slovenia', 'exposureCountry_Bulgaria']]  # example columns

# Then you can do missing value filling
  # Forward fill missing values
# Target column
target_col = 'ytdReturnCUR'
X = X.ffill()  # forward fill missing values
y = data[target_col].ffill()



scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Define kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(2.0, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

# Train the model
gpr.fit(X_train, y_train)
pred_mean, pred_std = gpr.predict(X_test, return_std=True)


plt.plot(y_test.values, label='Actual')
plt.plot(pred_mean, label='Predicted')
plt.fill_between(range(len(pred_mean)),
                 pred_mean - 1.96 * pred_std,
                 pred_mean + 1.96 * pred_std,
                 color='orange', alpha=0.3, label='Confidence Interval')
plt.legend()
plt.show()


# Train as before, then:
joblib.dump(gpr, 'gpr_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
