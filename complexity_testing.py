import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf

# Load your dataset
df = pd.read_csv('processed_hourly_data.csv')

# Features and target
X = df[['Global_reactive_power', 'Voltage', 'Global_intensity', 
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
y = df['Global_active_power']

# 1. Correlation Matrix
print("Correlation Matrix:")
corr_matrix = X.corr()
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# 2. Feature Distribution
print("Feature Distributions:")
X.hist(bins=20, figsize=(12, 8))
plt.suptitle('Feature Distributions')
plt.show()

# 3. Autocorrelation Plot for Time Series Data (y)
print("Autocorrelation of Target Variable:")
plot_acf(y, lags=50)
plt.show()

# 4. Baseline Model Performance
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression MSE: {mse_lr:.4f}, R^2: {r2_lr:.4f}")

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"Decision Tree MSE: {mse_dt:.4f}, R^2: {r2_dt:.4f}")

# 5. Cross-Validation Performance
cv_scores_lr = cross_val_score(lr, X, y, cv=5, scoring='neg_mean_squared_error')
cv_scores_dt = cross_val_score(dt, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Linear Regression CV MSE: {-cv_scores_lr.mean():.4f}")
print(f"Decision Tree CV MSE: {-cv_scores_dt.mean():.4f}")

# Interpretation:
# 1. If the correlation matrix shows high correlation between features, this may indicate redundancy.
# 2. Non-normal distributions or skewed histograms indicate potential complexity in data.
# 3. High autocorrelation suggests temporal dependencies, adding complexity.
# 4. Poor performance in linear regression vs decision tree may suggest non-linear relationships.
# 5. If cross-validation scores vary significantly, it indicates instability in model performance, pointing to complexity.
