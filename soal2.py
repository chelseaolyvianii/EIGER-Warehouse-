import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# POLYNOMIAL DEGREE 2
try:
    df = pd.read_csv("MATH6192031_AoL Data(in).csv")
    df.columns = ['month_year_str', 'y']
    df['x'] = np.arange(1, len(df) + 1)
except FileNotFoundError:
    print("Error: File 'MATH6192031_AoL Data(in).csv' tidak ditemukan.")
    exit()

train_df = df.iloc[:114]
test_df = df.iloc[114:]

X_train = train_df[['x']]
y_train = train_df['y']
X_test = test_df[['x']]
y_test = test_df['y']

print(f"Training Data: {len(train_df)}")
print(f"Testing Data: {len(test_df)}")

poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train_2 = poly_features_2.fit_transform(X_train)
model_2 = LinearRegression()
model_2.fit(X_poly_train_2, y_train)

X_poly_test_2 = poly_features_2.transform(X_test)
y_test_pred_2 = model_2.predict(X_poly_test_2)
mae_2_test = mean_absolute_error(y_test, y_test_pred_2)
mse_2_test = mean_squared_error(y_test, y_test_pred_2)
rmse_2_test = np.sqrt(mse_2_test)
r2_2_test = r2_score(y_test, y_test_pred_2)

print("\nEvaluation Metrics on Testing Data (Degree 2):")
print(f"Mean Absolute Error (MAE): {mae_2_test:.4f}")
print(f"Mean Squared Error (MSE): {mse_2_test:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_2_test:.4f}")
print(f"R-squared (R2): {r2_2_test:.4f}")

plt.figure(figsize=(12, 7))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data') 

x_plot = np.linspace(df['x'].min(), df['x'].max(), 400).reshape(-1, 1)
X_plot_poly_2 = poly_features_2.transform(x_plot)
y_plot_2 = model_2.predict(X_plot_poly_2)
plt.plot(x_plot, y_plot_2, color='red', linewidth=2, label='Polynomial Degree 2')

plt.title('Polynomial Degree 2 - Testing Data')
plt.xlabel('Month')
plt.ylabel('Number of Bags')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# POLYNOMIAL DEGREE 3
poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly_train_3 = poly_features_3.fit_transform(X_train)
model_3 = LinearRegression()
model_3.fit(X_poly_train_3, y_train)

X_poly_test_3 = poly_features_3.transform(X_test)
y_test_pred_3 = model_3.predict(X_poly_test_3)
mae_3_test = mean_absolute_error(y_test, y_test_pred_3)
mse_3_test = mean_squared_error(y_test, y_test_pred_3)
rmse_3_test = np.sqrt(mse_3_test)
r2_3_test = r2_score(y_test, y_test_pred_3)

print("\nEvaluation Metrics on Testing Data (Degree 3):")
print(f"Mean Absolute Error (MAE): {mae_3_test:.4f}")
print(f"Mean Squared Error (MSE): {mse_3_test:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_3_test:.4f}")
print(f"R-squared (R2): {r2_3_test:.4f}")

plt.figure(figsize=(12, 7))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')

x_plot = np.linspace(df['x'].min(), df['x'].max(), 400).reshape(-1, 1)
X_plot_poly_3 = poly_features_3.transform(x_plot)
y_plot_3 = model_3.predict(X_plot_poly_3)
plt.plot(x_plot, y_plot_3, color='purple', linewidth=2, label='Polynomial Degree 3')

plt.title('Polynomial Degree 3 - Testing Data')
plt.xlabel('Month')
plt.ylabel('Number of Bags')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()