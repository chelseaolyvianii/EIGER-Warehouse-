import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    df = pd.read_csv("MATH6192031_AoL Data(in).csv")
    df.columns = ['month_year_str', 'y']
    df['x'] = np.arange(1, len(df) + 1)
except FileNotFoundError:
    print("Error: File 'MATH6192031_AoL Data(in).csv' tidak ditemukan.")
    exit()

train_size = 114
train_df = df.iloc[:train_size]

X_train = train_df[['x']]
y_train = train_df['y']

print(f"Training Data: {len(train_df)}")

# POLYNOMIAL DEGREE 2
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train_2 = poly_features_2.fit_transform(X_train)

model_2 = LinearRegression()
model_2.fit(X_poly_train_2, y_train)

coef_2 = model_2.coef_
intercept_2 = model_2.intercept_

print("\nModel Polinomial Tingkat 2:")
print(f"Mathematical models:")
print(f"y = {coef_2[1]:.4f}x^2 + {coef_2[0]:.4f}x + {intercept_2:.4f}")

y_train_pred_2 = model_2.predict(X_poly_train_2)

mae_2 = mean_absolute_error(y_train, y_train_pred_2)
mse_2 = mean_squared_error(y_train, y_train_pred_2)
rmse_2 = np.sqrt(mse_2)
r2_2 = r2_score(y_train, y_train_pred_2)

print("\nEvaluation Metrics on Training Data (Degree 2):")
print(f"Mean Absolute Error (MAE): {mae_2:.4f}")
print(f"Mean Squared Error (MSE): {mse_2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_2:.4f}")
print(f"R-squared (R2): {r2_2:.4f}")

plt.figure(figsize=(12, 7))
plt.scatter(X_train['x'], y_train, color='blue', label='Training Data - 144 (80%)')

x_plot = np.linspace(X_train['x'].min(), X_train['x'].max(), 400).reshape(-1, 1)
X_plot_poly_2 = poly_features_2.transform(x_plot)
y_plot_2 = model_2.predict(X_plot_poly_2)
plt.plot(x_plot, y_plot_2, color='red', linewidth=2, label='Polynomial Degree 2')


plt.title('Polynomial Degree 2 - Training Data')
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

coef_3 = model_3.coef_
intercept_3 = model_3.intercept_

print("\nModel Polinomial Tingkat 3:")
print(f"Mathematical models:")
print(f"y = {coef_3[2]:.4f}x^3 + {coef_3[1]:.4f}x^2 + {coef_3[0]:.4f}x + {intercept_3:.4f}")

y_train_pred_3 = model_3.predict(X_poly_train_3)

mae_3 = mean_absolute_error(y_train, y_train_pred_3)
mse_3 = mean_squared_error(y_train, y_train_pred_3)
rmse_3 = np.sqrt(mse_3)
r2_3 = r2_score(y_train, y_train_pred_3)

print("\nEvaluation Metrics on Training Data (Degree 3):")
print(f"Mean Absolute Error (MAE): {mae_3:.4f}")
print(f"Mean Squared Error (MSE): {mse_3:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_3:.4f}")
print(f"R-squared (R2): {r2_3:.4f}")

plt.figure(figsize=(12, 7))
plt.scatter(X_train['x'], y_train, color='blue', label='Training Data - 144 (80%)')

x_plot = np.linspace(X_train['x'].min(), X_train['x'].max(), 400).reshape(-1, 1)
X_plot_poly_3 = poly_features_3.transform(x_plot)
y_plot_3 = model_3.predict(X_plot_poly_3)
plt.plot(x_plot, y_plot_3, color='red', linewidth=2, label='Polynomial Degree 3')

plt.title('Polynomial Degree 3 - Training Data')
plt.xlabel('Month')
plt.ylabel('Number of Bags')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()