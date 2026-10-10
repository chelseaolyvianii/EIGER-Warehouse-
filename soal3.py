import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from dateutil.relativedelta import relativedelta

try:
    df = pd.read_csv("MATH6192031_AoL Data(in).csv")
    df.columns = ['month_year_str', 'y']
    df['x'] = np.arange(1, len(df) + 1)
except FileNotFoundError:
    print("Error: File 'MATH6192031_AoL Data(in).csv' not found.")
    exit()

max_capacity = 25000
train_size = 114
train_df = df.iloc[:train_size]

X_train = train_df[['x']]
y_train = train_df['y']

poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly_train_3 = poly_features_3.fit_transform(X_train)

model_3 = LinearRegression()
model_3.fit(X_poly_train_3, y_train)

coef_3 = model_3.coef_
intercept_3 = model_3.intercept_


def f(x):
    return 0.0016*x**3 + 0.2298*x**2 + 31.5678*x + (1886.7842 - max_capacity) 

def f_derivative(x):
    return 0.0048*x**2 + 0.4596*x + 31.5678

def newton_raphson(initial_guess, tolerance=1e-6, max_iterations=100):
    
    x0 = initial_guess
    for i in range(max_iterations):
        fx = f(x0)
        f_prime_x = f_derivative(x0)

        if abs(f_prime_x) < 1e-12:
            print("Derivative is close to zero. Method failed.")
            return None
        
        x1 = x0 - fx / f_prime_x

        if abs(x1 - x0) < tolerance:
            return x1
        
        x0 = x1
    
    print("Maximum iterations reached. Method failed to converge.")
    return None

initial_guess = 150.0
month_full = newton_raphson(initial_guess)

if month_full is not None:
    print(f"\nWarehouse capacity is predicted to be exceeded at month number {month_full:.2f}")

    construction_time = 13
    start_build_month = month_full - construction_time

    def convert_to_date(month_number):
        start_date = datetime(2013, 1, 1)
        target_date = start_date + relativedelta(months=int(round(month_number)) - 1)
        return target_date.strftime("%B %Y")

    date_full = convert_to_date(month_full)
    start_build_date = convert_to_date(start_build_month)

    print(f"This is equivalent to the date: {date_full}")
    print(f"\nRequired construction time: {construction_time} months")
    print(f"Therefore, construction must begin at month number {start_build_month:.2f}")
    print(f"Construction Start Date: {start_build_date}")
    
    plt.figure(figsize=(12, 7))

    plt.scatter(X_train['x'], y_train, color='blue', label='Training Data')

    x_plot = np.linspace(1, month_full + 24, 500).reshape(-1, 1) 
    X_plot_poly = poly_features_3.transform(x_plot)
    y_plot = model_3.predict(X_plot_poly)
    plt.plot(x_plot, y_plot, color='purple', label='Polynomial Degree 3 Prediction')

    plt.axhline(y=max_capacity, color='red', linestyle='--', label=f'Max Capacity ({max_capacity} bags)')

    plt.plot(month_full, max_capacity, 'ro', markersize=10, label=f'Capacity Exceeded ({date_full})')
    
    y_start_build = model_3.predict(poly_features_3.transform([[start_build_month]]))
    
    plt.plot(start_build_month, y_start_build, 'g*', markersize=15, label=f'Start Construction ({start_build_date})')

    plt.vlines(x=start_build_month, ymin=0, ymax=y_start_build, color='green', linestyle=':', lw=2)
             
    plt.title('Warehouse Capacity Prediction', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Bags', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.ylim(bottom=0, top=max_capacity + 5000)
    plt.tight_layout()
    plt.show()

else:
    print("Could not find a solution with the Newton-Raphson method.")