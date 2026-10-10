# EIGER Warehouse Capacity & Bag Production Trend Analysis

An end-to-end numerical analysis, non-linear time-series trend modeling, and capacity forecasting project designed to determine the optimal timing for main warehouse expansion for **EIGER** (PT Eigerindo Multi Produk Industri).

---

## Project Overview

EIGER's primary manufacturing and distribution operations in Bandung require accurate forecasting of monthly bag production volumes to manage warehouse capacity effectively. The primary warehouse has a strict maximum operational storage capacity of **25,000 bags per month**. 

Using historical production data spanning 144 months (January 2018 to December 2023, denoted as $M_1$ to $M_{144}$), this project models the non-linear growth trend of production, evaluates model precision, predicts when storage limits will be breached, and determines the deadline to begin constructing a new warehouse considering a **13-month construction lead time**.

---

## 🛠️ Key Technical Problems & Solutions

### 1. Non-Linear Trend Modeling (Question 1)
- **Challenge:** Avoid linear approximation biases to capture non-linear manufacturing growth accurately over 144 periods.
- **Approach:** Fitted a higher-degree polynomial model $P(t)$ against historical time-series data to provide exact mathematical representation and precise long-term extrapolation.

### 2. Polynomial Transformation & Verification (Question 2)
- **Challenge:** Ensure the non-linear model remains computable and stable for forecasting beyond $M_{144}$.
- **Approach:** Evaluated model precision using residual metrics ($R^2$, RMSE, MSE) without relying on early data points as static references, ensuring model validity for future horizon extrapolation.

### 3. Warehouse Capacity & Lead Time Strategy (Question 3)
- **Challenge:** Identify the exact threshold month where production exceeds 25,000 bags/month and derive the mandatory construction start date.
- **Approach:** Expressed the capacity constraint as a root-finding problem:
  $$P(t) - 25,000 = 0$$
  Applied numerical root-finding algorithms (e.g., Newton-Raphson / Bisection) to locate $t_{\text{threshold}}$, then integrated operational lead time to calculate:
  $$t_{\text{construction}} = t_{\text{threshold}} - 13 \text{ months}$$

---

## Repository Structure

```text
.
├── MATH6192031_AoL Data(in)  # Historical monthly bag production dataset (M1 to M144)
├── soal1.py                  # Script for non-linear trend modeling & curve fitting
├── soal2.py                  # Script for polynomial transformation & accuracy proof
├── soal3.py                  # Script for root-finding analysis & lead-time planning
└── README.md                 # Project documentation

## Technical Requirements & Libraries

* **Language:** Python[cite: 1]
* **Key Libraries:** `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`[cite: 1]

---

## Analytical Methodology & Workflow

1. **Data Ingestion:** Load and clean 144 months of bag production metrics from `Data.csv`.
2. **Curve Fitting:** Determine polynomial coefficients minimizing mean squared error across the time-series range.
3. **Accuracy Proof:** Compute statistical fit metrics to validate extrapolation capability.
4. **Numerical Root Finding:** Calculate the exact fractional month $t$ where $P(t) = 25,000$.
5. **Operational Timeline Strategy:** Subtract 13 months from the capacity breach date to specify the required project launch date for the new warehouse facility.