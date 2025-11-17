import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# USER INPUT
file_path = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_track_results/results.xlsx"
target_col = "avg_train_delay_time"
group_cols = ["track_number", "cranes_per_track", "hostler_number",
              "train_batch_size", "trains_per_day"]

# Load and preprocess
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

for col in group_cols + [target_col]:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

df["container"] = df["train_batch_size"] * df["trains_per_day"]


def exp_func(x, a, b):
    return a * np.exp(b * x)    # Exponential curve function: delay = a * exp(b * trains_per_day)

coeff_list = []

# STEP 1: Fit exponential curve for each unique combination
unique_groups = df.groupby(group_cols)

for keys, group in unique_groups:
    t = group["trains_per_day"].values
    delay = group[target_col].values

    if len(t) < 3:
        continue  # Need at least 3 points for exponential fitting

    try:
        popt, _ = curve_fit(exp_func, t, delay, p0=(1, 0.01), maxfev=10000) # Fit delay(t) = a exp(b t)
        a, b = popt

        coeff_list.append({
            "track_number": keys[0],
            "cranes_per_track": keys[1],
            "hostler_number": keys[2],
            "train_batch_size": keys[3],
            "trains_per_day": keys[4],
            "container": keys[3] * keys[4],
            "coef_a": a,
            "coef_b": b
        })

    except Exception:
        continue

coeff_df = pd.DataFrame(coeff_list)
print("STEP 1 done: fitted exponential coefficients for each group.")

# STEP 2: Fit a, b as functions of terminal resources (multi-variate regression)
feature_cols = [
    "track_number",
    "cranes_per_track",
    "hostler_number",
    "train_batch_size",
    "trains_per_day",
    "container"
]

X = coeff_df[feature_cols].values

# Fit model for "a"
y_a = coeff_df["coef_a"].values
reg_a = LinearRegression().fit(X, y_a)
y_a_pred = reg_a.predict(X)
r2_a = r2_score(y_a, y_a_pred)

# Fit model for "b"
y_b = coeff_df["coef_b"].values
reg_b = LinearRegression().fit(X, y_b)
y_b_pred = reg_b.predict(X)
r2_b = r2_score(y_b, y_b_pred)

print("\n===== Exponential Model =====")
print("delay(t) = a * exp(b * t)")

print("\n===== Fitted Relationships =====")
print("\n---- Coefficient a regression ----")
for name, coef in zip(feature_cols, reg_a.coef_):
    print(f"{name}: {coef:.6f}")
print(f"(intercept): {reg_a.intercept_:.6f}")
print(f"R² for a: {r2_a:.4f}")

print("\n---- Coefficient b regression ----")
for name, coef in zip(feature_cols, reg_b.coef_):
    print(f"{name}: {coef:.6f}")
print(f"(intercept): {reg_b.intercept_:.6f}")
print(f"R² for b: {r2_b:.4f}")


out_path = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_delay_coefficients.xlsx"
coeff_df.to_excel(out_path, index=False)
print(f"\nExponential coefficients saved to:\n{out_path}")