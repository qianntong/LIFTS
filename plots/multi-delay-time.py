import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score


# ============================================================
# INPUT / OUTPUT
# ============================================================
file_path   = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_track_results/input/results.xlsx"
output_path = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_track_results/node_delay_inverse_output.xlsx"
delay_plot_dir = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_track_results/delay"


os.makedirs(os.path.dirname(output_path), exist_ok=True)
os.makedirs(delay_plot_dir, exist_ok=True)

target_col = "avg_train_delay_time"

group_cols = ["track_number", "cranes_per_track", "hostler_number", "train_batch_size"]


# ============================================================
# LOAD EXCEL (only track sheets)
# ============================================================
xls = pd.ExcelFile(file_path)
valid_sheets = ["1 track", "2 tracks", "3 tracks", "4 tracks", "5 tracks"]

selected_sheets = [s for s in xls.sheet_names if s.strip().lower() in valid_sheets]
print("Reading sheets:", selected_sheets)

df_list = [pd.read_excel(file_path, sheet_name=s) for s in selected_sheets]
df = pd.concat(df_list, ignore_index=True)

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

required_cols = group_cols + [target_col, "trains_per_day"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

print("Loaded merged df:", df.shape)


# ============================================================
# FUNCTION: log-linear exponential fitting
# delay = a * exp(b t)
# ln(delay) = ln(a) + b t
# ============================================================
def fit_log_linear(t, delay):
    eps = 1e-8
    mask = delay > eps
    t = t[mask]
    delay = delay[mask]

    if len(t) < 3:
        return None

    ln_delay = np.log(delay).reshape(-1, 1)
    t_input = t.reshape(-1, 1)

    model = LinearRegression().fit(t_input, ln_delay)
    ln_a = model.intercept_[0]
    b = model.coef_[0][0]
    a = np.exp(ln_a)

    pred_ln_delay = model.predict(t_input).flatten()
    r2 = r2_score(ln_delay, pred_ln_delay)

    return a, b, model, r2


# ============================================================
# STEP 1: Fit a,b for each resource combination
# ============================================================
coeff_list = []
unique_groups = df.groupby(group_cols)

for keys, group in unique_groups:
    t = group["trains_per_day"].values.astype(float)
    delay = group[target_col].values.astype(float)

    result = fit_log_linear(t, delay)
    if result is None:
        continue

    a, b, model, r2 = result

    coeff_list.append({
        "track_number": keys[0],
        "cranes_per_track": keys[1],
        "hostler_number": keys[2],
        "train_batch_size": keys[3],
        "coef_a": a,
        "coef_b": b,
        "fit_r2": r2
    })

    # Plot curve
    plt.figure(figsize=(6,4))
    plt.scatter(t, delay, c="black", label="data")

    t_fit = np.linspace(min(t), max(t), 200)
    delay_fit = a * np.exp(b * t_fit)
    plt.plot(t_fit, delay_fit, 'r-', label=f"a={a:.3f}, b={b:.3f}")

    plt.xlabel("Trains per day")
    plt.ylabel("Average train delay time")
    plt.title(f"Track={keys[0]}, Cranes={keys[1]}, Hostlers={keys[2]}, Batch={keys[3]}")
    plt.legend()

    fname = f"delay_track{keys[0]}_crane{keys[1]}_host{keys[2]}_batch{keys[3]}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(delay_plot_dir, fname))
    plt.close()


coeff_df = pd.DataFrame(coeff_list)
print("\nLOG-LINEAR FITTING DONE")
print(coeff_df.head())


# ============================================================
# STEP 2: Clean and DO NOT apply log-regression on a,b
# Only filter using log-based R²
# ============================================================
coeff_df = coeff_df.replace([np.inf, -np.inf], np.nan)
coeff_df = coeff_df[(coeff_df["coef_a"] > 0) & (coeff_df["coef_b"] > 0)]
coeff_df = coeff_df.dropna(subset=["coef_a", "coef_b"])
coeff_df = coeff_df[coeff_df["fit_r2"] >= 0.5]

print("\nValid points for polynomial regression:", coeff_df.shape[0])


# ============================================================
# STEP 3: Build inverse variables for polynomial regression
# ============================================================
coeff_df["inv_track"]   = 1 / coeff_df["track_number"]
coeff_df["inv_crane"]   = 1 / coeff_df["cranes_per_track"]
coeff_df["inv_hostler"] = 1 / coeff_df["hostler_number"]
coeff_df["batch"]       = coeff_df["train_batch_size"]

feature_cols = ["inv_track", "inv_crane", "inv_hostler", "batch"]

X = coeff_df[feature_cols].values

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(Xs)
poly_names = poly.get_feature_names_out(feature_cols)


# ============================================================
# STEP 4: Directly fit a and b (no log)
# ============================================================
y_a = coeff_df["coef_a"].values
y_b = coeff_df["coef_b"].values

reg_a = LinearRegression().fit(X_poly, y_a)
reg_b = LinearRegression().fit(X_poly, y_b)

coeff_df["a_pred"] = reg_a.predict(X_poly)
coeff_df["b_pred"] = reg_b.predict(X_poly)

r2_a = r2_score(y_a, coeff_df["a_pred"])
r2_b = r2_score(y_b, coeff_df["b_pred"])


# ============================================================
# PRINT results
# ============================================================
print("\n===== Polynomial NPF for delay coefficients (No log) =====")

print("\n--- Regression for a ---")
for name, coef in zip(poly_names, reg_a.coef_):
    print(f"{name}: {coef:.6f}")
print("Intercept:", reg_a.intercept_)
print("R²(a):", r2_a)

print("\n--- Regression for b ---")
for name, coef in zip(poly_names, reg_b.coef_):
    print(f"{name}: {coef:.6f}")
print("Intercept:", reg_b.intercept_)
print("R²(b):", r2_b)


# ============================================================
# SAVE everything
# ============================================================
with pd.ExcelWriter(output_path) as writer:
    df.to_excel(writer, sheet_name="raw_data", index=False)
    coeff_df.to_excel(writer, sheet_name="exp_coeffs", index=False)

    pd.DataFrame({
        "feature": list(poly_names) + ["intercept"],
        "coef_a": list(reg_a.coef_) + [reg_a.intercept_],
        "coef_b": list(reg_b.coef_) + [reg_b.intercept_],
    }).to_excel(writer, sheet_name="poly_regression_ab", index=False)

print("\nAll results saved to:", output_path)
print("All delay plots saved to:", delay_plot_dir)


# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import r2_score
#
#
# # ============================================================
# # INPUT / OUTPUT
# # ============================================================
# file_path   = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_track_results/input/results.xlsx"
# output_path = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_track_results/node_delay_output.xlsx"
# delay_plot_dir = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_track_results/delay"
#
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
# os.makedirs(delay_plot_dir, exist_ok=True)
#
# target_col = "avg_train_delay_time"
#
# # resources for grouping
# group_cols = ["track_number", "cranes_per_track", "hostler_number", "train_batch_size"]
#
#
# # ============================================================
# # LOAD EXCEL (only track sheets)
# # ============================================================
# xls = pd.ExcelFile(file_path)
# valid_sheets = ["1 track", "2 tracks", "3 tracks", "4 tracks", "5 tracks"]
#
# selected_sheets = [s for s in xls.sheet_names if s.strip().lower() in valid_sheets]
# print("Reading sheets:", selected_sheets)
#
# df_list = [pd.read_excel(file_path, sheet_name=s) for s in selected_sheets]
# df = pd.concat(df_list, ignore_index=True)
#
# df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
#
# required_cols = group_cols + [target_col, "trains_per_day"]
# for col in required_cols:
#     if col not in df.columns:
#         raise ValueError(f"Missing column: {col}")
#
# print("Loaded merged df:", df.shape)
#
#
# # ============================================================
# # FUNCTION: log-linear exponential fitting
# # delay = a * exp(b t)
# # ln(delay) = ln(a) + b t
# # ============================================================
# def fit_log_linear(t, delay):
#     eps = 1e-8
#     mask = delay > eps
#     t = t[mask]
#     delay = delay[mask]
#
#     if len(t) < 3:
#         return None
#
#     ln_delay = np.log(delay).reshape(-1, 1)
#     t_input = t.reshape(-1, 1)
#
#     model = LinearRegression().fit(t_input, ln_delay)
#     ln_a = model.intercept_[0]
#     b = model.coef_[0][0]
#     a = np.exp(ln_a)
#
#     pred_ln_delay = model.predict(t_input).flatten()
#     r2 = r2_score(ln_delay, pred_ln_delay)
#
#     return a, b, model, r2
#
#
# # ============================================================
# # STEP 1: Fit a,b for each resource combination
# # ============================================================
# coeff_list = []
#
# unique_groups = df.groupby(group_cols)
#
# for keys, group in unique_groups:
#     t = group["trains_per_day"].values.astype(float)
#     delay = group[target_col].values.astype(float)
#
#     result = fit_log_linear(t, delay)
#     if result is None:
#         continue
#
#     a, b, model, r2 = result
#
#     coeff_list.append({
#         "track_number": keys[0],
#         "cranes_per_track": keys[1],
#         "hostler_number": keys[2],
#         "train_batch_size": keys[3],
#         "coef_a": a,
#         "coef_b": b,
#         "fit_r2": r2
#     })
#
#     # ============================================================
#     # PLOT delay-t curve
#     # ============================================================
#     plt.figure(figsize=(6,4))
#     plt.scatter(t, delay, c="black", label="data")
#
#     t_fit = np.linspace(min(t), max(t), 200)
#     delay_fit = a * np.exp(b * t_fit)
#     plt.plot(t_fit, delay_fit, 'r-', label=f"fit: a={a:.3f}, b={b:.3f}")
#
#     plt.xlabel("Trains per day")
#     plt.ylabel("Average train delay time")
#     plt.title(f"Track={keys[0]}, Cranes={keys[1]}, Hostlers={keys[2]}, Batch={keys[3]}")
#     plt.legend()
#
#     fname = f"delay_track{keys[0]}_crane{keys[1]}_host{keys[2]}_batch{keys[3]}.png"
#     plt.tight_layout()
#     plt.savefig(os.path.join(delay_plot_dir, fname))
#     plt.close()
#
#
# coeff_df = pd.DataFrame(coeff_list)
# print("\nLOG-LINEAR FITTING DONE")
# print(coeff_df.head())
#
#
# # ============================================================
# # STEP 2: Fit polynomial + log-transform model for a,b
# # ============================================================
#
# # Clean invalid a,b values
# coeff_df = coeff_df.replace([np.inf, -np.inf], np.nan)
# coeff_df = coeff_df[(coeff_df["coef_a"] > 0) & (coeff_df["coef_b"] > 0)]
# coeff_df = coeff_df.dropna(subset=["coef_a", "coef_b"])
# coeff_df = coeff_df[coeff_df["fit_r2"] >= 0.5]  # Remove low-quality exponential fits
#
#
# print("\nValid data points for polynomial regression:", coeff_df.shape[0])
#
# # Now rebuild X using CLEAN coeff_df
# feature_cols = ["track_number", "cranes_per_track", "hostler_number", "train_batch_size"]
# X = coeff_df[feature_cols].values
#
# # rebuild polynomial features using CLEAN X
# poly_features = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly_features.fit_transform(X)
# poly_names = poly_features.get_feature_names_out(feature_cols)
#
# # ========= fit log(a) =========
# y_a_log = np.log(coeff_df["coef_a"].values + 1e-12)
# reg_a = LinearRegression().fit(X_poly, y_a_log)
# y_a_log_pred = reg_a.predict(X_poly)
# r2_a = r2_score(y_a_log, y_a_log_pred)
# coeff_df["a_pred"] = np.exp(y_a_log_pred)
#
# # ========= fit log(b) =========
# y_b_log = np.log(coeff_df["coef_b"].values + 1e-12)
# reg_b = LinearRegression().fit(X_poly, y_b_log)
# y_b_log_pred = reg_b.predict(X_poly)
# r2_b = r2_score(y_b_log, y_b_log_pred)
# coeff_df["b_pred"] = np.exp(y_b_log_pred)
#
# # PRINT RESULTS
# print("\n===== Polynomial + Log-Transform NPF =====\n")
#
# print("--- Regression for log(a) ---")
# for name, coef in zip(poly_names, reg_a.coef_):
#     print(f"{name}: {coef:.6f}")
# print("intercept:", reg_a.intercept_)
# print("R²(log(a)):", r2_a)
#
# print("\n--- Regression for log(b) ---")
# for name, coef in zip(poly_names, reg_b.coef_):
#     print(f"{name}: {coef:.6f}")
# print("intercept:", reg_b.intercept_)
# print("R²(log(b)):", r2_b)
#
#
# # ============================================================
# # STEP 3: SAVE RESULTS
# # ============================================================
# with pd.ExcelWriter(output_path) as writer:
#     df.to_excel(writer, sheet_name="raw_data", index=False)
#     coeff_df.to_excel(writer, sheet_name="exp_coeffs", index=False)
#
#     pd.DataFrame({
#         "feature": list(poly_names) + ["intercept"],
#         "coef_a": list(reg_a.coef_) + [reg_a.intercept_],
#         "coef_b": list(reg_b.coef_) + [reg_b.intercept_],
#     }).to_excel(writer, sheet_name="poly_log_regression_ab", index=False)
#
# print("\nAll results saved to:", output_path)
# print("All delay plots saved to:", delay_plot_dir)
