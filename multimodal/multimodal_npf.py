import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")

FILE_PATH = "/Users/qianqiantong/PycharmProjects/LIFTS/multimodal/output/experiment_results.csv"

MU_CRANE = 35
MU_HOSTLER = 3
RHO_MAX = 0.95

# =========================
# Load
# =========================
df = pd.read_csv(FILE_PATH)
print("Original rows:", len(df))

# =========================
# Weekly volumes
# =========================
df["train_weekly_volume"] = df["train_weekly_num"] * df["train_batch_size"]
df["vessel_weekly_volume"] = df["vessel_weekly_num"] * df["vessel_batch_size"]
df["truck_weekly_volume_in"] = df["truck_weekly_volume"]

# =========================
# Resources
# =========================
df["berth_cranes"] = df["cranes_per_berth"] * df["berth_number"]
df["track_cranes"] = df["cranes_per_track"] * df["track_number"]

# =========================
# OD flows
# =========================
s = df["vessel_to_train_share"]

df["V_T"] = df["vessel_weekly_volume"] * s
df["V_H"] = df["vessel_weekly_volume"] - df["V_T"]

df["T_V"] = df["V_T"]
df["T_H_raw"] = df["train_weekly_volume"] - df["T_V"]
df["T_H"] = df["T_H_raw"].clip(lower=0)

df["residual_truck"] = df["V_H"] + df["T_H"]
scale = np.where(
    df["residual_truck"] > 1e-9,
    df["truck_weekly_volume_in"] / df["residual_truck"],
    0.0
)

df["H_V"] = df["V_H"] * scale
df["H_T"] = df["T_H"] * scale

# =========================
# Schedule terms
# =========================
df["Delta_V"] = 168.0 / (df["vessel_weekly_num"] + 1e-12)
df["Delta_T"] = 168.0 / (df["train_weekly_num"] + 1e-12)
df["Delta_H"] = 168.0 / (df["truck_weekly_volume_in"] + 1e-12)

DEST_MODE = {
    "V_T": "T",
    "V_H": "H",
    "T_V": "V",
    "T_H": "H",
    "H_T": "T",
    "H_V": "V",
}

OD_TARGETS = {
    "V_T": "vessel_to_train.avg_container_processing_time.mean",
    "V_H": "vessel_to_truck.avg_container_processing_time.mean",
    "T_V": "train_to_vessel.avg_container_processing_time.mean",
    "T_H": "train_to_truck.avg_container_processing_time.mean",
    "H_T": "truck_to_train.avg_container_processing_time.mean",
    "H_V": "truck_to_vessel.avg_container_processing_time.mean",
}

BERTH_OD = {"V_T", "V_H", "T_V", "H_V"}
TRACK_OD = {"T_H", "H_T"}

# =========================
# Utilities
# =========================
def safe_rho(x):
    return np.clip(x, 1e-9, RHO_MAX)

def clean_data(df_in, y_col):
    df_sub = df_in.copy()
    df_sub = df_sub[np.isfinite(df_sub[y_col])]
    df_sub = df_sub[df_sub[y_col] > 0]
    return df_sub

# =========================
# Linear + interaction model
# =========================
def performance_model(X, b1, b2, b3, b4, a0):
    Delta, V, rho_c, rho_h = X
    return (
        a0
        + Delta
        + b1 * V
        + b2 * rho_c
        + b3 * rho_h
        + b4 * rho_c * rho_h
    )

print("\n========== Linear + Interaction Model ==========")

for od_key, y_col in OD_TARGETS.items():

    if y_col not in df.columns:
        continue

    df_sub = clean_data(df, y_col)
    if len(df_sub) < 30:
        print(f"{od_key}: Not enough samples")
        continue

    dest = DEST_MODE[od_key]

    V = df_sub[od_key].values.astype(float)
    Lambda = V / 168.0

    if od_key in BERTH_OD:
        N_crane = df_sub["berth_cranes"].values
    else:
        N_crane = df_sub["track_cranes"].values

    N_hostler = df_sub["hostler_number"].values

    rho_c = safe_rho(Lambda / (N_crane * MU_CRANE + 1e-12))
    rho_h = safe_rho(Lambda / (N_hostler * MU_HOSTLER + 1e-12))

    if dest == "V":
        Delta = df_sub["Delta_V"].values
    elif dest == "T":
        Delta = df_sub["Delta_T"].values
    else:
        Delta = df_sub["Delta_H"].values

    Xdata = np.vstack([Delta, V, rho_c, rho_h])
    ydata = df_sub[y_col].values.astype(float)

    bounds = (
        [-1e3, -1e4, -1e4, -1e4, -1e3],
        [1e3,  1e4,  1e4,  1e4,  1e3],
    )

    try:
        params, _ = curve_fit(
            performance_model,
            Xdata,
            ydata,
            bounds=bounds,
            maxfev=200000
        )
    except Exception as e:
        print(f"{od_key} fit failed:", e)
        continue

    y_pred = performance_model(Xdata, *params)

    ss_res = np.sum((ydata - y_pred) ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    r2 = abs(1 - ss_res / (ss_tot + 1e-12))

    print(f"\nOD: {od_key}")
    print("R^2:", round(r2, 4))
    print("Parameters [b1, b2, b3, b4, a0]:")
    print(np.round(params, 4))

print("\nDone.")



# import pandas as pd
# import numpy as np
# from scipy.optimize import curve_fit
# import warnings
#
# warnings.filterwarnings("ignore")
#
# FILE_PATH = "/Users/qianqiantong/PycharmProjects/LIFTS/multimodal/output/experiment_results_small.csv"
#
# # =========================
# # Config: service rates
# # =========================
# MU_CRANE = 30    # containers/hour per crane
# MU_HOSTLER = 2  # containers/hour per hostler
#
# # queue blow-up
# RHO_MAX = 0.97
#
# # =========================
# # Load data
# # =========================
# df = pd.read_csv(FILE_PATH)
# print("Original rows:", len(df))
#
# # =========================
# # Weekly volumes
# # =========================
# df["train_weekly_volume"] = df["train_weekly_num"] * df["train_batch_size"]
# df["vessel_weekly_volume"] = df["vessel_weekly_num"] * df["vessel_batch_size"]
# df["truck_weekly_volume_in"] = df["truck_weekly_volume"]
# df["total_weekly_volume"] = df["train_weekly_volume"] + df["vessel_weekly_volume"] + df["truck_weekly_volume_in"]
#
# # =========================
# # Resources
# # =========================
# df["berth_cranes"] = df["cranes_per_berth"] * df["berth_number"]
# df["track_cranes"] = df["cranes_per_track"] * df["track_number"]
#
# # =========================
# # Structure variable: share s
# # =========================
# s = df["vessel_to_train_share"].astype(float)
#
# # OD flows (weekly)
# df["V_T"] = df["vessel_weekly_volume"] * s
# df["V_H"] = df["vessel_weekly_volume"] - df["V_T"]
#
# df["T_V"] = df["V_T"]
# df["T_H_raw"] = df["train_weekly_volume"] - df["T_V"]
# df["T_H"] = df["T_H_raw"].clip(lower=0)
#
# df["residual_truck"] = df["V_H"] + df["T_H"]
# eps = 1e-9
# scale = np.where(df["residual_truck"] > eps, df["truck_weekly_volume_in"] / df["residual_truck"], 0.0)
#
# df["H_V"] = df["V_H"] * scale
# df["H_T"] = df["T_H"] * scale
#
# # =========================
# # Headway / arrival interval (structure term)
# # Weekly arrivals -> avg interval in hours: 168 / lambda_m
# # =========================
# df["Delta_V"] = 168.0 / (df["vessel_weekly_num"].astype(float) + 1e-12)
# df["Delta_T"] = 168.0 / (df["train_weekly_num"].astype(float) + 1e-12)
# # truck：如果你认为 truck 是连续流，就设 0；如果你要 gate headway，可以另建 Delta_H
# df["Delta_H"] = 0.0
#
# # =========================
# # Targets
# # =========================
# OD_TARGETS = {
#     "V_T": "vessel_to_train.avg_container_processing_time.mean",
#     "V_H": "vessel_to_truck.avg_container_processing_time.mean",
#     "T_V": "train_to_vessel.avg_container_processing_time.mean",
#     "T_H": "train_to_truck.avg_container_processing_time.mean",
#     "H_T": "truck_to_train.avg_container_processing_time.mean",
#     "H_V": "truck_to_vessel.avg_container_processing_time.mean",
# }
#
# # 哪些 OD 主要走 berth crane / track crane（沿用你原来划分）
# BERTH_OD = {"V_T", "V_H", "T_V", "H_V"}
# TRACK_OD = {"T_H", "H_T"}
#
# # downstream mode (决定结构等待项用哪个 Delta)
# DEST_MODE = {
#     "V_T": "T",
#     "V_H": "H",
#     "T_V": "V",
#     "T_H": "H",
#     "H_T": "T",
#     "H_V": "V",
# }
#
# # =========================
# # Utils
# # =========================
# def clean_processing_time(df_in, y_col):
#     df_sub = df_in.copy()
#     df_sub = df_sub[np.isfinite(df_sub[y_col])]
#     df_sub = df_sub[df_sub[y_col] > 0]
#
#     q1 = df_sub[y_col].quantile(0.25)
#     q3 = df_sub[y_col].quantile(0.75)
#     iqr = q3 - q1
#     lower = q1 - 1.5 * iqr
#     upper = q3 + 1.5 * iqr
#
#     return df_sub[(df_sub[y_col] >= lower) & (df_sub[y_col] <= upper)]
#
# def safe_clip_rho(rho, hi=RHO_MAX):
#     return np.clip(np.asarray(rho, dtype=float), 1e-9, hi)
#
# def qterm(rho):
#     """Queue-consistent congestion term: rho/(1-rho)"""
#     r = safe_clip_rho(rho)
#     return r / (1.0 - r)
#
# def get_delta_col(dest_mode):
#     if dest_mode == "V":
#         return "Delta_V"
#     if dest_mode == "T":
#         return "Delta_T"
#     return "Delta_H"
#
# # =========================
# # Separated model: structure + congestion
# #
# # t = a0 + w * Delta_dest + b1 * V_od + b2 * Q(rho_c) + b3 * Q(rho_h)
# # where:
# #   Lambda_od = V_od / 168
# #   rho_c = Lambda_od / (N_cranes * MU_CRANE)
# #   rho_h = Lambda_od / (N_hostlers * MU_HOSTLER)
# # =========================
# def model_struct_cong(X, a0, w, b1, b2, b3):
#     Delta_dest, V_od, rho_c, rho_h = X
#     return a0 + w * Delta_dest + b1 * V_od + b2 * qterm(rho_c) + b3 * qterm(rho_h)
#
# print("\n========== Separated Model Fit (structure + congestion) ==========")
#
# for od_key, y_col in OD_TARGETS.items():
#     if y_col not in df.columns:
#         continue
#
#     df_sub = clean_processing_time(df, y_col)
#
#     dest_mode = DEST_MODE[od_key]
#     delta_col = get_delta_col(dest_mode)
#
#     # choose crane pool size (berth vs track)
#     if od_key in BERTH_OD:
#         Ncr_col = "berth_cranes"
#     elif od_key in TRACK_OD:
#         Ncr_col = "track_cranes"
#     else:
#         # fallback: total cranes
#         Ncr_col = None
#
#     needed = [y_col, od_key, "hostler_number", delta_col]
#     if Ncr_col is not None:
#         needed.append(Ncr_col)
#
#     df_sub = df_sub.dropna(subset=needed)
#     if len(df_sub) < 30:
#         print(f"{od_key}: Not enough samples")
#         continue
#
#     V_od = df_sub[od_key].astype(float).values
#     Delta_dest = df_sub[delta_col].astype(float).values
#
#     # arrival rate (containers/hour)
#     Lambda_od = V_od / 168.0
#
#     # capacities
#     Nh = df_sub["hostler_number"].astype(float).values
#     cap_h = Nh * MU_HOSTLER + 1e-12
#
#     if Ncr_col is None:
#         # if you ever want total cranes, add it here; right now not used
#         Ncr = (df_sub["berth_cranes"].astype(float).values + df_sub["track_cranes"].astype(float).values)
#     else:
#         Ncr = df_sub[Ncr_col].astype(float).values
#
#     cap_c = Ncr * MU_CRANE + 1e-12
#
#     # utilizations
#     rho_h = safe_clip_rho(Lambda_od / cap_h, hi=RHO_MAX)
#     rho_c = safe_clip_rho(Lambda_od / cap_c, hi=RHO_MAX)
#
#     Xdata = np.vstack([Delta_dest, V_od, rho_c, rho_h])
#     ydata = df_sub[y_col].astype(float).values
#
#     # bounds for [a0, w, b1, b2, b3]
#     bounds = (
#         [-1e3, -1e3, -1e3, -1e4, -1e4],
#         [ 1e3,  1e3,  1e3,  1e4,  1e4],
#     )
#
#     try:
#         params, _ = curve_fit(
#             model_struct_cong,
#             Xdata,
#             ydata,
#             bounds=bounds,
#             maxfev=200000
#         )
#     except Exception as e:
#         print(f"\nOD: {od_key} fit failed: {e}")
#         continue
#
#     y_pred = model_struct_cong(Xdata, *params)
#     ss_res = np.sum((ydata - y_pred) ** 2)
#     ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
#     r2 = 1.0 - ss_res / (ss_tot + 1e-12)
#
#     a0, w, b1, b2, b3 = params
#
#     print(f"\nOD: {od_key} (dest={dest_mode}, crane_pool={Ncr_col})")
#     print("R^2:", round(r2, 4))
#     print("Parameters [a0, w, b1, b2, b3]:")
#     print(np.round(params, 6))
#
#     # 可直接放论文的公式（用你拟合出来的系数）
#     # t = a0 + w*Delta_dest + b1*V_od + b2*rho_c/(1-rho_c) + b3*rho_h/(1-rho_h)
#     print("Formula:")
#     print(
#         f"t = {a0:.6f} + ({w:.6f})*Delta_dest + ({b1:.6f})*V_od"
#         f" + ({b2:.6f})*(rho_c/(1-rho_c)) + ({b3:.6f})*(rho_h/(1-rho_h))"
#     )
#
# print("\nDone.")


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.optimize import curve_fit
# import warnings
# warnings.filterwarnings("ignore")
#
# FILE_PATH = "/Users/qianqiantong/PycharmProjects/LIFTS/multimodal/output/experiment_results_small.csv"
#
# # ======================================
# # Load data
# # ======================================
# df = pd.read_csv(FILE_PATH)
# print("Original rows:", len(df))
#
# # ======================================
# # Weekly volumes
# # ======================================
# df["train_weekly_volume"] = df["train_weekly_num"] * df["train_batch_size"]
# df["vessel_weekly_volume"] = df["vessel_weekly_num"] * df["vessel_batch_size"]
# df["truck_weekly_volume_in"] = df["truck_weekly_volume"]
#
# df["total_weekly_volume"] = (
#     df["train_weekly_volume"] +
#     df["vessel_weekly_volume"] +
#     df["truck_weekly_volume_in"]
# )
#
# # ======================================
# # Resource variables
# # ======================================
# df["berth_cranes"] = df["cranes_per_berth"] * df["berth_number"]
# df["track_cranes"] = df["cranes_per_track"] * df["track_number"]
#
# # ======================================
# # OD flow construction (share implicit)
# # ======================================
# s = df["vessel_to_train_share"].astype(float)
#
# df["V_T"] = df["vessel_weekly_volume"] * s
# df["V_H"] = df["vessel_weekly_volume"] - df["V_T"]
#
# df["T_V"] = df["V_T"]
# df["T_H_raw"] = df["train_weekly_volume"] - df["T_V"]
# df["T_H"] = df["T_H_raw"].clip(lower=0)
#
# df["residual_truck"] = df["V_H"] + df["T_H"]
#
# eps = 1e-9
# scale = np.where(df["residual_truck"] > eps,
#                  df["truck_weekly_volume_in"] / df["residual_truck"],
#                  0.0)
#
# df["H_V"] = df["V_H"] * scale
# df["H_T"] = df["T_H"] * scale
#
# # ======================================
# # Utilization variables (proxy)
# # ======================================
# CRANE_SERVICE_RATE = 1.0
# HOSTLER_SERVICE_RATE = 1.0
#
# df["hostler_util"] = df["total_weekly_volume"] / (df["hostler_number"] * HOSTLER_SERVICE_RATE + 1e-9)
# df["berth_util"] = df["vessel_weekly_volume"] / (df["berth_cranes"] * CRANE_SERVICE_RATE + 1e-9)
# df["track_util"] = df["train_weekly_volume"] / (df["track_cranes"] * CRANE_SERVICE_RATE + 1e-9)
#
# # ======================================
# # Clean processing time
# # ======================================
# def clean_processing_time(df_in, y_col):
#     df_sub = df_in.copy()
#     df_sub = df_sub[np.isfinite(df_sub[y_col])]
#     df_sub = df_sub[df_sub[y_col] > 0]
#
#     q1 = df_sub[y_col].quantile(0.25)
#     q3 = df_sub[y_col].quantile(0.75)
#     iqr = q3 - q1
#
#     lower = q1 - 1.5 * iqr
#     upper = q3 + 1.5 * iqr
#
#     df_sub = df_sub[(df_sub[y_col] >= lower) & (df_sub[y_col] <= upper)]
#     return df_sub
#
# # ======================================
# # OD targets
# # ======================================
# OD_TARGETS = {
#     "V_T": "vessel_to_train.avg_container_processing_time.mean",
#     "V_H": "vessel_to_truck.avg_container_processing_time.mean",
#     "T_V": "train_to_vessel.avg_container_processing_time.mean",
#     "T_H": "train_to_truck.avg_container_processing_time.mean",
#     "H_T": "truck_to_train.avg_container_processing_time.mean",
#     "H_V": "truck_to_vessel.avg_container_processing_time.mean",
# }
#
# BERTH_OD = {"V_T", "V_H", "T_V", "H_V"}
# TRACK_OD = {"T_H", "H_T"}
#
# # ======================================
# # Queue-consistent transforms
# # ======================================
# def _safe_clip_rho(rho, hi=0.97):
#     return np.clip(np.asarray(rho, dtype=float), 1e-9, hi)
#
# def queue_x(rho):
#     r = _safe_clip_rho(rho)
#     return r / (1.0 - r)
#
# # ======================================
# # Queue-consistent model with share modulating OD
# # T = a0 + b1*OD + b4*(OD*s) + b2*Q(rho_c) + b3*Q(rho_h)
# # ======================================
# def qc_share_mod_od(X, a0, b1, b4, b2, b3):
#     OD_flow, share, rho_c, rho_h = X
#     xc = queue_x(rho_c)
#     xh = queue_x(rho_h)
#     return a0 + b1 * OD_flow + b4 * (OD_flow * share) + b2 * xc + b3 * xh
#
# print("\n========== Queue-Consistent Fit (share modulates OD) ==========")
#
# for od_key, y_col in OD_TARGETS.items():
#     if y_col not in df.columns:
#         continue
#
#     crane_util_col = "berth_util" if od_key in BERTH_OD else "track_util"
#
#     df_sub = clean_processing_time(df, y_col)
#
#     needed = [y_col, od_key, "vessel_to_train_share", crane_util_col, "hostler_util"]
#     df_sub = df_sub.dropna(subset=needed)
#
#     # clip rho for stability
#     df_sub[crane_util_col] = _safe_clip_rho(df_sub[crane_util_col].values, hi=0.97)
#     df_sub["hostler_util"] = _safe_clip_rho(df_sub["hostler_util"].values, hi=0.97)
#
#     if len(df_sub) < 30:
#         print(f"{od_key}: Not enough samples")
#         continue
#
#     Xdata = np.vstack([
#         df_sub[od_key].values.astype(float),
#         df_sub["vessel_to_train_share"].values.astype(float),
#         df_sub[crane_util_col].values.astype(float),
#         df_sub["hostler_util"].values.astype(float),
#     ])
#     ydata = df_sub[y_col].values.astype(float)
#
#     # bounds: [a0, b1, b4, b2, b3]
#     bounds = (
#         [-1e3, -1e3, -1e3, -1e4, -1e4],
#         [ 1e3,  1e3,  1e3,  1e4,  1e4],
#     )
#
#     try:
#         params, _ = curve_fit(
#             qc_share_mod_od,
#             Xdata,
#             ydata,
#             bounds=bounds,
#             maxfev=200000
#         )
#     except Exception as e:
#         print(f"\nOD: {od_key} fit failed: {e}")
#         continue
#
#     y_pred = qc_share_mod_od(Xdata, *params)
#     ss_res = np.sum((ydata - y_pred) ** 2)
#     ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
#     r2 = 1 - ss_res / (ss_tot + 1e-12)
#
#     print(f"\nOD: {od_key}")
#     print("R^2:", round(r2, 4))
#     print("Parameters [a0, b1, b4, b2, b3]:")
#     print(np.round(params, 6))
#
# print("\nPipeline completed.")


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.optimize import curve_fit
# from scipy.stats import pearsonr
# import warnings
#
# warnings.filterwarnings("ignore")
#
# FILE_PATH = "/Users/qianqiantong/PycharmProjects/LIFTS/multimodal/output/experiment_results_small.csv"
#
# # ======================================
# # Load data
# # ======================================
# df = pd.read_csv(FILE_PATH)
# print("Original rows:", len(df))
#
# # ======================================
# # Weekly volumes
# # ======================================
# df["train_weekly_volume"] = df["train_weekly_num"] * df["train_batch_size"]
# df["vessel_weekly_volume"] = df["vessel_weekly_num"] * df["vessel_batch_size"]
# df["truck_weekly_volume_in"] = df["truck_weekly_volume"]
#
# df["total_weekly_volume"] = (
#     df["train_weekly_volume"] +
#     df["vessel_weekly_volume"] +
#     df["truck_weekly_volume_in"]
# )
#
# # ======================================
# # Resource variables
# # ======================================
# df["berth_cranes"] = df["cranes_per_berth"] * df["berth_number"]
# df["track_cranes"] = df["cranes_per_track"] * df["track_number"]
# df["total_cranes"] = df["berth_cranes"] + df["track_cranes"]
#
# # ======================================
# # OD flow construction (share implicit)
# # ======================================
# share = df["vessel_to_train_share"]
#
# df["V_T"] = df["vessel_weekly_volume"] * share
# df["V_H"] = df["vessel_weekly_volume"] - df["V_T"]
#
# df["T_V"] = df["V_T"]
# df["T_H_raw"] = df["train_weekly_volume"] - df["T_V"]
# df["T_H"] = df["T_H_raw"].clip(lower=0)
#
# df["residual_truck"] = df["V_H"] + df["T_H"]
#
# eps = 1e-9
# scale = np.where(
#     df["residual_truck"] > eps,
#     df["truck_weekly_volume_in"] / df["residual_truck"],
#     0
# )
#
# df["H_V"] = df["V_H"] * scale
# df["H_T"] = df["T_H"] * scale
#
# # ======================================
# # Utilization variables (improved proxy)
# # ======================================
# # Assume service rates (can adjust if known)
# CRANE_SERVICE_RATE = 1.0
# HOSTLER_SERVICE_RATE = 1.0
#
# df["hostler_util"] = (
#     df["total_weekly_volume"] /
#     (df["hostler_number"] * HOSTLER_SERVICE_RATE + 1e-9)
# )
#
# df["berth_util"] = (
#     df["vessel_weekly_volume"] /
#     (df["berth_cranes"] * CRANE_SERVICE_RATE + 1e-9)
# )
#
# df["track_util"] = (
#     df["train_weekly_volume"] /
#     (df["track_cranes"] * CRANE_SERVICE_RATE + 1e-9)
# )
#
# # ======================================
# # Clean processing time
# # ======================================
# def clean_processing_time(df, y_col):
#     df_sub = df.copy()
#     df_sub = df_sub[np.isfinite(df_sub[y_col])]
#     df_sub = df_sub[df_sub[y_col] > 0]
#
#     q1 = df_sub[y_col].quantile(0.25)
#     q3 = df_sub[y_col].quantile(0.75)
#     iqr = q3 - q1
#
#     lower = q1 - 1.5 * iqr
#     upper = q3 + 1.5 * iqr
#
#     df_sub = df_sub[(df_sub[y_col] >= lower) & (df_sub[y_col] <= upper)]
#     return df_sub
#
#
# # ======================================
# # OD targets
# # ======================================
# OD_TARGETS = {
#     "V_T": "vessel_to_train.avg_container_processing_time.mean",
#     "V_H": "vessel_to_truck.avg_container_processing_time.mean",
#     "T_V": "train_to_vessel.avg_container_processing_time.mean",
#     "T_H": "train_to_truck.avg_container_processing_time.mean",
#     "H_T": "truck_to_train.avg_container_processing_time.mean",
#     "H_V": "truck_to_vessel.avg_container_processing_time.mean",
# }
#
# BERTH_OD = {"V_T", "V_H", "T_V", "H_V"}
# TRACK_OD = {"T_H", "H_T"}
#
# # ======================================
# # Correlation Heatmap (no share)
# # ======================================
# corr_results = []
#
# for od_key, y_col in OD_TARGETS.items():
#
#     if y_col not in df.columns:
#         continue
#
#     crane_util_col = "berth_util" if od_key in BERTH_OD else "track_util"
#
#     df_sub = clean_processing_time(df, y_col)
#
#     needed = [
#         y_col,
#         od_key,
#         crane_util_col,
#         "hostler_util"
#     ]
#
#     df_sub = df_sub.dropna(subset=needed)
#
#     if len(df_sub) < 20:
#         continue
#
#     variables = [
#         od_key,
#         crane_util_col,
#         "hostler_util"
#     ]
#
#     for var in variables:
#         pear = pearsonr(df_sub[var], df_sub[y_col])[0]
#         corr_results.append({
#             "OD": od_key,
#             "Variable": var,
#             "Correlation": pear
#         })
#
# corr_df = pd.DataFrame(corr_results)
# pivot_table = corr_df.pivot(index="OD", columns="Variable", values="Correlation")
#
# plt.figure(figsize=(10,6))
# sns.heatmap(
#     pivot_table,
#     annot=True,
#     cmap="coolwarm",
#     center=0,
#     fmt=".2f"
# )
# plt.title("Correlation Heatmap Across 6 OD Modes")
# plt.tight_layout()
# plt.show()
#
# # ======================================
# # Nonlinear performance function
# # ======================================
# def nonlinear_model(X, a0, a1, a2, a3, g, d, h):
#     OD_flow, crane_util, hostler_util = X
#     return (
#         a0
#         + a1 * (OD_flow ** g)
#         + a2 * (crane_util ** d)
#         + a3 * (hostler_util ** h)
#     )
#
# print("\n========== Nonlinear Fit Results ==========")
#
# for od_key, y_col in OD_TARGETS.items():
#
#     if y_col not in df.columns:
#         continue
#
#     crane_util_col = "berth_util" if od_key in BERTH_OD else "track_util"
#
#     df_sub = clean_processing_time(df, y_col)
#
#     needed = [
#         y_col,
#         od_key,
#         crane_util_col,
#         "hostler_util"
#     ]
#
#     df_sub = df_sub.dropna(subset=needed)
#
#     if len(df_sub) < 30:
#         print(f"{od_key}: Not enough samples")
#         continue
#
#     Xdata = np.vstack([
#         df_sub[od_key].values,
#         df_sub[crane_util_col].values,
#         df_sub["hostler_util"].values
#     ])
#
#     ydata = df_sub[y_col].values
#
#     bounds = (
#         [0, -1e6, -1e6, -1e6, 0.5, 0.5, 0.5],
#         [1e5,  1e6,  1e6,  1e6, 3,   3,   3]
#     )
#
#     params, _ = curve_fit(
#         nonlinear_model,
#         Xdata,
#         ydata,
#         bounds=bounds,
#         maxfev=50000
#     )
#
#     y_pred = nonlinear_model(Xdata, *params)
#
#     ss_res = np.sum((ydata - y_pred) ** 2)
#     ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
#     r2 = 1 - ss_res / ss_tot
#
#     print(f"\nOD: {od_key}")
#     print("R^2:", round(r2,4))
#     print("Parameters [a0, a1, a2, a3, g, d, h]:")
#     print(np.round(params,4))
#
# print("\nPipeline completed.")
