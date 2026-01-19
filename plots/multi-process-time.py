# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# import os
#
#
# file_path   = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_track_results/input/results.xlsx"
# output_path = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_track_results/node_reversed_processing_output.xlsx"
#
# targets = {
#     "IC": "avg_ic_processing_time",
#     "OC": "avg_oc_processing_time"
# }
#
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#
# # ==========================================================
# # LOAD MULTI-SHEET DATA
# # ==========================================================
# xls = pd.ExcelFile(file_path)
# valid_sheets = [s for s in xls.sheet_names if s.strip().lower() in
#                 ["1 track", "2 tracks", "3 tracks", "4 tracks", "5 tracks"]]
#
# df_raw_list = []
# for sheet in valid_sheets:
#     t = pd.read_excel(file_path, sheet_name=sheet)
#     t["sheet"] = sheet
#     df_raw_list.append(t)
#
# df_raw = pd.concat(df_raw_list, ignore_index=True)
# df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(" ", "_")
#
#
# # ==========================================================
# # AGGREGATION + CALCULATE b1
# # ==========================================================
# agg_cols = [
#     "track_number",
#     "cranes_per_track",
#     "hostler_number",
#     "train_batch_size"
# ]
#
# df = df_raw.groupby(agg_cols).agg({
#     "avg_ic_processing_time": "mean",
#     "avg_oc_processing_time": "mean",
# }).reset_index()
#
# # ---- b1 = processing_time / batch_size ----
# df["ic_b1"] = df["avg_ic_processing_time"] / df["train_batch_size"]
# df["oc_b1"] = df["avg_oc_processing_time"] / df["train_batch_size"]
#
#
# # ==========================================================
# # FIT MODEL: b1 = f(1/T, 1/C, 1/H, 1/(TC), 1/(TH), 1/(CH), squares)
# # ==========================================================
# def fit_b1(df, b1_col):
#
#     df2 = df.copy()
#
#     # ---- We ONLY use inverse resource variables ----
#     df2["inv_track"]   = 1 / df2["track_number"]
#     df2["inv_crane"]   = 1 / df2["cranes_per_track"]
#     df2["inv_hostler"] = 1 / df2["hostler_number"]
#
#     df2["inv_track_crane"]   = df2["inv_track"] * df2["inv_crane"]
#     df2["inv_track_hostler"] = df2["inv_track"] * df2["inv_hostler"]
#     df2["inv_crane_hostler"] = df2["inv_crane"] * df2["inv_hostler"]
#
#     # allow quadratic inverse terms
#     df2["inv_track2"]   = df2["inv_track"] ** 2
#     df2["inv_crane2"]   = df2["inv_crane"] ** 2
#     df2["inv_hostler2"] = df2["inv_hostler"] ** 2
#
#     feature_cols = [
#         "inv_track",
#         "inv_crane",
#         "inv_hostler",
#         "inv_track_crane",
#         "inv_track_hostler",
#         "inv_crane_hostler",
#         "inv_track2",
#         "inv_crane2",
#         "inv_hostler2",
#     ]
#
#     X = df2[feature_cols].values
#     y = df2[b1_col].values
#
#     # Standardize
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)
#
#     # Linear regression (no polynomial since features already structured)
#     model = LinearRegression().fit(Xs, y)
#     y_pred = model.predict(Xs)
#     r2 = r2_score(y, y_pred)
#
#     coef_df = pd.DataFrame({
#         "feature": feature_cols + ["intercept"],
#         "coefficient": list(model.coef_) + [model.intercept_]
#     })
#
#     print("\n==============================")
#     print(f"b1 equation for {b1_col}")
#     print("==============================")
#     for _, row in coef_df.iterrows():
#         print(f"{row['coefficient']:.6f} * {row['feature']}")
#     print("==============================")
#     print(f"R² = {r2:.4f}\n")
#
#     return model, coef_df, y_pred, r2, feature_cols
#
#
# # Run both IC and OC
# ic_model, ic_coef_df, ic_pred, ic_r2, feature_cols = fit_b1(df, "ic_b1")
# oc_model, oc_coef_df, oc_pred, oc_r2, _            = fit_b1(df, "oc_b1")
#
#
# # ==========================================================
# # SAVE RESULTS
# # ==========================================================
# with pd.ExcelWriter(output_path) as writer:
#
#     df_raw.to_excel(writer, sheet_name="raw_data", index=False)
#     df.to_excel(writer, sheet_name="aggregated_data", index=False)
#
#     ic_coef_df.to_excel(writer, sheet_name="IC_b1_coefficients", index=False)
#     oc_coef_df.to_excel(writer, sheet_name="OC_b1_coefficients", index=False)
#
#     pd.DataFrame({"true": df["ic_b1"], "pred": ic_pred}).to_excel(
#         writer, sheet_name="IC_b1_pred_vs_true", index=False)
#
#     pd.DataFrame({"true": df["oc_b1"], "pred": oc_pred}).to_excel(
#         writer, sheet_name="OC_b1_pred_vs_true", index=False)
#
# print("\nDONE! Node Performance Function output saved to:")
# print(output_path)


# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# import os
#
#
# file_path   = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_track_results/input/results.xlsx"
# output_path = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_track_results/node_processing_output.xlsx"
#
# targets = {
#     "IC": "avg_ic_processing_time",
#     "OC": "avg_oc_processing_time"
# }
#
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#
# # ==========================================================
# # LOAD MULTI-SHEET DATA
# # ==========================================================
# xls = pd.ExcelFile(file_path)
# valid_sheets = [s for s in xls.sheet_names if s.strip().lower() in
#                 ["1 track", "2 tracks", "3 tracks", "4 tracks", "5 tracks"]]
#
# df_raw_list = []
# for sheet in valid_sheets:
#     t = pd.read_excel(file_path, sheet_name=sheet)
#     t["sheet"] = sheet
#     df_raw_list.append(t)
#
# df_raw = pd.concat(df_raw_list, ignore_index=True)
# df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(" ", "_")
#
#
# # ==========================================================
# # AGGREGATION
# # ==========================================================
# agg_cols = [
#     "track_number",
#     "cranes_per_track",
#     "hostler_number",
#     "train_batch_size"
# ]
#
# df = df_raw.groupby(agg_cols).agg({
#     "avg_ic_processing_time": "mean",
#     "avg_oc_processing_time": "mean",
# }).reset_index()
#
#
# # ==========================================================
# # FIT MODEL (2nd order polynomial)
# # ==========================================================
# def fit_model(df, target):
#     y = df[target].values
#
#     feature_cols = [
#         "track_number",
#         "cranes_per_track",
#         "hostler_number",
#         "train_batch_size"
#     ]
#
#     X = df[feature_cols].values
#
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)
#
#     poly = PolynomialFeatures(degree=2, include_bias=False)
#     X_poly = poly.fit_transform(Xs)
#
#     model = LinearRegression().fit(X_poly, y)
#     y_pred = model.predict(X_poly)
#
#     r2 = r2_score(y, y_pred)
#
#     # ========== Build Coefficient Table ==========
#     coef_names = poly.get_feature_names_out(feature_cols)
#     coef_df = pd.DataFrame({"feature": coef_names, "coefficient": model.coef_})
#     coef_df.loc[len(coef_df)] = ["intercept", model.intercept_]
#
#     # ========== Human-readable Equation Printing ==========
#     print("\n==============================")
#     print(f"Equation for {target}")
#     print("==============================")
#
#     print(f"{target} = ")
#
#     # Print each term
#     for _, row in coef_df.iterrows():
#         f = row["feature"]
#         c = row["coefficient"]
#         if f == "intercept":
#             print(f"  {c:.6f}")
#         else:
#             print(f"  + {c:.6f} * {f}")
#
#     print("==============================\n")
#
#     return model, poly, coef_df, y_pred, r2, feature_cols
#
# ic_model, ic_poly, ic_coef, ic_pred, ic_r2, feature_cols = fit_model(df, targets["IC"])
# oc_model, oc_poly, oc_coef, oc_pred, oc_r2, _            = fit_model(df, targets["OC"])
#
# print(f"[IC] R² = {ic_r2:.4f}")
# print(f"[OC] R² = {oc_r2:.4f}")
#
#
# # ==========================================================
# # (1) GENERATE EQUATION STRING
# # ==========================================================
# def generate_equation(coef_df, target_name):
#     """Convert coefficients into a readable polynomial equation"""
#     terms = []
#     for _, row in coef_df.iterrows():
#         f = row["feature"]
#         c = row["coefficient"]
#
#         if f == "intercept":
#             terms.append(f"{c:.6f}")
#         else:
#             terms.append(f"{c:.6f} * {f}")
#
#     eq = f"{target_name} = " + " +\n    ".join(terms)
#     return eq
#
#
# ic_equation = generate_equation(ic_coef, "avg_ic_processing_time")
# oc_equation = generate_equation(oc_coef, "avg_oc_processing_time")
#
#
# # ==========================================================
# # (2) NORMALIZED IMPORTANCE HEATMAP (sum to 1)
# # ==========================================================
# def var_heatmap(coef_df, tag):
#     lin = coef_df[coef_df["feature"].isin(feature_cols)].copy()
#     lin["raw_importance"] = lin["coefficient"].abs()
#
#     # normalize to 1
#     lin["importance"] = lin["raw_importance"] / lin["raw_importance"].sum()
#
#     heat_df = lin.pivot_table(index="feature", values="importance")
#
#     plt.figure(figsize=(6,4))
#     sns.heatmap(heat_df, cmap="viridis", annot=True, fmt=".3f")
#     plt.title(f"{tag} Normalized Significance (sum=1)")
#     plt.tight_layout()
#     plt.savefig(output_path.replace(".xlsx", f"_{tag}_heatmap.png"))
#     plt.close()
#
#     return heat_df, lin
#
#
# heatmap_ic, ic_importance_table = var_heatmap(ic_coef, "IC")
# heatmap_oc, oc_importance_table = var_heatmap(oc_coef, "OC")
#
#
# # ==========================================================
# # SAVE EVERYTHING TO EXCEL
# # ==========================================================
# with pd.ExcelWriter(output_path) as writer:
#     df_raw.to_excel(writer, sheet_name="raw_data", index=False)
#     df.to_excel(writer, sheet_name="aggregated_data", index=False)
#
#     pd.DataFrame({"true": df[targets["IC"]], "pred": ic_pred}).to_excel(
#         writer, sheet_name="ic_pred_vs_true", index=False)
#     ic_coef.to_excel(writer, sheet_name="ic_coefficients", index=False)
#     heatmap_ic.to_excel(writer, sheet_name="ic_heatmap", index=True)
#     ic_importance_table.to_excel(writer, sheet_name="ic_importance_table", index=False)
#
#     pd.DataFrame({"true": df[targets["OC"]], "pred": oc_pred}).to_excel(
#         writer, sheet_name="oc_pred_vs_true", index=False)
#     oc_coef.to_excel(writer, sheet_name="oc_coefficients", index=False)
#     heatmap_oc.to_excel(writer, sheet_name="oc_heatmap", index=True)
#     oc_importance_table.to_excel(writer, sheet_name="oc_importance_table", index=False)
#
# print("\nDONE! Node Performance Function output saved to:")
# print(output_path)


# reversed, full processing time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os


file_path   = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_track_results/input/results.xlsx"
output_path = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_track_results/node_reversed_processing_output.xlsx"

targets = {
    "IC": "avg_ic_processing_time",
    "OC": "avg_oc_processing_time"
}

os.makedirs(os.path.dirname(output_path), exist_ok=True)


# ==========================================================
# LOAD MULTI-SHEET DATA
# ==========================================================
xls = pd.ExcelFile(file_path)
valid_sheets = [s for s in xls.sheet_names if s.strip().lower() in
                ["1 track", "2 tracks", "3 tracks", "4 tracks", "5 tracks"]]

df_raw_list = []
for sheet in valid_sheets:
    t = pd.read_excel(file_path, sheet_name=sheet)
    t["sheet"] = sheet
    df_raw_list.append(t)

df_raw = pd.concat(df_raw_list, ignore_index=True)
df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(" ", "_")


# ==========================================================
# AGGREGATION
# ==========================================================
agg_cols = [
    "track_number",
    "cranes_per_track",
    "hostler_number",
    "train_batch_size"
]

df = df_raw.groupby(agg_cols).agg({
    "avg_ic_processing_time": "mean",
    "avg_oc_processing_time": "mean",
}).reset_index()


# ==========================================================
# FIT MODEL (倒数资源 + 二阶多项式)
# ==========================================================
def fit_model(df, target):
    y = df[target].values

    # -----------------------------
    # 资源项全部倒数
    # -----------------------------
    df_trans = df.copy()

    df_trans["inv_track"]   = 1 / df_trans["track_number"]
    df_trans["inv_crane"]   = 1 / df_trans["cranes_per_track"]
    df_trans["inv_hostler"] = 1 / df_trans["hostler_number"]

    # 非资源项保持线性
    df_trans["batch"] = df_trans["train_batch_size"]

    # 模型使用的新特征
    feature_cols = ["inv_track", "inv_crane", "inv_hostler", "batch"]

    # -----------------------------
    # 标准化 + 多项式
    # -----------------------------
    X = df_trans[feature_cols].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(Xs)

    # -----------------------------
    # 拟合线性模型
    # -----------------------------
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)

    # -----------------------------
    # 系数输出
    # -----------------------------
    coef_names = poly.get_feature_names_out(feature_cols)
    coef_df = pd.DataFrame({"feature": coef_names, "coefficient": model.coef_})
    coef_df.loc[len(coef_df)] = ["intercept", model.intercept_]

    # -----------------------------
    # 输出方程
    # -----------------------------
    print("\n==============================")
    print(f"Equation for {target}")
    print("==============================")
    print(f"{target} = ")
    for _, row in coef_df.iterrows():
        f = row["feature"]
        c = row["coefficient"]
        if f == "intercept":
            print(f"  {c:.6f}")
        else:
            print(f"  + {c:.6f} * {f}")
    print("==============================\n")

    return model, poly, coef_df, y_pred, r2, feature_cols


# Run models
ic_model, ic_poly, ic_coef, ic_pred, ic_r2, feature_cols = fit_model(df, targets["IC"])
oc_model, oc_poly, oc_coef, oc_pred, oc_r2, _            = fit_model(df, targets["OC"])

print(f"[IC] R² = {ic_r2:.4f}")
print(f"[OC] R² = {oc_r2:.4f}")


# ==========================================================
# (1) GENERATE EQUATION STRING
# ==========================================================
def generate_equation(coef_df, target_name):
    terms = []
    for _, row in coef_df.iterrows():
        f = row["feature"]
        c = row["coefficient"]

        if f == "intercept":
            terms.append(f"{c:.6f}")
        else:
            terms.append(f"{c:.6f} * {f}")

    eq = f"{target_name} = " + " +\n    ".join(terms)
    return eq


ic_equation = generate_equation(ic_coef, "avg_ic_processing_time")
oc_equation = generate_equation(oc_coef, "avg_oc_processing_time")


# ==========================================================
# (2) NORMALIZED IMPORTANCE HEATMAP
# ==========================================================
def var_heatmap(coef_df, tag):
    lin = coef_df[coef_df["feature"].isin(feature_cols)].copy()
    lin["raw_importance"] = lin["coefficient"].abs()
    lin["importance"] = lin["raw_importance"] / lin["raw_importance"].sum()

    heat_df = lin.pivot_table(index="feature", values="importance")

    plt.figure(figsize=(6,4))
    sns.heatmap(heat_df, cmap="viridis", annot=True, fmt=".3f")
    plt.title(f"{tag} Normalized Significance (sum=1)")
    plt.tight_layout()
    plt.savefig(output_path.replace(".xlsx", f"_{tag}_heatmap.png"))
    plt.close()

    return heat_df, lin


heatmap_ic, ic_importance_table = var_heatmap(ic_coef, "IC")
heatmap_oc, oc_importance_table = var_heatmap(oc_coef, "OC")


# ==========================================================
# SAVE EVERYTHING TO EXCEL
# ==========================================================
with pd.ExcelWriter(output_path) as writer:
    df_raw.to_excel(writer, sheet_name="raw_data", index=False)
    df.to_excel(writer, sheet_name="aggregated_data", index=False)

    pd.DataFrame({"true": df[targets["IC"]], "pred": ic_pred}).to_excel(
        writer, sheet_name="ic_pred_vs_true", index=False)
    ic_coef.to_excel(writer, sheet_name="ic_coefficients", index=False)
    heatmap_ic.to_excel(writer, sheet_name="ic_heatmap", index=True)
    ic_importance_table.to_excel(writer, sheet_name="ic_importance_table", index=False)

    pd.DataFrame({"true": df[targets["OC"]], "pred": oc_pred}).to_excel(
        writer, sheet_name="oc_pred_vs_true", index=False)
    oc_coef.to_excel(writer, sheet_name="oc_coefficients", index=False)
    heatmap_oc.to_excel(writer, sheet_name="oc_heatmap", index=True)
    oc_importance_table.to_excel(writer, sheet_name="oc_importance_table", index=False)

print("\nDONE! Node Performance Function output saved to:")
print(output_path)
