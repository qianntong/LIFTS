import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

file_path = "/Users/qianqiantong/PycharmProjects/LIFTS/output/multi_track_results/results.xlsx"
container_type = "IC"   # Choose "IC" or "OC"

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
if container_type.upper() == "IC":
    target_col = "avg_ic_processing_time"
else:
    target_col = "avg_oc_processing_time"

required_cols = [
    "track_number",
    "cranes_per_track",
    "hostler_number",
    "train_batch_size",
    "trains_per_day",
    target_col.lower()
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' is missing in the Excel file.")

df["container"] = df["train_batch_size"] * df["trains_per_day"]
df = df.dropna(subset=[target_col])

# Prepare data
X1 = df["track_number"].values
X2 = df["cranes_per_track"].values
X3 = df["hostler_number"].values
X4 = df["train_batch_size"].values
X5 = df["trains_per_day"].values
X6 = df["container"].values

Y = df[target_col].values

# Model form (multi-input nonlinear regression)
def model_func(Xdata, a0, a1, a2, a3, a4, a5, a6):
    x1, x2, x3, x4, x5, x6 = Xdata
    return (a0 +
            a1 * x1 +
            a2 * x2 +
            a3 * x3 +
            a4 * x4 +
            a5 * x6 +          # container column
            a6 * np.log(x5+1)  # log term helps fit exponential growth
            )

Xdata = np.vstack((X1, X2, X3, X4, X5, X6))

popt, pcov = curve_fit(model_func, Xdata, Y, maxfev=20000)
a0, a1, a2, a3, a4, a5, a6 = popt

Y_pred = model_func(Xdata, *popt)
r2 = r2_score(Y, Y_pred)

print("\n===== Fitted Model =====")
print(f"{target_col} = {a0:.4f}"
      f" + {a1:.4f}·track_number"
      f" + {a2:.4f}·cranes_per_track"
      f" + {a3:.4f}·hostler_number"
      f" + {a4:.4f}·train_batch_size"
      f" + {a5:.4f}·container"
      f" + {a6:.4f}·log(trains_per_day+1)")

print(f"R² = {r2:.4f}")

# Scatter plot versus prediction
plt.figure(figsize=(8, 6))
plt.scatter(Y, Y_pred, alpha=0.7, edgecolors='k')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title(f"Prediction Accuracy for {target_col} (R²={r2:.3f})")
plt.grid(True)
plt.tight_layout()
plt.show()

coef_df = pd.DataFrame({
    "parameter": ["a0", "a1", "a2", "a3", "a4", "a5", "a6"],
    "value": [a0, a1, a2, a3, a4, a5, a6]
})

output_path = f"/Users/qianqiantong/PycharmProjects/LIFTS/output/{container_type.lower()}_fitted_model.xlsx"
coef_df.to_excel(output_path, index=False)

print(f"\nCoefficients saved to: {output_path}")