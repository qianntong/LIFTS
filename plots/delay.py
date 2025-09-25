import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator

file_path = '/Users/qianqiantong/PycharmProjects/LIFTS/output/experiment_2.xlsx'
interested_batch_size = 20
container_type = "OC"  # choose "IC" or "OC"

sheet_name = "delay_ps"
if container_type == "IC":
    z = "ic_avg_delay_time"
elif container_type == "OC":
    z = "oc_avg_delay_time"
else:
    raise ValueError("container_type must be 'IC' or 'OC'")

x = "num_trains"
y = "train_batch_size"
group_cols = ["cranes", "hostlers"]

df = pd.read_excel(file_path, sheet_name=sheet_name)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

z = z.lower()
x = x.lower()
y = y.lower()
df = df[df[y] == interested_batch_size].copy()

def exponential_func(x, a, b):
    return a * np.exp(b * x)

plt.figure(figsize=(12, 8))
df["group_label"] = df.apply(lambda row: f"{int(row['cranes'])}C-{int(row['hostlers'])}H", axis=1)

unique_labels = df["group_label"].unique()
custom_colors = [
    "#e41a1c",  # 红
    "#377eb8",  # 蓝
    "#4daf4a",  # 绿
    "#ff7f00",  # 橙
    "#984ea3",  # 紫
    "#ffff33",  # 黄
    "#a65628",  # 棕
    "#f781bf",  # 粉
    "#999999",  # 灰
    "#66c2a5"   # 青绿
]

delay_coef_list = []

for idx, label in enumerate(unique_labels):
    group = df[df["group_label"] == label]

    x_data = pd.to_numeric(group[x], errors='coerce').values
    y_data = pd.to_numeric(group[z], errors='coerce').values

    valid = np.isfinite(x_data) & np.isfinite(y_data) & (y_data >= 0)
    x_data = x_data[valid]
    y_data = y_data[valid]

    if len(x_data) < 3:
        print(f"Skipping {label} due to insufficient data")
        continue

    try:
        popt, _ = curve_fit(exponential_func, x_data, y_data, p0=(1, 0.0001), maxfev=10000)
        a, b = popt
        y_pred = exponential_func(x_data, a, b)
        r2 = r2_score(y_data, y_pred)

        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = exponential_func(x_fit, a, b)

        color = custom_colors[idx % len(custom_colors)]

        plt.scatter(x_data, y_data, s=20, alpha=0.6, color=color)
        plt.plot(x_fit, y_fit, color=color, label=f'{label}: $y={a:.2f}e^{{{b:.4f}x}}$, $R^2={r2:.3f}$')
        plt.xlim(0, 21)
        plt.xticks(np.arange(0, 22, 1))
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().xaxis.set_major_locator(MultipleLocator(1))

        crane_str, hostler_str = label.split("C-")
        crane = int(crane_str)
        hostler = int(hostler_str.rstrip("H"))

        delay_coef_list.append({
            "crane": crane,
            "hostler": hostler,
            "a": a,
            "b": b,
            "r_2": r2
        })


    except Exception as e:
        print(f"Warning: fitting failed for {label}: {e}")
        continue

# === Final plot formatting ===
plt.xlabel("Trains/Platoons per Day", fontsize=14)
plt.ylabel(f"{container_type} Avg Delay Time (Hours)", fontsize=14)
# plt.title(f"Exponential Fit for Selected Resource Groups (Batch Size = {interested_batch_size})")
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# === Save delay coefficients ===
delay_coef_df = pd.DataFrame(delay_coef_list)
delay_coef_df.to_excel(f"/Users/qianqiantong/PycharmProjects/LIFTS/output/delay_coef_{container_type}.xlsx", index=False)
print("Fitting coefficients saved!")