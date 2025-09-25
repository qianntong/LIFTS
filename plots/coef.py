import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# === Step 1: Load data from Excel ===
mode = 'ic'
platoon = 'ps'
file_path = '/Users/qianqiantong/PycharmProjects/LIFTS/output/delay_coef_IC.xlsx'
df = pd.read_excel(file_path)

# === Step 2: Compute Pearson correlation matrix ===
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)
print("*"*10)

# # === Step 3: Plot correlation heatmap ===
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar=True)
plt.title(f"Correlation Heatmap: {mode} Delay Coefficient Data ({platoon})")
plt.tight_layout()
plt.show()

print("*"*10)


# # a: comb 1
# df['inv_hostlers'] = 1 / df['hostler']
# X = df[['batch_size', 'inv_hostlers']]
# y = df['a']
#
# model = LinearRegression()
# model.fit(X, y)
#
# print(f"Intercept (α₀): {model.intercept_:.4f}")
# print(f"batch_size coefficient (α₁): {abs(model.coef_[0]):.4f}")
# print(f"1/hostlers coefficient (α₂): {model.coef_[1]:.4f}")
# print(f"R²: {model.score(X, y):.4f}")
# print("*"*10)

# # a: comb 2
# df['inv_cranes'] = 1 / df['crane']
# df['inv_hostlers'] = 1 / df['hostler']
# X = df[['inv_cranes', 'inv_hostlers']]
# y = df['a']
#
# model = LinearRegression()
# model.fit(X, y)
#
# print(f"Intercept (α₀): {model.intercept_:.4f}")
# print(f"1/Cranes coefficient (α₁): {model.coef_[0]:.4f}")
# print(f"1/Hostlers coefficient (α₂): {model.coef_[1]:.4f}")
# # print(f"Batch size coefficient (α₃): {model.coef_[2]:.4f}")
# print(f"R²: {model.score(X, y):.4f}")
# print("*"*10)

# # a_combo: 3
# X_a1 = df[['batch_size']]
# y_a1 = df['a']
#
# model_a1 = LinearRegression()
# model_a1.fit(X_a1, y_a1)
#
# print(f"Intercept (α₀): {model_a1.intercept_:.4f}")
# print(f"Batch size coefficient (α₁): {abs(model_a1.coef_[0]):.4f}")
# print(f"R²: {model_a1.score(X_a1, y_a1):.4f}")
# print("*"*10)

# ## a: comb 4
# df['inv_hostlers'] = 1 / df['hostlers']
# X = df[['cranes', 'inv_hostlers', 'batch_size']]
# y = df['a']
#
# model = LinearRegression()
# model.fit(X, y)
#
# print(f"Intercept (α₀): {model.intercept_:.4f}")
# print(f"Cranes coefficient (α₁): {model.coef_[0]:.4f}")
# print(f"1/Hostlers coefficient (α₂): {model.coef_[1]:.4f}")
# print(f"Batch size coefficient (α₃): {abs(model.coef_[2]):.4f}")
# print(f"R²: {model.score(X, y):.4f}")
# print("*"*10)

# # b
# X_b1 = df[['batch_size']]  # X：二维
# y_b1 = df['b']
#
# model_b1 = LinearRegression()
# model_b1.fit(X_b1, y_b1)
#
# print(f"Intercept (β₀): {model_b1.intercept_:.4f}")
# print(f"Batch size coefficient (β₁): {model_b1.coef_[0]:.4f}")
# print(f"R²: {model_b1.score(X_b1, y_b1):.4f}")

# b: comb 2
df['inv_cranes'] = 1 / df['crane']
df['inv_hostlers'] = 1 / df['hostler']

X_b = df[['inv_cranes', 'inv_hostlers']]
y_b = df['b']

model_b = LinearRegression()
model_b.fit(X_b, y_b)

print(f"Intercept (β₀): {model_b.intercept_:.4f}")
print(f"1/Cranes coefficient (β₁): {model_b.coef_[0]:.4f}")
print(f"1/Hostlers coefficient (β₂): {model_b.coef_[1]:.4f}")
print(f"R²: {model_b.score(X_b, y_b):.4f}")
print("*" * 10)
