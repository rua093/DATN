import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Đọc file
df = pd.read_csv("../data/dataset_clean.csv")

# 1. Kiểm tra dữ liệu thiếu
print("===== Missing values (số lượng) =====")
print(df.isnull().sum())

print("\n===== Missing rows (chi tiết) =====")
missing_rows = df[df.isnull().any(axis=1)]
print(missing_rows)

# 2. Kiểm tra thống kê mô tả
print("\n===== Statistical Summary =====")
print(df.describe())

# 3. Vẽ boxplot để phát hiện outlier
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[["ENERGY", "TEMPERATURE", "HUMIDITY"]])
plt.title("Boxplot kiểm tra outlier")
plt.show()

# 4. Phát hiện outlier bằng IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return outliers

print("\n===== Outliers theo IQR =====")
for col in ["ENERGY", "HUMIDITY", "TEMPERATURE"]:
    outliers_iqr = detect_outliers_iqr(df, col)
    print(f"\n{col} có {len(outliers_iqr)} outlier (IQR):")
    print(outliers_iqr[["DATE", col]])

# 5. Phát hiện outlier bằng Z-score
print("\n===== Outliers theo Z-score =====")
for col in ["ENERGY", "HUMIDITY", "TEMPERATURE"]:
    # Bỏ NaN để tính z-score
    col_data = df[col].dropna()
    z_scores = np.abs(stats.zscore(col_data))
    outlier_index = col_data.index[np.where(z_scores > 3)[0]]
    outliers_z = df.loc[outlier_index, ["DATE", col]]
    print(f"\n{col} có {len(outliers_z)} outlier (Z-score > 3):")
    print(outliers_z)
