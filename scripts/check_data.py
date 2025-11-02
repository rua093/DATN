import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os

# Đọc file
df = pd.read_csv("data/dataset_clean.csv")

# Tạo thư mục lưu ảnh
os.makedirs("results", exist_ok=True)

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
plt.tight_layout()
plt.savefig("results/eda_boxplot_outliers.png", dpi=200)
plt.close()

# 3b. Phân bố các biến chính
for col in ["ENERGY", "TEMPERATURE", "HUMIDITY"]:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col].dropna(), kde=True, bins=50)
    plt.title(f"Phân bố {col}")
    plt.tight_layout()
    plt.savefig(f"results/eda_hist_{col.lower()}.png", dpi=200)
    plt.close()

# 3c. Time series ENERGY (nếu có DATE)
if "DATE" in df.columns:
    try:
        df_ts = df.copy()
        df_ts["DATE"] = pd.to_datetime(df_ts["DATE"], errors="coerce", dayfirst=True)
        plt.figure(figsize=(14, 4))
        plt.plot(df_ts["DATE"], df_ts["ENERGY"], linewidth=1)
        plt.title("ENERGY theo thời gian")
        plt.xlabel("Thời gian")
        plt.ylabel("ENERGY")
        plt.tight_layout()
        plt.savefig("results/eda_timeseries_energy.png", dpi=200)
        plt.close()
    except Exception:
        pass

# 3d. Boxplot theo WEEKDAY và HOUR nếu có
if set(["WEEKDAY", "HOUR"]).issubset(df.columns):
    plt.figure(figsize=(12, 5))
    sns.boxplot(x=df["WEEKDAY"], y=df["ENERGY"]) 
    plt.title("ENERGY theo ngày trong tuần (WEEKDAY)")
    plt.tight_layout()
    plt.savefig("results/eda_boxplot_energy_weekday.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.boxplot(x=df["HOUR"], y=df["ENERGY"]) 
    plt.title("ENERGY theo giờ (HOUR)")
    plt.tight_layout()
    plt.savefig("results/eda_boxplot_energy_hour.png", dpi=200)
    plt.close()

# 3e. Ma trận tương quan
cols_corr = [c for c in ["ENERGY", "TEMPERATURE", "HUMIDITY", "HOLIDAY"] if c in df.columns]
if len(cols_corr) >= 2:
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[cols_corr].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Ma trận tương quan")
    plt.tight_layout()
    plt.savefig("results/eda_correlation_heatmap.png", dpi=200)
    plt.close()

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
