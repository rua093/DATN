import pandas as pd

# đọc file
df = pd.read_csv("../data/dataset_clean.csv")

# ép kiểu DATE về datetime (rất quan trọng)
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", dayfirst=True)

# thêm cột WEEKDAY (0 = Thứ 2, ..., 6 = Chủ Nhật)
df["WEEKDAY"] = df["DATE"].dt.weekday

# nếu muốn tên thứ thay vì số:
df["WEEKDAY_NAME"] = df["DATE"].dt.day_name()

# lưu lại
df.to_csv("../data/dataset_clean.csv", index=False)
