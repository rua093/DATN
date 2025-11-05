from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pytz
import requests
import os

app = FastAPI(title="Energy Consumption Forecast API")

# Cho phép mobile app truy cập API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------- Config ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "my_lstm_model_woa.h5")
LAT, LON = 10.7769, 106.7009  # TP.HCM

# ---------------------------- Models ----------------------------
class HistoryItem(BaseModel):
    date: str
    energy_actual: float

class PredictionItem(BaseModel):
    date: str
    predicted: float

class HistoryResponse(BaseModel):
    history: list[HistoryItem]
    next_day: PredictionItem

# ---------------------------- Helpers ----------------------------
def preprocess_for_lstm(df):
    df_proc = df.copy()
    if "DATE" in df_proc.columns:
        df_proc = df_proc.drop(columns=["DATE"])
    
    # Chuyển sang int, NaN -> 1 (hoặc 0, tùy bạn)
    for col in ["DAY", "MONTH", "WEEKDAY"]:
        if col not in df_proc.columns:
            raise ValueError(f"Thiếu cột {col} trong dataset")
        df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(1).astype(int)
    
    # Encode cyclical features
    df_proc["DAY_sin"] = np.sin(2 * np.pi * df_proc["DAY"] / 31)
    df_proc["DAY_cos"] = np.cos(2 * np.pi * df_proc["DAY"] / 31)
    df_proc["MONTH_sin"] = np.sin(2 * np.pi * df_proc["MONTH"] / 12)
    df_proc["MONTH_cos"] = np.cos(2 * np.pi * df_proc["MONTH"] / 12)
    df_proc["WEEKDAY_sin"] = np.sin(2 * np.pi * df_proc["WEEKDAY"] / 7)
    df_proc["WEEKDAY_cos"] = np.cos(2 * np.pi * df_proc["WEEKDAY"] / 7)
    df_proc = df_proc.drop(columns=["DAY", "MONTH", "WEEKDAY"])
    
    target_col = "ENERGY_ADJ" if "ENERGY_ADJ" in df_proc.columns else "ENERGY"
    y = df_proc[[target_col]].values
    X = df_proc.values
    return X, y

def create_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
        ys.append(y[i+timesteps])
    return np.array(Xs), np.array(ys)

def split_scale_for_timesteps(X_raw, y_raw, timesteps):
    if len(X_raw) <= timesteps:
        raise ValueError(f"Dữ liệu quá ngắn để tạo sequence với timesteps={timesteps}")
    X_seq_raw, y_seq_raw = create_sequences(X_raw, y_raw, timesteps)
    num_features = X_seq_raw.shape[2]
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_seq_raw.reshape(-1, num_features)).reshape(X_seq_raw.shape)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_seq_raw)
    return X_scaled, y_scaled, scaler_X, scaler_y

def fetch_weather_next_day():
    """Lấy nhiệt độ và độ ẩm trung bình cho ngày mai từ Open-Meteo"""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}&longitude={LON}&hourly=temperature_2m,relativehumidity_2m&timezone=Asia/Ho_Chi_Minh"
    )
    res = requests.get(url, timeout=30).json()
    if "hourly" not in res:
        raise HTTPException(status_code=500, detail=f"Lỗi API thời tiết: {res}")
    
    tomorrow = (datetime.now() + timedelta(days=1)).date()
    times = pd.to_datetime(res["hourly"]["time"]).date
    temps = res["hourly"]["temperature_2m"]
    hums  = res["hourly"]["relativehumidity_2m"]
    
    temps_tomorrow = [temps[i] for i, t in enumerate(times) if t == tomorrow]
    hums_tomorrow  = [hums[i] for i, t in enumerate(times) if t == tomorrow]
    
    if not temps_tomorrow or not hums_tomorrow:
        raise HTTPException(status_code=500, detail="Không có dữ liệu thời tiết cho ngày mai")
    
    temp_avg = float(np.mean(temps_tomorrow))
    humidity_avg = float(np.mean(hums_tomorrow))
    return temp_avg, humidity_avg

def prepare_next_day_row(df_last, temp, humidity):
    """Tạo dòng mới cho ngày mai dựa trên ngày cuối cùng"""
    last_row = df_last.iloc[-1].copy()
    new_row = last_row.copy()
    new_row["DAY"] = (last_row["DAY"] % 31) + 1
    new_row["MONTH"] = (last_row["MONTH"] % 12) + 1
    new_row["WEEKDAY"] = (last_row["WEEKDAY"] % 7) + 1
    new_row["TEMPERATURE_AVG"] = temp
    new_row["HUMIDITY_AVG"] = humidity
    new_row["ENERGY"] = last_row["ENERGY"]  # placeholder
    new_row["ENERGY_ADJ"] = last_row.get("ENERGY_ADJ", last_row["ENERGY"])
    return new_row

def predict_next_day(model, df_full, timesteps=48):
    # Giảm timesteps nếu dataset quá nhỏ
    timesteps = min(timesteps, len(df_full)-1)
    X_raw, y_raw = preprocess_for_lstm(df_full)
    X_scaled, y_scaled, scaler_X, scaler_y = split_scale_for_timesteps(X_raw, y_raw, timesteps)
    last_seq = X_scaled[-1].reshape(1, timesteps, X_scaled.shape[2])
    y_pred_scaled = model.predict(last_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return float(y_pred[0][0])

# ---------------------------- API Endpoint ----------------------------
@app.get("/api/history", response_model=HistoryResponse)
def get_energy_history():
    """Trả về 7 ngày gần nhất + dự đoán ngày tiếp theo"""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Không tìm thấy file dữ liệu.")
    
    df = pd.read_csv(DATA_PATH, dtype={"DATE": str})
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["DATE"]).reset_index(drop=True)
    
    # Lấy 7 ngày gần nhất
    df_last = df.tail(7).copy()
    history = [
        HistoryItem(date=row["DATE"].strftime("%Y-%m-%d"), energy_actual=float(row["ENERGY"]))
        for _, row in df_last.iterrows()
    ]
    
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Không tìm thấy model đã huấn luyện.")
    model = load_model(MODEL_PATH)
    
    # Lấy dữ liệu thời tiết ngày mai
    temp_avg, humidity_avg = fetch_weather_next_day()
    new_row = prepare_next_day_row(df.tail(48), temp_avg, humidity_avg)  # dùng 48 ngày gần nhất
    
    df_for_pred = pd.concat([df.tail(48), new_row.to_frame().T], ignore_index=True)
    y_pred = predict_next_day(model, df_for_pred, timesteps=48)
    
    next_date = df_last["DATE"].max() + timedelta(days=1)
    next_item = PredictionItem(date=next_date.strftime("%Y-%m-%d"), predicted=y_pred)
    
    return HistoryResponse(history=history, next_day=next_item)
