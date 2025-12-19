from openai import OpenAI
from server.config import OPENAI_API_KEY, MODELS_DIR
from server.database import get_account_by_username, load_consumption_dataframe
from server.services.training_service import TrainingService
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Optional

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================================================
# 1. GỌI LLM – CHỈ DIỄN GIẢI, KHÔNG DỰ ĐOÁN
# =========================================================
def ask_energy_ai(question: str, context: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý phân tích điện năng. "
                    "CHỈ sử dụng dữ liệu được cung cấp. "
                    "KHÔNG suy đoán, KHÔNG tạo số liệu mới."
                )
            },
            {
                "role": "user",
                "content": f"DỮ LIỆU THỰC:\n{context}\n\nCÂU HỎI:\n{question}"
            }
        ],
        temperature=0.0  # tuyệt đối không sáng tạo
    )
    return response.choices[0].message.content.strip()


# =========================================================
# 2. LẤY TOÀN BỘ LỊCH SỬ TIÊU THỤ TỪ DB
# =========================================================
def get_all_history(db, evn_username: str) -> list:
    try:
        df = load_consumption_dataframe(db, evn_username)
        if df.empty:
            return []

        df["DATE_PARSED"] = pd.to_datetime(
            df["DATE"], dayfirst=True, errors="coerce"
        )
        df = df.dropna(subset=["DATE_PARSED"]).sort_values("DATE_PARSED")

        return [
            {
                "date": r["DATE_PARSED"].date().isoformat(),
                "consumption_kwh": float(r["ENERGY"])
            }
            for _, r in df.iterrows()
            if pd.notna(r["ENERGY"])
        ]
    except Exception:
        return []


# =========================================================
# 3. LẤY TOÀN BỘ KẾT QUẢ CUỐI CỦA LSTM
# =========================================================
def compute_lstm_forecast(
    evn_username: str,
    db,
    horizon: int
) -> Optional[list]:
    try:
        model_dir = MODELS_DIR / f"user_{evn_username}"
        model_path = model_dir / "lstm_model.h5"
        scaler_path = model_dir / "scaler_x.pkl"

        if not model_path.exists() or not scaler_path.exists():
            return None

        model = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)

        ts = TrainingService()
        acc = get_account_by_username(db, evn_username)
        location = acc.location if acc else "Ho Chi Minh City"

        df = ts.build_dataset_from_db(db, evn_username, location)
        if df.empty:
            return None

        df_processed = ts.preprocess_for_base_model(df)
        timesteps = 7
        if len(df_processed) < timesteps + 1:
            return None

        current_window = df_processed.iloc[-timesteps:].copy()
        predictions = []

        for _ in range(horizon):
            window_scaled = scaler.transform(current_window.values)
            x_in = np.expand_dims(window_scaled, axis=0)
            y_hat_scaled = model.predict(x_in, verbose=0)

            n_feat = window_scaled.shape[1] - 1
            dummy = np.zeros((1, n_feat + 1))
            dummy[0, -1] = y_hat_scaled[0, 0]
            y_hat = scaler.inverse_transform(dummy)[0, -1]
            predictions.append(float(y_hat))

            next_row = current_window.iloc[-1].copy()
            next_row["ENERGY_ADJ"] = y_hat
            current_window = pd.concat(
                [current_window.iloc[1:], next_row.to_frame().T],
                ignore_index=True
            )

        return predictions
    except Exception:
        return None


# =========================================================
# 4. BUILD CONTEXT = DB + LSTM (KHỚP UI)
# =========================================================
def build_user_context(
    evn_username: str,
    db,
    forecast_horizon: int
) -> str:
    acc = get_account_by_username(db, evn_username)
    location = acc.location if acc and acc.location else "Unknown"

    history = get_all_history(db, evn_username)
    forecasts = compute_lstm_forecast(
        evn_username, db, horizon=forecast_horizon
    )

    parts = []
    parts.append(f"USER: {evn_username}")
    parts.append(f"LOCATION: {location}")

    parts.append("\n=== FULL CONSUMPTION HISTORY (DATABASE) ===")
    if history:
        for r in history:
            parts.append(f"{r['date']}: {r['consumption_kwh']} kWh")
    else:
        parts.append("No consumption data available")

    parts.append("\n=== FINAL LSTM FORECAST RESULTS ===")
    if forecasts:
        for i, v in enumerate(forecasts, 1):
            parts.append(f"Day +{i}: {v:.2f} kWh")
    else:
        parts.append("No forecast available")

    return "\n".join(parts)


# =========================================================
# 5. API ENTRY
# =========================================================
def ask_energy_ai_for_user(
    evn_username: str,
    question: str,
    db,
    forecast_horizon: int
) -> str:
    context = build_user_context(
        evn_username=evn_username,
        db=db,
        forecast_horizon=forecast_horizon
    )
    return ask_energy_ai(question, context)
