from openai import OpenAI
from server.config import OPENAI_API_KEY, MODELS_DIR
from server.database import get_account_by_username, load_consumption_dataframe
from server.services.training_service import TrainingService
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Optional
from datetime import timedelta

client = OpenAI(api_key=OPENAI_API_KEY)


def ask_energy_ai(question: str, context: str = "") -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Bạn là trợ lý thân thiện phân tích tiêu thụ điện năng cho người dùng, gợi ý tiết kiệm điện"
            },
            {
                "role": "user",
                "content": f"""Dữ liệu:
{context}

Câu hỏi:
{question}
"""
            }
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def compute_short_forecast(evn_username: str, db, horizon: int = 1) -> Optional[list]:
    """Trả về danh sách dự đoán (float kWh) cho `horizon` ngày tiếp theo nếu có model và đủ dữ liệu."""
    try:
        model_dir = MODELS_DIR / f"user_{evn_username}"
        model_path = model_dir / "lstm_model.h5"
        scaler_path = model_dir / "scaler_x.pkl"
        if not (model_path.exists() and scaler_path.exists()):
            return None

        model = tf.keras.models.load_model(str(model_path), compile=False)
        scaler = joblib.load(str(scaler_path))

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

        # iterative forecast
        current_window = df_processed.iloc[-timesteps:].copy().reset_index(drop=True)
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

            # prepare next row (fill non-target features from last row)
            next_row = {}
            for col in df_processed.columns:
                if col in ("ENERGY_ADJ", "ENERGY"):
                    next_row[col] = y_hat
                else:
                    next_row[col] = float(current_window.iloc[-1][col]) if col in current_window.columns else 0.0
            next_row_df = pd.DataFrame([next_row], columns=df_processed.columns)
            current_window = pd.concat([current_window.iloc[1:].reset_index(drop=True), next_row_df], ignore_index=True)

        return predictions
    except Exception:
        return None


def get_recent_history(db, evn_username: str, days: int = 30) -> list:
    """Lấy `days` dòng gần nhất từ daily_consumption, trả về list dict {date, consumption_kwh}."""
    try:
        df = load_consumption_dataframe(db, evn_username)
        if df.empty:
            return []
        df['DATE_PARSED'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['DATE_PARSED']).sort_values('DATE_PARSED', ascending=False).head(days)
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "date": r['DATE_PARSED'].date().isoformat(),
                "consumption_kwh": float(r['ENERGY']) if pd.notna(r['ENERGY']) else 0.0
            })
        return rows
    except Exception:
        return []


def compute_electricity_bill(kwh: float) -> dict:
    """
    Tính tiền điện theo bậc lũy tiến (mẫu). Trả về tổng và breakdown.
    Giá mẫu (VND/kWh) — chỉnh theo biểu giá thực tế nếu cần.
    """
    tiers = [
        (50, 1678.6),
        (50, 1734.7),
        (100, 2014.9),
        (100, 2536.9),
        (100, 2834.4),
        (float('inf'), 2927.8)
    ]
    remaining = float(kwh)
    breakdown = []
    total = 0.0
    for cap, price in tiers:
        if remaining <= 0:
            break
        use = min(remaining, cap)
        cost = use * price
        breakdown.append({
            "tier_kwh": round(use, 2),
            "unit_price_vnd": price,
            "cost_vnd": round(cost, 2)
        })
        total += cost
        remaining -= use
    return {"total_vnd": round(total, 2), "breakdown": breakdown}


def build_user_context(evn_username: str, db, history_days: int = 30, forecast_horizon: int = 1) -> str:
    """
    Tạo context text gồm:
      - User info
      - Lịch sử tiêu thụ (last `history_days`)
      - Dự báo ngắn hạn (horizon)
      - Ước tính tiền điện
    """
    acc = get_account_by_username(db, evn_username)
    location = acc.location if acc and acc.location else "Unknown"
    model_dir = MODELS_DIR / f"user_{evn_username}"
    has_model = (model_dir / "lstm_model.h5").exists() and (model_dir / "scaler_x.pkl").exists()

    recent = get_recent_history(db, evn_username, days=history_days)
    total_recent_kwh = sum(r["consumption_kwh"] for r in recent) if recent else 0.0

    forecasts = compute_short_forecast(evn_username, db, horizon=forecast_horizon)
    forecast_text = "Không có dự báo (thiếu model hoặc dữ liệu)" if not forecasts else ", ".join([f"{v:.2f} kWh" for v in forecasts])

    bill_recent = compute_electricity_bill(total_recent_kwh)
    next_day_cost = None
    if forecasts and len(forecasts) >= 1:
        next_day_cost = compute_electricity_bill(forecasts[0])

    parts = []
    parts.append(f"User: {evn_username}")
    parts.append(f"Location: {location}")
    parts.append(f"Model available: {'yes' if has_model else 'no'}")
    parts.append("")
    parts.append(f"Recent consumption (last {history_days} days) — total: {total_recent_kwh:.2f} kWh")
    if recent:
        for r in recent[:10]:
            parts.append(f"- {r['date']}: {r['consumption_kwh']:.2f} kWh")
    else:
        parts.append("- No consumption history available")
    parts.append("")
    parts.append(f"Short-term forecast ({forecast_horizon} day(s)): {forecast_text}")
    parts.append("")
    parts.append("Estimated bill for recent period:")
    parts.append(f"- Total (VND): {bill_recent['total_vnd']:,}")
    for i, b in enumerate(bill_recent['breakdown']):
        parts.append(f"  * Tier {i+1}: {b['tier_kwh']} kWh x {b['unit_price_vnd']} VND/kWh = {b['cost_vnd']:,} VND")
    if next_day_cost is not None:
        parts.append("")
        parts.append(f"Estimated cost for next day forecast ({forecasts[0]:.2f} kWh): {next_day_cost['total_vnd']:,} VND")

    return "\n".join(parts)


def ask_energy_ai_for_user(evn_username: str, question: str, db, history_days: int = 30, forecast_horizon: int = 1) -> str:
    """Dựng context tự động cho user rồi gọi LLM."""
    context = build_user_context(evn_username, db, history_days=history_days, forecast_horizon=forecast_horizon)
    return ask_energy_ai(question, context)