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

# ======================================================
# 1. GỌI OPENAI – CHỈ DIỄN GIẢI, KHÔNG DỰ ĐOÁN
# ======================================================
def ask_energy_ai(question: str, context: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý phân tích điện năng.\n"
                )
            },
            {
                "role": "user",
                "content": f"""
{context}

CÂU HỎI:
{question}
"""
            }
        ],
        temperature=0.0  # ép AI không sáng tạo
    )
    return response.choices[0].message.content.strip()


# ======================================================
# 2. LẤY TOÀN BỘ LỊCH SỬ TIÊU THỤ TỪ DB
# ======================================================
def get_all_history(db, evn_username: str) -> pd.DataFrame:
    """
    Trả về toàn bộ lịch sử tiêu thụ điện từ database
    """
    df = load_consumption_dataframe(db, evn_username)
    if df.empty:
        return df

    df["DATE_PARSED"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["DATE_PARSED"])
    df = df.sort_values("DATE_PARSED")
    return df


# ======================================================
# 3. DỰ ĐOÁN – CHỈ DÙNG OUTPUT CUỐI CÙNG CỦA LSTM
# ======================================================
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

        # --- BƯỚC 1: TIỀN XỬ LÝ CHẶT CHẼ TRONG AI_SERVICE ---
        df_processed = ts.preprocess_for_base_model(df)
        
        # Đảm bảo không còn giá trị rỗng (NaN) trước khi đưa vào model
        # Fillna giúp loại bỏ lỗi 'nan kWh' nếu DB thiếu dữ liệu thời tiết
        df_processed = df_processed.ffill().bfill()
        
        # Nếu sau khi fill vẫn còn NaN (do cột đó trống hoàn toàn), điền bằng 0 để tránh lỗi toán học
        df_processed = df_processed.fillna(0)
        
        feature_cols = df_processed.columns.tolist()
        
        # Xác định cột năng lượng
        energy_key = "ENERGY_ADJ" if "ENERGY_ADJ" in feature_cols else "ENERGY"
        energy_col_idx = feature_cols.index(energy_key)

        timesteps = 7
        if len(df_processed) < timesteps:
            return None

        # Lấy cửa sổ 7 ngày gần nhất
        current_window = df_processed.iloc[-timesteps:].copy().reset_index(drop=True)
        predictions = []

        # --- BƯỚC 2: VÒNG LẶP DỰ BÁO ---
        for i in range(horizon):
            # Tạo DataFrame có tên cột để tránh lỗi UserWarning của Scaler
            window_df = pd.DataFrame(current_window.values, columns=feature_cols)
            
            # Scale dữ liệu đầu vào
            window_scaled = scaler.transform(window_df)
            x_in = np.expand_dims(window_scaled, axis=0)

            # Dự báo (y_scaled)
            y_scaled = model.predict(x_in, verbose=0)[0][0]

            # Kiểm tra nếu model trả về nan (do model bị hỏng lúc train hoặc gradient nổ)
            if np.isnan(y_scaled):
                # Backup: Lấy giá trị trung bình của cửa sổ hiện tại để không bị đứt mạch nan
                y_hat = current_window[energy_key].mean()
            else:
                # Inverse scale để lấy số kWh thực
                dummy = np.zeros((1, len(feature_cols)))
                dummy[0, energy_col_idx] = y_scaled
                y_hat = scaler.inverse_transform(dummy)[0, energy_col_idx]

            predictions.append(float(y_hat))

            # Xây dựng dòng dữ liệu tiếp theo
            next_row = current_window.iloc[-1].copy()
            next_row[energy_key] = y_hat
            
            # Cập nhật các cột energy khác nếu có
            for col in ["ENERGY", "ENERGY_ADJ"]:
                if col in next_row:
                    next_row[col] = y_hat

            # Trượt cửa sổ (Slide window)
            current_window = pd.concat(
                [current_window.iloc[1:], next_row.to_frame().T],
                ignore_index=True
            )

        return predictions

    except Exception as e:
        import traceback
        print(f"❗ LỖI AI SERVICE: {str(e)}")
        traceback.print_exc()
        return None



# ======================================================
# 4. BUILD CONTEXT – DỮ LIỆU ĐƯA CHO AI
# ======================================================
def build_user_context(
    evn_username: str,
    db,
    forecast_horizon: int
) -> str:
    acc = get_account_by_username(db, evn_username)
    location = acc.location if acc and acc.location else "Unknown"

    # ---- LỊCH SỬ ----
    history_df = get_all_history(db, evn_username)

    # ---- DỰ ĐOÁN (OUTPUT CUỐI LSTM) ----
    forecasts = compute_lstm_forecast(
        evn_username=evn_username,
        db=db,
        horizon=forecast_horizon
    )

    parts = []
    parts.append(f"User: {evn_username}")
    parts.append(f"Location: {location}")
    parts.append("")

    # ===== HISTORY =====
    parts.append("HISTORY (TOÀN BỘ DỮ LIỆU DB):")
    if history_df.empty:
        parts.append("- Không có dữ liệu lịch sử")
    else:
        for _, r in history_df.iterrows():
            parts.append(
                f"- {r['DATE_PARSED'].date().isoformat()}: {float(r['ENERGY']):.2f} kWh"
            )

    parts.append("")

    # ===== FORECAST =====
    parts.append("FORECAST (KẾT QUẢ CUỐI TỪ LSTM):")
    if not forecasts:
        parts.append("- Không có dự báo")
    else:
        last_date = history_df["DATE_PARSED"].iloc[-1].date()
        for i, v in enumerate(forecasts):
            d = last_date + timedelta(days=i + 1)
            parts.append(f"- {d.isoformat()}: {v:.2f} kWh")

    return "\n".join(parts)


# ======================================================
# 5. API GỌI CUỐI
# ======================================================
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
