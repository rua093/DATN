from openai import OpenAI
from server.config import OPENAI_API_KEY, MODELS_DIR
from server.database import get_account_by_username, load_consumption_dataframe
from server.services.training_service import TrainingService
from datetime import datetime, timedelta
import dateparser

import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Optional

client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================================
# 0. HÀM CHUẨN HÓA NGÀY THÁNG TRONG CÂU HỎI
# ======================================================
def normalize_date_in_question(question: str) -> str:
    """
    Sử dụng dateparser để nhận diện các định dạng ngày (như 25/11, 25/11/2025)
    và chuyển về chuẩn YYYY-MM-DD để AI dễ tra cứu trong context.
    """
    # languages=['vi'] giúp ưu tiên hiểu tiếng Việt, DATE_ORDER='DMY' để tránh nhầm ngày/tháng
    parsed_date = dateparser.parse(question, languages=['vi'], settings={'DATE_ORDER': 'DMY'})
    if parsed_date:
        standard_date = parsed_date.strftime('%Y-%m-%d')
        # Ghi chú thêm vào câu hỏi để AI không bị nhầm lẫn
        return f"{question} (Ngày cần tra cứu trong dữ liệu: {standard_date})"
    return question

# ======================================================
# 1. GỌI OPENAI – DIỄN GIẢI DỮ LIỆU
# ======================================================
def ask_energy_ai(question: str, context: str) -> str:
    # Chuẩn hóa câu hỏi trước khi đưa vào Prompt
    processed_question = normalize_date_in_question(question)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
            "Bạn là chuyên gia phân tích dữ liệu điện năng thông minh. "
            "Dữ liệu hệ thống cung cấp bao gồm HISTORY (quá khứ) và FORECAST (dự báo) theo định dạng YYYY-MM-DD.\n\n"
    
            "QUY TẮC XỬ LÝ DỮ LIỆU:\n"
            "1. Đối chiếu ngày tháng: Khi người dùng hỏi về một ngày, hãy sử dụng 'Ngày cần tra cứu' đã được chuẩn hóa để tìm con số chính xác trong dữ liệu.\n"
            "2. Phân biệt phạm vi: \n"
            "   - Nếu hỏi về 'Tuần qua', 'Đã dùng', 'Lịch sử': Chỉ tìm trong phần HISTORY.\n"
            "   - Nếu hỏi về 'Sắp tới', 'Dự báo', 'Ngày mai': Chỉ tìm trong phần FORECAST.\n"
            "3. So sánh dữ liệu: Khi tìm ngày 'thấp nhất' hoặc 'cao nhất', phải duyệt qua toàn bộ giá trị số trong phạm vi yêu cầu trước khi kết luận. Luôn ghi kèm con số kWh để minh chứng.\n"
            "4. Mốc thời gian: Luôn lấy giá trị 'Today' trong context làm mốc hiện tại để xác định phạm vi thời gian.\n\n"
    
            "QUY TẮC PHẢN HỒI:\n"
            "1. Tính minh bạch dự báo: Nếu trả lời bằng dữ liệu từ FORECAST, bắt buộc phải đính kèm câu: "
            "'Lưu ý: Kết quả dự báo có độ chính xác cao nhất cho ngày kế tiếp; sai số sẽ tích lũy và tăng dần đối với các ngày xa hơn trong tương lai.'\n"
            "2. Trình bày: Sử dụng định dạng danh sách (bullet points) cho các chuỗi dữ liệu. Trình bày ngắn gọn, súc tích.\n"
            "3. Xử lý thiếu dữ liệu: Nếu không tìm thấy ngày người dùng yêu cầu trong cả History và Forecast, hãy báo không có dữ liệu cho ngày đó thay vì tự đoán."
)
            },
            {
                "role": "user",
                "content": f"""
DỮ LIỆU HỆ THỐNG CUNG CẤP:
{context}

CÂU HỎI NGƯỜI DÙNG:
{processed_question}
"""
            }
        ],
        temperature=0.3  # Đảm bảo câu trả lời nhất quán, không sáng tạo số liệu
    )
    return response.choices[0].message.content.strip()


# ======================================================
# 2. LẤY TOÀN BỘ LỊCH SỬ TIÊU THỤ TỪ DB
# ======================================================
def get_all_history(db, evn_username: str) -> pd.DataFrame:
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

        df_processed = ts.preprocess_for_base_model(df)
        df_processed = df_processed.ffill().bfill()
        df_processed = df_processed.fillna(0)
        
        feature_cols = df_processed.columns.tolist()
        energy_key = "ENERGY_ADJ" if "ENERGY_ADJ" in feature_cols else "ENERGY"
        energy_col_idx = feature_cols.index(energy_key)

        timesteps = 7
        if len(df_processed) < timesteps:
            return None

        current_window = df_processed.iloc[-timesteps:].copy().reset_index(drop=True)
        predictions = []

        for i in range(horizon):
            window_df = pd.DataFrame(current_window.values, columns=feature_cols)
            window_scaled = scaler.transform(window_df)
            x_in = np.expand_dims(window_scaled, axis=0)

            y_scaled = model.predict(x_in, verbose=0)[0][0]

            if np.isnan(y_scaled):
                y_hat = current_window[energy_key].mean()
            else:
                dummy = np.zeros((1, len(feature_cols)))
                dummy[0, energy_col_idx] = y_scaled
                y_hat = scaler.inverse_transform(dummy)[0, energy_col_idx]

            predictions.append(float(y_hat))

            next_row = current_window.iloc[-1].copy()
            next_row[energy_key] = y_hat
            
            for col in ["ENERGY", "ENERGY_ADJ"]:
                if col in next_row:
                    next_row[col] = y_hat

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

    history_df = get_all_history(db, evn_username)
    forecasts = compute_lstm_forecast(
        evn_username=evn_username,
        db=db,
        horizon=forecast_horizon
    )

    parts = []
    parts.append(f"User: {evn_username}")
    parts.append(f"Location: {location}")
    parts.append(f"Today: {datetime.now().strftime('%Y-%m-%d')}")
    parts.append("")

    parts.append("HISTORY (DỮ LIỆU QUÁ KHỨ):")
    if history_df.empty:
        parts.append("- Không có dữ liệu lịch sử")
    else:
        # Lấy 30 ngày gần nhất để context gọn gàng
        for _, r in history_df.tail(30).iterrows():
            parts.append(
                f"- {r['DATE_PARSED'].date().isoformat()}: {float(r['ENERGY']):.2f} kWh"
            )

    parts.append("")

    parts.append("FORECAST (KẾT QUẢ DỰ BÁO TƯƠNG LAI):")
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