from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from server.services.ai_services import ask_energy_ai, ask_energy_ai_for_user

from server.services.training_service import TrainingService

import os

from server.database import (
    init_db, get_db, SessionLocal, get_account_by_username, create_account,
    get_active_model, create_model, EvnAccount, TrainingJob, CrawlJob, DailyConsumption,
    update_account_password
)
from server.services.crawler_service import CrawlerService
from server.services.training_service import TrainingService
from server.config import (
    MODELS_DIR, FINE_TUNE_LR, FINE_TUNE_EPOCHS
)
from fastapi import FastAPI, Body
from server.services.ai_services import ask_energy_ai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="EVN Energy Prediction API", version="1.0.0")
init_db()


def get_user_by_username(evn_username: str = Query(..., description="EVN username"), db: Session = Depends(get_db)) -> EvnAccount:
    """Get user by evn_username from query parameter"""
    user = get_account_by_username(db, evn_username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

def handle_first_login(evn_username: str, evn_password: str):
    logger.info(f"🚀 handle_first_login được gọi cho user: {evn_username}")
    db = SessionLocal()
    job = None
    try:
        job = TrainingJob(evn_username=evn_username, status="running")
        db.add(job)
        db.commit()
        logger.info(f"Bắt đầu thiết lập lần đầu cho người dùng {evn_username}")
        # Tạo CrawlJob và cập nhật trạng thái theo tiến trình
        crawl_job = CrawlJob(evn_username=evn_username, status="running")
        db.add(crawl_job)
        db.commit()
        crawler_service = CrawlerService(evn_username, evn_password)
        crawl_result = crawler_service.crawl_initial_data(evn_username, years_back=3)
        if not crawl_result["success"]:
            if crawl_job:
                crawl_job.status = "failed"
                crawl_job.error_message = crawl_result.get("error")
                crawl_job.completed_at = datetime.utcnow()
                db.commit()
            if job:
                job.status = "failed"
                job.error_message = f"Crawl thất bại: {crawl_result.get('error')}"
                db.commit()
            acc = get_account_by_username(db, evn_username)
            if acc:
                acc.crawl_status = "failed"
                db.commit()
            return
        else:
            if crawl_job:
                crawl_job.status = "completed"
                crawl_job.completed_at = datetime.utcnow()
                crawl_job.records_crawled = crawl_result.get("records")
                db.commit()
        training_service = TrainingService()
        train_result = training_service.train_model(
            evn_username, db,
            fine_tune_lr=FINE_TUNE_LR,
            fine_tune_epochs=FINE_TUNE_EPOCHS
        )
        if not train_result["success"]:
            if job:
                job.status = "failed"
                job.error_message = f"Train thất bại: {train_result.get('error')}"
                db.commit()
            acc = get_account_by_username(db, evn_username)
            if acc:
                acc.crawl_status = "failed"
                db.commit()
            return
        model = create_model(
            db=db, evn_username=evn_username, model_path=train_result["model_path"],
            scaler_x_path=train_result.get("scaler_x_path"), scaler_y_path=train_result.get("scaler_y_path"),
            metrics=train_result.get("metrics"), training_params=train_result.get("training_params")
        )
        u = get_account_by_username(db, evn_username)
        if u:
            u.crawl_status = "success"
            u.model_path = train_result["model_path"]
            db.commit()
        if job:
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.model_id = model.id
            db.commit()
        logger.info(f"Thiết lập lần đầu hoàn tất cho người dùng {evn_username}")
    except Exception as e:
        logger.error(f"Lỗi trong handle_first_login: {str(e)}", exc_info=True)
        try:
            if job:
                job.status = "failed"
                job.error_message = str(e)
                db.commit()
            # đánh dấu crawl job gần nhất là failed nếu có
            try:
                last_crawl = db.query(CrawlJob).filter(CrawlJob.evn_username == evn_username).order_by(CrawlJob.started_at.desc()).first()
                if last_crawl and last_crawl.status == "running":
                    last_crawl.status = "failed"
                    last_crawl.error_message = str(e)
                    last_crawl.completed_at = datetime.utcnow()
                    db.commit()
            except Exception:
                pass
            acc = get_account_by_username(db, evn_username)
            if acc:
                acc.crawl_status = "failed"
                db.commit()
        except Exception as e2:
            logger.error(f"Lỗi khi cập nhật job status: {str(e2)}", exc_info=True)
    finally:
        try:
            db.close()
        except Exception:
            pass

@app.get("/")
async def root():
    return {"message": "EVN Energy Prediction API", "version": "1.0.0"}

# POST /api/auth/login with immediate EVN verification
class AuthLoginRequest(BaseModel):
    evn_username: str
    evn_password: str
    location: Optional[str] = None

@app.post("/api/auth/login")
async def auth_login(request: AuthLoginRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    user = get_account_by_username(db, request.evn_username)
    if not user:
        # User mới: tạo account và crawl + train
        user = create_account(db=db, evn_username=request.evn_username, evn_password=request.evn_password, location=request.location)
        user.crawl_status = "pending"
        db.commit()
        background_tasks.add_task(handle_first_login, request.evn_username, request.evn_password)
        return {"status": "pending", "message": "Xác thực thành công. Đang xử lý dữ liệu..."}
    
    # User đã tồn tại: cập nhật password và location
    password_changed = user.evn_password != request.evn_password
    update_account_password(db, request.evn_username, request.evn_password)
    if request.location:
        user.location = request.location
    db.commit()
    
    # Refresh user từ DB để lấy crawl_status mới nhất
    db.refresh(user)

    if password_changed:
        user.crawl_status = "pending"
        db.commit()
        background_tasks.add_task(handle_first_login, request.evn_username, request.evn_password)
        return {"status": "pending", "message": "Mật khẩu đã thay đổi. Đang xác thực lại dữ liệu..."}
    
    # Kiểm tra xem đã có model và crawl_status = "success" chưa
    active_model = get_active_model(db, user.evn_username)
    crawl_status = user.crawl_status or "pending"
    
    logger.info(f"User {request.evn_username} - crawl_status: {crawl_status}, has_model: {active_model is not None}")
    
    if not active_model or crawl_status != "success":
        # Chưa có model hoặc crawl_status chưa success → crawl lại
        logger.info(f"User {request.evn_username} chưa có model hoặc crawl_status != success, bắt đầu crawl lại...")
        if crawl_status != "pending":
            user.crawl_status = "pending"
            db.commit()
        background_tasks.add_task(handle_first_login, request.evn_username, request.evn_password)
        return {"status": "pending", "message": "Đang crawl và train lại dữ liệu..."}
    
    # Đã có model thành công
    return {"status": "success", "message": "Đăng nhập thành công!"}

@app.get("/api/model/status")
async def get_model_status(user: EvnAccount = Depends(get_user_by_username), db: Session = Depends(get_db)):
    active_model = get_active_model(db, user.evn_username)
    if not active_model:
        return {"has_model": False, "message": "No active model found"}
    model_exists = os.path.exists(active_model.model_path)
    return {
        "has_model": True, "model_id": active_model.id, "trained_at": active_model.trained_at.isoformat(),
        "metrics": {"mae": active_model.metrics_mae, "rmse": active_model.metrics_rmse, "r2": active_model.metrics_r2},
        "model_exists": model_exists
    }

@app.get("/api/model/download")
async def download_model(user: EvnAccount = Depends(get_user_by_username), db: Session = Depends(get_db)):
    active_model = get_active_model(db, user.evn_username)
    if not active_model:
        raise HTTPException(status_code=404, detail="No active model found")
    if not os.path.exists(active_model.model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    return FileResponse(active_model.model_path, media_type="application/octet-stream", filename="lstm_model.h5")

@app.get("/api/model/scalers/download")
async def download_scalers(user: EvnAccount = Depends(get_user_by_username), db: Session = Depends(get_db)):
    active_model = get_active_model(db, user.evn_username)
    if not active_model:
        raise HTTPException(status_code=404, detail="No active model found")
    return {"scaler_x_path": active_model.scaler_x_path, "scaler_y_path": active_model.scaler_y_path}

@app.get("/api/training/jobs")
async def get_training_jobs(user: EvnAccount = Depends(get_user_by_username), db: Session = Depends(get_db)):
    jobs = db.query(TrainingJob).filter(TrainingJob.evn_username == user.evn_username).order_by(TrainingJob.started_at.desc()).limit(10).all()
    return [{"id": job.id, "status": job.status, "started_at": job.started_at.isoformat(),
             "completed_at": job.completed_at.isoformat() if job.completed_at else None, "error_message": job.error_message} for job in jobs]

@app.get("/api/crawl/jobs")
async def get_crawl_jobs(user: EvnAccount = Depends(get_user_by_username), db: Session = Depends(get_db)):
    jobs = db.query(CrawlJob).filter(
        CrawlJob.evn_username == user.evn_username
    ).order_by(CrawlJob.started_at.desc()).limit(10).all()
    return [
        {
            "id": job.id,
            "status": job.status,
            "crawl_date": job.crawl_date.isoformat() if job.crawl_date else None,
            "started_at": job.started_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "records_crawled": job.records_crawled,
            "error_message": job.error_message
        }
        for job in jobs
    ]

@app.get("/api/data/forecast")
async def get_forecast(user: EvnAccount = Depends(get_user_by_username), db: Session = Depends(get_db)):
    import joblib
    import numpy as np
    from server.services.training_service import TrainingService
    from server.database import get_account_by_username
    model_dir = MODELS_DIR / f"user_{user.evn_username}"
    model_path = model_dir / "lstm_model.h5"
    sx_path = model_dir / "scaler_x.pkl"
    sy_path = model_dir / "scaler_y.pkl"
    if not (model_path.exists() and sx_path.exists() and sy_path.exists()):
        raise HTTPException(status_code=404, detail="Model chưa sẵn sàng")
    acc = get_account_by_username(db, user.evn_username)
    if not acc:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài khoản")
    location = acc.location if acc.location else "Ho Chi Minh City"
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    timesteps = 7
    scaler = joblib.load(sx_path)
    ts = TrainingService()
    df = ts.build_dataset_from_db(db, user.evn_username, location)
    if df.empty:
        raise HTTPException(status_code=404, detail="Không có dữ liệu để dự báo")
    df_processed = ts.preprocess_for_base_model(df)
    if len(df_processed) < timesteps + 1:
        raise HTTPException(status_code=400, detail=f"Không đủ dữ liệu để dự báo. Cần ít nhất {timesteps + 1} mẫu")
    # Lấy ngày cuối cùng từ dữ liệu gốc để tính ngày dự báo
    last_date = None
    import pandas as pd
    if 'DATE_ONLY' in df.columns:
        last_date = df['DATE_ONLY'].iloc[-1]
    elif 'DATE' in df.columns:
        # Parse từ cột DATE nếu DATE_ONLY không có
        try:
            date_parsed = pd.to_datetime(df['DATE'].iloc[-1], errors='coerce', dayfirst=True)
            if pd.notna(date_parsed):
                last_date = date_parsed.date()
            else:
                logger.warning(f"Không parse được ngày từ cột DATE: {df['DATE'].iloc[-1]}")
        except Exception as e:
            logger.warning(f"Lỗi khi parse ngày từ cột DATE: {str(e)}")
    else:
        logger.warning(f"Không tìm thấy cột DATE hoặc DATE_ONLY trong dataframe. Các cột có sẵn: {list(df.columns)}")
    if last_date:
        from datetime import timedelta
        forecast_date = last_date + timedelta(days=1)
        forecast_date_str = forecast_date.strftime('%Y-%m-%d')
    else:
        forecast_date_str = None
        logger.warning("Không lấy được ngày cuối cùng từ dữ liệu, forecast_date sẽ là null")
    window = df_processed.iloc[-timesteps:].values
    window_scaled = scaler.transform(window)
    x_in = np.expand_dims(window_scaled, axis=0)
    y_hat_scaled = model.predict(x_in, verbose=0)
    n_feat = window_scaled.shape[1] - 1
    dummy = np.zeros((1, n_feat + 1))
    dummy[0, -1] = y_hat_scaled[0, 0]
    y_hat = scaler.inverse_transform(dummy)[0, -1]
    return {
        "horizon": 1, 
        "unit": "days", 
        "forecast_date": forecast_date_str,
        "predictions": [float(y_hat)]
    }

def forecast_multiple_days(
    model, scaler, df_processed, df_original, timesteps, num_days, location, db
):
    """Helper function để dự báo nhiều ngày"""
    import pandas as pd
    import numpy as np
    from datetime import timedelta
    from server.services.training_service import TrainingService
    from scripts.preprocess import fetch_open_meteo_weather, add_holiday_and_calendar_cols
    
    # Lấy ngày cuối cùng
    last_date = None
    if 'DATE_ONLY' in df_original.columns:
        last_date = df_original['DATE_ONLY'].iloc[-1]
    elif 'DATE' in df_original.columns:
        try:
            date_parsed = pd.to_datetime(df_original['DATE'].iloc[-1], errors='coerce', dayfirst=True)
            if pd.notna(date_parsed):
                last_date = date_parsed.date()
        except Exception:
            pass
    
    if not last_date:
        raise ValueError("Không lấy được ngày cuối cùng từ dữ liệu")
    
    # Lấy dữ liệu thời tiết cho các ngày dự báo
    start_forecast = last_date + timedelta(days=1)
    end_forecast = last_date + timedelta(days=num_days)
    start_str = start_forecast.strftime('%Y-%m-%d')
    end_str = end_forecast.strftime('%Y-%m-%d')
    
    try:
        weather_forecast = fetch_open_meteo_weather(start_str, end_str, location=location)
    except Exception as e:
        logger.warning(f"Không lấy được dữ liệu thời tiết từ API: {str(e)}, sử dụng giá trị trung bình")
        # Sử dụng giá trị trung bình từ dữ liệu lịch sử
        temp_avg_mean = df_original['TEMPERATURE_AVG'].mean() if 'TEMPERATURE_AVG' in df_original.columns else 28.0
        humidity_avg_mean = df_original['HUMIDITY_AVG'].mean() if 'HUMIDITY_AVG' in df_original.columns else 75.0
        weather_forecast = pd.DataFrame({
            'DATE_ONLY': [start_forecast + timedelta(days=i) for i in range(num_days)],
            'TEMPERATURE_AVG': [temp_avg_mean] * num_days,
            'TEMPERATURE_MAX': [temp_avg_mean + 3] * num_days,
            'HUMIDITY_AVG': [humidity_avg_mean] * num_days
        })
    
    # Chuẩn hóa DATE_ONLY trong weather_forecast về date object
    weather_forecast['DATE_ONLY'] = weather_forecast['DATE_ONLY'].apply(
        lambda x: x if isinstance(x, type(start_forecast)) else pd.to_datetime(x).date() if pd.notna(pd.to_datetime(x, errors='coerce')) else None
    )
    
    # Khởi tạo window từ dữ liệu hiện tại
    current_window = df_processed.iloc[-timesteps:].copy()
    predictions = []
    forecast_dates = []
    
    ts = TrainingService()
    
    for day_idx in range(num_days):
        forecast_date = start_forecast + timedelta(days=day_idx)
        forecast_dates.append(forecast_date.strftime('%Y-%m-%d'))
        
        # Lấy dữ liệu thời tiết cho ngày này
        weather_row = weather_forecast[weather_forecast['DATE_ONLY'] == forecast_date]
        if weather_row.empty:
            # Nếu không có, dùng giá trị trung bình
            temp_avg = df_original['TEMPERATURE_AVG'].mean() if 'TEMPERATURE_AVG' in df_original.columns else 28.0
            humidity_avg = df_original['HUMIDITY_AVG'].mean() if 'HUMIDITY_AVG' in df_original.columns else 75.0
        else:
            temp_avg = float(weather_row.iloc[0]['TEMPERATURE_AVG']) if pd.notna(weather_row.iloc[0].get('TEMPERATURE_AVG')) else 28.0
            humidity_avg = float(weather_row.iloc[0]['HUMIDITY_AVG']) if pd.notna(weather_row.iloc[0].get('HUMIDITY_AVG')) else 75.0
        
        # Tạo row mới với features cho ngày dự báo
        new_row_data = {
            'TEMPERATURE_AVG': temp_avg,
            'HUMIDITY_AVG': humidity_avg,
        }
        
        # Tính calendar features
        try:
            import holidays
            vn_holidays = holidays.country_holidays('VN')
            new_row_data['HOLIDAY'] = 1 if forecast_date in vn_holidays else 0
        except Exception:
            new_row_data['HOLIDAY'] = 0
        
        month = forecast_date.month
        weekday = forecast_date.weekday()  # Monday=0, Sunday=6
        new_row_data['month_sin'] = np.sin(2 * np.pi * month / 12)
        new_row_data['month_cos'] = np.cos(2 * np.pi * month / 12)
        new_row_data['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
        new_row_data['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
        
        # Tạo row mới với đầy đủ features theo thứ tự của df_processed
        new_row_dict = {}
        for col in df_processed.columns:
            if col == 'ENERGY_ADJ' or col == 'ENERGY':
                new_row_dict[col] = 0.0  # Tạm thời, sẽ được thay thế sau
            elif col in new_row_data:
                new_row_dict[col] = new_row_data[col]
            else:
                # Nếu thiếu feature, dùng giá trị trung bình từ window hiện tại
                new_row_dict[col] = current_window[col].mean() if col in current_window.columns else 0.0
        
        # Tạo DataFrame với đúng thứ tự cột
        new_row_df = pd.DataFrame([new_row_dict], columns=df_processed.columns)
        
        # Dự báo: scale window hiện tại và dự báo
        window_scaled = scaler.transform(current_window.values)
        x_in = np.expand_dims(window_scaled, axis=0)
        y_hat_scaled = model.predict(x_in, verbose=0)
        n_feat = window_scaled.shape[1] - 1
        dummy = np.zeros((1, n_feat + 1))
        dummy[0, -1] = y_hat_scaled[0, 0]
        y_hat = scaler.inverse_transform(dummy)[0, -1]
        predictions.append(float(y_hat))
        
        # Cập nhật window: thêm row mới với giá trị dự báo
        new_row_df['ENERGY_ADJ'] = y_hat
        # Cập nhật window: bỏ row đầu, thêm row mới (giữ ở dạng unprocessed)
        current_window = pd.concat([
            current_window.iloc[1:],
            new_row_df
        ], ignore_index=True)
    
    return forecast_dates, predictions

@app.get("/api/data/forecast/week")
async def get_forecast_week(user: EvnAccount = Depends(get_user_by_username), db: Session = Depends(get_db)):
    """Dự báo cho 7 ngày tiếp theo"""
    import joblib
    import numpy as np
    from server.services.training_service import TrainingService
    from server.database import get_account_by_username
    model_dir = MODELS_DIR / f"user_{user.evn_username}"
    model_path = model_dir / "lstm_model.h5"
    sx_path = model_dir / "scaler_x.pkl"
    sy_path = model_dir / "scaler_y.pkl"
    if not (model_path.exists() and sx_path.exists() and sy_path.exists()):
        raise HTTPException(status_code=404, detail="Model chưa sẵn sàng")
    acc = get_account_by_username(db, user.evn_username)
    if not acc:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài khoản")
    location = acc.location if acc.location else "Ho Chi Minh City"
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    timesteps = 7
    scaler = joblib.load(sx_path)
    ts = TrainingService()
    df = ts.build_dataset_from_db(db, user.evn_username, location)
    if df.empty:
        raise HTTPException(status_code=404, detail="Không có dữ liệu để dự báo")
    df_processed = ts.preprocess_for_base_model(df)
    if len(df_processed) < timesteps + 1:
        raise HTTPException(status_code=400, detail=f"Không đủ dữ liệu để dự báo. Cần ít nhất {timesteps + 1} mẫu")
    
    try:
        forecast_dates, predictions = forecast_multiple_days(
            model, scaler, df_processed, df, timesteps, 7, location, db
        )
        return {
            "horizon": 7,
            "unit": "days",
            "forecast_dates": forecast_dates,
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Lỗi khi dự báo 7 ngày: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự báo: {str(e)}")

@app.get("/api/data/forecast/month")
async def get_forecast_month(user: EvnAccount = Depends(get_user_by_username), db: Session = Depends(get_db)):
    """Dự báo cho 30 ngày tiếp theo"""
    import joblib
    import numpy as np
    from server.services.training_service import TrainingService
    from server.database import get_account_by_username
    model_dir = MODELS_DIR / f"user_{user.evn_username}"
    model_path = model_dir / "lstm_model.h5"
    sx_path = model_dir / "scaler_x.pkl"
    sy_path = model_dir / "scaler_y.pkl"
    if not (model_path.exists() and sx_path.exists() and sy_path.exists()):
        raise HTTPException(status_code=404, detail="Model chưa sẵn sàng")
    acc = get_account_by_username(db, user.evn_username)
    if not acc:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài khoản")
    location = acc.location if acc.location else "Ho Chi Minh City"
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    timesteps = 7
    scaler = joblib.load(sx_path)
    ts = TrainingService()
    df = ts.build_dataset_from_db(db, user.evn_username, location)
    if df.empty:
        raise HTTPException(status_code=404, detail="Không có dữ liệu để dự báo")
    df_processed = ts.preprocess_for_base_model(df)
    if len(df_processed) < timesteps + 1:
        raise HTTPException(status_code=400, detail=f"Không đủ dữ liệu để dự báo. Cần ít nhất {timesteps + 1} mẫu")
    
    try:
        forecast_dates, predictions = forecast_multiple_days(
            model, scaler, df_processed, df, timesteps, 30, location, db
        )
        return {
            "horizon": 30,
            "unit": "days",
            "forecast_dates": forecast_dates,
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Lỗi khi dự báo 30 ngày: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự báo: {str(e)}")

# History from MySQL (paged)
@app.get("/api/data/history/db")
async def get_history_db(
    user: EvnAccount = Depends(get_user_by_username),
    db: Session = Depends(get_db),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    page_size: int = 100
):
    from datetime import datetime as dt
    q = db.query(DailyConsumption).filter(DailyConsumption.evn_username == user.evn_username)
    try:
        if start_date:
            sd = dt.fromisoformat(start_date).date()
            q = q.filter(DailyConsumption.date >= sd)
        if end_date:
            ed = dt.fromisoformat(end_date).date()
            q = q.filter(DailyConsumption.date <= ed)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    total = q.count()
    page = max(1, page)
    page_size = max(1, min(page_size, 1000))
    rows = q.order_by(DailyConsumption.date.asc()).offset((page - 1) * page_size).limit(page_size).all()
    items = [
        {
            "date": r.date.isoformat() if r.date else None,
            "consumption_kwh": float(r.consumption_kwh) if r.consumption_kwh is not None else None
        }
        for r in rows
    ]
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": items
    }
    

from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str

@app.post("/api/chat")
def chat_with_ai(
    evn_username: str,
    req: ChatRequest,
    db: Session = Depends(get_db)
):
    answer = ask_energy_ai_for_user(
        evn_username=evn_username,
        question=req.question,
        db=db,
        forecast_horizon=30
    )
    return {
        "question": req.question,
        "answer": answer
    }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

