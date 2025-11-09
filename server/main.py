from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime
import os

from server.database import (
    init_db, get_db, SessionLocal, get_account_by_username, create_account,
    get_active_model, create_model, EvnAccount, TrainingJob, CrawlJob, DailyConsumption,
    update_account_password
)
from server.services.crawler_service import CrawlerService
from server.services.training_service import TrainingService
from server.config import MODELS_DIR, USE_WOA_BY_DEFAULT, WOA_N_AGENTS, WOA_MAX_ITER

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

async def handle_first_login(evn_username: str, evn_password: str):
    logger.info(f"🚀 handle_first_login được gọi cho user: {evn_username}")
    db = SessionLocal()
    job = None
    try:
        job = TrainingJob(evn_username=evn_username, status="running")
        db.add(job)
        db.commit()
        logger.info(f"Bắt đầu thiết lập lần đầu cho người dùng {evn_username}")
        crawler_service = CrawlerService(evn_username, evn_password)
        crawl_result = crawler_service.crawl_initial_data(evn_username, years_back=3)
        if not crawl_result["success"]:
            if job:
                job.status = "failed"
                job.error_message = f"Crawl thất bại: {crawl_result.get('error')}"
                db.commit()
            return
        training_service = TrainingService()
        train_result = training_service.train_model(
            evn_username, db,
            use_woa=USE_WOA_BY_DEFAULT, woa_n_agents=WOA_N_AGENTS, woa_max_iter=WOA_MAX_ITER
        )
        if not train_result["success"]:
            if job:
                job.status = "failed"
                job.error_message = f"Train thất bại: {train_result.get('error')}"
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
    # hot verification against EVN
    crawler = CrawlerService(request.evn_username, request.evn_password)
    try:
        ok = crawler.crawler is None  # ensure construction
        # use only login check
        crawler.crawler = None
        crawler.crawler = None
        evn = CrawlerService(request.evn_username, request.evn_password)
        evn.crawler = None
        # direct check
        from scripts.evn_crawler import EVNCrawler
        bot = EVNCrawler(headless=True, username=request.evn_username, password=request.evn_password)
        success = bot.login()
        bot.close()
        if not success:
            raise HTTPException(status_code=401, detail="EVN login failed")
    except Exception:
        # attempt graceful close
        try:
            bot.close()
        except Exception:
            pass
        raise HTTPException(status_code=401, detail="EVN login failed")

    user = get_account_by_username(db, request.evn_username)
    if not user:
        # User mới: tạo account và crawl + train
        user = create_account(db=db, evn_username=request.evn_username, evn_password=request.evn_password, location=request.location)
        db.commit()
        background_tasks.add_task(handle_first_login, request.evn_username, request.evn_password)
        return {"status": "pending", "message": "Xác thực thành công. Đang xử lý dữ liệu..."}
    
    # User đã tồn tại: cập nhật password và location
    from server.database import update_account_password
    update_account_password(db, request.evn_username, request.evn_password)
    if request.location:
        user.location = request.location
    db.commit()
    
    # Refresh user từ DB để lấy crawl_status mới nhất
    db.refresh(user)
    
    # Kiểm tra xem đã có model và crawl_status = "success" chưa
    active_model = get_active_model(db, user.evn_username)
    crawl_status = user.crawl_status or "pending"
    
    logger.info(f"User {request.evn_username} - crawl_status: {crawl_status}, has_model: {active_model is not None}")
    
    if not active_model or crawl_status != "success":
        # Chưa có model hoặc crawl_status chưa success → crawl lại
        logger.info(f"User {request.evn_username} chưa có model hoặc crawl_status != success, bắt đầu crawl lại...")
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
    ts = TrainingService()
    df = ts.build_dataset_from_db(db, user.evn_username, location)
    if df.empty:
        raise HTTPException(status_code=404, detail="Không có dữ liệu để dự báo")
    X_raw, y_raw = ts.preprocess_for_lstm(df)
    timesteps = 24
    if len(X_raw) < timesteps + 1:
        raise HTTPException(status_code=400, detail="Không đủ dữ liệu để dự báo")
    window = X_raw[-timesteps:]
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    scaler_y = joblib.load(sy_path)
    preds = []
    cur_window = window.copy()
    for _ in range(1):
        x_in = np.expand_dims(cur_window, axis=0)
        y_hat_scaled = model.predict(x_in, verbose=0)
        y_hat = scaler_y.inverse_transform(y_hat_scaled)[0, 0]
        preds.append(float(y_hat))
        next_feat = cur_window[-1].copy()
        next_feat[0] = y_hat_scaled[0, 0]
        cur_window = np.vstack([cur_window[1:], next_feat])
    return {"horizon": 1, "unit": "days", "predictions": preds}

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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
