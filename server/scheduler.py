from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta, date
import logging
from typing import List
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from preprocess import fetch_open_meteo_weather
from server.database import SessionLocal, EvnAccount, CrawlJob, TrainingJob, create_model, save_daily_weather_rows
from server.services.crawler_service import CrawlerService
from server.services.training_service import TrainingService
from server.config import (
    CRAWLER_DAILY_TIME, RETRAIN_YEARLY_DAY, RETRAIN_YEARLY_MONTH, RETRAIN_YEARLY_TIME,
    YEARS_BACK_FOR_TRAINING, USE_WOA_BY_DEFAULT, WOA_N_AGENTS, WOA_MAX_ITER,
    USE_FINE_TUNE_BY_DEFAULT, FINE_TUNE_LR, FINE_TUNE_EPOCHS
)

logger = logging.getLogger(__name__)
scheduler = BackgroundScheduler()

def get_all_active_users() -> List[EvnAccount]:
    db = SessionLocal()
    try:
        users = db.query(EvnAccount).all()
        return users
    finally:
        db.close()

def daily_crawl_task():
    logger.info("Bắt đầu nhiệm vụ crawl hàng ngày")
    users = get_all_active_users()
    logger.info(f"Tìm thấy {len(users)} người dùng đang hoạt động")
    for user in users:
        if not user.evn_username or not user.evn_password:
            logger.warning(f"Tài khoản {user.evn_username} thiếu thông tin đăng nhập EVN, bỏ qua")
            continue
        db = SessionLocal()
        job = None
        try:
            job = CrawlJob(evn_username=user.evn_username, status="running", crawl_date=datetime.now() - timedelta(days=1))
            db.add(job)
            db.commit()
            logger.info(f"Đang crawl dữ liệu hôm qua cho {user.evn_username}")
            crawler_service = CrawlerService(user.evn_username, user.evn_password)
            result = crawler_service.crawl_yesterday_data(user.evn_username)
            if result["success"]:
                if job:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    job.records_crawled = result.get("records", 0)
                    db.commit()
                logger.info(f"Crawl thành công {result.get('records', 0)} bản ghi cho {user.evn_username}")
            else:
                if job:
                    job.status = "failed"
                    job.error_message = result.get("error", "Lỗi không xác định")
                    db.commit()
                logger.error(f"Crawl thất bại cho {user.evn_username}: {result.get('error')}")
        except Exception as e:
            logger.error(f"Lỗi trong crawl hàng ngày cho {user.evn_username}: {str(e)}", exc_info=True)
            try:
                if job:
                    job.status = "failed"
                    job.error_message = str(e)
                    db.commit()
            except Exception:
                pass
        finally:
            try:
                db.close()
            except Exception:
                pass

def daily_weather_update_task():
    logger.info("Bắt đầu nhiệm vụ cập nhật thời tiết hàng ngày")
    db = SessionLocal()
    try:
        # Lấy ngày hôm qua
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_str = yesterday.strftime('%Y-%m-%d')
        
        # Lấy tất cả locations từ users (unique)
        users = db.query(EvnAccount).all()
        locations = set()
        for user in users:
            location = user.location if user.location else "Ho Chi Minh City"
            locations.add(location)
        
        if not locations:
            logger.info("Không có location nào để cập nhật thời tiết")
            return
        
        logger.info(f"Đang cập nhật thời tiết cho {len(locations)} location(s) - ngày {yesterday_str}")
        
        # Fetch và lưu thời tiết cho từng location riêng biệt
        for location in locations:
            try:
                logger.info(f"Đang fetch thời tiết cho location '{location}' - ngày {yesterday_str}")
                weather_df = fetch_open_meteo_weather(yesterday_str, yesterday_str, location=location)
                if weather_df.empty:
                    logger.warning(f"Không lấy được dữ liệu thời tiết cho location '{location}' - ngày {yesterday_str}")
                    continue
                
                # Chuyển đổi DataFrame thành rows để lưu
                weather_rows = []
                for _, row in weather_df.iterrows():
                    date_val = row.get('DATE_ONLY')
                    if date_val:
                        # DATE_ONLY từ fetch_open_meteo_weather đã là date object
                        if not isinstance(date_val, date):
                            # Nếu là string hoặc datetime, convert sang date
                            if isinstance(date_val, str):
                                date_val = datetime.strptime(date_val, '%Y-%m-%d').date()
                            elif isinstance(date_val, datetime):
                                date_val = date_val.date()
                        weather_rows.append({
                            "date": date_val,
                            "max_temp_c": float(row.get("TEMPERATURE_MAX")) if pd.notna(row.get("TEMPERATURE_MAX")) else None,
                            "avg_temp_c": float(row.get("TEMPERATURE_AVG")) if pd.notna(row.get("TEMPERATURE_AVG")) else None,
                            "avg_humidity": float(row.get("HUMIDITY_AVG")) if pd.notna(row.get("HUMIDITY_AVG")) else None,
                        })
                
                if weather_rows:
                    inserted = save_daily_weather_rows(db, location, weather_rows)
                    logger.info(f"Đã cập nhật {inserted} dòng thời tiết cho location '{location}' - ngày {yesterday_str}")
                else:
                    logger.warning(f"Không có dữ liệu thời tiết để lưu cho location '{location}' - ngày {yesterday_str}")
            except Exception as e:
                logger.error(f"Lỗi khi fetch thời tiết cho location '{location}': {str(e)}", exc_info=True)
                continue
    except Exception as e:
        logger.error(f"Lỗi trong daily_weather_update_task: {str(e)}", exc_info=True)
    finally:
        try:
            db.close()
        except Exception:
            pass

def yearly_retrain_task():
    logger.info("Bắt đầu nhiệm vụ train lại hàng năm")
    users = get_all_active_users()
    logger.info(f"Tìm thấy {len(users)} người dùng đang hoạt động")
    for user in users:
        if not user.evn_username or not user.evn_password:
            logger.warning(f"Tài khoản {user.evn_username} thiếu thông tin đăng nhập EVN, bỏ qua")
            continue
        db = SessionLocal()
        job = None
        try:
            job = TrainingJob(evn_username=user.evn_username, status="running")
            db.add(job)
            db.commit()
            logger.info(f"Đang train lại model cho {user.evn_username}")
            crawler_service = CrawlerService(user.evn_username, user.evn_password)
            crawl_result = crawler_service.crawl_initial_data(user.evn_username, years_back=YEARS_BACK_FOR_TRAINING)
            if not crawl_result["success"]:
                if job:
                    job.status = "failed"
                    job.error_message = f"Crawl thất bại: {crawl_result.get('error')}"
                    db.commit()
                continue
            training_service = TrainingService()
            train_result = training_service.train_model(
                user.evn_username, db,
                use_woa=USE_WOA_BY_DEFAULT,
                woa_n_agents=WOA_N_AGENTS,
                woa_max_iter=WOA_MAX_ITER,
                use_fine_tune=USE_FINE_TUNE_BY_DEFAULT,
                fine_tune_lr=FINE_TUNE_LR,
                fine_tune_epochs=FINE_TUNE_EPOCHS
            )
            if not train_result["success"]:
                if job:
                    job.status = "failed"
                    job.error_message = f"Train thất bại: {train_result.get('error')}"
                    db.commit()
                continue
            model = create_model(
                db=db, evn_username=user.evn_username, model_path=train_result["model_path"],
                scaler_x_path=train_result.get("scaler_x_path"), scaler_y_path=train_result.get("scaler_y_path"),
                metrics=train_result.get("metrics"), training_params=train_result.get("training_params")
            )
            if job:
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                job.model_id = model.id
                db.commit()
            logger.info(f"Train lại thành công cho {user.evn_username}")
        except Exception as e:
            logger.error(f"Lỗi trong train lại hàng năm cho {user.evn_username}: {str(e)}", exc_info=True)
            try:
                if job:
                    job.status = "failed"
                    job.error_message = str(e)
                    db.commit()
            except Exception:
                pass
        finally:
            try:
                db.close()
            except Exception:
                pass

def start_scheduler():
    hour, minute = map(int, CRAWLER_DAILY_TIME.split(":"))
    scheduler.add_job(daily_crawl_task, trigger=CronTrigger(hour=hour, minute=minute), id="daily_crawl", name="Daily Crawl - Yesterday's Data", replace_existing=True)
    logger.info(f"Đã lên lịch crawl hàng ngày lúc {CRAWLER_DAILY_TIME}")
    
    # Cập nhật thời tiết sau crawl 30 phút (để đảm bảo có dữ liệu consumption trước)
    weather_hour = hour
    weather_minute = minute + 30
    if weather_minute >= 60:
        weather_hour = (weather_hour + 1) % 24
        weather_minute = weather_minute - 60
    scheduler.add_job(daily_weather_update_task, trigger=CronTrigger(hour=weather_hour, minute=weather_minute), id="daily_weather_update", name="Daily Weather Update - Yesterday's Data", replace_existing=True)
    logger.info(f"Đã lên lịch cập nhật thời tiết hàng ngày lúc {weather_hour:02d}:{weather_minute:02d}")
    
    retrain_hour, retrain_minute = map(int, RETRAIN_YEARLY_TIME.split(":"))
    scheduler.add_job(yearly_retrain_task, trigger=CronTrigger(day=RETRAIN_YEARLY_DAY, month=RETRAIN_YEARLY_MONTH, hour=retrain_hour, minute=retrain_minute), id="yearly_retrain", name="Yearly Retrain - 3 Years Data", replace_existing=True)
    logger.info(f"Đã lên lịch train lại hàng năm vào ngày {RETRAIN_YEARLY_DAY}/{RETRAIN_YEARLY_MONTH} lúc {RETRAIN_YEARLY_TIME}")
    scheduler.start()
    logger.info("Scheduler đã khởi động")

def stop_scheduler():
    scheduler.shutdown()
    logger.info("Scheduler đã dừng")
