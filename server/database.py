from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float, Text, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from datetime import datetime
from typing import Optional
import logging

from server.config import DATABASE_URL

logger = logging.getLogger(__name__)

Base = declarative_base()

class EvnAccount(Base):
    __tablename__ = "evn_accounts"
    evn_username = Column(String(64), primary_key=True, index=True)
    evn_password = Column(String(255), nullable=False)
    location = Column(String(128), nullable=True)
    crawl_status = Column(String(32), default="pending")
    model_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    evn_username = Column(String(64), ForeignKey("evn_accounts.evn_username"), index=True, nullable=False)
    model_path = Column(String(255), nullable=False)
    scaler_x_path = Column(String(255), nullable=True)
    scaler_y_path = Column(String(255), nullable=True)
    trained_at = Column(DateTime, default=datetime.utcnow)
    metrics_mae = Column(Float, nullable=True)
    metrics_rmse = Column(Float, nullable=True)
    metrics_r2 = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True)
    training_params = Column(Text, nullable=True)

class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id = Column(Integer, primary_key=True, index=True)
    evn_username = Column(String(64), ForeignKey("evn_accounts.evn_username"), index=True, nullable=False)
    status = Column(String(32), default="pending")
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    model_id = Column(Integer, nullable=True)

class CrawlJob(Base):
    __tablename__ = "crawl_jobs"
    id = Column(Integer, primary_key=True, index=True)
    evn_username = Column(String(64), ForeignKey("evn_accounts.evn_username"), index=True, nullable=False)
    status = Column(String(32), default="pending")
    crawl_date = Column(DateTime, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    records_crawled = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_account_by_username(db: Session, evn_username: str) -> Optional[EvnAccount]:
    return db.query(EvnAccount).filter(EvnAccount.evn_username == evn_username).first()

def get_active_model(db: Session, evn_username: str) -> Optional[Model]:
    return db.query(Model).filter(Model.evn_username == evn_username, Model.is_active == True).order_by(Model.trained_at.desc()).first()

def create_account(db: Session, evn_username: str, evn_password: str, location: str = None) -> EvnAccount:
    acc = EvnAccount(evn_username=evn_username, evn_password=evn_password, location=location)
    db.add(acc)
    db.commit()
    db.refresh(acc)
    return acc

def update_account_password(db: Session, evn_username: str, evn_password: str) -> bool:
    try:
        db.query(EvnAccount).filter(EvnAccount.evn_username == evn_username).update({"evn_password": evn_password})
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Lỗi khi cập nhật password: {str(e)}", exc_info=True)
        return False

def create_model(db: Session, evn_username: str, model_path: str, scaler_x_path: str = None, scaler_y_path: str = None, metrics: dict = None, training_params: dict = None) -> Model:
    db.query(Model).filter(Model.evn_username == evn_username).update({"is_active": False})
    model = Model(
        evn_username=evn_username, model_path=model_path, scaler_x_path=scaler_x_path, scaler_y_path=scaler_y_path,
        metrics_mae=metrics.get("mae") if metrics else None, metrics_rmse=metrics.get("rmse") if metrics else None,
        metrics_r2=metrics.get("r2") if metrics else None, training_params=str(training_params) if training_params else None, is_active=True
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model

class DailyConsumption(Base):
    __tablename__ = "daily_consumption"
    data_id = Column(Integer, primary_key=True, autoincrement=True)
    evn_username = Column(String(64), ForeignKey("evn_accounts.evn_username"), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    consumption_kwh = Column(Float, nullable=False)
    __table_args__ = (
        {'mysql_engine': 'InnoDB'},
    )

class DailyWeather(Base):
    __tablename__ = "daily_weather"
    weather_id = Column(Integer, primary_key=True, autoincrement=True)
    location = Column(String(128), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    max_temp_c = Column(Float, nullable=True)
    avg_temp_c = Column(Float, nullable=True)
    avg_humidity = Column(Float, nullable=True)


def save_daily_consumption_rows(db: Session, evn_username: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    try:
        mappings = []
        for r in rows:
            try:
                mappings.append({
                    "evn_username": evn_username,
                    "date": r["date"],
                    "consumption_kwh": float(r["consumption_kwh"])
                })
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Bỏ qua dòng không hợp lệ: {r}, lỗi: {str(e)}")
                continue
        if not mappings:
            return 0
        
        # Kiểm tra duplicate với DB trước khi insert
        existing_dates = set()
        if len(mappings) > 0:
            # Lấy danh sách date đã tồn tại trong DB
            existing_records = db.query(DailyConsumption.date).filter(
                DailyConsumption.evn_username == evn_username,
                DailyConsumption.date.in_([m["date"] for m in mappings])
            ).all()
            existing_dates = {r.date for r in existing_records}
        
        # Chỉ insert các dòng chưa tồn tại
        new_mappings = [m for m in mappings if m["date"] not in existing_dates]
        duplicate_count = len(mappings) - len(new_mappings)
        
        if duplicate_count > 0:
            logger.info(f"Phát hiện {duplicate_count} dòng đã tồn tại trong DB, chỉ insert {len(new_mappings)} dòng mới")
        
        if not new_mappings:
            logger.info(f"Tất cả {len(mappings)} dòng đã tồn tại trong DB, không cần insert")
            return 0
        
        db.bulk_insert_mappings(DailyConsumption, new_mappings)
        db.commit()
        inserted = len(new_mappings)
        logger.info(f"Đã lưu batch {inserted} dòng mới vào daily_consumption cho {evn_username}")
        return inserted
    except IntegrityError as e:
        db.rollback()
        logger.warning(f"Lỗi duplicate khi insert batch, thử insert từng dòng: {str(e)}")
        inserted = 0
        duplicate_count = 0
        for r in rows:
            try:
                dc = DailyConsumption(
                    evn_username=evn_username,
                    date=r["date"],
                    consumption_kwh=float(r["consumption_kwh"])
                )
                db.merge(dc)
                db.commit()
                inserted += 1
            except IntegrityError:
                db.rollback()
                duplicate_count += 1
                continue
            except Exception as e2:
                db.rollback()
                logger.warning(f"Lỗi khi insert dòng: {str(e2)}")
                continue
        if duplicate_count > 0:
            logger.info(f"Có {duplicate_count} dòng duplicate (đã tồn tại trong DB)")
        return inserted
    except Exception as e:
        db.rollback()
        logger.error(f"Lỗi khi lưu daily_consumption: {str(e)}", exc_info=True)
        raise

def save_daily_weather_rows(db: Session, location: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    try:
        mappings = []
        for r in rows:
            try:
                mappings.append({
                    "location": location,
                    "date": r["date"],
                    "max_temp_c": r.get("max_temp_c"),
                    "avg_temp_c": r.get("avg_temp_c"),
                    "avg_humidity": r.get("avg_humidity")
                })
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Bỏ qua dòng thời tiết không hợp lệ: {r}, lỗi: {str(e)}")
                continue
        if not mappings:
            return 0
        # loại duplicate trong batch theo date
        seen_dates = set()
        unique_mappings = []
        dup_in_batch = 0
        for m in mappings:
            d = m["date"]
            if d in seen_dates:
                dup_in_batch += 1
                continue
            seen_dates.add(d)
            unique_mappings.append(m)
        if dup_in_batch > 0:
            logger.info(f"save_daily_weather_rows: bỏ {dup_in_batch} dòng duplicate trong batch cho '{location}'")

        if not unique_mappings:
            return 0

        # loại ngày đã tồn tại trong DB
        existing = db.query(DailyWeather.date).filter(
            DailyWeather.location == location,
            DailyWeather.date.in_([m["date"] for m in unique_mappings])
        ).all()
        existing_dates = {r.date for r in existing}
        new_rows = [m for m in unique_mappings if m["date"] not in existing_dates]
        skipped_existing = len(unique_mappings) - len(new_rows)
        if skipped_existing > 0:
            logger.info(f"save_daily_weather_rows: bỏ {skipped_existing} dòng do đã tồn tại trong DB cho '{location}'")

        if not new_rows:
            return 0

        db.bulk_insert_mappings(DailyWeather, new_rows)
        db.commit()
        inserted = len(new_rows)
        logger.info(f"Đã lưu batch {inserted} dòng vào daily_weather cho '{location}'")
        return inserted
    except IntegrityError as e:
        db.rollback()
        logger.warning(f"Lỗi duplicate khi insert batch thời tiết, thử insert từng dòng: {str(e)}")
        inserted = 0
        for r in rows:
            try:
                dw = DailyWeather(
                    location=location,
                    date=r["date"],
                    max_temp_c=r.get("max_temp_c"),
                    avg_temp_c=r.get("avg_temp_c"),
                    avg_humidity=r.get("avg_humidity")
                )
                db.merge(dw)
                db.commit()
                inserted += 1
            except IntegrityError:
                db.rollback()
                continue
            except Exception as e2:
                db.rollback()
                logger.warning(f"Lỗi khi insert dòng thời tiết: {str(e2)}")
                continue
        return inserted
    except Exception as e:
        db.rollback()
        logger.error(f"Lỗi khi lưu daily_weather: {str(e)}", exc_info=True)
        raise

def load_consumption_dataframe(db: Session, evn_username: str) -> "pd.DataFrame":
    import pandas as pd
    rows = db.query(DailyConsumption).filter(
        DailyConsumption.evn_username == evn_username
    ).order_by(DailyConsumption.date.asc()).all()
    data = []
    for r in rows:
        data.append({
            "DATE": r.date.strftime("%d/%m/%Y") if r.date else None,
            "ENERGY": float(r.consumption_kwh) if r.consumption_kwh is not None else 0.0
        })
    df = pd.DataFrame(data)
    return df

def load_weather_dataframe(db: Session, location: str) -> "pd.DataFrame":
    import pandas as pd
    rows = db.query(DailyWeather).filter(
        DailyWeather.location == location
    ).order_by(DailyWeather.date.asc()).all()
    data = []
    for r in rows:
        data.append({
            "DATE_ONLY": r.date,
            "TEMPERATURE_AVG": float(r.avg_temp_c) if r.avg_temp_c is not None else None,
            "TEMPERATURE_MAX": float(r.max_temp_c) if r.max_temp_c is not None else None,
            "HUMIDITY_AVG": float(r.avg_humidity) if r.avg_humidity is not None else None
        })
    df = pd.DataFrame(data)
    return df
