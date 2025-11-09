import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evn_crawler import EVNCrawler
from server.config import CRAWLER_HEADLESS

logger = logging.getLogger(__name__)

class CrawlerService:
    def __init__(self, username: str = None, password: str = None):
        self.username = username
        self.password = password
        self.crawler = None
    
    def crawl_initial_data(self, evn_username: str, years_back: int = 3) -> dict:
        try:
            logger.info(f"Bắt đầu crawl dữ liệu ban đầu cho {evn_username}, {years_back} năm trước")
            self.crawler = EVNCrawler(headless=CRAWLER_HEADLESS, username=self.username, password=self.password)
            if not self.crawler.login():
                return {"success": False, "error": "Đăng nhập thất bại"}
            df = self.crawler.crawl_3_years_data(years_back=years_back)
            if df.empty:
                return {"success": False, "error": "Không crawl được dữ liệu"}
            logger.info(f"Crawl thành công {len(df)} bản ghi cho {evn_username}")
            try:
                from server.database import SessionLocal, save_daily_consumption_rows
                import datetime as dt
                db = SessionLocal()
                try:
                    rows = []
                    skipped_parse = 0
                    for _, row in df.iterrows():
                        date_str = str(row["DATE"]) if "DATE" in df.columns else None
                        energy = row.get("ENERGY") if "ENERGY" in df.columns else None
                        if date_str is None or energy is None:
                            skipped_parse += 1
                            continue
                        try:
                            if ":" in date_str:
                                d = dt.datetime.strptime(date_str, "%d/%m/%Y %H:%M").date()
                            else:
                                d = dt.datetime.strptime(date_str, "%d/%m/%Y").date()
                        except Exception as e:
                            skipped_parse += 1
                            logger.debug(f"Bỏ qua dòng không parse được date: {date_str}, lỗi: {e}")
                            continue
                        rows.append({"date": d, "consumption_kwh": energy})
                    logger.info(f"Đã parse {len(rows)} dòng hợp lệ từ {len(df)} bản ghi (bỏ qua {skipped_parse} dòng khi parse)")
                    
                    # Loại bỏ duplicate trong rows (cùng date)
                    seen_dates = set()
                    unique_rows = []
                    duplicate_in_df = 0
                    for r in rows:
                        date_key = (evn_username, r["date"])
                        if date_key in seen_dates:
                            duplicate_in_df += 1
                            continue
                        seen_dates.add(date_key)
                        unique_rows.append(r)
                    
                    if duplicate_in_df > 0:
                        logger.info(f"Phát hiện {duplicate_in_df} dòng duplicate trong DataFrame (cùng date), chỉ lưu {len(unique_rows)} dòng unique")
                    
                    if unique_rows:
                        inserted = save_daily_consumption_rows(db, evn_username, unique_rows)
                        logger.info(f"Đã lưu {inserted} dòng vào daily_consumption cho {evn_username} (từ {len(unique_rows)} dòng unique, có thể có {len(unique_rows) - inserted} dòng đã tồn tại trong DB)")
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Không thể lưu daily_consumption: {str(e)}", exc_info=True)
            return {"success": True, "records": len(df)}
        except Exception as e:
            logger.error(f"Lỗi trong crawl_initial_data: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
        finally:
            if self.crawler:
                self.crawler.close()
    
    def crawl_yesterday_data(self, evn_username: str) -> dict:
        try:
            logger.info(f"Bắt đầu crawl dữ liệu hàng ngày cho {evn_username}")
            yesterday = datetime.now() - timedelta(days=1)
            self.crawler = EVNCrawler(headless=CRAWLER_HEADLESS, username=self.username, password=self.password)
            if not self.crawler.login():
                return {"success": False, "error": "Đăng nhập thất bại"}
            df = self.crawler.crawl_date_range(start_date=yesterday, end_date=yesterday)
            if df.empty:
                return {"success": False, "error": "Không crawl được dữ liệu"}
            logger.info(f"Crawl thành công {len(df)} bản ghi cho {evn_username} (ngày: {yesterday.date()})")
            try:
                from server.database import SessionLocal, save_daily_consumption_rows
                import datetime as dt
                db = SessionLocal()
                try:
                    rows = []
                    skipped_parse = 0
                    for _, row in df.iterrows():
                        date_str = str(row["DATE"]) if "DATE" in df.columns else None
                        energy = row.get("ENERGY") if "ENERGY" in df.columns else None
                        if date_str is None or energy is None:
                            skipped_parse += 1
                            continue
                        try:
                            if ":" in date_str:
                                d = dt.datetime.strptime(date_str, "%d/%m/%Y %H:%M").date()
                            else:
                                d = dt.datetime.strptime(date_str, "%d/%m/%Y").date()
                        except Exception as e:
                            skipped_parse += 1
                            logger.debug(f"Bỏ qua dòng không parse được date: {date_str}, lỗi: {e}")
                            continue
                        rows.append({"date": d, "consumption_kwh": energy})
                    logger.info(f"Đã parse {len(rows)} dòng hợp lệ từ {len(df)} bản ghi (bỏ qua {skipped_parse} dòng khi parse)")
                    
                    # Loại bỏ duplicate trong rows (cùng date)
                    seen_dates = set()
                    unique_rows = []
                    duplicate_in_df = 0
                    for r in rows:
                        date_key = (evn_username, r["date"])
                        if date_key in seen_dates:
                            duplicate_in_df += 1
                            continue
                        seen_dates.add(date_key)
                        unique_rows.append(r)
                    
                    if duplicate_in_df > 0:
                        logger.info(f"Phát hiện {duplicate_in_df} dòng duplicate trong DataFrame (cùng date), chỉ lưu {len(unique_rows)} dòng unique")
                    
                    if unique_rows:
                        inserted = save_daily_consumption_rows(db, evn_username, unique_rows)
                        logger.info(f"Đã lưu {inserted} dòng vào daily_consumption cho {evn_username} (từ {len(unique_rows)} dòng unique, có thể có {len(unique_rows) - inserted} dòng đã tồn tại trong DB)")
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Không thể lưu daily_consumption: {str(e)}", exc_info=True)
            return {"success": True, "records": len(df), "date": yesterday.date().isoformat()}
        except Exception as e:
            logger.error(f"Lỗi trong crawl_yesterday_data: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
        finally:
            if self.crawler:
                self.crawler.close()
