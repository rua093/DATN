import time
import re
import pandas as pd
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging
import os
from typing import List, Dict

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EVNCrawler:
    
    def __init__(self, headless: bool = False, wait_timeout: int = 30, username: str = None, password: str = None):
        self.wait_timeout = wait_timeout if wait_timeout >= 60 else 60  # Minimum 60 seconds
        self.driver = None
        self.headless = headless
        self.base_url = "https://www.evnhcmc.vn/"
        # Store credentials if provided, otherwise use defaults
        self.username = username if username else "0983716898"
        self.password = password if password else "BacNLH#201103"
        
    def _setup_driver(self):
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument('--headless')
            
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument(
                'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
            
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            except ImportError:
                logger.warning("webdriver_manager không được cài đặt. Sử dụng ChromeDriver mặc định.")
                self.driver = webdriver.Chrome(options=chrome_options)
            
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            logger.info("✅ Đã khởi tạo Chrome WebDriver thành công")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi khởi tạo WebDriver: {str(e)}")
            raise
    
    def login(self) -> bool:
        username = self.username
        password = self.password
        
        try:
            if self.driver is None:
                self._setup_driver()
            
            logger.info(f"🔐 Truy cập: {self.base_url}")
            self.driver.get(self.base_url)
            time.sleep(3)
            
            wait = WebDriverWait(self.driver, self.wait_timeout)
            
            login_trigger_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//*[contains(@class, 'btn-dangnhap')]"))
            )
            login_trigger_button.click()
            time.sleep(1)
            
            username_field = wait.until(
                EC.presence_of_element_located((By.XPATH, "//*[@class='form-control f-w-normal f-size-14 input-user']"))
            )
            username_field.clear()
            
            username_field.send_keys(username)
            
            password_field = wait.until(
                EC.presence_of_element_located((By.XPATH, "//*[@id='modalFormLogin']/div/div/div[2]/form/div[2]/div/div[1]/input"))
            )
            password_field.clear()
            password_field.send_keys(password)
            
            login_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//*[@id='modalFormLogin']/div/div/div[2]/form/div[5]/button"))
            )
            login_button.click()
            
            time.sleep(5)
            
            current_url = self.driver.current_url
            if "login" not in current_url.lower() or "dashboard" in current_url.lower():
                logger.info("✅ Đăng nhập thành công")
                return True
            else:
                logger.warning("⚠️ Đăng nhập thất bại")
                return False
                
        except TimeoutException:
            logger.error("❌ Timeout đăng nhập")
            return False
        except NoSuchElementException as e:
            logger.error(f"❌ Không tìm thấy element: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Lỗi đăng nhập: {str(e)}")
            return False
    
    def navigate_to_data_page(self, retry_count: int = 3):
        """Navigate to data page by direct URL"""
        data_page_url = "https://www.evnhcmc.vn/Tracuu/dienNangTieuThu"
        
        for attempt in range(retry_count):
        try:
                # Điều hướng trực tiếp đến URL
                self.driver.get(data_page_url)
                time.sleep(3)  # Chờ trang load
                
                # Kiểm tra xem đã đến đúng trang chưa bằng cách kiểm tra URL hoặc element đặc trưng
            wait = WebDriverWait(self.driver, self.wait_timeout)
                current_url = self.driver.current_url
                
                # Kiểm tra URL có chứa "dienNangTieuThu" hoặc "Tracuu"
                if "dienNangTieuThu" in current_url or "Tracuu" in current_url:
                    # Chờ một chút để đảm bảo trang đã load hoàn toàn
                    time.sleep(2)
            return True
                else:
                    if attempt < retry_count - 1:
                        time.sleep(5)
                        continue
                    else:
                        return False
            
            except TimeoutException as e:
                if attempt < retry_count - 1:
                    time.sleep(5)  # Wait before retry
                    continue
                else:
            return False
        except Exception as e:
                if attempt < retry_count - 1:
                    time.sleep(5)
                    continue
                else:
                    return False
        
            return False
    
    def select_date_range(self, start_date: datetime, end_date: datetime) -> bool:
        try:
            wait = WebDriverWait(self.driver, self.wait_timeout)
            
            time.sleep(2)
            
            date_picker_input = wait.until(
                EC.presence_of_element_located((By.XPATH, "//*[@id='input-thoigian']"))
            )
            
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", date_picker_input)
            time.sleep(0.5)
            
            date_range_str = f"{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"
            date_picker_input.clear()
            date_picker_input.click()
            time.sleep(0.5)
            date_picker_input.send_keys(date_range_str)
            time.sleep(1)
            
            apply_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//*[@id='c-app']/div[13]/div[4]/button[2]"))
            )
            apply_button.click()
            time.sleep(3)
            return True
            
        except TimeoutException:
            logger.error("❌ Timeout tìm input date range")
            return False
        except NoSuchElementException as e:
            logger.error(f"❌ Không tìm thấy input: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Lỗi điền date range: {str(e)}")
            return False
    
    def scrape_table_data(self) -> List[Dict]:
        data_list = []
        all_seen_dates = set()
        
        try:
            wait = WebDriverWait(self.driver, self.wait_timeout)
            time.sleep(2)
            
            table_body = wait.until(
                EC.presence_of_element_located((By.XPATH, "//*[@id='c-app']/main/div/div[2]/div[2]/div[2]/div/div[2]/div[2]/div[5]/table/tbody"))
            )
            
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", table_body)
            time.sleep(1)
            
            rows = table_body.find_elements(By.TAG_NAME, 'tr')
            
            if not rows:
                logger.warning("⚠️ Không tìm thấy dữ liệu trong bảng")
                return data_list
            
            logger.info(f"📊 Tìm thấy {len(rows)} hàng dữ liệu")
            
            record_idx = 0
            record_count = 0
            skipped_no_cells = 0
            skipped_empty = 0
            skipped_parse_error = 0
            skipped_duplicate = 0
            error_records = []
            
            while True:
                date_str = None
                energy_str = None
                date_row_idx = None
                energy_row_idx = None
                
                try:
                    date_row_idx = record_idx * 4  # 0, 4, 8, 12... (tr[1], tr[5], tr[9]...)
                    energy_row_idx = date_row_idx + 3  # 3, 7, 11, 15... (tr[4], tr[8], tr[12]...)
                    
                    if energy_row_idx >= len(rows):
                        break
                    
                    if date_row_idx >= len(rows):
                        break
                    
                    date_row = rows[date_row_idx]
                    energy_row = rows[energy_row_idx]
                    
                    date_cells = date_row.find_elements(By.TAG_NAME, 'td')
                    energy_cells = energy_row.find_elements(By.TAG_NAME, 'td')
                    
                    if len(date_cells) < 3:
                        skipped_no_cells += 1
                        record_idx += 1
                        continue
                    
                    if len(energy_cells) < 4:
                        skipped_no_cells += 1
                        record_idx += 1
                        continue
                    
                    date_cell = date_cells[1]
                    energy_cell = energy_cells[2]
                    
                    date_str = date_cell.text.strip()
                    energy_str = energy_cell.text.strip()
                    
                    if not date_str or not energy_str:
                        skipped_empty += 1
                        record_idx += 1
                        continue
                    
                    try:
                        if energy_str.strip() == '!' or energy_str.strip() == '-':
                            energy = 0.0
                        else:
                            energy_str_clean = energy_str.replace(',', '')
                            energy = float(energy_str_clean)
                    except ValueError as e:
                        error_records.append({
                            'RECORD_IDX': record_idx + 1,
                            'DATE_ROW': date_row_idx + 1,
                            'ENERGY_ROW': energy_row_idx + 1,
                            'DATE_RAW': date_str,
                            'ENERGY_RAW': energy_str,
                            'ERROR_TYPE': 'Parse Energy Error',
                            'ERROR_MSG': str(e)
                        })
                        skipped_parse_error += 1
                        record_idx += 1
                        continue
                    
                    try:
                        date_str_clean = date_str.strip()
                        
                        if 'Từ' in date_str_clean and 'đến' in date_str_clean:
                            pattern = r'(\d{1,2}/\d{1,2}/\d{4})'
                            dates = re.findall(pattern, date_str_clean)
                            if dates:
                                full_datetime = datetime.strptime(dates[-1], "%d/%m/%Y")
                                date_output = full_datetime.strftime('%d/%m/%Y')
                            else:
                                error_records.append({
                                    'RECORD_IDX': record_idx + 1,
                                    'DATE_ROW': date_row_idx + 1,
                                    'ENERGY_ROW': energy_row_idx + 1,
                                    'DATE_RAW': date_str,
                                    'ENERGY_RAW': energy_str,
                                    'ERROR_TYPE': 'Parse Date Range Error',
                                    'ERROR_MSG': f"Không tìm thấy date trong range"
                                })
                                skipped_parse_error += 1
                                record_idx += 1
                                continue
                        elif date_str_clean and date_str_clean[0].isdigit() and ' ' in date_str_clean:
                            parts = date_str_clean.split(' ', 1)
                            if len(parts) == 2 and '/' in parts[1]:
                                date_part = parts[1].strip()
                                if ':' in date_part:
                                    full_datetime = datetime.strptime(date_part, "%d/%m/%Y %H:%M")
                                    date_output = full_datetime.strftime('%d/%m/%Y %H:%M')
                                else:
                                    full_datetime = datetime.strptime(date_part, "%d/%m/%Y")
                                    date_output = full_datetime.strftime('%d/%m/%Y')
                            else:
                                raise ValueError(f"Format không hợp lệ: {date_str}")
                        elif '/' in date_str_clean and ':' in date_str_clean:
                            full_datetime = datetime.strptime(date_str_clean, "%d/%m/%Y %H:%M")
                            date_output = full_datetime.strftime('%d/%m/%Y %H:%M')
                        elif '-' in date_str_clean and ':' in date_str_clean:
                            full_datetime = datetime.strptime(date_str_clean, "%d-%m-%Y %H:%M")
                            date_output = full_datetime.strftime('%d/%m/%Y %H:%M')
                        elif '/' in date_str_clean:
                            if date_str_clean and date_str_clean[0].isdigit():
                                full_datetime = datetime.strptime(date_str_clean, "%d/%m/%Y")
                                date_output = full_datetime.strftime('%d/%m/%Y')
                            else:
                                raise ValueError(f"Invalid date format: {date_str}")
                        elif '-' in date_str_clean:
                            if date_str_clean and date_str_clean[0].isdigit():
                                full_datetime = datetime.strptime(date_str_clean, "%d-%m-%Y")
                                date_output = full_datetime.strftime('%d/%m/%Y')
                            else:
                                raise ValueError(f"Invalid date format: {date_str}")
                        else:
                            error_records.append({
                                'RECORD_IDX': record_idx + 1,
                                'DATE_ROW': date_row_idx + 1,
                                'ENERGY_ROW': energy_row_idx + 1,
                                'DATE_RAW': date_str,
                                'ENERGY_RAW': energy_str,
                                'ERROR_TYPE': 'Unknown Date Format',
                                'ERROR_MSG': f"Format không nhận dạng được"
                            })
                            skipped_parse_error += 1
                            record_idx += 1
                            continue
                    except ValueError as e:
                        error_records.append({
                            'RECORD_IDX': record_idx + 1,
                            'DATE_ROW': date_row_idx + 1,
                            'ENERGY_ROW': energy_row_idx + 1,
                            'DATE_RAW': date_str,
                            'ENERGY_RAW': energy_str,
                            'ERROR_TYPE': 'Parse Date Error',
                            'ERROR_MSG': str(e)
                        })
                        skipped_parse_error += 1
                        record_idx += 1
                        continue
                    
                    if date_output in all_seen_dates:
                        skipped_duplicate += 1
                        record_idx += 1
                        continue
                    
                    all_seen_dates.add(date_output)
                    
                    data_dict = {
                        'DATE': date_output,
                        'ENERGY': energy
                    }
                    
                    data_list.append(data_dict)
                    record_count += 1
                    
                    if record_count % 500 == 0:
                        logger.info(f"📊 Đã parse {record_count} bản ghi")
                    
                    record_idx += 1
                    
                except Exception as e:
                    logger.error(f"❌ Lỗi parse record {record_idx + 1}: {str(e)}")
                    error_records.append({
                        'RECORD_IDX': record_idx + 1,
                        'DATE_ROW': date_row_idx + 1 if date_row_idx is not None else 'N/A',
                        'ENERGY_ROW': energy_row_idx + 1 if energy_row_idx is not None else 'N/A',
                        'DATE_RAW': date_str if date_str is not None else 'N/A',
                        'ENERGY_RAW': energy_str if energy_str is not None else 'N/A',
                        'ERROR_TYPE': 'Exception',
                        'ERROR_MSG': str(e)
                    })
                    record_idx += 1
                    continue
            
            logger.info(f"✅ Parse thành công: {len(data_list)} bản ghi")
            if error_records:
                logger.info(f"⚠️ Lỗi parse: {len(error_records)} dòng")
            
        except Exception as e:
            logger.error(f"❌ Lỗi crawl bảng: {str(e)}")
        
        return data_list, error_records
    
    def crawl_3_years_data(
        self,
        years_back: int = 3
    ) -> pd.DataFrame:
        now = datetime.now()
        
        end_date = now - timedelta(days=1)
        start_year = now.year - years_back
        start_date = datetime(start_year, 1, 1)
        
        total_days = (end_date - start_date).days
        logger.info(f"🚀 Crawl {total_days} ngày: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
        
        if not self.navigate_to_data_page():
            logger.error("❌ Lỗi điều hướng")
            return pd.DataFrame()
        
        if not self.select_date_range(start_date, end_date):
            logger.error("❌ Lỗi chọn date range")
            return pd.DataFrame()
        
        data, error_records = self.scrape_table_data()
        
        if not data:
            logger.warning("⚠️ Không có dữ liệu")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.drop_duplicates(subset=['DATE'], keep='first')
        
        if ':' in df['DATE'].iloc[0] if len(df) > 0 else False:
            df['DATE_PARSED'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y %H:%M', errors='coerce')
        else:
            df['DATE_PARSED'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')
        
        df = df.sort_values('DATE_PARSED')
        df = df.drop(columns=['DATE_PARSED'])
        
        return df
    
    def crawl_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        if not self.navigate_to_data_page():
            logger.error("❌ Lỗi điều hướng")
            return pd.DataFrame()
        
        total_days = (end_date - start_date).days
        logger.info(f"🚀 Crawl {total_days} ngày: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
        
        if not self.select_date_range(start_date, end_date):
            logger.error("❌ Lỗi chọn date range")
            return pd.DataFrame()
        
        data, error_records = self.scrape_table_data()
        
        if not data:
            logger.warning("⚠️ Không có dữ liệu")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.drop_duplicates(subset=['DATE'], keep='first')
        
        if ':' in df['DATE'].iloc[0] if len(df) > 0 else False:
            df['DATE_PARSED'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y %H:%M', errors='coerce')
        else:
            df['DATE_PARSED'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')
        
        df = df.sort_values('DATE_PARSED')
        df = df.drop(columns=['DATE_PARSED'])
        
        return df
    
    def close(self):
        if self.driver:
            self.driver.quit()
            logger.info("✅ Đã đóng trình duyệt")


def main():
    crawler = EVNCrawler(headless=False)
    
    try:
        if not crawler.login():
            logger.error("❌ Đăng nhập thất bại. Dừng crawler.")
            return
        
        df = crawler.crawl_3_years_data(years_back=3)
        
        if not df.empty:
            logger.info(f"✅ Hoàn tất: {len(df)} bản ghi")
        else:
            logger.warning("⚠️ Không có dữ liệu")
            
    except Exception as e:
        logger.error(f"❌ Lỗi: {str(e)}")
    finally:
        crawler.close()


if __name__ == "__main__":
    main()
