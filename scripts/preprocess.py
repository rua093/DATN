import os
import json
from datetime import datetime
from typing import Tuple

import pandas as pd
import requests


def load_raw_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    if input_path.lower().endswith('.xlsx'):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    # Chuẩn hóa cột tên
    df.columns = [c.strip().upper() for c in df.columns]

    # Kỳ vọng có DATE và ENERGY
    if 'DATE' not in df.columns:
        raise ValueError("Thiếu cột DATE trong dữ liệu đầu vào")
    if 'ENERGY' not in df.columns:
        raise ValueError("Thiếu cột ENERGY trong dữ liệu đầu vào")

    # Parse ngày (hỗ trợ cả dd/mm/yyyy và dd/mm/yyyy HH:MM)
    df['DATE_PARSED'] = pd.to_datetime(df['DATE'], errors='coerce', dayfirst=True)
    if df['DATE_PARSED'].isna().all():
        raise ValueError("Không parse được cột DATE")

    # Lấy phần ngày (bỏ giờ) để merge với thời tiết
    df['DATE_ONLY'] = df['DATE_PARSED'].dt.date
    df = df.drop_duplicates(subset=['DATE_ONLY']).sort_values('DATE_ONLY').reset_index(drop=True)
    return df


def get_date_range(df: pd.DataFrame) -> Tuple[str, str]:
    start_date = df['DATE_ONLY'].min()
    end_date = df['DATE_ONLY'].max()
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


# Mapping location names to coordinates (latitude, longitude, timezone)
LOCATION_COORDINATES = {
    "Ho Chi Minh City": (10.8231, 106.6297, "Asia/Bangkok"),
    "HCMC": (10.8231, 106.6297, "Asia/Bangkok"),
    "Ho Chi Minh": (10.8231, 106.6297, "Asia/Bangkok"),
    "Hà Nội": (21.0285, 105.8542, "Asia/Bangkok"),
    "Hanoi": (21.0285, 105.8542, "Asia/Bangkok"),
    "Ha Noi": (21.0285, 105.8542, "Asia/Bangkok"),
    "Đà Nẵng": (16.0544, 108.2022, "Asia/Bangkok"),
    "Da Nang": (16.0544, 108.2022, "Asia/Bangkok"),
    "Cần Thơ": (10.0452, 105.7469, "Asia/Bangkok"),
    "Can Tho": (10.0452, 105.7469, "Asia/Bangkok"),
    "Hải Phòng": (20.8449, 106.6881, "Asia/Bangkok"),
    "Hai Phong": (20.8449, 106.6881, "Asia/Bangkok"),
}


def get_location_coordinates(location: str) -> Tuple[float, float, str]:
    """
    Get coordinates for a location name.
    Returns (latitude, longitude, timezone)
    Defaults to Ho Chi Minh City if location not found.
    """
    location_normalized = location.strip()
    # Try exact match first
    if location_normalized in LOCATION_COORDINATES:
        return LOCATION_COORDINATES[location_normalized]
    # Try case-insensitive match
    for key, coords in LOCATION_COORDINATES.items():
        if key.lower() == location_normalized.lower():
            return coords
    # Default to Ho Chi Minh City
    return LOCATION_COORDINATES["Ho Chi Minh City"]


def fetch_open_meteo_weather(start_date: str, end_date: str, location: str = "Ho Chi Minh City") -> pd.DataFrame:
    """
    Fetch weather data from Open-Meteo API.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        location: Location name (default: "Ho Chi Minh City")
    
    Returns:
        DataFrame with columns: DATE_ONLY, TEMPERATURE_AVG, TEMPERATURE_MAX, HUMIDITY_AVG
    """
    latitude, longitude, timezone = get_location_coordinates(location)
    timezone_encoded = timezone.replace("/", "%2F")
    
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relativehumidity_2m&timezone={timezone_encoded}"
    )

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    hourly = payload.get('hourly', {})
    times = hourly.get('time', [])
    temps = hourly.get('temperature_2m', [])
    hums = hourly.get('relativehumidity_2m', [])

    if not times:
        raise ValueError("API thời tiết trả về rỗng")

    weather = pd.DataFrame({
        'time': pd.to_datetime(times),
        'temperature_2m': temps,
        'relativehumidity_2m': hums,
    })
    weather['DATE_ONLY'] = weather['time'].dt.date

    # Tổng hợp theo ngày
    daily = weather.groupby('DATE_ONLY').agg(
        TEMPERATURE_AVG=('temperature_2m', 'mean'),
        TEMPERATURE_MAX=('temperature_2m', 'max'),
        HUMIDITY_AVG=('relativehumidity_2m', 'mean'),
    ).reset_index()

    # Làm sạch số liệu
    daily['TEMPERATURE_AVG'] = daily['TEMPERATURE_AVG'].round(2)
    daily['TEMPERATURE_MAX'] = daily['TEMPERATURE_MAX'].round(2)
    daily['HUMIDITY_AVG'] = daily['HUMIDITY_AVG'].round(2)
    return daily


def add_holiday_and_calendar_cols(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import holidays  # type: ignore
        vn_holidays = holidays.country_holidays('VN')
        df['HOLIDAY'] = df['DATE_PARSED'].dt.date.apply(lambda d: d in vn_holidays)
    except Exception:
        # Nếu chưa cài thư viện, đánh dấu không nghỉ lễ để không chặn pipeline
        df['HOLIDAY'] = False

    # Chuyển HOLIDAY sang 0/1
    df['HOLIDAY'] = df['HOLIDAY'].astype(int)

    df['MONTH'] = df['DATE_PARSED'].dt.month
    df['DAY'] = df['DATE_PARSED'].dt.day
    df['WEEKDAY'] = df['DATE_PARSED'].dt.weekday  # Monday=0 ... Sunday=6

    # TIME_INDEX: chỉ số thời gian tăng dần theo ngày
    df = df.sort_values('DATE_ONLY').reset_index(drop=True)
    df['TIME_INDEX'] = range(len(df))
    return df


def build_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    raw_df = load_raw_data(input_path)
    start_date, end_date = get_date_range(raw_df)

    weather_daily = fetch_open_meteo_weather(start_date, end_date)

    # Merge theo ngày
    merged = pd.merge(raw_df, weather_daily, on='DATE_ONLY', how='left')

    # Không thêm cột dư thừa TEMPERATURE/HUMIDITY (đã có *_AVG)

    # Bổ sung holiday và các cột lịch
    merged = add_holiday_and_calendar_cols(merged)

    # Xử lý ngày ENERGY = 0 do mất tín hiệu: tạo flag và biến mục tiêu điều chỉnh
    merged = merged.sort_values('DATE_PARSED').reset_index(drop=True)
    merged['ENERGY_ZERO_FLAG'] = (merged['ENERGY'] == 0).astype(int)

    # Impute ENERGY = 0 bằng rolling median của 14 ngày trước (chỉ tính ngày >0)
    energy_series = merged['ENERGY'].copy()
    energy_nonzero = energy_series.replace(0, pd.NA)
    rolling_med = energy_nonzero.rolling(window=14, min_periods=3).median()
    energy_adj = energy_series.copy()
    energy_adj.loc[energy_adj == 0] = rolling_med.loc[energy_adj == 0]
    # Fallback: nếu vẫn NaN (đầu chuỗi), dùng median toàn cục (bỏ 0)
    global_med = energy_nonzero.median()
    energy_adj = energy_adj.fillna(global_med if pd.notna(global_med) else energy_series.median())
    merged['ENERGY_ADJ'] = energy_adj

    # Sắp xếp, cột cuối cùng
    final_cols = [
        'DATE', 'ENERGY', 'ENERGY_ZERO_FLAG', 'ENERGY_ADJ',
        'TEMPERATURE_AVG', 'TEMPERATURE_MAX', 'HUMIDITY_AVG',
        'HOLIDAY', 'MONTH', 'DAY', 'WEEKDAY', 'TIME_INDEX'
    ]
    existing = [c for c in final_cols if c in merged.columns]
    merged = merged[existing]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def main():
    # Đường dẫn mặc định: dùng file crawler tạo ra
    # Bạn có thể đổi sang CSV nếu cần
    default_input_candidates = [
        # data/ (root)
        os.path.join('data', 'crawled_data_3years.xlsx'),
        os.path.join('data', 'crawled_data_raw.xlsx'),
        os.path.join('data', 'crawled_data_3years.csv'),
        os.path.join('data', 'crawled_data_raw.csv'),
        # scripts/data/ (trường hợp bạn lưu ở đây)
        os.path.join('scripts', 'data', 'crawled_data_3years.xlsx'),
        os.path.join('scripts', 'data', 'crawled_data_raw.xlsx'),
        os.path.join('scripts', 'data', 'crawled_data_3years.csv'),
        os.path.join('scripts', 'data', 'crawled_data_raw.csv'),
    ]

    input_path = None
    for p in default_input_candidates:
        if os.path.exists(p):
            input_path = p
            break

    if input_path is None:
        raise FileNotFoundError(
            "Không tìm thấy file đầu vào trong data/ hoặc scripts/data/. Hãy đặt file vào đó hoặc chỉ định đường dẫn."
        )

    # Lưu ra data/ (root). Nếu không tồn tại sẽ tự tạo
    output_path = os.path.join('data', 'dataset_clean.csv')
    merged = build_dataset(input_path, output_path)
    print(f"✅ Đã lưu dữ liệu sạch: {output_path} ({len(merged)} dòng)")


if __name__ == '__main__':
    main()


