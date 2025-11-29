import sys
import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from preprocess import fetch_open_meteo_weather, add_holiday_and_calendar_cols
from server.config import MODELS_DIR
from server.database import SessionLocal, save_daily_weather_rows, get_account_by_username, load_consumption_dataframe, load_weather_dataframe
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self):
        self.model = None
    
    def preprocess_for_base_model(self, df: pd.DataFrame):
        data = df.copy()
        if "DATE" in data.columns:
            data = data.drop(columns=["DATE"])
        if "TEMPERATURE_MAX" in data.columns:
            data = data.drop(columns=["TEMPERATURE_MAX"])
        if "MONTH" in data.columns:
            data["month_sin"] = np.sin(2 * np.pi * data["MONTH"] / 12)
            data["month_cos"] = np.cos(2 * np.pi * data["MONTH"] / 12)
            data = data.drop(columns=["MONTH"])
        if "WEEKDAY" in data.columns:
            data["weekday_sin"] = np.sin(2 * np.pi * data["WEEKDAY"] / 7)
            data["weekday_cos"] = np.cos(2 * np.pi * data["WEEKDAY"] / 7)
            data = data.drop(columns=["WEEKDAY"])
        if "DAY" in data.columns:
            data = data.drop(columns=["DAY"])
        target_col = "ENERGY_ADJ" if "ENERGY_ADJ" in data.columns else "ENERGY"
        cols = [c for c in data.columns if c != target_col] + [target_col]
        data = data[cols]
        
        return data
    
    def _handle_accumulated_energy(
        self, 
        energy_series: pd.Series, 
        rolling_med: pd.Series, 
        rolling_mean: pd.Series,
        energy_nonzero: pd.Series
    ) -> pd.Series:
        energy_adj = energy_series.copy()
        n = len(energy_adj)
        
        # Tính global median để dùng khi thiếu dữ liệu
        global_med = energy_nonzero.median()
        if pd.isna(global_med):
            global_med = energy_series.median()
        
        # Duyệt qua từng ngày để phát hiện và xử lý
        i = 0
        while i < n:
            if energy_adj.iloc[i] > 0:  # Ngày có tín hiệu
                # Đếm số ngày mất tín hiệu liên tiếp trước đó
                zero_count = 0
                j = i - 1
                while j >= 0 and energy_series.iloc[j] == 0:
                    zero_count += 1
                    j -= 1
                
                # Nếu có ngày mất tín hiệu trước đó → chắc chắn là cộng dồn, luôn phân bổ lại
                if zero_count > 0:
                    current_value = energy_adj.iloc[i]
                    
                    # Chia đều toàn bộ giá trị cho tất cả các ngày (bao gồm cả ngày có điện)
                    total_days = zero_count + 1  # Số ngày mất tín hiệu + 1 ngày có điện
                    energy_per_day = current_value / total_days
                    
                    # Phân bổ đều cho tất cả các ngày
                    for k in range(i - zero_count, i + 1):  # Bao gồm cả ngày hiện tại
                        if k >= 0:
                            energy_adj.iloc[k] = energy_per_day
                    
                    logger.debug(
                        f"Đã phân bổ lại năng lượng: {total_days} ngày (bao gồm {zero_count} ngày mất tín hiệu) "
                        f"nhận {energy_per_day:.2f} kWh mỗi ngày từ tổng {current_value:.2f} kWh"
                    )
            i += 1
        
        # Xử lý các ngày còn lại có ENERGY = 0 (không được phân bổ)
        energy_adj.loc[energy_adj == 0] = rolling_med.loc[energy_adj == 0]
        energy_adj = energy_adj.fillna(global_med if pd.notna(global_med) else energy_series.median())
        
        return energy_adj
    
    def build_dataset_from_db(self, db: Session, evn_username: str, location: str) -> pd.DataFrame:
        raw_df = load_consumption_dataframe(db, evn_username)
        if raw_df.empty:
            raise ValueError(f"Không có dữ liệu tiêu thụ cho {evn_username}")
        raw_df.columns = [c.strip().upper() for c in raw_df.columns]
        if 'DATE' not in raw_df.columns or 'ENERGY' not in raw_df.columns:
            raise ValueError("Thiếu cột DATE hoặc ENERGY trong dữ liệu")
        raw_df['DATE_PARSED'] = pd.to_datetime(raw_df['DATE'], errors='coerce', dayfirst=True)
        if raw_df['DATE_PARSED'].isna().all():
            raise ValueError("Không parse được cột DATE")
        raw_df['DATE_ONLY'] = raw_df['DATE_PARSED'].dt.date
        raw_df = raw_df.drop_duplicates(subset=['DATE_ONLY']).sort_values('DATE_ONLY').reset_index(drop=True)
        start_date = raw_df['DATE_ONLY'].min().strftime('%Y-%m-%d')
        end_date = raw_df['DATE_ONLY'].max().strftime('%Y-%m-%d')
        weather_db = load_weather_dataframe(db, location)
        if weather_db.empty:
            logger.info(f"Không có dữ liệu thời tiết trong DB, lấy từ API cho location '{location}'")
            weather_daily = fetch_open_meteo_weather(start_date, end_date, location=location)
        else:
            if 'DATE_ONLY' in weather_db.columns:
                if isinstance(weather_db['DATE_ONLY'].iloc[0], str):
                    weather_db['DATE_ONLY'] = pd.to_datetime(weather_db['DATE_ONLY']).dt.date
            weather_daily = weather_db
        if 'DATE_ONLY' in weather_daily.columns:
            weather_daily = weather_daily.drop_duplicates(subset=['DATE_ONLY'])
        elif 'date' in weather_daily.columns:
            weather_daily = weather_daily.drop_duplicates(subset=['date'])
        merged = pd.merge(raw_df, weather_daily, on='DATE_ONLY', how='left')
        merged = add_holiday_and_calendar_cols(merged)
        merged = merged.sort_values('DATE_PARSED').reset_index(drop=True)
        
        # Đảm bảo ENERGY là numeric
        merged['ENERGY'] = pd.to_numeric(merged['ENERGY'], errors='coerce')
        
        merged['ENERGY_ZERO_FLAG'] = (merged['ENERGY'] == 0).astype(int)
        energy_series = merged['ENERGY'].copy()
        energy_nonzero = energy_series.replace(0, pd.NA)
        
        # Đảm bảo energy_nonzero là numeric trước khi tính rolling median
        energy_nonzero = pd.to_numeric(energy_nonzero, errors='coerce')
        rolling_med = energy_nonzero.rolling(window=14, min_periods=3).median()
        rolling_mean = energy_nonzero.rolling(window=14, min_periods=3).mean()
        
        # Xử lý giá trị cao bất thường do cộng dồn từ các ngày mất tín hiệu
        energy_adj = self._handle_accumulated_energy(
            energy_series, rolling_med, rolling_mean, energy_nonzero
        )
        
        merged['ENERGY_ADJ'] = energy_adj
        final_cols = [
            'DATE', 'ENERGY_ADJ',
            'TEMPERATURE_AVG', 'TEMPERATURE_MAX', 'HUMIDITY_AVG',
            'HOLIDAY', 'MONTH', 'DAY', 'WEEKDAY'
        ]
        existing = [c for c in final_cols if c in merged.columns]
        merged = merged[existing]
        weather_rows = []
        for _, row in merged.iterrows():
            date_val = row.get('DATE_ONLY') or (pd.to_datetime(row.get('DATE'), dayfirst=True).date() if pd.notna(row.get('DATE')) else None)
            if date_val:
                weather_rows.append({
                    "date": date_val,
                    "max_temp_c": float(row.get("TEMPERATURE_MAX")) if pd.notna(row.get("TEMPERATURE_MAX")) else None,
                    "avg_temp_c": float(row.get("TEMPERATURE_AVG")) if pd.notna(row.get("TEMPERATURE_AVG")) else None,
                    "avg_humidity": float(row.get("HUMIDITY_AVG")) if pd.notna(row.get("HUMIDITY_AVG")) else None,
                })
        if weather_rows:
            inserted = save_daily_weather_rows(db, location, weather_rows)
            logger.info(f"Đã lưu {inserted} dòng thời tiết vào daily_weather cho '{location}'")
        return merged
    
    def train_model(
        self, 
        evn_username: str, 
        db: Session,
        fine_tune_lr: float = 0.0001,
        fine_tune_epochs: int = 30
    ) -> dict:
        try:
            logger.info(f"Bắt đầu fine-tune từ base model cho {evn_username}")
            return self.fine_tune_from_base_model(
                evn_username=evn_username,
                db=db,
                fine_tune_lr=fine_tune_lr,
                epochs=fine_tune_epochs
            )
        except Exception as e:
            logger.error(f"Lỗi trong train_model: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
        finally:
            K.clear_session()
    
    def fine_tune_from_base_model(
        self, 
        evn_username: str, 
        db: Session,
        base_model_path: Path = None,
        fine_tune_lr: float = 0.0001,
        epochs: int = 30,
        batch_size: int = 32,
        patience: int = 10
    ) -> dict:
        try:
            logger.info(f"Bắt đầu fine-tune từ base model cho {evn_username}")
            
            if base_model_path is None:
                base_model_path = BASE_DIR / "models" / "woa_lstm_final.keras"
            if not base_model_path.exists():
                return {"success": False, "error": f"Không tìm thấy base model tại {base_model_path}"}
            logger.info(f"Đang load base model từ {base_model_path}")
            base_model = tf.keras.models.load_model(str(base_model_path), compile=False)
            
            acc = get_account_by_username(db, evn_username)
            if not acc:
                return {"success": False, "error": f"Không tìm thấy tài khoản {evn_username}"}
            location = acc.location if acc.location else "Ho Chi Minh City"
            df = self.build_dataset_from_db(db, evn_username, location)
            logger.info(f"Đã tải {len(df)} bản ghi cho user {evn_username}")
            
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)
                df = df.set_index('DATE')
            
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            target_col = "ENERGY_ADJ" if "ENERGY_ADJ" in train_df.columns else "ENERGY"
            train_df[target_col] = train_df[target_col].apply(lambda x: np.nan if x < 0.1 else x)
            train_df[target_col] = train_df[target_col].interpolate(method='time')
            p99 = train_df[target_col].quantile(0.99)
            train_df[target_col] = train_df[target_col].clip(upper=p99)
            test_df[target_col] = test_df[target_col].apply(lambda x: np.nan if x < 0.1 else x)
            test_df[target_col] = test_df[target_col].ffill()
            test_df[target_col] = test_df[target_col].fillna(train_df[target_col].iloc[-1])
            
            train_df_processed = self.preprocess_for_base_model(train_df.reset_index())
            test_df_processed = self.preprocess_for_base_model(test_df.reset_index())
            n_features_expected = base_model.input_shape[2]
            n_features_actual = len(train_df_processed.columns)
            if n_features_actual != n_features_expected:
                logger.warning(f"Số features không khớp: expected {n_features_expected}, actual {n_features_actual}.")
            TIME_STEPS = 7
            if len(train_df_processed) < TIME_STEPS + 1:
                return {"success": False, "error": f"Không đủ dữ liệu. Cần ít nhất {TIME_STEPS + 1} mẫu, hiện có {len(train_df_processed)}"}
            
            # Kiểm tra kích thước dữ liệu - cần đủ mẫu để fine-tune hiệu quả
            min_samples_for_finetune = 100
            if len(train_df_processed) < min_samples_for_finetune:
                logger.warning(f"Dữ liệu user chỉ có {len(train_df_processed)} mẫu, ít hơn {min_samples_for_finetune} mẫu khuyến nghị. Fine-tuning có thể không hiệu quả.")
            
            # Tạo scaler mới cho dữ liệu user
            # Lưu ý: Scaler này sẽ khác với scaler của base model, nhưng cần thiết vì phân phối dữ liệu user có thể khác
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_scaled = scaler.fit_transform(train_df_processed)
            test_scaled = scaler.transform(test_df_processed)
            def create_sequences(data, time_steps):
                X, y = [], []
                for i in range(len(data) - time_steps):
                    X.append(data[i:(i + time_steps), :])
                    y.append(data[i + time_steps, -1])
                return np.array(X), np.array(y)
            X_train, y_train = create_sequences(train_scaled, TIME_STEPS)
            X_test, y_test = create_sequences(test_scaled, TIME_STEPS)
            val_size = int(len(X_train) * 0.2)
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train = X_train[:-val_size]
            y_train = y_train[:-val_size]
            
            logger.info(f"Dữ liệu đã sẵn sàng: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
            if X_train.shape[1] != base_model.input_shape[1] or X_train.shape[2] != base_model.input_shape[2]:
                return {"success": False, "error": f"Input shape không khớp: model expects {base_model.input_shape[1:]}, got {X_train.shape[1:]}"}
            
            # Định nghĩa hàm inverse transform
            n_feat = X_train.shape[2] - 1
            def inverse_target(pred, scaler, n_features):
                dummy = np.zeros((len(pred), n_features + 1))
                dummy[:, -1] = pred.flatten()
                return scaler.inverse_transform(dummy)[:, -1]
            
            # Đánh giá base model trên validation set để so sánh
            logger.info("Đang đánh giá base model trên validation set...")
            base_pred_val = base_model.predict(X_val, verbose=0)
            y_val_true = inverse_target(y_val, scaler, n_feat)
            y_val_base_pred = inverse_target(base_pred_val, scaler, n_feat)
            base_val_mae = mean_absolute_error(y_val_true, y_val_base_pred)
            base_val_rmse = np.sqrt(mean_squared_error(y_val_true, y_val_base_pred))
            logger.info(f"Base model - Val MAE: {base_val_mae:.4f}, Val RMSE: {base_val_rmse:.4f}")
            
            K.clear_session()
            fine_tuned_model = tf.keras.models.clone_model(base_model)
            fine_tuned_model.set_weights(base_model.get_weights())
            
            # Freeze các layers đầu (LSTM layers) để giữ kiến thức từ base model
            # Chỉ train các layers cuối (Dense layers)
            num_layers = len(fine_tuned_model.layers)
            layers_to_freeze = max(0, num_layers - 2)  # Freeze tất cả trừ 2 layers cuối
            
            for i in range(layers_to_freeze):
                fine_tuned_model.layers[i].trainable = False
                logger.info(f"Đã freeze layer {i}: {fine_tuned_model.layers[i].name}")
            
            # Đếm số layers trainable
            trainable_count = sum([1 for layer in fine_tuned_model.layers if layer.trainable])
            logger.info(f"Số layers trainable: {trainable_count}/{num_layers}")
            
            fine_tuned_model.compile(optimizer=Adam(learning_rate=fine_tune_lr), loss='mse')
            
            # Callbacks: Early stopping và Reduce learning rate on plateau
            es = EarlyStopping(
                monitor='val_loss', 
                patience=patience, 
                restore_best_weights=True, 
                verbose=1,
                min_delta=1e-6
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
            
            logger.info(f"Bắt đầu fine-tuning với lr={fine_tune_lr}, epochs={epochs}, batch_size={batch_size}")
            logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
            
            history = fine_tuned_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[es, reduce_lr],
                verbose=1
            )
            
            y_pred_scaled = fine_tuned_model.predict(X_test, verbose=0)
            y_true = inverse_target(y_test, scaler, n_feat)
            y_pred = inverse_target(y_pred_scaled, scaler, n_feat)
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            
            metrics = {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "r2": float(r2),
                "mape": float(mape)
            }
            
            model_dir = MODELS_DIR / f"user_{evn_username}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "lstm_model.h5"
            scaler_path = model_dir / "scaler_x.pkl"
            scaler_y_path = model_dir / "scaler_y.pkl"
            fine_tuned_model.save(str(model_path))
            joblib.dump(scaler, scaler_path)
            joblib.dump(scaler, scaler_y_path)
            
            logger.info(f"Fine-tuning hoàn tất. Metrics: {metrics}")
            
            # So sánh với base model trên test set
            logger.info("Đang so sánh với base model trên test set...")
            base_pred_test = base_model.predict(X_test, verbose=0)
            y_test_true = inverse_target(y_test, scaler, n_feat)
            y_test_base_pred = inverse_target(base_pred_test, scaler, n_feat)
            base_test_mae = mean_absolute_error(y_test_true, y_test_base_pred)
            base_test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_base_pred))
            base_test_r2 = r2_score(y_test_true, y_test_base_pred)
            
            logger.info(f"Base model - Test MAE: {base_test_mae:.4f}, Test RMSE: {base_test_rmse:.4f}, Test R2: {base_test_r2:.4f}")
            logger.info(f"Fine-tuned model - Test MAE: {metrics['mae']:.4f}, Test RMSE: {metrics['rmse']:.4f}, Test R2: {metrics['r2']:.4f}")
            
            improvement_mae = ((base_test_mae - metrics['mae']) / base_test_mae) * 100 if base_test_mae > 0 else 0
            improvement_rmse = ((base_test_rmse - metrics['rmse']) / base_test_rmse) * 100 if base_test_rmse > 0 else 0
            improvement_r2 = metrics['r2'] - base_test_r2
            
            logger.info(f"Cải thiện: MAE {improvement_mae:+.2f}%, RMSE {improvement_rmse:+.2f}%, R2 {improvement_r2:+.4f}")
            
            if metrics['r2'] < base_test_r2:
                logger.warning(f"Fine-tuned model có R2 ({metrics['r2']:.4f}) thấp hơn base model ({base_test_r2:.4f}). Có thể do dữ liệu quá ít hoặc learning rate chưa tối ưu.")
            
            logger.info(f"Model đã lưu tại: {model_path}")
            
            return {
                "success": True,
                "model_path": str(model_path),
                "scaler_x_path": str(scaler_path),
                "scaler_y_path": str(scaler_y_path),
                "metrics": metrics,
                "training_params": {
                    "method": "fine_tune_from_base",
                    "base_model": str(base_model_path),
                    "fine_tune_lr": fine_tune_lr,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "timesteps": TIME_STEPS
                }
            }
            
        except Exception as e:
            logger.error(f"Lỗi trong fine_tune_from_base_model: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
        finally:
            K.clear_session()
