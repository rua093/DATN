import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import time

# ======== 1. Tiền xử lý dữ liệu ========
np.random.seed(42)
tf.random.set_seed(42)

def preprocess_for_lstm(df):
    """Tiền xử lý dữ liệu với cyclic encoding"""
    if "DATE" in df.columns:
        df = df.drop(columns=["DATE"])

    data = df.copy()

    # Cyclic encoding cho các biến thời gian
    data["HOUR_sin"] = np.sin(2 * np.pi * data["HOUR"] / 24)
    data["HOUR_cos"] = np.cos(2 * np.pi * data["HOUR"] / 24)
    data["DAY_sin"] = np.sin(2 * np.pi * data["DAY"] / 31)
    data["DAY_cos"] = np.cos(2 * np.pi * data["DAY"] / 31)
    data["MONTH_sin"] = np.sin(2 * np.pi * data["MONTH"] / 12)
    data["MONTH_cos"] = np.cos(2 * np.pi * data["MONTH"] / 12)
    data["WEEKDAY_sin"] = np.sin(2 * np.pi * data["WEEKDAY"] / 7)
    data["WEEKDAY_cos"] = np.cos(2 * np.pi * data["WEEKDAY"] / 7)
    data = data.drop(columns=["DAY", "MONTH", "HOUR", "WEEKDAY"])

    target_col = "ENERGY"
    y = data[[target_col]].values
    X = data.drop(columns=[target_col]).values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_X, scaler_y

def create_sequences(X, y, timesteps=24):
    """Tạo sequences cho LSTM"""
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
        ys.append(y[i+timesteps])
    return np.array(Xs), np.array(ys)

# ======== 2. Đọc dữ liệu ========
print("🔄 Đang tải dữ liệu...")
df = pd.read_csv("data/dataset_clean.csv")
X_scaled, y_scaled, scaler_X, scaler_y = preprocess_for_lstm(df)

timesteps = 24
X_seq, y_seq = create_sequences(X_scaled, y_scaled, timesteps)

# Chia dữ liệu
total_size = len(X_seq)
train_size = int(total_size * 0.7)
val_size = int(total_size * 0.15)

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]
X_val = X_seq[train_size:train_size+val_size]
y_val = y_seq[train_size:train_size+val_size]
X_test = X_seq[train_size+val_size:]
y_test = y_seq[train_size+val_size:]

print(f"✅ Dữ liệu: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

# ======== 3. Hàm tạo mô hình nâng cao ========
def create_advanced_lstm_model(params, input_shape):
    """
    Tạo mô hình LSTM nâng cao với nhiều layer và Bidirectional
    params: [lstm_units_1, lstm_units_2, dropout_1, dropout_2, batch_norm, bidirectional, lr]
    """
    lstm_units_1 = int(params[0])
    lstm_units_2 = int(params[1])
    dropout_1 = params[2]
    dropout_2 = params[3]
    batch_norm = params[4] > 0.5  # Boolean
    bidirectional = params[5] > 0.5  # Boolean
    lr = params[6]
    
    model = Sequential()
    
    # Layer 1: LSTM hoặc Bidirectional LSTM
    if bidirectional:
        model.add(Bidirectional(LSTM(lstm_units_1, return_sequences=True), input_shape=input_shape))
    else:
        model.add(LSTM(lstm_units_1, return_sequences=True, input_shape=input_shape))
    
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_1))
    
    # Layer 2: LSTM thứ hai
    if bidirectional:
        model.add(Bidirectional(LSTM(lstm_units_2, return_sequences=False)))
    else:
        model.add(LSTM(lstm_units_2, return_sequences=False))
    
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_2))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    
    return model

# ======== 4. Hàm fitness tối ưu hóa ========
def fitness_optimized(params):
    """
    Hàm fitness tối ưu hóa với ít dữ liệu để giảm thời gian
    Chỉ sử dụng 20% dữ liệu train và 5 epochs
    """
    # Lấy subset nhỏ để training nhanh
    subset_size = int(len(X_train) * 0.2)  # Chỉ 20% dữ liệu
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    
    subset_val_size = int(len(X_val) * 0.2)  # Chỉ 20% validation
    X_val_subset = X_val[:subset_val_size]
    y_val_subset = y_val[:subset_val_size]
    
    try:
        K.clear_session()
        model = create_advanced_lstm_model(params, (X_train.shape[1], X_train.shape[2]))
        
        # Early stopping với patience thấp
        es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        # Chỉ train 5 epochs để nhanh
        history = model.fit(
            X_train_subset, y_train_subset,
            validation_data=(X_val_subset, y_val_subset),
            epochs=5,  # Rất ít epochs
            batch_size=32,
            verbose=0,
            callbacks=[es]
        )
        
        # Trả về validation loss cuối cùng
        return min(history.history['val_loss'])
        
    except Exception as e:
        print(f"❌ Lỗi trong fitness: {e}")
        return float('inf')

# ======== 5. Thuật toán WOA cải tiến ========
def WOA_improved(fitness, bounds, n_agents=10, max_iter=20):
    """
    WOA cải tiến với nhiều agents và iterations hơn
    """
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # Khởi tạo whales
    whales = np.random.rand(n_agents, dim) * (ub - lb) + lb
    best_pos = whales[0].copy()
    best_score = fitness(best_pos)
    
    print(f"🎯 Bắt đầu WOA với {n_agents} agents, {max_iter} iterations")
    print(f"⏱️  Ước tính thời gian: ~{n_agents * max_iter * 5 * 0.2 * 0.1:.1f} phút")

    for i in range(max_iter):
        a = 2 - i * (2 / max_iter)  # giảm dần từ 2 xuống 0
        
        for j in range(n_agents):
            r = np.random.rand(dim)
            A = 2 * a * r - a
            C = 2 * r
            p = np.random.rand()
            b = 1
            l = np.random.uniform(-1, 1, dim)

            if p < 0.5:
                if np.linalg.norm(A) < 1:
                    # Exploitation: tìm kiếm xung quanh best
                    D = abs(C * best_pos - whales[j])
                    whales[j] = best_pos - A * D
                else:
                    # Exploration: tìm kiếm ngẫu nhiên
                    rand_pos = whales[np.random.randint(n_agents)]
                    D = abs(C * rand_pos - whales[j])
                    whales[j] = rand_pos - A * D
            else:
                # Spiral updating
                D = abs(best_pos - whales[j])
                whales[j] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos

            # Giới hạn trong bounds
            whales[j] = np.clip(whales[j], lb, ub)
            
            # Đánh giá fitness
            score = fitness(whales[j])
            if score < best_score:
                best_score = score
                best_pos = whales[j].copy()

        print(f"Iter {i+1}/{max_iter} - Best val_loss: {best_score:.6f}")

    return best_pos, best_score

# ======== 6. Chạy tối ưu hóa ========
print("\n🚀 BẮT ĐẦU TỐI ƯU HÓA MÔ HÌNH NÂNG CAO")
print("=" * 60)

# Bounds cho các tham số
bounds = [
    (32, 256),      # lstm_units_1
    (16, 128),      # lstm_units_2  
    (0.1, 0.6),     # dropout_1
    (0.1, 0.6),     # dropout_2
    (0, 1),         # batch_norm (0 hoặc 1)
    (0, 1),         # bidirectional (0 hoặc 1)
    (1e-4, 1e-2)    # learning_rate
]

print("📋 Tham số tối ưu hóa:")
print("   • LSTM units 1: 32-256")
print("   • LSTM units 2: 16-128")
print("   • Dropout 1: 0.1-0.6")
print("   • Dropout 2: 0.1-0.6")
print("   • Batch Normalization: Có/Không")
print("   • Bidirectional: Có/Không")
print("   • Learning Rate: 1e-4 đến 1e-2")

start_time = time.time()
best_params, best_score = WOA_improved(fitness_optimized, bounds, n_agents=10, max_iter=20)
optimization_time = time.time() - start_time

print(f"\n✅ HOÀN THÀNH TỐI ƯU HÓA trong {optimization_time:.1f} giây")
print("=" * 50)
print("🎯 KẾT QUẢ TỐI ƯU:")
print(f"   • Best val_loss: {best_score:.6f}")
print(f"   • LSTM units 1: {int(best_params[0])}")
print(f"   • LSTM units 2: {int(best_params[1])}")
print(f"   • Dropout 1: {best_params[2]:.3f}")
print(f"   • Dropout 2: {best_params[3]:.3f}")
print(f"   • Batch Norm: {'Có' if best_params[4] > 0.5 else 'Không'}")
print(f"   • Bidirectional: {'Có' if best_params[5] > 0.5 else 'Không'}")
print(f"   • Learning Rate: {best_params[6]:.6f}")

# ======== 7. Train mô hình cuối cùng ========
print(f"\n🏋️  TRAIN MÔ HÌNH CUỐI CÙNG")
print("=" * 40)

final_model = create_advanced_lstm_model(best_params, (X_train.shape[1], X_train.shape[2]))
es_final = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("📊 Kiến trúc mô hình cuối cùng:")
final_model.summary()

# Train với toàn bộ dữ liệu
train_start = time.time()
history = final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=[es_final]
)
train_time = time.time() - train_start

print(f"✅ Hoàn thành training trong {train_time:.1f} giây")

# Lưu mô hình
final_model.save("models/advanced_lstm_model.h5")
print("💾 Mô hình đã được lưu: models/advanced_lstm_model.h5")

# ======== 8. Đánh giá mô hình ========
print(f"\n📈 ĐÁNH GIÁ MÔ HÌNH")
print("=" * 30)

y_pred_scaled = final_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_orig = scaler_y.inverse_transform(y_test)

# Tính metrics
mae = mean_absolute_error(y_test_orig, y_pred)
mse = mean_squared_error(y_test_orig, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_orig, y_pred)
mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100

print(f"📊 KẾT QUẢ ĐÁNH GIÁ:")
print(f"   • MAE: {mae:.4f}")
print(f"   • MSE: {mse:.4f}")
print(f"   • RMSE: {rmse:.4f}")
print(f"   • R²: {r2:.4f}")
print(f"   • MAPE: {mape:.2f}%")

# Vẽ đồ thị
plt.figure(figsize=(15, 10))

# Subplot 1: So sánh dự đoán
plt.subplot(2, 2, 1)
plt.plot(y_test_orig[:200], label='Thực tế', linewidth=2)
plt.plot(y_pred[:200], label='Dự đoán', linewidth=2)
plt.title('So sánh Dự đoán vs Thực tế (200 điểm đầu)')
plt.xlabel('Thời điểm')
plt.ylabel('Năng lượng tiêu thụ')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Scatter plot
plt.subplot(2, 2, 2)
plt.scatter(y_test_orig, y_pred, alpha=0.6)
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.title('Scatter Plot: Thực tế vs Dự đoán')
plt.grid(True, alpha=0.3)

# Subplot 3: Training history
plt.subplot(2, 2, 3)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 4: Error distribution
plt.subplot(2, 2, 4)
errors = y_test_orig.flatten() - y_pred.flatten()
plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
plt.title('Phân bố Sai số Dự đoán')
plt.xlabel('Sai số')
plt.ylabel('Tần suất')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Lưu kết quả
results_df = pd.DataFrame({
    'ThucTe': y_test_orig.flatten(),
    'DuDoan': y_pred.flatten()
})
results_df.to_csv("data/advanced_lstm_predictions.csv", index=False)

# Lưu metrics
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE'],
    'Value': [mae, mse, rmse, r2, mape]
})
metrics_df.to_csv("data/advanced_lstm_metrics.csv", index=False)

print(f"\n🎉 HOÀN THÀNH!")
print("=" * 20)
print(f"⏱️  Tổng thời gian: {optimization_time + train_time:.1f} giây")
print(f"📁 Files đã tạo:")
print(f"   • models/advanced_lstm_model.h5")
print(f"   • data/advanced_lstm_predictions.csv")
print(f"   • data/advanced_lstm_metrics.csv")
