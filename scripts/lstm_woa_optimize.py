import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# ======== 1. Tiền xử lý dữ liệu ========
np.random.seed(42)
tf.random.set_seed(42)

def preprocess_for_lstm(df):
    if "DATE" in df.columns:
        df = df.drop(columns=["DATE"])

    data = df.copy()

    data["DAY_sin"] = np.sin(2 * np.pi * data["DAY"] / 31)
    data["DAY_cos"] = np.cos(2 * np.pi * data["DAY"] / 31)
    data["MONTH_sin"] = np.sin(2 * np.pi * data["MONTH"] / 12)
    data["MONTH_cos"] = np.cos(2 * np.pi * data["MONTH"] / 12)
    data["WEEKDAY_sin"] = np.sin(2 * np.pi * data["WEEKDAY"] / 7)
    data["WEEKDAY_cos"] = np.cos(2 * np.pi * data["WEEKDAY"] / 7)
    data = data.drop(columns=["DAY", "MONTH", "WEEKDAY"])

    target_col = "ENERGY_ADJ" if "ENERGY_ADJ" in data.columns else "ENERGY"
    y = data[[target_col]].values
    # Giữ cả cột mục tiêu trong đặc trưng để LSTM nhìn thấy giá trị quá khứ của đích
    # giúp mô hình nắm bắt cấu trúc tự hồi quy (AR)
    X = data.values
    return X, y

def create_sequences(X, y, timesteps=24):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
        ys.append(y[i+timesteps])
    return np.array(Xs), np.array(ys)

# ======== 2. Đọc dữ liệu (raw, chưa tạo sequence) ========
df = pd.read_csv("data/dataset_clean.csv")
os.makedirs("results", exist_ok=True)
X_raw, y_raw = preprocess_for_lstm(df)

# ======== 3. Hàm tạo model và hàm fitness ========
def create_lstm_model(units, dropout, lr, input_shape):
    model = Sequential()
    # Tăng khả năng biểu diễn: 2 lớp LSTM + Dense ẩn
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(max(16, units // 2)))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss=tf.keras.losses.Huber())
    return model

# Mở rộng cửa sổ để bao phủ mùa vụ ngày/tuần (tùy tần suất dữ liệu)
candidate_timesteps = [14, 21, 24, 28, 30, 45, 48, 72, 96, 120, 168]

def split_scale_for_timesteps(X_raw, y_raw, timesteps):
    X_seq_raw, y_seq_raw = create_sequences(X_raw, y_raw, timesteps)
    total_size = len(X_seq_raw)
    train_size = int(total_size * 0.7)
    val_size   = int(total_size * 0.15)

    X_train_raw = X_seq_raw[:train_size]
    y_train_raw = y_seq_raw[:train_size]
    X_val_raw   = X_seq_raw[train_size:train_size+val_size]
    y_val_raw   = y_seq_raw[train_size:train_size+val_size]
    X_test_raw  = X_seq_raw[train_size+val_size:]
    y_test_raw  = y_seq_raw[train_size+val_size:]

    num_features = X_train_raw.shape[2]
    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train_raw.reshape(-1, num_features)).reshape(X_train_raw.shape)
    X_val   = scaler_X.transform(X_val_raw.reshape(-1, num_features)).reshape(X_val_raw.shape)
    X_test  = scaler_X.transform(X_test_raw.reshape(-1, num_features)).reshape(X_test_raw.shape)

    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train_raw)
    y_val   = scaler_y.transform(y_val_raw)
    y_test  = scaler_y.transform(y_test_raw)

    return (X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y)

def fitness(params):
    """
    params: [units, dropout, batch_size, lr, timestep_idx]
    Trả về val_loss nhỏ nhất (cần minimize)
    """
    units     = int(params[0])
    dropout   = params[1]
    batch_sz  = int(params[2])
    lr        = params[3]
    # map timestep index -> discrete candidates
    idx_raw   = int(round(params[4]))
    idx_clamped = max(0, min(len(candidate_timesteps)-1, idx_raw))
    timesteps = candidate_timesteps[idx_clamped]

    X_train, y_train, X_val, y_val, _, _, _, _ = split_scale_for_timesteps(X_raw, y_raw, timesteps)

    K.clear_session()
    model = create_lstm_model(units, dropout, lr,
                              input_shape=(X_train.shape[1], X_train.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,                 # tăng nhẹ epoch để đánh giá đủ tốt
        batch_size=batch_sz,
        verbose=0,
        callbacks=[es]
    )
    return min(history.history['val_loss'])

# ======== 4. Thuật toán WOA đơn giản ========
def WOA(fitness, bounds, n_agents=5, max_iter=10):
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    whales = np.random.rand(n_agents, dim) * (ub - lb) + lb
    best_pos = whales[0].copy()
    best_score = fitness(best_pos)

    best_history = []
    for i in range(max_iter):
        a = 2 - i * (2 / max_iter)  # giảm dần
        for j in range(n_agents):
            r = np.random.rand(dim)
            A = 2 * a * r - a
            C = 2 * r
            p = np.random.rand()
            b = 1
            l = np.random.uniform(-1, 1, dim)

            if p < 0.5:
                if np.linalg.norm(A) < 1:
                    D = abs(C * best_pos - whales[j])
                    whales[j] = best_pos - A * D
                else:
                    rand_pos = whales[np.random.randint(n_agents)]
                    D = abs(C * rand_pos - whales[j])
                    whales[j] = rand_pos - A * D
            else:
                D = abs(best_pos - whales[j])
                whales[j] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos

            whales[j] = np.clip(whales[j], lb, ub)
            score = fitness(whales[j])
            if score < best_score:
                best_score = score
                best_pos = whales[j].copy()

        print(f"Iter {i+1}/{max_iter} - Best val_loss: {best_score:.4f}")
        best_history.append(best_score)

    # Save convergence plot
    try:
        plt.figure(figsize=(8,4))
        plt.plot(best_history, marker='o')
        plt.title('WOA Convergence (Best val_loss per iteration)')
        plt.xlabel('Iteration')
        plt.ylabel('Best val_loss')
        plt.tight_layout()
        plt.savefig("results/woa_convergence.png", dpi=200)
        plt.close()
    except Exception:
        pass
    return best_pos, best_score

# ======== 5. Chạy tối ưu WOA ========
bounds = [
    (32, 256),    # units
    (0.1, 0.5),   # dropout
    (16, 64),     # batch size
    (1e-4, 1e-2), # learning rate
    (0, len(candidate_timesteps)-1)  # index của timesteps rời rạc
]

best_params, best_score = WOA(fitness, bounds, n_agents=5, max_iter=10)

print("\n===== Kết quả tối ưu WOA =====")
print("Best raw params:", best_params)
best_units    = int(best_params[0])
best_dropout  = best_params[1]
best_batch    = int(best_params[2])
best_lr       = best_params[3]
best_idx      = int(round(best_params[4]))
best_idx      = max(0, min(len(candidate_timesteps)-1, best_idx))
best_timesteps = candidate_timesteps[best_idx]
print("Best (units, dropout, batch_size, lr, timesteps):", best_units, best_dropout, best_batch, best_lr, best_timesteps)
print("Best val_loss:", best_score)

# ======== 6. Train lại mô hình cuối cùng với tham số tối ưu (tạo lại split/scale theo timesteps tốt nhất) ========
X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = split_scale_for_timesteps(X_raw, y_raw, best_timesteps)

final_model = create_lstm_model(best_units, best_dropout, best_lr,
                                input_shape=(X_train.shape[1], X_train.shape[2]))
es_final = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=best_batch,
    verbose=1,
    callbacks=[es_final]
)

final_model.save("models/my_lstm_model_woa.h5")
print("Model đã được lưu thành công!")

# ======== 7. Đánh giá trên tập test ========
y_pred_scaled = final_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_orig = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_test_orig, y_pred)
mse = mean_squared_error(y_test_orig, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_orig, y_pred)

print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

plt.figure(figsize=(12,6))
plt.plot(y_test_orig[:200], label='Thực tế')
plt.plot(y_pred[:200], label='Dự đoán')
plt.title("So sánh giá trị thực tế và dự đoán (200 điểm đầu)")
plt.xlabel("Thời điểm")
plt.ylabel("ENERGY")
plt.legend()
plt.tight_layout()
plt.savefig("results/woa_actual_vs_pred_200.png", dpi=200)
plt.close()

# Scatter plot
plt.figure(figsize=(6,6))
plt.scatter(y_test_orig, y_pred, alpha=0.6)
y_min, y_max = y_test_orig.min(), y_test_orig.max()
plt.plot([y_min, y_max], [y_min, y_max], 'r--')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.title('Scatter: Thực tế vs Dự đoán (WOA)')
plt.tight_layout()
plt.savefig("results/woa_scatter_actual_vs_pred.png", dpi=200)
plt.close()

# Residual histogram
residuals = y_test_orig.flatten() - y_pred.flatten()
plt.figure(figsize=(8,4))
plt.hist(residuals, bins=50, alpha=0.8, edgecolor='black')
plt.title('Phân bố Sai số Dự đoán (WOA)')
plt.xlabel('Sai số')
plt.ylabel('Tần suất')
plt.tight_layout()
plt.savefig("results/woa_residual_hist.png", dpi=200)
plt.close()

df_result = pd.DataFrame({
    "ThucTe": y_test_orig.flatten(),
    "DuDoan": y_pred.flatten()
})
df_result.to_csv("data/ketqua_du_doan_woa.csv", index=False)
print("Kết quả dự đoán đã lưu vào 'ketqua_du_doan_woa.csv'.")
