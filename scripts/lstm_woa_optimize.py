import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
        ys.append(y[i+timesteps])
    return np.array(Xs), np.array(ys)

# ======== 2. Đọc dữ liệu ========
df = pd.read_csv("../data/dataset_clean.csv")
X_scaled, y_scaled, scaler_X, scaler_y = preprocess_for_lstm(df)

timesteps = 24
X_seq, y_seq = create_sequences(X_scaled, y_scaled, timesteps)

total_size = len(X_seq)
train_size = int(total_size * 0.7)
val_size   = int(total_size * 0.15)

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]
X_val   = X_seq[train_size:train_size+val_size]
y_val   = y_seq[train_size:train_size+val_size]
X_test  = X_seq[train_size+val_size:]
y_test  = y_seq[train_size+val_size:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ======== 3. Hàm tạo model và hàm fitness ========
def create_lstm_model(units, dropout, lr, input_shape):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model

def fitness(params):
    """
    params: [units, dropout, batch_size, lr]
    Trả về val_loss nhỏ nhất (cần minimize)
    """
    units     = int(params[0])
    dropout   = params[1]
    batch_sz  = int(params[2])
    lr        = params[3]

    K.clear_session()
    model = create_lstm_model(units, dropout, lr,
                              input_shape=(X_train.shape[1], X_train.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,                 # ít epoch để tìm nhanh
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

    return best_pos, best_score

# ======== 5. Chạy tối ưu WOA ========
bounds = [
    (32, 256),    # units
    (0.1, 0.5),   # dropout
    (16, 64),     # batch size
    (1e-4, 1e-2)  # learning rate
]

best_params, best_score = WOA(fitness, bounds, n_agents=5, max_iter=10)

print("\n===== Kết quả tối ưu WOA =====")
print("Best params (units, dropout, batch_size, lr):", best_params)
print("Best val_loss:", best_score)

best_units    = int(best_params[0])
best_dropout  = best_params[1]
best_batch    = int(best_params[2])
best_lr       = best_params[3]

# ======== 6. Train lại mô hình cuối cùng với tham số tối ưu ========
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

final_model.save("../models/my_lstm_model_woa.h5")
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
plt.show()

df_result = pd.DataFrame({
    "ThucTe": y_test_orig.flatten(),
    "DuDoan": y_pred.flatten()
})
df_result.to_csv("../data/ketqua_du_doan_woa.csv", index=False)
print("Kết quả dự đoán đã lưu vào 'ketqua_du_doan_woa.csv'.")
