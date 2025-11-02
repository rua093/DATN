import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
import os

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

if __name__ == "__main__":

    df = pd.read_csv("data/dataset_clean.csv")
    os.makedirs("results", exist_ok=True)
    X_scaled, y_scaled, scaler_X, scaler_y = preprocess_for_lstm(df)

    timesteps = 24
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, timesteps)
    print("Shape X_seq:", X_seq.shape)
    print("Shape y_seq:", y_seq.shape)

    total_size = len(X_seq)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)

    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    X_val = X_seq[train_size:train_size+val_size]
    y_val = y_seq[train_size:train_size+val_size]
    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[es]
    )

    model.save("models/my_lstm_model_optimized.h5")
    print("Model đã được lưu thành công!")

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_orig, y_pred)
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_orig, y_pred)

    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Lưu biểu đồ: Actual vs Predicted (200 điểm đầu)
    plt.figure(figsize=(12,6))
    plt.plot(y_test_orig[:200], label='Thực tế')
    plt.plot(y_pred[:200], label='Dự đoán')
    plt.title("So sánh giá trị thực tế và dự đoán (200 điểm đầu)")
    plt.xlabel("Thời điểm")
    plt.ylabel("ENERGY")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/train_actual_vs_pred_200.png", dpi=200)
    plt.close()

    # Lưu biểu đồ: Training/Validation Loss
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/train_history_loss.png", dpi=200)
    plt.close()

    # Lưu biểu đồ: Scatter Plot Actual vs Predicted
    plt.figure(figsize=(6,6))
    plt.scatter(y_test_orig, y_pred, alpha=0.6)
    y_min, y_max = y_test_orig.min(), y_test_orig.max()
    plt.plot([y_min, y_max], [y_min, y_max], 'r--')
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title('Scatter: Thực tế vs Dự đoán')
    plt.tight_layout()
    plt.savefig("results/train_scatter_actual_vs_pred.png", dpi=200)
    plt.close()

    # Lưu biểu đồ: Phân bố sai số (Residuals)
    residuals = y_test_orig.flatten() - y_pred.flatten()
    plt.figure(figsize=(8,4))
    plt.hist(residuals, bins=50, alpha=0.8, edgecolor='black')
    plt.title('Phân bố Sai số Dự đoán')
    plt.xlabel('Sai số')
    plt.ylabel('Tần suất')
    plt.tight_layout()
    plt.savefig("results/train_residual_hist.png", dpi=200)
    plt.close()

    df_result = pd.DataFrame({
        "ThucTe": y_test_orig.flatten(),
        "DuDoan": y_pred.flatten()
    })
    df_result.to_csv("data/ketqua_du_doan_optimized.csv", index=False)
    print("Kết quả dự đoán đã lưu vào 'ketqua_du_doan_optimized.csv'.")
