import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import warnings
import os
warnings.filterwarnings('ignore')

# Thiết lập font cho tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'

def preprocess_for_lstm(df, include_target_in_X=False):
    """Tiền xử lý dữ liệu.
    include_target_in_X: nếu True, giữ cả cột mục tiêu trong X (AR inputs) để
    phù hợp với mô hình được train theo kiểu này.
    """
    if "DATE" in df.columns:
        df = df.drop(columns=["DATE"])
    
    data = df.copy()
    
    # Cyclic encoding cho các biến thời gian
    data["DAY_sin"] = np.sin(2 * np.pi * data["DAY"] / 31)
    data["DAY_cos"] = np.cos(2 * np.pi * data["DAY"] / 31)
    data["MONTH_sin"] = np.sin(2 * np.pi * data["MONTH"] / 12)
    data["MONTH_cos"] = np.cos(2 * np.pi * data["MONTH"] / 12)
    data["WEEKDAY_sin"] = np.sin(2 * np.pi * data["WEEKDAY"] / 7)
    data["WEEKDAY_cos"] = np.cos(2 * np.pi * data["WEEKDAY"] / 7)
    data = data.drop(columns=["DAY", "MONTH", "WEEKDAY"])
    
    target_col = "ENERGY_ADJ" if "ENERGY_ADJ" in data.columns else "ENERGY"
    y = data[[target_col]].values
    if include_target_in_X:
        X = data.values
    else:
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

def calculate_metrics(y_true, y_pred):
    """Tính toán các metrics đánh giá"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    }

def plot_comparison(y_true, y_pred_basic, y_pred_woa, title="So sánh dự đoán"):
    """Vẽ đồ thị so sánh các mô hình"""
    os.makedirs("../results", exist_ok=True)
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: So sánh tổng thể
    plt.subplot(2, 2, 1)
    plt.plot(y_true[:200], label='Thực tế', linewidth=2, alpha=0.8)
    plt.plot(y_pred_basic[:200], label='LSTM Cơ bản', linewidth=1.5, alpha=0.8)
    plt.plot(y_pred_woa[:200], label='LSTM + WOA', linewidth=1.5, alpha=0.8)
    plt.title('So sánh dự đoán (200 điểm đầu)', fontsize=14, fontweight='bold')
    plt.xlabel('Thời điểm')
    plt.ylabel('Năng lượng tiêu thụ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Scatter plot - LSTM Cơ bản
    plt.subplot(2, 2, 2)
    plt.scatter(y_true, y_pred_basic, alpha=0.6, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title('LSTM Cơ bản - Scatter Plot')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Scatter plot - LSTM + WOA
    plt.subplot(2, 2, 3)
    plt.scatter(y_true, y_pred_woa, alpha=0.6, s=20, color='orange')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title('LSTM + WOA - Scatter Plot')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Histogram của sai số
    plt.subplot(2, 2, 4)
    errors_basic = y_true.flatten() - y_pred_basic.flatten()
    errors_woa = y_true.flatten() - y_pred_woa.flatten()
    
    plt.hist(errors_basic, bins=50, alpha=0.7, label='LSTM Cơ bản', density=True)
    plt.hist(errors_woa, bins=50, alpha=0.7, label='LSTM + WOA', density=True)
    plt.xlabel('Sai số dự đoán')
    plt.ylabel('Mật độ')
    plt.title('Phân bố sai số dự đoán')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/compare_time_scatter_error.png", dpi=200)
    plt.close()

def plot_metrics_comparison(metrics_basic, metrics_woa):
    """Vẽ biểu đồ so sánh metrics"""
    metrics_names = list(metrics_basic.keys())
    basic_values = list(metrics_basic.values())
    woa_values = list(metrics_woa.values())
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, basic_values, width, label='LSTM Cơ bản', alpha=0.8)
    bars2 = ax.bar(x + width/2, woa_values, width, label='LSTM + WOA', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Giá trị')
    ax.set_title('So sánh hiệu suất các mô hình', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Thêm giá trị lên các cột
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("results/compare_metrics_bar.png", dpi=200)
    plt.close()

def print_detailed_evaluation(metrics_basic, metrics_woa):
    """In báo cáo đánh giá chi tiết"""
    print("="*80)
    print("                    ĐÁNH GIÁ MÔ HÌNH LSTM SAU KHI ÁP DỤNG WOA")
    print("="*80)
    
    print("\n📊 BẢNG SO SÁNH METRICS:")
    print("-" * 60)
    print(f"{'Metric':<15} {'LSTM Cơ bản':<15} {'LSTM + WOA':<15} {'Cải thiện':<15}")
    print("-" * 60)
    
    for metric in metrics_basic.keys():
        basic_val = metrics_basic[metric]
        woa_val = metrics_woa[metric]
        
        if metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:  # Càng thấp càng tốt
            improvement = ((basic_val - woa_val) / basic_val) * 100
            status = "✅ Tốt hơn" if improvement > 0 else "❌ Kém hơn"
        else:  # R² - càng cao càng tốt
            improvement = ((woa_val - basic_val) / basic_val) * 100
            status = "✅ Tốt hơn" if improvement > 0 else "❌ Kém hơn"
        
        print(f"{metric:<15} {basic_val:<15.4f} {woa_val:<15.4f} {improvement:>+10.2f}% {status}")
    
    print("\n🎯 PHÂN TÍCH HIỆU SUẤT:")
    print("-" * 40)
    
    # Phân tích từng metric
    mae_improvement = ((metrics_basic['MAE'] - metrics_woa['MAE']) / metrics_basic['MAE']) * 100
    rmse_improvement = ((metrics_basic['RMSE'] - metrics_woa['RMSE']) / metrics_basic['RMSE']) * 100
    r2_improvement = ((metrics_woa['R²'] - metrics_basic['R²']) / metrics_basic['R²']) * 100
    
    print(f"• MAE: {'Cải thiện' if mae_improvement > 0 else 'Giảm'} {abs(mae_improvement):.2f}%")
    print(f"• RMSE: {'Cải thiện' if rmse_improvement > 0 else 'Giảm'} {abs(rmse_improvement):.2f}%")
    print(f"• R²: {'Cải thiện' if r2_improvement > 0 else 'Giảm'} {abs(r2_improvement):.2f}%")
    
    # Đánh giá tổng thể
    print(f"\n🏆 KẾT LUẬN:")
    print("-" * 20)
    if mae_improvement > 0 and rmse_improvement > 0 and r2_improvement > 0:
        print("✅ Thuật toán WOA đã cải thiện hiệu suất mô hình LSTM")
        print("✅ Mô hình tối ưu hóa cho kết quả dự đoán tốt hơn")
    elif mae_improvement > 0 or rmse_improvement > 0 or r2_improvement > 0:
        print("⚠️  Thuật toán WOA cải thiện một số metrics nhưng không toàn diện")
    else:
        print("❌ Thuật toán WOA không cải thiện hiệu suất mô hình")
        print("❌ Cần xem xét lại tham số tối ưu hóa")

def main():
    """Hàm chính để đánh giá mô hình"""
    print("🔄 Đang tải dữ liệu và chuẩn bị đánh giá...")
    
    # 1. Load dữ liệu gốc (chưa scale)
    df = pd.read_csv("data/dataset_clean.csv")
    
    # 2. Load các mô hình
    import os
    
    # Kiểm tra các file model có sẵn
    model_files = [f for f in os.listdir('models/') if f.endswith('.h5')]
    print(f"📁 Các file model có sẵn: {model_files}")
    
    # Tìm mô hình cơ bản
    basic_model_path = None
    for model_file in model_files:
        if 'optimized' in model_file.lower() or 'basic' in model_file.lower():
            basic_model_path = model_file
            break
    
    if basic_model_path:
        try:
            # Thử load với custom_objects để xử lý lỗi version
            model_basic = load_model(f"models/{basic_model_path}", compile=False)
            print(f"✅ Đã tải mô hình LSTM cơ bản: {basic_model_path}")
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình cơ bản: {e}")
            print("💡 Thử load model với compile=False...")
            try:
                model_basic = load_model(f"models/{basic_model_path}", compile=False)
                print(f"✅ Đã tải mô hình LSTM cơ bản (compile=False): {basic_model_path}")
            except Exception as e2:
                print(f"❌ Vẫn không thể load model: {e2}")
                return
    else:
        print("❌ Không tìm thấy mô hình LSTM cơ bản")
        print("💡 Hãy chạy train_lstm.py trước để tạo mô hình cơ bản")
        return
    
    # Tìm mô hình WOA
    woa_model_path = None
    for model_file in model_files:
        if 'woa' in model_file.lower():
            woa_model_path = model_file
            break
    
    if woa_model_path:
        try:
            model_woa = load_model(f"models/{woa_model_path}", compile=False)
            print(f"✅ Đã tải mô hình LSTM + WOA: {woa_model_path}")
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình WOA: {e}")
            print("💡 Thử load model với compile=False...")
            try:
                model_woa = load_model(f"models/{woa_model_path}", compile=False)
                print(f"✅ Đã tải mô hình LSTM + WOA (compile=False): {woa_model_path}")
            except Exception as e2:
                print(f"❌ Vẫn không thể load model WOA: {e2}")
                return
    else:
        print("❌ Không tìm thấy mô hình LSTM + WOA")
        print("💡 Hãy chạy lstm_woa_optimize.py trước để tạo mô hình WOA")
        return
    
    # 3. Chuẩn bị dữ liệu test theo đúng cấu hình từng mô hình
    #    - Suy ra timesteps và số đặc trưng từ input_shape
    t_basic = model_basic.input_shape[1]
    f_basic = model_basic.input_shape[2]
    t_woa   = model_woa.input_shape[1]
    f_woa   = model_woa.input_shape[2]

    # Dựng pipeline riêng cho từng mô hình để tránh lệch số đặc trưng
    # a) Cho mô hình cơ bản (không giữ target trong X nếu số đặc trưng khớp như vậy)
    keep_target_basic = False
    # Thử 2 khả năng: nếu số cột khi có target khớp thì đặt True
    X_all_keep, y_all, scaler_X_keep, scaler_y_keep = preprocess_for_lstm(df, include_target_in_X=True)
    X_all_drop, _,      scaler_X_drop, _            = preprocess_for_lstm(df, include_target_in_X=False)
    if X_all_keep.shape[1] == f_basic:
        keep_target_basic = True
        Xb_scaled, yb_scaled, sXb, sY = X_all_keep, y_all, scaler_X_keep, scaler_y_keep
    else:
        Xb_scaled, yb_scaled, sXb, sY = X_all_drop, y_all, scaler_X_drop, scaler_y_keep

    Xb_seq, yb_seq = create_sequences(Xb_scaled, yb_scaled, t_basic)
    total_b = len(Xb_seq)
    train_b = int(total_b * 0.7)
    val_b   = int(total_b * 0.15)
    Xb_test = Xb_seq[train_b+val_b:]
    yb_test = yb_seq[train_b+val_b:]

    # b) Cho mô hình WOA
    keep_target_woa = (X_all_keep.shape[1] == f_woa)
    if keep_target_woa:
        Xw_scaled, yw_scaled, sXw, sYw = X_all_keep, y_all, scaler_X_keep, scaler_y_keep
    else:
        Xw_scaled, yw_scaled, sXw, sYw = X_all_drop, y_all, scaler_X_drop, scaler_y_keep
    Xw_seq, yw_seq = create_sequences(Xw_scaled, yw_scaled, t_woa)
    total_w = len(Xw_seq)
    train_w = int(total_w * 0.7)
    val_w   = int(total_w * 0.15)
    Xw_test = Xw_seq[train_w+val_w:]
    yw_test = yw_seq[train_w+val_w:]

    # Lấy y_test gốc theo mô hình WOA để dùng chung khi so sánh
    y_test_orig = sYw.inverse_transform(yw_test)
    print(f"✅ Dữ liệu test: {Xw_test.shape[0]} mẫu")

    # 4. Dự đoán
    print("🔄 Đang thực hiện dự đoán...")
    y_pred_basic_scaled = model_basic.predict(Xb_test)
    y_pred_woa_scaled   = model_woa.predict(Xw_test)
    
    y_pred_basic = sY.inverse_transform(y_pred_basic_scaled)
    y_pred_woa   = sYw.inverse_transform(y_pred_woa_scaled)
    
    # 6. Tính metrics
    print("🔄 Đang tính toán metrics...")
    metrics_basic = calculate_metrics(y_test_orig, y_pred_basic)
    metrics_woa = calculate_metrics(y_test_orig, y_pred_woa)
    
    # 7. In báo cáo chi tiết
    print_detailed_evaluation(metrics_basic, metrics_woa)
    
    # 8. Vẽ đồ thị so sánh
    print("\n📈 Đang tạo biểu đồ so sánh...")
    plot_comparison(y_test_orig, y_pred_basic, y_pred_woa)
    plot_metrics_comparison(metrics_basic, metrics_woa)
    
    # 9. Lưu kết quả đánh giá
    evaluation_results = pd.DataFrame({
        'Metric': list(metrics_basic.keys()),
        'LSTM_Basic': list(metrics_basic.values()),
        'LSTM_WOA': list(metrics_woa.values())
    })
    
    evaluation_results['Improvement_%'] = [
        ((metrics_basic[metric] - metrics_woa[metric]) / metrics_basic[metric]) * 100 
        if metric in ['MAE', 'MSE', 'RMSE', 'MAPE'] 
        else ((metrics_woa[metric] - metrics_basic[metric]) / metrics_basic[metric]) * 100
        for metric in metrics_basic.keys()
    ]
    
    evaluation_results.to_csv("data/evaluation_results_woa.csv", index=False)
    print("\n💾 Kết quả đánh giá đã được lưu vào 'evaluation_results_woa.csv'")
    
    print("\n🎉 Hoàn thành đánh giá mô hình!")

if __name__ == "__main__":
    main()
