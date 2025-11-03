#!/usr/bin/env python3
"""
Script so sánh các mô hình LSTM khác nhau
"""

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

def load_and_compare_models():
    """So sánh tất cả các mô hình đã train"""
    
    print("🔄 Đang tải và so sánh các mô hình...")
    
    # Đọc dữ liệu
    df = pd.read_csv("data/dataset_clean.csv")
    
    # Tiền xử lý (giống như trong training)
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
    X = data.drop(columns=[target_col]).values
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Tạo sequences
    def create_sequences(X, y, timesteps=24):
        Xs, ys = [], []
        for i in range(len(X) - timesteps):
            Xs.append(X[i:i+timesteps])
            ys.append(y[i+timesteps])
        return np.array(Xs), np.array(ys)
    
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, 24)
    
    # Chia dữ liệu test
    total_size = len(X_seq)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    
    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]
    y_test_orig = scaler_y.inverse_transform(y_test)
    
    print(f"✅ Dữ liệu test: {X_test.shape[0]} mẫu")
    
    # Danh sách các mô hình để so sánh
    models_to_compare = [
        {
            'name': 'LSTM Cơ bản',
            'path': 'models/my_lstm_model_optimized.h5',
            'color': 'blue'
        },
        {
            'name': 'LSTM + WOA',
            'path': 'models/my_lstm_model_woa.h5',
            'color': 'orange'
        },
        {
            'name': 'Advanced LSTM',
            'path': 'models/advanced_lstm_model.h5',
            'color': 'green'
        }
    ]
    
    results = []
    
    for model_info in models_to_compare:
        try:
            print(f"🔄 Đang đánh giá {model_info['name']}...")
            
            # Load model
            model = load_model(model_info['path'], compile=False)
            
            # Dự đoán
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
            # Tính metrics
            mae = mean_absolute_error(y_test_orig, y_pred)
            mse = mean_squared_error(y_test_orig, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_orig, y_pred)
            mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100
            
            results.append({
                'Model': model_info['name'],
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape,
                'Color': model_info['color'],
                'Predictions': y_pred
            })
            
            print(f"   ✅ MAE: {mae:.4f}, R²: {r2:.4f}")
            
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
    
    return results, y_test_orig

def create_comparison_plots(results, y_test_orig):
    """Tạo biểu đồ so sánh"""
    
    print("📊 Đang tạo biểu đồ so sánh...")
    
    # Tạo DataFrame cho kết quả
    df_results = pd.DataFrame([{
        'Model': r['Model'],
        'MAE': r['MAE'],
        'MSE': r['MSE'],
        'RMSE': r['RMSE'],
        'R²': r['R²'],
        'MAPE': r['MAPE']
    } for r in results])
    
    # Tạo subplot
    os.makedirs("../results", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('So sánh Hiệu suất các Mô hình LSTM', fontsize=16, fontweight='bold')
    
    # 1. Bar chart metrics
    metrics = ['MAE', 'RMSE', 'R²', 'MAPE']
    colors = ['blue', 'orange', 'green']
    
    for i, metric in enumerate(metrics):
        ax = axes[0, i] if i < 2 else axes[1, i-2]
        
        bars = ax.bar(df_results['Model'], df_results[metric], 
                     color=colors[:len(df_results)], alpha=0.7)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        # Thêm giá trị lên bars
        for bar, value in zip(bars, df_results[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
    
    # 2. Prediction comparison (200 điểm đầu)
    ax = axes[1, 2]
    ax.plot(y_test_orig[:200], label='Thực tế', linewidth=2, color='black')
    
    for i, result in enumerate(results):
        ax.plot(result['Predictions'][:200], 
               label=result['Model'], 
               linewidth=1.5, 
               color=result['Color'],
               alpha=0.8)
    
    ax.set_title('So sánh Dự đoán (200 điểm đầu)')
    ax.set_xlabel('Thời điểm')
    ax.set_ylabel('Năng lượng tiêu thụ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/compare_metrics_and_predictions.png", dpi=200)
    plt.close()
    
    # 3. Scatter plots cho từng mô hình
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for i, result in enumerate(results):
        ax = axes[i]
        ax.scatter(y_test_orig, result['Predictions'], alpha=0.6, color=result['Color'])
        ax.plot([y_test_orig.min(), y_test_orig.max()], 
               [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
        ax.set_xlabel('Giá trị thực tế')
        ax.set_ylabel('Giá trị dự đoán')
        ax.set_title(f'{result["Model"]}\nR² = {result["R²"]:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/compare_scatter_each_model.png", dpi=200)
    plt.close()

def print_detailed_comparison(results):
    """In báo cáo so sánh chi tiết"""
    
    print("\n" + "="*80)
    print("                    BÁO CÁO SO SÁNH MÔ HÌNH CHI TIẾT")
    print("="*80)
    
    # Tạo bảng so sánh
    df_results = pd.DataFrame([{
        'Model': r['Model'],
        'MAE': r['MAE'],
        'MSE': r['MSE'],
        'RMSE': r['RMSE'],
        'R²': r['R²'],
        'MAPE': r['MAPE']
    } for r in results])
    
    print("\n📊 BẢNG SO SÁNH METRICS:")
    print("-" * 80)
    print(df_results.to_string(index=False, float_format='%.4f'))
    
    # Tìm mô hình tốt nhất cho từng metric
    print(f"\n🏆 MÔ HÌNH TỐT NHẤT CHO TỪNG METRIC:")
    print("-" * 50)
    
    for metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
        if metric == 'R²':
            best_idx = df_results[metric].idxmax()
        else:
            best_idx = df_results[metric].idxmin()
        best_model = df_results.loc[best_idx, 'Model']
        best_value = df_results.loc[best_idx, metric]
        print(f"   • {metric}: {best_model} ({best_value:.4f})")
    
    # Phân tích tổng quan
    print(f"\n📈 PHÂN TÍCH TỔNG QUAN:")
    print("-" * 30)
    
    # Tính điểm tổng hợp (lower is better cho MAE, MSE, RMSE, MAPE; higher is better cho R²)
    df_results['Score'] = (
        -df_results['MAE']/df_results['MAE'].max() +  # Normalize và đảo ngược
        -df_results['MSE']/df_results['MSE'].max() +
        -df_results['RMSE']/df_results['RMSE'].max() +
        -df_results['MAPE']/df_results['MAPE'].max() +
        df_results['R²']/df_results['R²'].max()
    )
    
    best_overall_idx = df_results['Score'].idxmax()
    best_overall = df_results.loc[best_overall_idx, 'Model']
    
    print(f"   • Mô hình tổng thể tốt nhất: {best_overall}")
    print(f"   • Điểm tổng hợp: {df_results.loc[best_overall_idx, 'Score']:.3f}")
    
    # Khuyến nghị
    print(f"\n💡 KHUYẾN NGHỊ:")
    print("-" * 20)
    print(f"   • Sử dụng {best_overall} cho dự án")
    print(f"   • Cân nhắc ensemble nếu cần độ chính xác cao")
    print(f"   • Thử thêm features nếu muốn cải thiện hơn")

def save_comparison_results(results):
    """Lưu kết quả so sánh"""
    
    # Tạo DataFrame
    df_results = pd.DataFrame([{
        'Model': r['Model'],
        'MAE': r['MAE'],
        'MSE': r['MSE'],
        'RMSE': r['RMSE'],
        'R²': r['R²'],
        'MAPE': r['MAPE']
    } for r in results])
    
    # Lưu CSV
    df_results.to_csv("data/model_comparison_results.csv", index=False)
    print("💾 Kết quả đã lưu: data/model_comparison_results.csv")
    
    # Lưu JSON cho web
    import json
    comparison_data = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'models': df_results.to_dict('records')
    }
    
    with open("data/model_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    print("💾 JSON đã lưu: data/model_comparison_results.json")

def main():
    """Hàm chính"""
    print("🔄 BẮT ĐẦU SO SÁNH MÔ HÌNH")
    print("=" * 50)
    
    # Load và so sánh mô hình
    results, y_test_orig = load_and_compare_models()
    
    if not results:
        print("❌ Không có mô hình nào để so sánh")
        return
    
    # Tạo biểu đồ
    create_comparison_plots(results, y_test_orig)
    
    # In báo cáo chi tiết
    print_detailed_comparison(results)
    
    # Lưu kết quả
    save_comparison_results(results)
    
    print(f"\n🎉 HOÀN THÀNH SO SÁNH!")
    print("=" * 30)
    print(f"📊 Đã so sánh {len(results)} mô hình")
    print(f"📁 Kết quả đã lưu trong data/")

if __name__ == "__main__":
    main()
