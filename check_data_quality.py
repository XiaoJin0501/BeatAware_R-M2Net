import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal

# ================= 配置区域 =================
# 指向您刚刚生成的 H5 文件路径 (确保路径正确)
H5_PATH = 'data_preprocessing/processed_to_h5/experiment_A_SubjectIndependent/train.h5'
# 想要检查的样本数量
NUM_SAMPLES = 5 
# ===========================================

def compute_lag(x, y):
    """计算两个信号的相位差 (Lag)"""
    correlation = signal.correlate(x, y, mode='full')
    lags = signal.correlation_lags(len(x), len(y), mode='full')
    lag = lags[np.argmax(correlation)]
    return lag

def inspect_data():
    if not os.path.exists(H5_PATH):
        print(f"❌ Error: File not found at {H5_PATH}")
        return

    print(f"🔍 Loading data from {H5_PATH}...")
    with h5py.File(H5_PATH, 'r') as f:
        # 随机抽取几个样本，或者固定取前几个
        total_len = len(f['radar'])
        indices = np.linspace(0, total_len-1, NUM_SAMPLES, dtype=int)
        
        radars = f['radar'][indices]
        ecgs = f['ecg'][indices]
        masks = f['mask'][indices]

    print(f"✅ Data Loaded. Shape: {radars.shape}")
    
    # 开始绘图
    fig, axes = plt.subplots(NUM_SAMPLES, 2, figsize=(15, 4 * NUM_SAMPLES))
    if NUM_SAMPLES == 1: axes = [axes] # 兼容单样本情况

    for i, idx in enumerate(indices):
        radar = radars[i, 0] # [1600]
        ecg = ecgs[i, 0]     # [1600]
        mask = masks[i, 0]   # [1600]
        
        # --- 1. 左图: 雷达 vs ECG (检查滤波和物理对齐) ---
        ax_left = axes[i][0]
        
        # 为了画图好看，把 ECG 也标准化到 Z-Score (和 Radar 同尺度对比)
        r_plot = (radar - np.mean(radar)) / (np.std(radar) + 1e-6)
        e_plot = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-6)
        
        # 计算当前样本的滞后
        lag = compute_lag(r_plot, e_plot)
        
        ax_left.plot(r_plot, label='Radar (Input)', alpha=0.8, linewidth=1)
        ax_left.plot(e_plot, label='ECG (Target)', alpha=0.6, linewidth=1, linestyle='--')
        ax_left.set_title(f"Sample {idx} | Lag: {lag} pts (Should be small/stable)")
        ax_left.legend(loc='upper right')
        ax_left.grid(True, alpha=0.3)
        
        # --- 2. 右图: ECG vs Mask (检查 Ground Truth 生成) ---
        ax_right = axes[i][1]
        ax_right.plot(ecg, label='ECG (Norm [0,1])', color='red')
        # Mask 乘以一个系数稍微放大一点，方便看
        ax_right.plot(mask, label='Anchor Mask', color='green', linestyle='-', linewidth=1.5)
        ax_right.fill_between(range(len(mask)), mask, color='green', alpha=0.1) # 填充颜色
        ax_right.set_title(f"Sample {idx} | Mask Alignment Check")
        ax_right.legend()
        ax_right.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    save_path = 'data_quality_report.png'
    plt.savefig(save_path, dpi=150)
    print(f"\n🎉 Diagnosis complete! Please open '{save_path}' to inspect.")

if __name__ == "__main__":
    inspect_data()