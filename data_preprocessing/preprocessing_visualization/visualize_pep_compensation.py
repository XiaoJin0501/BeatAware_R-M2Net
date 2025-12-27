import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys

# 路径处理：确保能找到父目录中的 config 和 src
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from config import Config
from src import radar_dsp, ecg_dsp, utils

def visualize_pep_compensation(file_path):
    """
    视觉对比脚本：展示 PEP 补偿前后的波形相位锁定效果
    """
    data = utils.load_mat_file(file_path)
    if data is None:
        print(f"❌ 无法读取文件: {file_path}")
        return

    # 1. 基础信号处理
    radar = radar_dsp.process_radar_signal(data['radar_i'].flatten(), data['radar_q'].flatten(), 
                                         Config.FS_RADAR_RAW, Config.FS_TARGET, Config.RADAR_BANDPASS)
    ecg = ecg_dsp.process_ecg_signal(data['tfm_ecg1'].flatten(), Config.FS_ECG_RAW, 
                                    Config.FS_TARGET, Config.ECG_BANDPASS)
    
    # 2. 截取 3 秒片段用于清晰展示 (避开开头)
    start_sec, duration = 10, 3
    s, e = int(start_sec * Config.FS_TARGET), int((start_sec + duration) * Config.FS_TARGET)
    r_seg, e_seg = radar[s:e], ecg[s:e]

    # 3. 归一化便于对比
    r_norm = (r_seg - np.mean(r_seg)) / (np.std(r_seg) + 1e-6)
    e_norm = (e_seg - np.mean(e_seg)) / (np.std(e_seg) + 1e-6)
    
    # 4. 计算最佳滞后 (Lag)
    import scipy.signal as signal
    corr = signal.correlate(r_norm, e_norm, mode='full')
    lags = signal.correlation_lags(len(r_norm), len(e_norm), mode='full')
    best_lag = lags[np.argmax(corr)]

    # 5. 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- 对齐前 (展示 PEP 延迟) ---
    ax1.plot(e_norm, color='red', label='ECG (Electrical)', alpha=0.5)
    ax1.plot(r_norm, color='blue', label='Radar (Mechanical)', alpha=0.8)
    ax1.set_title(f"Before Alignment: Systematic Lag (Detected PEP $\\approx$ {int(best_lag * (1000/Config.FS_TARGET))}ms)")
    ax1.set_ylabel("Normalized Amplitude")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # --- 对齐后 (展示相位锁定) ---
    # 使用 roll 模拟对齐效果
    r_aligned = np.roll(r_norm, -best_lag)
    ax2.plot(e_norm, color='red', label='ECG', alpha=0.5)
    ax2.plot(r_aligned, color='blue', label='Radar (Synchronized)', alpha=0.8)
    ax2.set_title("After Alignment: Phase-Locked Modalities")
    ax2.set_xlabel(f"Samples (at {Config.FS_TARGET}Hz)")
    ax2.set_ylabel("Normalized Amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    output_img = os.path.join(CURRENT_DIR, 'pep_compensation_study.png')
    plt.savefig(output_img, dpi=300)
    print(f"🎬 视觉对比图已生成: {output_img}")

if __name__ == "__main__":
    # 搜索一个受试者文件进行测试
    search_pattern = os.path.join(Config.RAW_DATA_DIR, "**", "*Resting.mat")
    files = glob.glob(search_pattern, recursive=True)
    if files:
        # 修改这里：确保函数名与上面定义的一致
        visualize_pep_compensation(files[0])
    else:
        print(f"❌ 在 {Config.RAW_DATA_DIR} 下未找到数据文件")