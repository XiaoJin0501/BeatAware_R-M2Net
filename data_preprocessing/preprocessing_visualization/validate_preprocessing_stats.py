import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import sys

# 关键：确保能找到父目录中的 config 和 src
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from config import Config
from src import radar_dsp, ecg_dsp, utils

def calculate_metrics(radar, ecg, fs):
    # 提取心跳频段用于计算相关性 (0.8-3.0Hz)
    nyquist = 0.5 * fs
    b, a = signal.butter(4, [0.8 / nyquist, 3.0 / nyquist], btype='band')
    r_h = signal.filtfilt(b, a, radar)
    e_h = signal.filtfilt(b, a, ecg)
    
    pcc_raw = np.corrcoef(r_h, e_h)[0, 1]
    
    # 执行对齐寻找最佳滞后
    max_lag = int(1.5 * fs)
    correlation = signal.correlate(r_h, e_h, mode='full')
    lags = signal.correlation_lags(len(r_h), len(e_h), mode='full')
    mask = (lags >= -max_lag) & (lags <= max_lag)
    best_lag = lags[mask][np.argmax(correlation[mask])]
    
    # 计算对齐后的 PCC
    if best_lag > 0:
        pcc_aligned = np.corrcoef(r_h[best_lag:], e_h[:len(r_h)-best_lag])[0, 1]
    elif best_lag < 0:
        pcc_aligned = np.corrcoef(r_h[:len(r_h)-abs(best_lag)], e_h[abs(best_lag):])[0, 1]
    else:
        pcc_aligned = pcc_raw
        
    return pcc_raw, pcc_aligned, best_lag

def main():
    search_pattern = os.path.join(Config.RAW_DATA_DIR, "**", "*Resting.mat")
    raw_files = glob.glob(search_pattern, recursive=True)
    stats = []

    print("📊 正在量化全量数据的预处理性能...")
    for fpath in tqdm(raw_files[:50]): # 建议先抽样50个受试者
        data = utils.load_mat_file(fpath)
        if data is None: continue
        
        # 信号处理 (调用 dsp 模块)
        radar = radar_dsp.process_radar_signal(data['radar_i'].flatten(), data['radar_q'].flatten(), 
                                             Config.FS_RADAR_RAW, Config.FS_TARGET, Config.RADAR_BANDPASS)
        ecg = ecg_dsp.process_ecg_signal(data['tfm_ecg1'].flatten(), Config.FS_ECG_RAW, 
                                        Config.FS_TARGET, Config.ECG_BANDPASS)
        
        min_len = min(len(radar), len(ecg))
        p_raw, p_aligned, lag = calculate_metrics(radar[:min_len], ecg[:min_len], Config.FS_TARGET)
        
        stats.append({
            'file': os.path.basename(fpath),
            'pcc_raw': p_raw,
            'pcc_aligned': p_aligned,
            'lag_ms': (lag / Config.FS_TARGET) * 1000
        })

    df = pd.DataFrame(stats)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    df[['pcc_raw', 'pcc_aligned']].boxplot()
    plt.title("Correlation Coefficient (PCC) Improvement")
    
    plt.subplot(1, 2, 2)
    plt.hist(df['lag_ms'], bins=20, color='skyblue', edgecolor='black')
    plt.axvline(df['lag_ms'].mean(), color='red', linestyle='dashed', label=f"Mean: {df['lag_ms'].mean():.1f}ms")
    plt.title("Physiological Lag (PEP) Distribution")
    plt.legend()

    plt.tight_layout()
    output_img = os.path.join(CURRENT_DIR, 'preprocessing_quantification.png')
    plt.savefig(output_img)
    print(f"\n✅ 量化统计完成！图表已保存至: {output_img}")

if __name__ == "__main__":
    main()