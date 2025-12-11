# data_preprocessing/debug_view.py
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# 路径感知
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src import utils, ecg_dsp, radar_dsp

def debug_subject(file_path):
    print(f"--- Debugging: {os.path.basename(file_path)} ---")
    data = utils.load_mat_file(file_path)
    
    # 1. 检查数据加载
    if data is None:
        print("Error: Load failed.")
        return
        
    try:
        raw_ecg = data['tfm_ecg1'].flatten()
        print(f"Loaded Raw ECG: shape={raw_ecg.shape}, min={raw_ecg.min():.4f}, max={raw_ecg.max():.4f}")
    except KeyError:
        print("Key 'tfm_ecg1' not found!")
        return

    # 2. 检查 ECG 预处理
    ecg_clean = ecg_dsp.process_ecg_signal(
        raw_ecg, Config.FS_ECG_RAW, Config.FS_TARGET, Config.ECG_BANDPASS
    )
    print(f"Clean ECG: shape={ecg_clean.shape}, min={ecg_clean.min():.4f}, max={ecg_clean.max():.4f}")

    # 3. 检查 R 峰检测 (这是最容易出错的地方)
    # 复用 src/ecg_dsp.py 里的逻辑
    distance = int(0.3 * Config.FS_TARGET)
    # 打印一下计算出的 prominence 阈值
    prominence_threshold = 0.4 * (np.max(ecg_clean) - np.min(ecg_clean))
    print(f"Peak Detection Params: distance={distance}, prominence={prominence_threshold:.4f}")
    
    r_peaks, _ = find_peaks(ecg_clean, distance=distance, prominence=prominence_threshold)
    print(f"Found {len(r_peaks)} R-peaks.")
    
    if len(r_peaks) < 2:
        print("!!! FAIL: Less than 2 peaks found.")
    else:
        # 计算一下瞬时心率看看是否正常
        rr_intervals = np.diff(r_peaks) / Config.FS_TARGET
        hr = 60.0 / rr_intervals
        print(f"Heart Rate (BPM): mean={np.mean(hr):.2f}, min={np.min(hr):.2f}, max={np.max(hr):.2f}")

    # 4. 画图 (只画前 10 秒)
    plt.figure(figsize=(12, 6))
    
    # 画原始 vs 处理后
    limit = 10 * Config.FS_TARGET # 10秒
    t = np.arange(limit) / Config.FS_TARGET
    
    plt.plot(t, ecg_clean[:limit], label='Clean ECG (200Hz)')
    
    # 画 R 峰
    valid_peaks = [p for p in r_peaks if p < limit]
    plt.plot(np.array(valid_peaks)/Config.FS_TARGET, ecg_clean[valid_peaks], "x", color='r', label='Detected R-peaks')
    
    plt.title(f"Debug View: {os.path.basename(file_path)}\nFound {len(r_peaks)} peaks in total")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 找一个被 Rejected 的文件来测试 (例如 GDN0001)
    # 请修改下面的路径为你本地的实际路径
    test_file = os.path.join(Config.RAW_DATA_DIR, "GDN0001", "GDN0001_1_Resting.mat")
    
    # 如果你的文件结构没有子文件夹，直接用:
    # test_file = os.path.join(Config.RAW_DATA_DIR, "GDN0001_1_Resting.mat")
    
    # 自动搜索第一个文件来测试
    import glob
    files = glob.glob(os.path.join(Config.RAW_DATA_DIR, "**", "*Resting.mat"), recursive=True)
    if files:
        debug_subject(files[0]) # 测试第一个文件
    else:
        print("No files found to debug.")