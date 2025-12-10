import numpy as np
from scipy.signal import butter, filtfilt, resample_poly, find_peaks
from scipy.stats import norm

def process_ecg_signal(ecg_raw, fs_raw, fs_target, bandpass_freqs):
    """ECG 滤波与降采样"""
    # 1. 降采样
    gcd_fs = np.gcd(fs_raw, fs_target)
    up = fs_target // gcd_fs
    down = fs_raw // gcd_fs
    
    if up == 1 and down == 1:
        ecg_res = ecg_raw
    else:
        ecg_res = resample_poly(ecg_raw, up, down)
        
    # 2. 带通滤波
    nyquist = 0.5 * fs_target
    low = bandpass_freqs[0] / nyquist
    high = bandpass_freqs[1] / nyquist
    b, a = butter(4, [low, high], btype='band')
    ecg_clean = filtfilt(b, a, ecg_res)
    
    return ecg_clean

def generate_anchor_mask(ecg_signal, fs, sigma_points=5):
    """
    生成 Beat-Aware Anchor Mask (高斯热图)
    :param ecg_signal: 200Hz 的 ECG 信号
    :param sigma_points: 高斯核的标准差 (默认5点 -> 25ms)
    :return: mask (与 ecg 等长), r_peaks_indices
    """
    # 1. R峰检测 (基于 Scipy, 简单鲁棒)
    # 距离限制: 假设心率 < 200bpm, 也就是间隔 > 0.3s (60点)
    distance = int(0.3 * fs)
    prominence = 0.4 * (np.max(ecg_signal) - np.min(ecg_signal)) # 动态阈值
    
    r_peaks, _ = find_peaks(ecg_signal, distance=distance, prominence=prominence)
    
    # 2. 生成高斯 Mask
    mask = np.zeros_like(ecg_signal)
    x = np.arange(-3*sigma_points, 3*sigma_points + 1)
    gaussian_kernel = np.exp(-(x**2) / (2 * sigma_points**2))
    # 归一化到 [0, 1]
    gaussian_kernel = gaussian_kernel / np.max(gaussian_kernel) 
    
    for r in r_peaks:
        # 边界处理
        start_k = max(0, - (r - 3*sigma_points))
        end_k = min(len(gaussian_kernel), len(ecg_signal) - (r - 3*sigma_points))
        
        start_m = max(0, r - 3*sigma_points)
        end_m = min(len(ecg_signal), r + 3*sigma_points + 1)
        
        # 叠加 (防止重叠处的数值爆炸，取最大值)
        mask[start_m:end_m] = np.maximum(mask[start_m:end_m], gaussian_kernel[start_k:end_k])
        
    return mask, r_peaks