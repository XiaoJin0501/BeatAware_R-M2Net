import numpy as np
from scipy.signal import butter, filtfilt, resample_poly, find_peaks
from scipy.stats import norm

def process_ecg_signal(ecg_raw, fs_raw, fs_target, bandpass_freqs):
    """ECG 滤波与降采样：加入极性修正与稳健归一化"""
    # 1. 降采样 resample_poly 自带抗混叠滤波
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
    
    # 3. [核心微调] 极性自动纠正
    # 如果向下脉冲的绝对值大于向上脉冲，说明信号反转了
    if np.abs(np.max(ecg_clean)) < np.abs(np.min(ecg_clean)):
        ecg_clean = -ecg_clean
    
    # 4. [新增] 局部 Z-Score 归一化，使信号处于标准量级，提升寻峰成功率
    ecg_clean = (ecg_clean - np.mean(ecg_clean)) / (np.std(ecg_clean) + 1e-8)
    
    return ecg_clean

def generate_anchor_mask(ecg_signal, fs, sigma_points=5):
    """
    生成 Beat-Aware Anchor Mask (高斯热图)
    :param ecg_signal: 200Hz 的 ECG 信号
    :param sigma_points: 高斯核的标准差 (默认5点 -> 25ms)
    :return: mask (与 ecg 等长), r_peaks_indices
    """
    # 1. R峰检测 (基于 Scipy, 简单鲁棒)
    # 距离限制: 假设心率 < 200bpm, 也就是间隔 > 0.3s
    # fs = 200Hz 时, distance = 60 点
    distance = int(0.3 * fs)
    q_high, q_low = np.percentile(ecg_signal, [99, 1])
    # [修改点] 降低阈值系数到 0.5 (50%)
    prominence = 0.5 * (q_high - q_low)
    # 防止信号是直线导致 prominence 为 0
    if prominence < 1e-6:
        prominence = 0.1 # 给个默认值
    r_peaks, _ = find_peaks(ecg_signal, distance=distance, prominence=prominence)
    
    # 2. 生成高斯 Mask
    mask = np.zeros_like(ecg_signal)
    # 定义核的半径
    radius = 4 * sigma_points
    x = np.arange(-radius, radius + 1)
    
    gaussian_kernel = np.exp(-(x**2) / (2 * sigma_points**2))
    gaussian_kernel = gaussian_kernel / np.max(gaussian_kernel) # 归一化到最大值为1
    
    for r in r_peaks:
        # 计算在 mask 中的起止位置
        m_start = max(0, r - radius)
        m_end = min(len(ecg_signal), r + radius + 1)
        
        # 对应在 gaussian_kernel 中的范围
        # 如果 r-radius < 0，说明左边越界，核也要从后面截取
        k_start = radius - (r - m_start)
        # 长度必须保持一致
        k_end = k_start + (m_end - m_start)
        
        # 执行叠加
        if (m_end - m_start) > 0:
            mask[m_start:m_end] = np.maximum(mask[m_start:m_end], gaussian_kernel[k_start:k_end])
    
    return mask, r_peaks