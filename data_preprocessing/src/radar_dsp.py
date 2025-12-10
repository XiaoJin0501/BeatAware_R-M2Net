import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
from scipy.optimize import least_squares

def fit_ellipse(i_signal, q_signal):
    """
    基于最小二乘法的直接椭圆拟合 (Direct Least Squares Ellipse Fitting)
    用于校准 I/Q 信号的幅相失配和直流偏移
    """
    X = np.array(i_signal)
    Y = np.array(q_signal)
    
    # 简化的代数拟合: 拟合圆心 (xc, yc) 和 缩放因子
    # 目标: (x-xc)^2 + (y-yc)^2 * (scale)^2 = R^2
    # 这比完整的椭圆拟合更鲁棒，适合雷达I/Q圆校准
    
    def residuals(params, x, y):
        xc, yc, scale, r = params
        return np.sqrt((x - xc)**2 + (scale * (y - yc))**2) - r

    # 初始猜测: 均值作为圆心，标准差比作为缩放
    x_m, y_m = np.mean(X), np.mean(Y)
    scale0 = np.std(X) / (np.std(Y) + 1e-6)
    r0 = np.std(X)
    
    res = least_squares(residuals, x0=[x_m, y_m, scale0, r0], args=(X, Y), loss='soft_l1')
    xc, yc, scale, _ = res.x
    
    # 校准信号
    i_corr = X - xc
    q_corr = (Y - yc) * scale
    
    return i_corr, q_corr

def process_radar_signal(radar_i, radar_q, fs_raw, fs_target, bandpass_freqs):
    """
    完整的雷达信号处理流水线
    1. 椭圆拟合校准
    2. 相位解调 (Arctan + Unwrap)
    3. 降采样
    4. 带通滤波
    """
    # 1. 椭圆拟合校准
    i_calib, q_calib = fit_ellipse(radar_i, radar_q)
    
    # 2. 相位解调
    phase = np.unwrap(np.arctan2(q_calib, i_calib))
    
    # 3. 降采样 (防止混叠，使用 polyphase 重采样)
    # 计算最大公约数以获得整数倍率
    gcd_fs = np.gcd(fs_raw, fs_target)
    up = fs_target // gcd_fs
    down = fs_raw // gcd_fs
    
    if up == 1 and down == 1:
        phase_resampled = phase
    else:
        phase_resampled = resample_poly(phase, up, down)
        
    # 4. 带通滤波 (提取胸部机械位移)
    # 注意: 差分(Diff) 实际上是一个高通滤波器，这里直接用带通代替，保留位移波形
    nyquist = 0.5 * fs_target
    low = bandpass_freqs[0] / nyquist
    high = bandpass_freqs[1] / nyquist
    b, a = butter(4, [low, high], btype='band')
    displacement = filtfilt(b, a, phase_resampled)
    
    return displacement