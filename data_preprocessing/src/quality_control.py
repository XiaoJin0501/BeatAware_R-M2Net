import numpy as np

def check_sqi(r_peaks, signal_len, fs, hr_min=40, hr_max=140):
    """
    简单的信号质量指数 (Signal Quality Index) 检查
    如果这段信号的心率极度不正常，或者检测不到 R 峰，视为坏数据
    """
    if len(r_peaks) < 2:
        return False # 甚至没有两个心跳
        
    # 计算 R-R 间期 (秒)
    rr_intervals = np.diff(r_peaks) / fs
    
    # 计算平均心率
    mean_rr = np.mean(rr_intervals)
    mean_hr = 60.0 / mean_rr
    
    if mean_hr < hr_min or mean_hr > hr_max:
        return False # 心率异常 (过快或过慢)
    
    # 检查 R-R 间期变异性 (简单的伪影检测)
    # 如果标准差太大，说明可能有漏检或误检
    if np.std(rr_intervals) > 0.2 * mean_rr:
        return False
        
    return True