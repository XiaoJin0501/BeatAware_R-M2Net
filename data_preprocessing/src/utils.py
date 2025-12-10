import scipy.io as sio
import numpy as np

def load_mat_file(filepath):
    """读取 .mat 文件，返回字典"""
    try:
        data = sio.loadmat(filepath)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def min_max_normalize(signal):
    """
    归一化到 [-1, 1] (适配 Tanh)
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val < 1e-6:
        return signal # 避免除零，如果是直线就不动了
        
    # 映射到 [0, 1]
    norm_01 = (signal - min_val) / (max_val - min_val)
    # 映射到 [-1, 1]
    norm_n11 = 2 * norm_01 - 1
    return norm_n11