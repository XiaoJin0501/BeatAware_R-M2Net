import torch
import numpy as np
import neurokit2 as nk
from scipy.stats import pearsonr

def calculate_metrics(pred, target):
    """计算基础波形相似度指标"""
    # 1. 转换为 Numpy并展平为 1D 向量
    pred = pred.detach().cpu().numpy().reshape(pred.shape[0], -1)
    target = target.detach().cpu().numpy().reshape(target.shape[0], -1)
    
    batch_size = pred.shape[0]
    batch_mae = []
    batch_rmse = []
    batch_pcc = []
    
    # 2. 逐样本循环计算，确保指标反映的是每个片段的重构质量
    for i in range(batch_size):
        curr_p = pred[i]
        curr_t = target[i]
        
    # 3. MAE & RMSE
    batch_mae.append(np.mean(np.abs(curr_p - curr_t)))
    batch_rmse.append(np.sqrt(np.mean((curr_p - curr_t) ** 2)))
    
    # 5. 计算 Pearson 相关系数 (核心修正点)
    # 使用 np.corrcoef 得到相关矩阵，取 [0, 1] 元素
    # 注意：如果信号是平线（标准差为0），corrcoef 会返回 NaN
    
    std_p = np.std(curr_p)
    std_t = np.std(curr_t)
    if std_p < 1e-6 or std_t < 1e-6:
            batch_pcc.append(0.0)
    else:
            # np.corrcoef 返回 2x2 矩阵，取第 0 行第 1 列
            pcc = np.corrcoef(curr_p, curr_t)[0, 1]
            batch_pcc.append(pcc)
    
    
    
    
    if np.std(pred) < 1e-6 or np.std(target) < 1e-6:
        pcc = 0.0
    else:
        pcc = np.corrcoef(pred, target)[0, 1]
    
    # 3. 返回 Batch 平均值
    return {
        'MAE': np.mean(batch_mae),
        'RMSE': np.mean(batch_rmse),
        'Pearson': np.mean(batch_pcc)
    }

def extract_clinical_features_nk(signal, fs=200):
    """
    [核心新增] 使用 NeuroKit2 提取临床生理指标
    """
    try:
        # 1. 寻峰 (R-peaks)
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
        
        # 2. 波形解析 (Delineation): 检测 P, Q, S, T 特征点
        # method="peak" 比较稳健，适用于重构信号
        _, waves = nk.ecg_delineate(signal, rpeaks, sampling_rate=fs, method="peak")
        
        # 3. 计算基础心率指标
        r_indices = rpeaks['ECG_R_Peaks']
        if len(r_indices) < 2:
            return {"HR": np.nan, "RR": np.nan, "QRS": np.nan, "QT": np.nan}
        
        rr_intervals = np.diff(r_indices) * (1000.0 / fs) # 转为 ms
        rr_mean = np.nanmean(rr_intervals)
        hr = 60000.0 / rr_mean
        
        # 4. 计算细致间期 (QRS & QT)
        # QRS: S波偏移 - Q波起始; QT: T波偏移 - Q波起始
        # 注意: 如果信号质量差，nk检测不到某些波形，这里会返回 NaN
        qrs = np.nanmean(waves['ECG_S_Offsets'] - waves['ECG_Q_Onsets']) * (1000.0 / fs)
        qt = np.nanmean(waves['ECG_T_Offsets'] - waves['ECG_Q_Onsets']) * (1000.0 / fs)
        
        return {"HR": hr, "RR": rr_mean, "QRS": qrs, "QT": qt}
    except Exception:
        return {"HR": np.nan, "RR": np.nan, "QRS": np.nan, "QT": np.nan}