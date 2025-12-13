import torch
import numpy as np

def calculate_metrics(pred, target):
    """
    计算 ECG 重建的三大核心指标
    Input:
        pred: [B, 1, L]
        target: [B, 1, L]
    Output:
        dict: {mae, rmse, pearson}
    """
    # 转为 Numpy 方便计算 (尤其是 Pearson)
    pred = pred.detach().cpu().numpy().squeeze()
    target = target.detach().cpu().numpy().squeeze()
    
    # 1. MAE (L1)
    mae = np.mean(np.abs(pred - target))
    
    # 2. RMSE (L2)
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    
    # 3. Pearson Correlation (逐样本计算后取平均)
    pearsons = []
    for p, t in zip(pred, target):
        # 防止分母为0
        if np.std(p) < 1e-6 or np.std(t) < 1e-6:
            pearsons.append(0)
        else:
            corr = np.corrcoef(p, t)[0, 1]
            pearsons.append(corr)
    
    avg_pearson = np.mean(pearsons)
    
    return {"MAE": mae, "RMSE": rmse, "Pearson": avg_pearson}