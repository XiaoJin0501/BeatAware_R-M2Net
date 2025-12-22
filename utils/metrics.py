import torch
import numpy as np

def calculate_metrics(pred, target):
    """
    计算 ECG 重建的三大核心指标
    Input:
        pred: [B, 1, L]
        target: [B, 1, L]
    Output:
        dict: {MAE, RMSE, Pearson}
    """
    # 1. 转换为 Numpy
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    # 2. 形状调整: [B, 1, L] -> [B, L]
    # ⚠️ 关键修改: 使用 reshape 而不是 squeeze，确保 Batch=1 时维度不丢失
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
    
    # 此时 pred 和 target 的形状必定是 [Batch, Length]
    # 即使 Batch=1，也是 [1, 1600]，zip 会正确迭代 1 次，拿到完整的 (1600,) 数组
    
    # 3. MAE (L1)
    mae = np.mean(np.abs(pred - target))
    
    # 4. RMSE (L2)
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    
    # 5. Pearson Correlation (逐样本计算后取平均)
    pearsons = []
    for p, t in zip(pred, target):
        # p, t 现在是形状为 (L,) 的完整波形
        if np.std(p) < 1e-6 or np.std(t) < 1e-6:
            pearsons.append(0)
        else:
            corr = np.corrcoef(p, t)[0, 1]
            pearsons.append(corr)
    
    avg_pearson = np.mean(pearsons)
    
    # 返回字典 (Key首字母大写以匹配 test.py 的 defaultdict)
    return {"MAE": mae, "RMSE": rmse, "Pearson": avg_pearson}