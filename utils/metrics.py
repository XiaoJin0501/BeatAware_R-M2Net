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
        
    # 3. MAE (L1)
    batch_mae.append(np.mean(np.abs(curr_p - curr_t)))
    
    # 4. RMSE (L2)
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
