import os
import torch
import h5py
import numpy as np
import pandas as pd
import argparse  # <--- [核心修复] 补上这一行
from torch.utils.data import DataLoader
from tqdm import tqdm
# ✅ [新增] 引入 defaultdict，自动处理所有指标 Key
from collections import defaultdict 

# --- 引入项目模块 ---
from config import Config
from dataset import RadarDataset
from models.BA_M2Net import BeatAwareRM2Net
from utils.metrics import calculate_metrics, extract_clinical_features_nk
from utils.seeding import seed_everything
from tools.plotting import plot_reconstruction 

def test():
    # --- [新增] 动态参数解析 ---
    parser = argparse.ArgumentParser(description="Test BeatAware R-M2Net")
    parser.add_argument('--alpha', type=float, default=0.5, help='STFT loss weight used in training')
    parser.add_argument('--beta', type=float, default=1.0, help='Anchor loss weight used in training')
    parser.add_argument('--gamma', type=float, default=0.1, help='Smooth loss weight used in training')
    parser.add_argument('--exp_tag', type=str, default="Default", help='Tag used for this experiment')
    args = parser.parse_args()
    
    # 根据参数动态更新 Config 中的实验名称和路径
    # 必须与 train.py 中的命名规则完全一致
    new_exp_name = f"Exp_a{args.alpha}_b{args.beta}_g{args.gamma}_{args.exp_tag}"
    Config.update_paths(new_exp_name) # 调用 config.py 中的方法更新所有路径
    # --------------------------
    # 1. 设置环境
    seed_everything(Config.SEED)
    device = Config.DEVICE
    
    # 2. 准备路径
    ckpt_path = os.path.join(Config.CKPT_DIR, f"{Config.EXP_NAME}_best.pth")
    result_dir = Config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)
    
    # ✅ [修改] 使用 defaultdict(list)
    # 这样无论 calculate_metrics 返回 'MAE' 还是 'HR_Error'，都不会报错
    all_metrics = defaultdict(list)
    
    print(f"🚀 Starting Test for Experiment: {Config.EXP_NAME}")
    print(f"   Reading test data from: {Config.TEST_H5}")

    # 3. 加载数据
    test_set = RadarDataset(Config.TEST_H5)
    test_loader = DataLoader(
        test_set, 
        batch_size=1,  # 测试通常逐个样本进行
        shuffle=False, 
        num_workers=0
    )

    # 4. 加载模型
    model = BeatAwareRM2Net(
        in_channels=Config.IN_CHANNELS, 
        base_channels=Config.BASE_CHANNELS
    ).to(device)
    
    print(f"✅ Loading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # --- 数据导出准备 ---
    clinical_records = []  # 用于导出 A. test_comprehensive.csv
    vis_data_pool = []     # 用于导出 B. visualization_data.npz
    
    print(f"🚀 Starting Comprehensive Inference for: {Config.EXP_NAME}")
    
    with torch.no_grad():
        for i, (radar, ecg, mask, subject_id) in enumerate(tqdm(test_loader)):
            radar, ecg, mask = radar.to(device), ecg.to(device), mask.to(device)
            
            # [核心：捕获 pred_mask 以证明 Anchor 机制]
            pred_ecg, pred_mask = model(radar)
            
            # 转为 Numpy 方便指标计算和保存
            p_np = pred_ecg.cpu().numpy().squeeze()
            t_np = ecg.cpu().numpy().squeeze()
            pm_np = pred_mask.cpu().numpy().squeeze()
            tm_np = mask.cpu().numpy().squeeze()
            r_np = radar.cpu().numpy().squeeze()

            # A. 计算波形和临床指标
            wave_m = calculate_metrics(pred_ecg, ecg)
            p_feat = extract_clinical_features_nk(p_np, fs=200)
            t_feat = extract_clinical_features_nk(t_np, fs=200)

            # B. 记录受试者级详细数据
            record = {
                "Subject_ID": int(subject_id.item()), 
                "PCC": wave_m['Pearson'],
                "MAE": wave_m['MAE'],
                "RMSE": wave_m['RMSE'],
                "HR_True": t_feat['HR'], "HR_Pred": p_feat['HR'],
                "RR_True": t_feat['RR'], "RR_Pred": p_feat['RR'],
                "QRS_Err": abs(t_feat['QRS'] - p_feat['QRS']),
                "QT_Err": abs(t_feat['QT'] - p_feat['QT'])
            }
            clinical_records.append(record)

            # C. 缓存原始矩阵数据 (用于绘图)
            vis_data_pool.append({
                "radar": r_np,
                "ecg_true": t_np, "ecg_pred": p_np,
                "mask_true": tm_np, "mask_pred": pm_np,
                "pcc": wave_m['Pearson']
            })

    # --- 后处理与导出 ---
    
    # 1. 保存 A. test_comprehensive.csv (作图灵魂)
    df_full = pd.DataFrame(clinical_records)
    csv_path = os.path.join(Config.RESULT_DIR, "test_comprehensive.csv")
    df_full.to_csv(csv_path, index=False)
    
    # 2. 计算并输出 Table III 的统计量 (Mean ± Std)
    # 先按受试者取平均，再算所有人之间的标准差
    subject_wise_mean = df_full.groupby('Subject_ID').mean()
    print("\n📊 Table III Statistics (Inter-subject):")
    for col in ['PCC', 'MAE', 'HR_Pred', 'HR_True']:
        print(f"   {col}: {subject_wise_mean[col].mean():.4f} ± {subject_wise_mean[col].std():.4f}")

    # 3. 保存 B. visualization_data.npz (选出 Best/Worst Case)
    vis_data_pool.sort(key=lambda x: x['pcc']) # 按 PCC 排序
    # 提取典型案例
    best_case = vis_data_pool[-1]
    worst_case = vis_data_pool[0]
    median_case = vis_data_pool[len(vis_data_pool)//2]
    
    npz_path = os.path.join(Config.RESULT_DIR, "visualization_data.npz")
    np.savez(npz_path, 
             best_radar=best_case['radar'], best_ecg_true=best_case['ecg_true'], 
             best_ecg_pred=best_case['ecg_pred'], best_mask_pred=best_case['mask_pred'],
             worst_radar=worst_case['radar'], worst_ecg_true=worst_case['ecg_true'],
             worst_ecg_pred=worst_case['ecg_pred'], worst_mask_pred=worst_case['mask_pred'])
    
    print(f"\n✅ 全量原始数据导出成功！")
    print(f"   - 详细清单 (CSV): {csv_path}")
    print(f"   - 绘图矩阵 (NPZ): {npz_path}")

if __name__ == "__main__":
    test()