# test.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sio

from config import Config
from .dataset import RadarDataset
from models.BA_M2Net import BeatAwareRM2Net
from utils.metrics import calculate_metrics

def test():
    print(f"🚀 Starting Inference on {Config.DEVICE}...")
    
    # 1. 加载模型
    model = BeatAwareRM2Net(in_channels=Config.IN_CHANNELS, base_channels=Config.BASE_CHANNELS).to(Config.DEVICE)
    
    ckpt_path = os.path.join(Config.CKPT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        
    checkpoint = torch.load(ckpt_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict']) # 假设保存时加了 key
    model.eval()
    print(f"✅ Loaded weights from {ckpt_path}")

    # 2. 数据加载
    test_set = RadarDataset(Config.TEST_PATH)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False) # Batch=1 方便逐个分析
    
    # 3. 推理循环
    all_metrics = {"MAE": [], "RMSE": [], "Pearson": []}
    predictions = []
    ground_truths = []
    inputs = []
    
    with torch.no_grad():
        for i, (radar, ecg, mask) in tqdm(enumerate(test_loader), total=len(test_loader)):
            radar, ecg, mask = radar.to(Config.DEVICE), ecg.to(Config.DEVICE), mask.to(Config.DEVICE)
            
            # Forward
            pred = model(radar, mask)
            
            # Metrics
            met = calculate_metrics(pred, ecg)
            for k, v in met.items():
                all_metrics[k].append(v)
            
            # 收集数据用于保存
            inputs.append(radar.cpu().numpy().squeeze())
            predictions.append(pred.cpu().numpy().squeeze())
            ground_truths.append(ecg.cpu().numpy().squeeze())

    # 4. 统计结果
    print("\n" + "="*30)
    print("📊 Final Test Results")
    print("="*30)
    for k, v in all_metrics.items():
        mean_val = np.mean(v)
        std_val = np.std(v)
        print(f"{k}: {mean_val:.4f} ± {std_val:.4f}")
    print("="*30)
    
    # 5. 保存结果 (方便用 MATLAB/Python 画论文图)
    save_file = os.path.join(Config.RESULT_DIR, "test_results.mat")
    sio.savemat(save_file, {
        "radar": np.array(inputs),
        "ecg_true": np.array(ground_truths),
        "ecg_pred": np.array(predictions),
        "mae": np.array(all_metrics["MAE"]),
        "pearson": np.array(all_metrics["Pearson"])
    })
    print(f"💾 Results saved to {save_file}")
    
    # 6. 可视化分析 (Best & Worst Cases)
    # 根据 Pearson 排序
    pearsons = np.array(all_metrics["Pearson"])
    sorted_indices = np.argsort(pearsons)
    
    worst_idx = sorted_indices[:3] # 最差的3个
    best_idx = sorted_indices[-3:] # 最好的3个
    
    plot_cases(worst_idx, inputs, ground_truths, predictions, pearsons, "Worst")
    plot_cases(best_idx, inputs, ground_truths, predictions, pearsons, "Best")

def plot_cases(indices, inputs, truths, preds, scores, tag):
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        plt.subplot(3, 1, i+1)
        plt.plot(truths[idx], 'r', label='Ground Truth', alpha=0.6)
        plt.plot(preds[idx], 'b--', label='Prediction', alpha=0.8)
        plt.title(f"{tag} Case #{i+1} (Sample {idx}) - Pearson: {scores[idx]:.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULT_DIR, f"{tag}_cases_analysis.png"))
    plt.close()
    print(f"📈 Saved {tag} cases plot.")

if __name__ == "__main__":
    test()