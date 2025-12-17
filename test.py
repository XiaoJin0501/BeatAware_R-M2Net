import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import scipy.io as sio
from tqdm import tqdm

from config import Config
from .dataset import RadarDataset
from .models.BA_M2Net import BeatAwareRM2Net
# 假设 metrics.py 在 utils 下
from .utils.metrics import calculate_metrics 
from .utils.logger import setup_logger

def test():
    # 初始化 Logger (单独的 test.log)
    logger = setup_logger(Config.LOG_DIR, name="test")
    logger.info(f"🚀 Starting Test for Experiment: {Config.EXP_NAME}")
    
    device = torch.device(Config.DEVICE)
    
    # 1. 加载数据
    logger.info(f"Reading test data from: {Config.TEST_H5}")
    test_dataset = RadarDataset(Config.TEST_H5)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. 加载模型
    model = BeatAwareRM2Net(in_channels=Config.IN_CHANNELS, base_channels=Config.BASE_CHANNELS).to(device)
    
    # 加载权重
    ckpt_path = os.path.join(Config.CKPT_DIR, f"{Config.EXP_NAME}_best.pth")
    if not os.path.exists(ckpt_path):
        logger.warning(f"⚠️ Checkpoint not found at {ckpt_path}! Using random weights.")
    else:
        logger.info(f"✅ Loading weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # 3. 推理循环
    all_metrics = {"MAE": [], "RMSE": [], "Pearson": []}
    saved_results = {"radar": [], "ecg_true": [], "ecg_pred": [], "pearson": []}
    
    logger.info("⏳ Running Inference...")
    with torch.no_grad():
        for radar, ecg, mask in tqdm(test_loader):
            radar, ecg, mask = radar.to(device), ecg.to(device), mask.to(device)
            
            # Forward
            pred = model(radar, mask)
            
            # Metrics
            metrics = calculate_metrics(pred, ecg)
            for k, v in metrics.items():
                all_metrics[k].append(v)
            
            # Collect data for .mat
            saved_results["radar"].append(radar.cpu().numpy().squeeze())
            saved_results["ecg_true"].append(ecg.cpu().numpy().squeeze())
            saved_results["ecg_pred"].append(pred.cpu().numpy().squeeze())
            saved_results["pearson"].append(metrics["Pearson"])

    # 4. 结果汇总与保存
    logger.info("="*40)
    logger.info(f"📊 Final Test Results ({len(test_loader)} samples)")
    logger.info("="*40)
    
    for k, v in all_metrics.items():
        logger.info(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    
    save_path = os.path.join(Config.RESULT_DIR, "test_results.mat")
    sio.savemat(save_path, {k: np.array(v) for k, v in saved_results.items()})
    logger.info(f"💾 Detailed results saved to: {save_path}")

if __name__ == "__main__":
    test()