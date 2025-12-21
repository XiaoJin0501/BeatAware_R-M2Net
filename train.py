import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# --- 引入我们搭建的基础设施 ---
from config import Config
from dataset import RadarDataset
from models.BA_M2Net import BeatAwareRM2Net
from utils.losses import TotalLoss
# logger 和 seeding 放在 utils 下 
from utils.logger import setup_logger
from utils.seeding import seed_everything

def train():
    # 1. 第一件事：固定随机种子 (保证每次跑结果一样)
    seed_everything(Config.SEED)
    
    # 2. 初始化日志系统 (同时输出到屏幕和文件)
    # 日志会保存在 experiments/Exp_Name/logs/train.log
    logger = setup_logger(Config.LOG_DIR, name="train")
    
    logger.info(f"🚀 Experiment Started: {Config.EXP_NAME}")
    logger.info(f"📂 Data Directory: {Config.DATA_DIR}")
    logger.info(f"💻 Device: {Config.DEVICE}")

    # 3. 数据准备
    logger.info("⏳ Loading datasets...")
    train_set = RadarDataset(Config.TRAIN_H5)
    test_set = RadarDataset(Config.TEST_H5)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS
    )
    
    logger.info(f"✅ Data loaded. Train samples: {len(train_set)}, Test samples: {len(test_set)}")

    # 4. 模型与 Loss 构建
    model = BeatAwareRM2Net(
        in_channels=Config.IN_CHANNELS, 
        base_channels=Config.BASE_CHANNELS
    ).to(Config.DEVICE)
    
    criterion = TotalLoss(alpha=Config.ALPHA).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    logger.info(f"🧠 Model initialized. Alpha for STFT Loss: {Config.ALPHA}")

    # 5. 训练主循环
    best_val_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss_avg = 0
        
        # 使用 tqdm 显示进度条，但不要刷屏 log 文件
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        
        for radar, ecg, mask in loop:
            radar, ecg, mask = radar.to(Config.DEVICE), ecg.to(Config.DEVICE), mask.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward (传入 Mask 用于 Anchor Branch)
            pred = model(radar, mask) 
            
            # Loss Calculation
            loss, l_time, l_freq = criterion(pred, ecg)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss_avg += loss.item()
            
            # 进度条显示实时 Loss
            loop.set_postfix(loss=loss.item(), L1=l_time.item(), STFT=l_freq.item())
            
        train_loss_avg /= len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss_avg = 0
        with torch.no_grad():
            for radar, ecg, mask in test_loader:
                radar, ecg, mask = radar.to(Config.DEVICE), ecg.to(Config.DEVICE), mask.to(Config.DEVICE)
                pred = model(radar, mask)
                loss, _, _ = criterion(pred, ecg)
                val_loss_avg += loss.item()
                
        val_loss_avg /= len(test_loader)
        
        # --- Logging & Saving ---
        # 这一行信息会被永久记录到 log 文件中
        logger.info(f"Epoch {epoch+1:03d} | Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f}")
        
        # 保存最佳模型
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            save_path = os.path.join(Config.CKPT_DIR, f"{Config.EXP_NAME}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, save_path)
            logger.info(f"🌟 Best model saved to {save_path}")

    logger.info("🎉 Training Finished Successfully!")

if __name__ == "__main__":
    train()