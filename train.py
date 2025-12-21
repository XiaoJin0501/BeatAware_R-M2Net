import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
# 引入 TensorBoard
from torch.utils.tensorboard import SummaryWriter 

# --- 引入搭建的基础设施 ---
from config import Config
from dataset import RadarDataset
from models.BA_M2Net import BeatAwareRM2Net
from utils.losses import TotalLoss
# logger 和 seeding 放在 utils 下 
from utils.logger import setup_logger
from utils.seeding import seed_everything

def train():
    # 1. 固定随机种子 (保证每次跑结果一样)
    seed_everything(Config.SEED)
    
    # 2. 初始化日志系统 (同时输出到屏幕和文件) 日志会保存在 experiments/Exp_Name/logs/train.log
    logger = setup_logger(Config.LOG_DIR, name="train")
    writer = SummaryWriter(log_dir=os.path.join(Config.LOG_DIR, 'tensorboard'))
    
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
    
    logger.info(f"Data loaded. Train samples: {len(train_set)}, Test samples: {len(test_set)}")

    # 4. 模型与 Loss 构建
    model = BeatAwareRM2Net(
        in_channels=Config.IN_CHANNELS, 
        base_channels=Config.BASE_CHANNELS
    ).to(Config.DEVICE)
    
    # 初始化 Loss (beta=0.1 是 anchor loss 的权重)
    criterion = TotalLoss(alpha=Config.ALPHA, beta=0.1).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    logger.info(f"Model initialized. Alpha for STFT Loss: {Config.ALPHA}")

    # 5. 训练主循环
    best_val_loss = float('inf')
    # 新增：初始化计数器
    epochs_no_improve = 0
    
    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss_avg = 0
        
        # 使用 tqdm 显示进度条，不刷屏 log 文件
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        
        # 加上 enumerate，这样才能拿到 step (i) 用于画连续曲线
        for i, (radar, ecg, mask) in enumerate(loop):
            radar, ecg, mask = radar.to(Config.DEVICE), ecg.to(Config.DEVICE), mask.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # ✅ [修正点 1] 只调用一次模型
            # 但 forward 签名里有 mask=None，所以传进去也没错，或者写 model(radar) 也可以
            pred_ecg, anchor_pred = model(radar)
            
            # ✅ 修改 2: 不需要额外的 anchor_label 变量
            # dataset 返回的 'mask' 就是 Ground Truth
            anchor_target = mask.to(Config.DEVICE)

            # Loss Calculation
            # ✅ 修改 3: 传入 4 个参数，接收 4 个返回值
            loss, l_time, l_freq, l_anchor = criterion(pred_ecg, ecg, pred_mask, anchor_target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss_avg += loss.item()
            
            # 进度条显示实时 Loss
            loop.set_postfix(loss=loss.item(), L1=l_time.item(), STFT=l_freq.item(), Anchor=l_anchor.item())
            
            # 记录 Training Loss (每 10 个 batch 记一次，减少开销)
            if i % 10 == 0:
                current_step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/Train_Total', loss.item(), current_step)
                writer.add_scalar('Loss/Train_L1', l_time.item(), current_step)
                writer.add_scalar('Loss/Train_STFT', l_freq.item(), current_step)
                # 只有当 anchor_loss 有效时才记录
                if anchor_target is not None:
                    writer.add_scalar('Loss/Train_Anchor', l_anchor.item(), current_step)
            
        train_loss_avg /= len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss_avg = 0
        with torch.no_grad():
            for radar, ecg, mask in test_loader:
                radar, ecg, mask = radar.to(Config.DEVICE), ecg.to(Config.DEVICE), mask.to(Config.DEVICE)
                # ✅ 验证集也需要接收两个返回值 (pred_mask 被忽略)
                pred_ecg, _ = model(radar)
                # 验证集通常只看重建 Loss，不需要算 Anchor Loss
                loss, _, _, _ = criterion(pred_ecg, ecg, None, None)
                val_loss_avg += loss.item()
                
        val_loss_avg /= len(test_loader)
        
        # TensorBoard Val Loss
        writer.add_scalar('Loss/Val_Total', val_loss_avg, epoch)
        
        # --- Logging & Saving ---
        logger.info(f"Epoch {epoch+1:03d} | Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f}")
        
        # 保存最佳模型+早停逻辑
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0 # 重置计数器
            save_path = os.path.join(Config.CKPT_DIR, f"{Config.EXP_NAME}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, save_path)
            logger.info(f"Best model saved to {save_path}")
        else:
            epochs_no_improve += 1 # 计数器加 1
            logger.info(f"No improvement for {epochs_no_improve}/{Config.PATIENCE} epochs.")
            
            # 检查是否触发早停
            if epochs_no_improve >= Config.PATIENCE:
                logger.info(f"Early stopping triggered after {Config.PATIENCE} epochs with no improvement.")
                break
            
    # ✅ [修改点 6] 训练结束关闭 writer
    writer.close()
    logger.info("Training Finished Successfully!")

if __name__ == "__main__":
    train()