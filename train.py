import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
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
    # --- [新增] 命令行参数解析：解决您“如何管理不同系数实验”的疑问 ---
    parser = argparse.ArgumentParser(description="Train BeatAware R-M2Net")
    parser.add_argument('--alpha', type=float, default=Config.ALPHA, help='STFT loss weight')
    parser.add_argument('--beta', type=float, default=Config.BETA, help='Anchor loss weight')
    parser.add_argument('--gamma', type=float, default=Config.GAMMA) # 新增命令行参数
    parser.add_argument('--exp_tag', type=str, default="Default", help='Tag for this experiment')
    args = parser.parse_args()

    # 动态更新 Config 参数和路径
    # 例如：Exp_Alpha0.5_Beta1.0_Default
    new_name = f"Exp_a{args.alpha}_b{args.beta}_g{args.gamma}_{args.exp_tag}"
    Config.ALPHA = args.alpha
    Config.BETA = args.beta
    Config.GAMMA = args.gamma  # 更新 gamma 参数
    Config.update_paths(new_name)
    Config.makedirs()
    # =========================================================================
    
    # 1. 固定随机种子 
    seed_everything(Config.SEED)
    # 2. 初始化日志系统 (同时输出到屏幕和文件) 日志会保存在 experiments/Exp_Name/logs/train.log
    logger = setup_logger(Config.LOG_DIR, name="train")
    writer = SummaryWriter(log_dir=os.path.join(Config.LOG_DIR, 'tensorboard'))
    
    logger.info(f"🚀 Experiment Started: {Config.EXP_NAME}")
    logger.info(f"📊 Hyperparams: Alpha={Config.ALPHA}, Beta={Config.BETA}")
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
    model = BeatAwareRM2Net(in_channels=Config.IN_CHANNELS, base_channels=Config.BASE_CHANNELS).to(Config.DEVICE)
    logger.info("⏳ Initializing model and optimizer...")
    
    # 初始化 Loss (包含 STFT Loss 和 Anchor Loss)
    criterion = TotalLoss(alpha=Config.ALPHA, beta=Config.BETA, gamma=Config.GAMMA).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    
    # 🔍  断点续训逻辑 (Resume Logic)
    # =========================================================================
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # 定义断点路径: last.pth 用于记录最新的训练状态
    last_ckpt_path = os.path.join(Config.CKPT_DIR, f"{Config.EXP_NAME}_last.pth")
    best_ckpt_path = os.path.join(Config.CKPT_DIR, f"{Config.EXP_NAME}_best.pth")

    # 检查是否存在上次中断的断点
    if os.path.exists(last_ckpt_path):
        logger.info(f"Found checkpoint at {last_ckpt_path}. Resuming training...")
        checkpoint = torch.load(last_ckpt_path, map_location=Config.DEVICE)
        
        # 1. 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        # 2. 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 3. 恢复 Epoch 计数
        start_epoch = checkpoint['epoch'] + 1
        # 4. 恢复 Loss 记录
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        
        logger.info(f"Resumed from Epoch {start_epoch}. Best Val Loss so far: {best_val_loss:.6f}")
    else:
        logger.info("No checkpoint found. Starting fresh training.")

    # =========================================================================

    # 5. 训练主循环
    best_val_loss = float('inf')
    # 新增：初始化计数器
    epochs_no_improve = 0
    
    for epoch in range(start_epoch, Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        
        # ✅ [新增] 增加 Smooth 统计变量
        train_loss_avg, train_L1_avg, train_STFT_avg, train_Anchor_avg, train_Smooth_avg = 0, 0, 0, 0, 0
        
        # 使用 tqdm 显示进度条，不刷屏 log 文件
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        
        # 加上 enumerate，这样才能拿到 step (i) 用于画连续曲线
        for i, (radar, ecg, mask, subject_id) in enumerate(loop):
            current_step = epoch * len(train_loader) + i
            radar, ecg, mask = radar.to(Config.DEVICE), ecg.to(Config.DEVICE), mask.to(Config.DEVICE)
            optimizer.zero_grad()
            
            # ✅ [修正点 1] 只调用一次模型
            # 但 forward 签名里有 mask=None，所以传进去也没错，或者写 model(radar) 也可以
            pred_ecg, pred_mask = model(radar)
            
            # ✅ 修改 2: 不需要额外的 anchor_label 变量
            # dataset 返回的 'mask' 就是 Ground Truth
            anchor_target = mask.to(Config.DEVICE)

            # Loss Calculation 用 Config 注入的权重计算 Loss
            # ✅ 修改 3: 传入 4 个参数，接收 4 个返回值
            loss, l_time, l_freq, l_anchor, l_smooth= criterion(pred_ecg, ecg, pred_mask, anchor_target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss_avg += loss.item()
            train_L1_avg += l_time.item()
            train_STFT_avg += l_freq.item()
            train_Anchor_avg += l_anchor.item()
            train_Smooth_avg += l_smooth.item() # ✅ 统计 Smooth Loss
            
            # 进度条显示实时 Loss
            loop.set_postfix(loss=loss.item(), L1=l_time.item(), STFT=l_freq.item(), Anchor=l_anchor.item(), Smooth=l_smooth.item())
            writer.add_scalar('Loss/Train_Smooth', l_smooth.item(), current_step)
            
            # TensorBoard Logging every 10 steps
            # 使用 epoch * len(train_loader) + i 作为全局 step
            if i % 10 == 0:
                current_step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/Train_Total', loss.item(), current_step)
                writer.add_scalar('Loss/Train_L1', l_time.item(), current_step)
                writer.add_scalar('Loss/Train_STFT', l_freq.item(), current_step)
                writer.add_scalar('Loss/Train_Anchor', l_anchor.item(), current_step)
                writer.add_scalar('Loss/Train_Smooth', l_smooth.item(), current_step)

        # ✅ [修改 3] 计算所有 Loss 的平均值
        train_loss_avg /= len(train_loader)
        train_L1_avg /= len(train_loader)
        train_STFT_avg /= len(train_loader)
        train_Anchor_avg /= len(train_loader)
        train_Smooth_avg /= len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss_avg = 0
        with torch.no_grad():
            for radar, ecg, mask, subject_id in test_loader:
                radar, ecg, mask = radar.to(Config.DEVICE), ecg.to(Config.DEVICE), mask.to(Config.DEVICE)
                # ✅ 验证集也需要接收两个返回值 (pred_mask 被忽略)
                pred_ecg, _ = model(radar)
                # 验证集通常只看重建 Loss，不需要算 Anchor Loss
                loss, _, _, _, _ = criterion(pred_ecg, ecg, None, None)
                val_loss_avg += loss.item()
                
        val_loss_avg /= len(test_loader)
        
        # TensorBoard Val Loss
        writer.add_scalar('Loss/Val_Total', val_loss_avg, epoch)
        
        # --- Logging & Saving ---
        logger.info(
            f"Epoch {epoch+1:03d} | "
            f"Train Total: {train_loss_avg:.4f} "
            f"(L1: {train_L1_avg:.4f}, STFT: {train_STFT_avg:.4f}, Anchor: {train_Anchor_avg:.4f}) | "
            f"Val Loss: {val_loss_avg:.4f}"
        )
        
        # 🔍 [新增功能] 保存 Best 和 Last Checkpoint
        # =====================================================================
        
        # 1. 始终保存当前 Epoch 为 "last.pth" (覆盖式)
        # 这样无论何时断电，下次都可以从这里恢复
        last_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss_avg,
            'best_val_loss': best_val_loss,      # 记住当前最好的成绩
            'epochs_no_improve': epochs_no_improve # 记住早停计数器
        }
        torch.save(last_checkpoint, last_ckpt_path)
        logger.info(f"Last checkpoint saved to {last_ckpt_path}")
        
        # # 2. 如果是历史最佳，额外保存一份 "best.pth"
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0 # 重置计数器
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss_avg
            }
            torch.save(best_checkpoint, best_ckpt_path)
            logger.info(f"Best model saved to {best_ckpt_path}")
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