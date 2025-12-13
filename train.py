import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# 引用我们写好的模块
from .dataset import RadarDataset
from models.BA_M2Net import BeatAwareRM2Net
from utils.losses import TotalLoss

# --- 配置参数 (Hyperparameters) ---
CONFIG = {
    "exp_name": "Experiment_A_Baseline",
    "batch_size": 32,      # 显存不够可以改小 (e.g., 16)
    "learning_rate": 1e-4, # 初始学习率
    "epochs": 100,         # 训练轮数
    "alpha": 1.0,          # STFT Loss 权重
    "num_workers": 4,      # Mac 上设为 0 如果报错，Linux 上设为 4 或 8
    
    # 路径配置 (请修改为你实际的 h5 路径)
    "train_path": "data_preprocessing/processed_to_h5/experiment_A_SubjectIndependent/train.h5",
    "test_path":  "data_preprocessing/processed_to_h5/experiment_A_SubjectIndependent/test.h5",
    "save_dir":   "checkpoints/"
}

def train():
    # 1. 环境设置 (自动适配 Mac/Linux)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Training on NVIDIA CUDA (Linux)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Training on Apple MPS (MacBook)")
    else:
        device = torch.device("cpu")
        print("⚠️ Training on CPU (Slow)")

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # 2. 数据准备
    print(f"Loading data from: {CONFIG['train_path']}")
    train_set = RadarDataset(CONFIG['train_path'])
    test_set = RadarDataset(CONFIG['test_path'])
    
    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")

    # 3. 模型与 Loss
    model = BeatAwareRM2Net(in_channels=1, base_channels=32).to(device)
    criterion = TotalLoss(alpha=CONFIG['alpha']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-2)

    # 4. 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        # --- Training ---
        model.train()
        train_loss_avg = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
        
        for radar, ecg, mask in loop:
            radar, ecg, mask = radar.to(device), ecg.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            # 注意：如果你的模型需要 mask 输入 (BA_M2Net), 这里传入 mask
            pred = model(radar, mask) 
            
            # Loss
            loss, l_time, l_freq = criterion(pred, ecg)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss_avg += loss.item()
            loop.set_postfix(loss=loss.item(), L1=l_time.item(), STFT=l_freq.item())
            
        train_loss_avg /= len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss_avg = 0
        with torch.no_grad():
            for radar, ecg, mask in test_loader:
                radar, ecg, mask = radar.to(device), ecg.to(device), mask.to(device)
                pred = model(radar, mask)
                loss, _, _ = criterion(pred, ecg)
                val_loss_avg += loss.item()
                
        val_loss_avg /= len(test_loader)
        
        print(f"Epoch {epoch+1} Result: Train Loss={train_loss_avg:.4f}, Val Loss={val_loss_avg:.4f}")
        
        # --- Save Best Model ---
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            save_path = os.path.join(CONFIG["save_dir"], f"{CONFIG['exp_name']}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved to {save_path}")

    print("🎉 Training Finished!")

if __name__ == "__main__":
    train()