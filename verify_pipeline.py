import torch
import os
from torch.utils.data import DataLoader, Subset
from config import Config
from dataset import RadarDataset
from models.BA_M2Net import BeatAwareRM2Net
from utils.losses import TotalLoss
from utils.logger import setup_logger

def verify_pipeline():
    print("🔍 开始项目全流程快速验证...\n")
    
    # 1. 环境与路径检查
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/6] 运行环境: {device}")
    
    test_exp_name = "Verify_Run_Temp"
    Config.update_paths(test_exp_name)
    Config.makedirs()
    logger = setup_logger(Config.LOG_DIR, name="verify")
    
    # 2. 数据加载检查
    print(f"[2/6] 正在尝试读取 H5 文件: {Config.TRAIN_H5}")
    try:
        full_dataset = RadarDataset(Config.TRAIN_H5)
        # 只取前 2 个样本进行快速测试
        mini_dataset = Subset(full_dataset, range(2))
        loader = DataLoader(mini_dataset, batch_size=2)
        radar, ecg, mask, subject_id = next(iter(loader))
        print(f"✅ 数据解包成功！当前样本来自受试者 ID: {subject_id[0].item()}")
    except Exception as e:
        print("❌ 报错：解包数量不匹配，请检查 verify_pipeline.py 是否也补上了 subject_id。")
        return

    # 3. 模型初始化与前向传播检查
    print("[3/6] 正在初始化模型与前向传播...")
    try:
        model = BeatAwareRM2Net(in_channels=Config.IN_CHANNELS, base_channels=Config.BASE_CHANNELS).to(device)
        radar = radar.to(device)
        pred_ecg, pred_mask = model(radar)
        print(f"✅ 前向传播成功! Pred_ECG shape: {pred_ecg.shape}")
    except Exception as e:
        print(f"❌ 模型运行失败: {e}")
        return

    # 4. Loss 计算检查 (验证 5 个返回值解包)
    print("[4/6] 正在验证 TotalLoss 计算与解包 (5个返回值)...")
    try:
        criterion = TotalLoss(alpha=0.5, beta=1.0, gamma=0.1).to(device)
        ecg, mask = ecg.to(device), mask.to(device)
        
        # 核心：检查是否能正确解包 5 个值
        loss_outputs = criterion(pred_ecg, ecg, pred_mask, mask)
        
        if len(loss_outputs) != 5:
            print(f"❌ Loss 解包错误: 预期 5 个返回值，实际得到 {len(loss_outputs)} 个")
            return
        
        loss, l_time, l_freq, l_anchor, l_smooth = loss_outputs
        print(f"✅ Loss 计算成功! Total: {loss.item():.4f}, Smooth: {l_smooth.item():.4f}")
    except Exception as e:
        print(f"❌ Loss 计算失败: {e}")
        return

    # 5. 反向传播检查
    print("[5/6] 正在验证反向传播...")
    try:
        loss.backward()
        print("✅ 反向传播成功!")
    except Exception as e:
        print(f"❌ 反向传播失败: {e}")
        return

    # 6. 权重保存检查
    print("[6/6] 正在验证模型保存...")
    try:
        save_path = os.path.join(Config.CKPT_DIR, "verify_test.pth")
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        if os.path.exists(save_path):
            print(f"✅ 权重保存成功: {save_path}")
            os.remove(save_path) # 清理测试文件
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return

    print("\n" + "="*40)
    print("🎉 恭喜！项目核心流水线验证通过。")
    print("你可以放心地提交到服务器运行 run_ablation_study.sh 了。")
    print("="*40)

if __name__ == "__main__":
    verify_pipeline()