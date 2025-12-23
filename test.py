import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
# ✅ [新增] 引入 defaultdict，自动处理所有指标 Key
from collections import defaultdict 

# --- 引入项目模块 ---
from config import Config
from dataset import RadarDataset
from models.BA_M2Net import BeatAwareRM2Net
from utils.metrics import calculate_metrics
from utils.seeding import seed_everything
from tools.plotting import plot_reconstruction 

def test():
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

    # 5. 推理循环
    print("⏳ Running Inference...")
    
    # 只画前 10 张图看看效果
    num_plots = 10 
    
    with torch.no_grad():
        for i, (radar, ecg, mask) in enumerate(tqdm(test_loader)):
            radar = radar.to(device)
            ecg = ecg.to(device)
            
            # 解包模型的两个返回值
            pred_ecg, _ = model(radar)  # 忽略 anchor 分支的输出
            
            # 计算指标
            metrics = calculate_metrics(pred_ecg, ecg)
            
            # 可视化 (只画前 num_plots 张)
            if i < num_plots:
                plot_reconstruction(
                    radar=radar, 
                    ecg_true=ecg, 
                    ecg_pred=pred_ecg, 
                    epoch=999,      # 测试阶段标记为 999
                    save_dir=result_dir, 
                    sample_idx=0
                )
                
            # ✅ 记录指标 (现在不会报错了)
            for k, v in metrics.items():
                all_metrics[k].append(v)

    # 6. 统计并保存结果
    print("\n📊 Test Results Summary:")
    final_results = {}
    for k, v in all_metrics.items():
        mean_val = np.mean(v)
        std_val = np.std(v)
        # 打印平均值 ± 标准差
        print(f"   {k}: {mean_val:.4f} ± {std_val:.4f}")
    
    # 保存详细结果到 CSV
    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(result_dir, "test_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Detailed metrics saved to: {csv_path}")
    print(f"🖼️  Visualization images saved to: {result_dir}")

if __name__ == "__main__":
    test()