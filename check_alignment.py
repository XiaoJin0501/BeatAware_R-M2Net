import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# 引入项目根目录的主 Config
from config import Config

def check_alignment():
    # ✅ [修正] 直接使用 Config.TRAIN_H5，它已经在 config.py 里定义好了完整路径
    h5_path = Config.TRAIN_H5
    
    print(f"Reading from: {h5_path}")
    
    if not os.path.exists(h5_path):
        print(f"❌ Error: File not found at {h5_path}")
        print("Please run data_preprocessing/build_dataset.py first!")
        return
    
    with h5py.File(h5_path, 'r') as f:
        # 读取全部数据
        radars = f['radar'][:]  # [N, 1, 1600]
        ecgs = f['ecg'][:]      # [N, 1, 1600]
        masks = f['mask'][:]    # [N, 1, 1600]

    print(f"Total samples found: {len(radars)}")
    
    # 随机抽取 5 个样本画图
    indices = np.random.choice(len(radars), 5, replace=False)
    
    plt.figure(figsize=(15, 12)) #稍微调大一点画布
    for i, idx in enumerate(indices):
        radar = radars[idx, 0, :]
        ecg = ecgs[idx, 0, :]
        mask = masks[idx, 0, :]
        
        plt.subplot(5, 1, i+1)
        
        # 1. 画雷达 (蓝色) - 稍微加粗一点
        plt.plot(radar, color='#1f77b4', linewidth=1.5, alpha=0.8, label='Radar Input')
        
        # 2. 画 ECG (红色)
        plt.plot(ecg, color='#d62728', linewidth=1.5, alpha=0.9, label='ECG Ground Truth')
        
        # 3. 画 Mask (绿色虚线) - 乘以 0.5 方便在同一坐标轴显示
        plt.plot(mask * 0.5, color='#2ca02c', linestyle='--', alpha=0.6, label='Anchor Mask (Scaled 0.5)')
        
        plt.title(f"Sample Index: {idx}")
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 只在第一张图显示图例，避免遮挡
        if i == 0:
            plt.legend(loc='upper right')
    
    plt.tight_layout()
    save_path = "alignment_check.png"
    plt.savefig(save_path)
    print(f"✅ Visualization saved to: {os.path.abspath(save_path)}")
    print("👉 请立即打开这张图，检查红色尖峰（ECG R波）是否总是对应蓝色波形（Radar）的特定特征！")

if __name__ == "__main__":
    check_alignment()