import h5py
import numpy as np
import os
import matplotlib
# 关键：强制使用无界面后端，防止 SSH 环境下绘图失败
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ================== 配置区 ==================
H5_PATH = "data_preprocessing/processed_to_h5/experiment_A_SubjectIndependent/train.h5"
# 明确指定要检查的样本索引
VISUALIZE_IDXS = [0, 1, 8] 
# ============================================

def verify_and_plot(path):
    if not os.path.exists(path):
        print(f"❌ 错误: 找不到文件 {path}")
        return

    print(f"🔍 正在读取数据集: {path}")
    
    with h5py.File(path, 'r') as f:
        print("\n" + "="*80)
        print(f"{'Sample':<8} | {'Type':<8} | {'Mean':<10} | {'Min':<10} | {'Max':<10} | {'Status'}")
        print("-" * 80)

        # 1. 统计数值检查 (前10个)
        num_to_check = min(10, f['radar'].shape[0])
        for i in range(num_to_check):
            for key in ['radar', 'ecg', 'mask']:
                sample = f[key][i].flatten()
                s_mean, s_min, s_max = np.mean(sample), np.min(sample), np.max(sample)
                status = "✅ OK"
                if key in ['ecg', 'mask'] and (s_min < -0.01 or s_max > 1.01):
                    status = "❌ 范围错误"
                elif key == 'radar' and abs(s_mean) > 0.1:
                    status = "⚠️ 偏移预警"
                print(f"{i:<8} | {key:<8} | {s_mean:>10.4f} | {s_min:>10.4f} | {s_max:>10.4f} | {status}")
            print("-" * 80)

        # 2. 绘图验证逻辑
        print(f"\n🎨 正在为样本 {VISUALIZE_IDXS} 生成可视化图...")
        for idx in VISUALIZE_IDXS:
            if idx >= f['radar'].shape[0]: continue
            
            radar = f['radar'][idx].flatten()
            ecg = f['ecg'][idx].flatten()
            
            fig, ax1 = plt.subplots(figsize=(12, 5))
            
            # 绘制绿色雷达速度信号
            ax1.set_ylabel('Radar Velocity (Z-Score)', color='tab:green')
            ax1.plot(radar, color='tab:green', label='Radar Input', alpha=0.8)
            ax1.tick_params(axis='y', labelcolor='tab:green')
            ax1.grid(True, alpha=0.2)

            # 绘制红色 ECG 标签
            ax2 = ax1.twinx()
            ax2.set_ylabel('ECG Label (0-1)', color='tab:red')
            ax2.plot(ecg, color='tab:red', linestyle='--', label='ECG Label', alpha=0.7)
            ax2.tick_params(axis='y', labelcolor='tab:red')

            plt.title(f"Sample {idx} - Radar vs ECG Alignment Check")
            
            # 强制保存到当前运行的目录下，方便查找
            save_name = f"check_sample_{idx}.png"
            plt.savefig(save_name)
            plt.close()
            print(f"  -> ✅ 图片已保存至: {os.getcwd()}/{save_name}")

if __name__ == "__main__":
    verify_and_plot(H5_PATH)