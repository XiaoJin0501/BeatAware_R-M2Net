import pandas as pd
import numpy as np
import os
import argparse

def verify_ablation_results(exp_path):
    csv_path = os.path.join(exp_path, "results", "test_comprehensive.csv")
    npz_path = os.path.join(exp_path, "results", "visualization_data.npz")
    
    print(f"🔍 正在验证实验路径: {exp_path}")

    # 1. 验证 CSV 指标文件
    if not os.path.exists(csv_path):
        print(f"❌ 错误: 未找到 CSV 文件 {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"✅ CSV 加载成功。总样本数: {len(df)}")

    # 检查 Q1 论文必需的列
    required = ['Subject_ID', 'PCC', 'HR_True', 'HR_Pred']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ 错误: CSV 缺失关键列: {missing}")
    else:
        print(f"✅ 关键字段完整: {required}")

    # 2. 统计受试者级性能 (Inter-subject Stats)
    subjects = df['Subject_ID'].unique()
    print(f"👤 包含受试者: {sorted(subjects.tolist())}")
    
    sub_stats = df.groupby('Subject_ID')['PCC'].mean()
    print("\n📊 受试者平均 PCC 预览:")
    print(sub_stats)
    print(f"\n📈 全局受试者间一致性: {sub_stats.mean():.4f} ± {sub_stats.std():.4f}")

    # 3. 验证 NPZ 绘图矩阵
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        print(f"\n✅ NPZ 验证成功。包含字段: {list(data.keys())}")
        if 'best_mask_pred' in data:
            print("✨ 检测到 Anchor 概率图数据，可用于机制可视化图。")
    else:
        print(f"⚠️ 警告: 未找到 NPZ 绘图文件 {npz_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 例如传入: experiments/Exp_a0.5_b1.0_g0.1_Baseline
    parser.add_argument('--path', type=str, required=True, help='实验文件夹路径')
    args = parser.parse_args()
    verify_ablation_results(args.path)