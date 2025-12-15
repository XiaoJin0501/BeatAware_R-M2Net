import matplotlib.pyplot as plt
import os

def plot_reconstruction(radar, ecg_true, ecg_pred, epoch, save_dir, sample_idx=0):
    """
    画出对比图：上方是 Radar 输入，下方是 ECG 对比 (红=真实, 蓝=重建)
    """
    # 取 Batch 中的第 sample_idx 个样本
    r = radar[sample_idx, 0].detach().cpu().numpy()
    t = ecg_true[sample_idx, 0].detach().cpu().numpy()
    p = ecg_pred[sample_idx, 0].detach().cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    
    # 子图 1: Radar Input
    plt.subplot(2, 1, 1)
    plt.plot(r, color='green', alpha=0.7, label='Radar Input')
    plt.title(f"Epoch {epoch} - Radar Input")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图 2: ECG Comparison
    plt.subplot(2, 1, 2)
    plt.plot(t, color='red', alpha=0.6, label='Ground Truth', linewidth=1.5)
    plt.plot(p, color='blue', alpha=0.8, label='Reconstruction', linewidth=1.0, linestyle='--')
    plt.title("ECG Reconstruction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch}_sample_{sample_idx}.png")
    plt.savefig(save_path)
    plt.close()