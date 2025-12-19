import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RadarDataset(Dataset):
    def __init__(self, h5_file_path):
        """
        Beat-Aware R-M2Net 的专用数据集加载器
        :param h5_file_path: .h5 文件的路径 (例如 '.../train.h5')
        """
        self.h5_file_path = h5_file_path
        
        # 预先读取数据长度，但不保持文件打开 (避免多进程冲突)
        with h5py.File(self.h5_file_path, 'r') as f:
            self.length = len(f['radar'])
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 在每次 getitem 时打开文件，确保多线程安全
        with h5py.File(self.h5_file_path, 'r') as f:
            # 读取数据
            # 形状都是 (1, 1600) -> 对应 (Channels, Time)
            radar = f['radar'][idx]
            ecg = f['ecg'][idx]
            mask = f['mask'][idx]

        # 转换为 PyTorch Tensor 并确保是 Float32
        radar_tensor = torch.from_numpy(radar).float()
        ecg_tensor = torch.from_numpy(ecg).float()
        mask_tensor = torch.from_numpy(mask).float()

        return radar_tensor, ecg_tensor, mask_tensor

# --- 简单的测试代码 (Test Block) ---
if __name__ == '__main__':
    from pathlib import Path
    from torch.utils.data import DataLoader

    # ============================================================
    # 1. 自动定位项目根目录
    # 假设当前文件位于：
    #   Radar2ECGNet/src/... 或 Radar2ECGNet/data_preprocessing/...
    # ============================================================
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # ============================================================
    # 2. 构建 HDF5 数据路径（与运行位置无关）
    # ============================================================
    base_path = (
        PROJECT_ROOT
        / 'data_preprocessing'
        / 'processed_to_h5'
        / 'experiment_A_SubjectIndependent'
        / 'train.h5'
    )

    # ============================================================
    # 3. 基本路径检查
    # ============================================================
    if not base_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Could not find test file:\n{base_path}\n"
            "Please check your directory structure."
        )

    print(f"\n[INFO] Testing dataset with:\n{base_path}")

    # ============================================================
    # 4. 构建 Dataset / DataLoader
    # ============================================================
    dataset = RadarDataset(str(base_path))
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,   # Test 阶段建议 0，便于 debug
        drop_last=True
    )

    # ============================================================
    # 5. 取一个 batch 做 sanity check
    # ============================================================
    radar, ecg, mask = next(iter(dataloader))

    print("\n================ Data Shapes Check ================")
    print(f"Radar Batch: {radar.shape} (Expected: [4, 1, 1600])")
    print(f"ECG Batch  : {ecg.shape}   (Expected: [4, 1, 1600])")
    print(f"Mask Batch : {mask.shape}  (Expected: [4, 1, 1600])")

    print("\n================ Value Range Check ================")
    print(f"Radar: min={radar.min():.4f}, max={radar.max():.4f}")
    print(f"ECG  : min={ecg.min():.4f}, max={ecg.max():.4f}")
    print(f"Mask : min={mask.min():.4f}, max={mask.max():.4f}")

    # ============================================================
    # 6. Mask 合法性检查（对 QRS / 有效心搏非常关键）
    # ============================================================
    if mask.max() > 0.5:
        print("\n✅ Mask looks good! (Contains valid peaks)")
    else:
        print("\n⚠️ Warning: Mask seems empty or flat.")
        print("   → Please check QRS detection or preprocessing pipeline.")

    print("\n[INFO] Test Block finished successfully.\n")