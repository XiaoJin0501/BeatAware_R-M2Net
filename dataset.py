import torch
import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class RadarDataset(Dataset):
    def __init__(self, h5_file_path, bad_indices_path: str = None):
        """
        Beat-Aware R-M2Net 的专用数据集加载器（支持 QC bad indices 过滤）

        Args:
            h5_file_path: .h5 文件路径（train.h5 / test.h5）
            bad_indices_path: bad_indices.npy 路径（可选）。
                若提供，则会在 Dataset 层剔除这些样本，使训练/测试完全可复现
        """
        self.h5_file_path = h5_file_path
        
        # 读取基础信息（只读一次，不保持文件句柄，避免多进程冲突）
        with h5py.File(self.h5_file_path, 'r') as f:
            self.total_length = len(f['radar'])
            
            # subject_id：如果 H5 没有，则用 -1 占位（或按需改为 idx->subject 映射）
            if "subject_id" in f:
                self.subject_ids = f["subject_id"][:].astype(np.int32)
            else:
                self.subject_ids = np.full((self.total_length,), -1, dtype=np.int32)
            
        # 2) 构造初始索引
        self.indices = np.arange(self.total_length, dtype=np.int64)
        
        # 3) 加载并应用 bad_indices 过滤
        self.bad_indices = None
        if bad_indices_path is not None:
            if not os.path.exists(bad_indices_path):
                raise FileNotFoundError(f"[ERROR] bad_indices_path not found: {bad_indices_path}")

            bad = np.load(bad_indices_path).astype(np.int64)
            bad = np.unique(bad)
            bad = bad[(bad >= 0) & (bad < self.total_length)]  # 边界保护

            self.bad_indices = bad

            # setdiff: 保留不在 bad 中的样本
            self.indices = np.setdiff1d(self.indices, bad, assume_unique=False)
        
        # 4) 打印一次摘要（可选，但强烈建议保留，便于实验日志）
        print(
            f"[RadarDataset] Loaded: {self.h5_file_path}\n"
            f"  total_length = {self.total_length}\n"
            f"  kept_length  = {len(self.indices)}\n"
            f"  removed_bad  = {0 if self.bad_indices is None else len(self.bad_indices)}"
        )
    
    
          
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        # DataLoader 的 idx -> H5 的真实索引
        real_idx = int(self.indices[idx])

        # 每次 getitem 打开文件，确保多线程安全
        with h5py.File(self.h5_file_path, "r") as f:
            radar = f["radar"][real_idx]  # (1, 1600)
            ecg = f["ecg"][real_idx]
            mask = f["mask"][real_idx]

        radar_tensor = torch.from_numpy(radar).float()
        ecg_tensor = torch.from_numpy(ecg).float()
        mask_tensor = torch.from_numpy(mask).float()

        subject_id = int(self.subject_ids[real_idx])
        return radar_tensor, ecg_tensor, mask_tensor, subject_id

# --- 简单的测试代码 (Test Block) ---
if __name__ == '__main__':
    from pathlib import Path
    from torch.utils.data import DataLoader

    # ============================================================
    # 1. 自动定位项目根目录
    # 假设当前文件位于：
    #   Radar2ECGNet/src/... 或 Radar2ECGNet/data_preprocessing/...
    # ============================================================
    PROJECT_ROOT = Path(__file__).resolve().parent
    while PROJECT_ROOT.name not in {'BeatAware_R-M2Net', 'Radar2ECGNet'}:
        if PROJECT_ROOT.parent == PROJECT_ROOT:
            raise RuntimeError("Cannot locate project root directory.")
        PROJECT_ROOT = PROJECT_ROOT.parent

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
    
    bad_path = (
        PROJECT_ROOT
        / "data_preprocessing"
        / "qc_indices"
        / "bad_indices.npy"
    )


    # ============================================================
    # 3. 基本路径检查（在 raise 之前打印全部关键信息）
    # ============================================================
    print("\n================ PATH DEBUG ================")
    print("[DEBUG] __file__        =", __file__)
    print("[DEBUG] resolved file   =", Path(__file__).resolve())
    print("[DEBUG] PROJECT_ROOT    =", PROJECT_ROOT)
    print("[DEBUG] base_path       =", base_path)
    print("[DEBUG] bad_path        =", bad_path)
    print("[DEBUG] base exists?    =", base_path.exists())
    print("[DEBUG] bad exists?     =", bad_path.exists())

    # 额外：列出你 processed_to_h5 下到底有什么（非常关键）
    p_root = PROJECT_ROOT / "data_preprocessing" / "processed_to_h5"
    print("[DEBUG] processed_to_h5 =", p_root, "exists?", p_root.exists())
    if p_root.exists():
        print("[DEBUG] processed_to_h5 children:")
        for x in sorted(p_root.iterdir()):
            if x.is_dir():
                print("   -", x.name)

    print("============================================\n")

    if not base_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Could not find test file:\n{base_path}\n"
            "Please check your directory structure."
        )


    # ============================================================
    # 4. 构建 Dataset / DataLoader
    # ============================================================
    dataset = RadarDataset(str(base_path), bad_indices_path=str(bad_path))
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
    radar, ecg, mask, sid = next(iter(dataloader))

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
