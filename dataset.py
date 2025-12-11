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
    import os
    
    # 假设你现在的路径结构，自动找一个文件来测试
    # 注意：这里路径可能需要根据你实际存放 HDF5 的位置微调
    base_path = r'data_preprocessing/processed_to_h5/experiment_A_SubjectIndependent/train.h5'
    
    if os.path.exists(base_path):
        print(f"Testing dataset with: {base_path}")
        dataset = RadarDataset(base_path)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # 取一个 Batch 看看长什么样
        radar, ecg, mask = next(iter(dataloader))
        
        print("\n[Data Shapes Check]")
        print(f"Radar Batch: {radar.shape} (Expected: [4, 1, 1600])")
        print(f"ECG Batch  : {ecg.shape}   (Expected: [4, 1, 1600])")
        print(f"Mask Batch : {mask.shape}  (Expected: [4, 1, 1600])")
        
        print("\n[Value Range Check]")
        print(f"Radar: min={radar.min():.4f}, max={radar.max():.4f}")
        print(f"ECG  : min={ecg.min():.4f}, max={ecg.max():.4f}")
        print(f"Mask : min={mask.min():.4f}, max={mask.max():.4f}")
        
        if mask.max() > 0.5:
            print("\n✅ Mask looks good! (Contains peaks)")
        else:
            print("\n⚠️ Warning: Mask seems empty/flat. Check preprocessing?")
    else:
        print(f"Error: Could not find test file at {base_path}")
        print("Please check your path.")