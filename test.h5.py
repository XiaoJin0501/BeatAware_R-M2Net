import neurokit2 as nk
import h5py
import numpy as np

with h5py.File('data_preprocessing/processed_to_h5/experiment_A_SubjectIndependent/test.h5', 'r') as f:
    # 抽查第 100 个样本的真值
    ecg_gt = f['ecg'][600].flatten() 
    
    # [自检] 必须确保归一化后的信号 R 峰依然清晰
    try:
        _, rpeaks = nk.ecg_peaks(ecg_gt, sampling_rate=200)
        print(f"✅ 预处理自检通过！R 峰检测成功：{len(rpeaks['ECG_R_Peaks'])} 个峰")
    except:
        print("❌ 预处理致命错误：真值 ECG 无法寻峰！请检查 build_dataset.py 中的降采样和归一化逻辑。")