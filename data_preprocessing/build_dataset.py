import os
import glob
import h5py
import numpy as np
from tqdm import tqdm
import sys

# 将当前脚本所在的目录添加到 sys.path，确保能找到 config 和 src 包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config 
from src import radar_dsp, ecg_dsp, quality_control, utils

def process_subject(file_path):
    """处理单个受试者，返回该受试者所有的合格片段列表"""
    data = utils.load_mat_file(file_path)
    if data is None: return []
    
    try:
        r_i = data['radar_i'].flatten()
        r_q = data['radar_q'].flatten()
        ecg = data['tfm_ecg1'].flatten() 
    except KeyError:
        if 'tfm_ecg2' in data:
            ecg = data['tfm_ecg2'].flatten()
        else:
            return []
        
    # --- 信号处理 ---
    # [修正2] 使用 Config 类名访问参数 (注意大写 C)
    radar_clean = radar_dsp.process_radar_signal(
        r_i, r_q, Config.FS_RADAR_RAW, Config.FS_TARGET, Config.RADAR_BANDPASS
    )
    ecg_clean = ecg_dsp.process_ecg_signal(
        ecg, Config.FS_ECG_RAW, Config.FS_TARGET, Config.ECG_BANDPASS
    )
    
    min_len = min(len(radar_clean), len(ecg_clean))
    radar_clean = radar_clean[:min_len]
    ecg_clean = ecg_clean[:min_len]
    
    # --- Anchor 生成 ---
    mask, r_peaks = ecg_dsp.generate_anchor_mask(ecg_clean, Config.FS_TARGET, Config.ANCHOR_SIGMA)
    
    # --- 切片与筛选 ---
    win_pts = int(Config.WINDOW_SECONDS * Config.FS_TARGET)
    stride_pts = int(Config.STRIDE_SECONDS * Config.FS_TARGET)
    
    segments = []
    
    for start in range(0, min_len - win_pts, stride_pts):
        end = start + win_pts
        
        # SQI 检查 (只检查这一小段内的 R 峰)
        seg_r_peaks = [p - start for p in r_peaks if start <= p < end]
        if not quality_control.check_sqi(seg_r_peaks, win_pts, Config.FS_TARGET, 
                                       Config.SQI_HR_MIN, Config.SQI_HR_MAX):
            continue # 丢弃坏片段
            
        # 归一化
        seg_radar = utils.min_max_normalize(radar_clean[start:end])
        seg_ecg = utils.min_max_normalize(ecg_clean[start:end])
        seg_mask = mask[start:end]
        
        segments.append({
            'radar': seg_radar,
            'ecg': seg_ecg,
            'mask': seg_mask
        })
        
    return segments

def save_h5(segments, filename):
    """保存列表到 HDF5"""
    if not segments: return
    
    radar_stack = np.stack([s['radar'] for s in segments])[:, np.newaxis, :]
    ecg_stack = np.stack([s['ecg'] for s in segments])[:, np.newaxis, :]
    mask_stack = np.stack([s['mask'] for s in segments])[:, np.newaxis, :]
    
    print(f"  -> Saving {len(segments)} samples to {filename}")
    with h5py.File(filename, 'w') as f:
        f.create_dataset('radar', data=radar_stack)
        f.create_dataset('ecg', data=ecg_stack)
        f.create_dataset('mask', data=mask_stack)

def main():
    # 创建目录
    for sub_dir in ['experiment_A_SubjectIndependent', 'experiment_B_Mixed']:
        os.makedirs(os.path.join(Config.PROCESSED_DATA_DIR, sub_dir), exist_ok=True)
        
    print(f"Searching for data in: {Config.RAW_DATA_DIR}")
    
    # 递归搜索
    search_pattern = os.path.join(Config.RAW_DATA_DIR, "**", "*Resting.mat")
    raw_files = glob.glob(search_pattern, recursive=True)
    raw_files.sort()
    
    if not raw_files:
        print(f"Error: No '*Resting.mat' files found in {Config.RAW_DATA_DIR}")
        return

    print(f"Found {len(raw_files)} files. Starting Quality Control & Processing...")
    
    # --- 第一阶段：处理并筛选受试者 ---
    valid_subjects_data = [] 
    rejected_subjects = []
    
    for fpath in tqdm(raw_files):
        fname = os.path.basename(fpath)
        try:
            sid = int(fname[3:7]) # 解析 ID
        except:
            continue
            
        sub_segs = process_subject(fpath)
        
        # 核心筛选逻辑
        if len(sub_segs) >= Config.MIN_VALID_SEGMENTS_PER_SUBJECT:
            valid_subjects_data.append({'sid': sid, 'segs': sub_segs})
        else:
            rejected_subjects.append(sid)
            
    print(f"\n[Quality Report]")
    print(f"  Total Subjects: {len(raw_files)}")
    print(f"  Valid Subjects: {len(valid_subjects_data)}")
    print(f"  Rejected Subjects (Low Quality): {rejected_subjects}")
    
    # 按 ID 排序
    valid_subjects_data.sort(key=lambda x: x['sid'])
    
    # ================= 实验 A: Subject Independent (按人切分) =================
    num_valid = len(valid_subjects_data)
    num_test = int(num_valid * Config.TEST_RATIO_A)
    num_train = num_valid - num_test
    
    # 划分
    train_subs_A = valid_subjects_data[:num_train]
    test_subs_A = valid_subjects_data[num_train:]
    
    print(f"\n[Experiment A Split]")
    print(f"  Train Subjects ({len(train_subs_A)}): {[s['sid'] for s in train_subs_A]}")
    print(f"  Test Subjects  ({len(test_subs_A)}): {[s['sid'] for s in test_subs_A]}")
    
    train_segs_A = [seg for sub in train_subs_A for seg in sub['segs']]
    test_segs_A = [seg for sub in test_subs_A for seg in sub['segs']]
    
    save_h5(train_segs_A, os.path.join(Config.PROCESSED_DATA_DIR, 'experiment_A_SubjectIndependent', 'train.h5'))
    save_h5(test_segs_A, os.path.join(Config.PROCESSED_DATA_DIR, 'experiment_A_SubjectIndependent', 'test.h5'))
    
    # ================= 实验 B: Mixed (混合切分) =================
    all_segs_mixed = [seg for sub in valid_subjects_data for seg in sub['segs']]
    
    np.random.seed(42) 
    np.random.shuffle(all_segs_mixed)
    
    split_idx_B = int(len(all_segs_mixed) * (1 - Config.TEST_RATIO_B))
    train_segs_B = all_segs_mixed[:split_idx_B]
    test_segs_B = all_segs_mixed[split_idx_B:]
    
    print(f"\n[Experiment B Split]")
    print(f"  Total Segments: {len(all_segs_mixed)}")
    print(f"  Train Segments: {len(train_segs_B)}")
    print(f"  Test Segments : {len(test_segs_B)}")
    
    save_h5(train_segs_B, os.path.join(Config.PROCESSED_DATA_DIR, 'experiment_B_Mixed', 'train.h5'))
    save_h5(test_segs_B, os.path.join(Config.PROCESSED_DATA_DIR, 'experiment_B_Mixed', 'test.h5'))

if __name__ == "__main__":
    main()