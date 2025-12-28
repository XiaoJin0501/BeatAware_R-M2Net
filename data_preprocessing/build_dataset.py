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
from scipy import signal

# --- 新增：对齐辅助函数 ---
def align_signals_robust(radar, ecg, fs):
    """
    计算雷达和ECG的互相关，找到最佳滞后并对齐。
    注意：为了计算准确，我们临时对信号进行强滤波(0.8-3Hz)来提取心跳包络进行匹配，
    但返回的是根据该滞后对齐后的原始输入信号。
    """
    # 1. 临时强滤波 (只取心跳频段 0.8-3.0Hz，排除呼吸干扰)
    b, a = signal.butter(4, [0.8 / (0.5 * fs), 3.0 / (0.5 * fs)], btype='band')
    r_heart = signal.filtfilt(b, a, radar)
    e_heart = signal.filtfilt(b, a, ecg) # ECG本身就很干净，但这能保持相位一致性

    # 2. 归一化 (Z-Score)
    r_heart = (r_heart - np.mean(r_heart)) / (np.std(r_heart) + 1e-6)
    e_heart = (e_heart - np.mean(e_heart)) / (np.std(e_heart) + 1e-6)

    # 3. 互相关计算
    # 限制最大搜索范围为 +/- 1.0 秒 (即 fs 个点)，避免匹配到错误的周期
    max_lag = int(1.0 * fs)
    correlation = signal.correlate(r_heart, e_heart, mode='full')
    lags = signal.correlation_lags(len(r_heart), len(e_heart), mode='full')

    # 只看中心窗口内的相关性
    mask = (lags >= -max_lag) & (lags <= max_lag)
    valid_lags = lags[mask]
    valid_corr = correlation[mask]

    # 4.-- 极性检测逻辑 ---
    # 如果负相关峰值强于正相关，说明雷达相位极性反转了
    max_idx = np.argmax(valid_corr)
    min_idx = np.argmin(valid_corr)
    
    if abs(valid_corr[min_idx]) > abs(valid_corr[max_idx]):
        # 极性反转处理：将雷达信号翻转，并取负相关峰值位置作为对齐点
        radar = -radar
        best_lag = valid_lags[min_idx]
    else:
        best_lag = valid_lags[max_idx]
    
    # 5. 执行对齐裁切
    # best_lag > 0: 雷达信号滞后 (Radar is delayed) -> 雷达需要往左移(丢掉开头)
    # best_lag < 0: ECG 滞后 -> ECG 需要往左移
    if best_lag > 0:
        radar_aligned = radar[best_lag:]
        ecg_aligned = ecg[:len(radar_aligned)]
    elif best_lag < 0:
        ecg_aligned = ecg[abs(best_lag):]
        radar_aligned = radar[:len(ecg_aligned)]
    else:
        radar_aligned = radar
        ecg_aligned = ecg
        
    # 再次确保长度一致
    min_len = min(len(radar_aligned), len(ecg_aligned))
    return radar_aligned[:min_len], ecg_aligned[:min_len], best_lag

def z_score_normalize(data):
    """[新增] Z-Score 归一化 (用于 Radar 输入)"""
    std_val = np.std(data)
    if std_val < 1e-6: # 防止除零 (死线)
        return np.zeros_like(data)
    return (data - np.mean(data)) / std_val

def min_max_normalize_strict(data):
    """[核心修正] 严格 Min-Max 归一化到 [0, 1]，确保适配 Sigmoid 激活函数"""
    d_min = np.min(data)
    d_max = np.max(data)
    return (data - d_min) / (d_max - d_min + 1e-8)

# ==========================================
# 数据处理核心循环
# ==========================================

def process_subject(file_path):
    """处理单个受试者，返回该受试者所有的合格片段列表"""
    fname = os.path.basename(file_path) # 先定义 fname
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
    
    
    # 2. 鲁棒对齐与异常值过滤
    try:
        radar_aligned, ecg_aligned, best_lag = align_signals_robust(radar_clean, ecg_clean, Config.FS_TARGET)
        
    # 【修改点】将阈值从 100 放大到 200 (即 1.0秒)，适配您数据中 660ms 的偏移
        if abs(best_lag) > 200: 
            print(f"  [Debug] {fname}: Rejected by Lag ({best_lag} pts)")
            return []
    except Exception as e:
        print(f"  [Debug] {fname}: Alignment crash - {e}")
        return []
    
    # # 2. 长度对齐 (先截断到相同长度)
    # min_len = min(len(radar_clean), len(ecg_clean))
    # radar_clean = radar_clean[:min_len]
    # ecg_clean = ecg_clean[:min_len]
    
    # # ========== 加入相位对齐 ==========
    # # 雷达的心跳分量通常滞后于ECG的R波，如果不强制对齐，模型学不到东西
    # try:
    #     radar_clean, ecg_clean = align_signals(radar_clean, ecg_clean, Config.FS_TARGET)
    # except Exception as e:
    #     print(f"Skipping {os.path.basename(file_path)}: Alignment error {e}")
    #     return []
    
    # --- Anchor 生成 ---
    mask, r_peaks = ecg_dsp.generate_anchor_mask(ecg_clean, Config.FS_TARGET, Config.ANCHOR_SIGMA)
    
    # --- 切片与筛选 ---
    win_pts = int(Config.WINDOW_SECONDS * Config.FS_TARGET)
    stride_pts = int(Config.STRIDE_SECONDS * Config.FS_TARGET)
    
    segments = []
    
    for start in range(0, len(radar_clean) - win_pts, stride_pts):
        end = start + win_pts
        
        # SQI 检查 (基于对齐后的 R peaks) 
        seg_r_peaks = [p - start for p in r_peaks if start <= p < end]
        if not quality_control.check_sqi(seg_r_peaks, win_pts, Config.FS_TARGET, Config.SQI_HR_MIN, Config.SQI_HR_MAX):
            continue
        
        segments.append({
            'radar': z_score_normalize(radar_aligned[start:end]),
            'ecg': min_max_normalize_strict(ecg_aligned[start:end]), # 必须归一化到 [0, 1]
            'mask': mask[start:end]
        })
        
    return segments

# ==========================================
# 主程序逻辑 (Experiment A/B 划分)
# ==========================================

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