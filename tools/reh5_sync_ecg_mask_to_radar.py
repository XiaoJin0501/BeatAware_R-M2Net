import h5py
import numpy as np
from scipy import signal
from tqdm import tqdm

# ========== 你需要改的 ==========
IN_H5  = "data_preprocessing/processed_to_h5/experiment_A_SubjectIndependent/train.h5"
OUT_H5 = "data_preprocessing/processed_to_h5/experiment_A_SubjectIndependent/train.h5"
FS = 200  # Config.FS_TARGET
# =================================

# 对齐频段（与你 align_signals_robust 保持一致）
BAND = (0.8, 3.0)

# 允许的最大局部偏移（samples）；你原来 local_lag 用的是 ±40（200ms），这里保持一致
MAX_LAG = 40

def bandpass(x, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype="band")
    return signal.filtfilt(b, a, x)

def estimate_lag(ecg_seg, radar_seg, fs, max_lag):
    """
    估计 radar 相对 ecg 的偏移：返回 lag（samples）
    lag > 0 表示 radar 需要向右移才能对齐 ecg（或者等价：ecg 向左移 lag）
    我们这里用“心跳频段”互相关，取绝对相关最大值。
    """
    e = bandpass(ecg_seg, fs, BAND[0], BAND[1])
    r = bandpass(radar_seg, fs, BAND[0], BAND[1])

    e = e - np.mean(e)
    r = r - np.mean(r)

    corr = signal.correlate(e, r, mode="full")
    lags = signal.correlation_lags(len(e), len(r), mode="full")

    sel = (lags >= -max_lag) & (lags <= max_lag)
    lags = lags[sel]
    corr = corr[sel]

    lag = lags[np.argmax(np.abs(corr))]
    return int(lag)

def shift_crop_same_length(ecg, mask, lag):
    """
    将 ecg/mask 按 lag 与 radar 对齐，并保持长度不变（通过裁切+padding）
    约定：我们“移动 ecg/mask”去匹配 radar。
    lag > 0：ecg 相对 radar 领先 -> ecg/mask 向右移 lag（前面补零）
    lag < 0：ecg 相对 radar 滞后 -> ecg/mask 向左移 |lag|（丢掉开头）
    """
    T = len(ecg)
    if lag == 0:
        return ecg, mask

    if lag > 0:
        # ecg/mask 向右移：前面 pad，尾部截断
        ecg_new = np.pad(ecg, (lag, 0), mode="constant")[:T]
        mask_new = np.pad(mask, (lag, 0), mode="constant")[:T]
    else:
        s = abs(lag)
        ecg_new = np.pad(ecg[s:], (0, s), mode="constant")[:T]
        mask_new = np.pad(mask[s:], (0, s), mode="constant")[:T]
    return ecg_new, mask_new

with h5py.File(IN_H5, "r") as fin:
    radar = fin["radar"][:]      # (N,1,T)
    ecg   = fin["ecg"][:]        # (N,1,T)
    mask  = fin["mask"][:]       # (N,1,T)
    sid   = fin["subject_id"][:] if "subject_id" in fin else None

N, _, T = radar.shape
print(f"Loaded {IN_H5} | N={N}, T={T}, FS={FS}")

ecg_new = np.empty_like(ecg)
mask_new = np.empty_like(mask)
lags_out = np.zeros((N,), dtype=np.int32)

for i in tqdm(range(N), desc="Re-aligning ecg/mask to radar"):
    r = radar[i,0].astype(np.float64)
    e = ecg[i,0].astype(np.float64)
    m = mask[i,0].astype(np.float64)

    # 估计 lag（ecg vs radar）
    lag = estimate_lag(e, r, FS, MAX_LAG)
    lags_out[i] = lag

    # 同步移动 ecg/mask
    e2, m2 = shift_crop_same_length(e, m, lag)

    ecg_new[i,0] = e2
    mask_new[i,0] = m2

# 写出新 h5
with h5py.File(OUT_H5, "w") as fout:
    fout.create_dataset("radar", data=radar)
    fout.create_dataset("ecg", data=ecg_new)
    fout.create_dataset("mask", data=mask_new)
    fout.create_dataset("local_lag_sync", data=lags_out)
    if sid is not None:
        fout.create_dataset("subject_id", data=sid)

print("Saved:", OUT_H5)
print("Lag summary (samples): median=", np.median(lags_out), "IQR=",
      np.percentile(lags_out,75)-np.percentile(lags_out,25))
