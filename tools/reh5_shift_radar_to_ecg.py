import h5py
import numpy as np
from scipy import signal
from tqdm import tqdm

IN_H5  = "data_preprocessing/processed_to_h5/experiment_A_SubjectIndependent/test.h5"
OUT_H5 = "data_preprocessing/processed_to_h5/experiment_A_SubjectIndependent/test.h5"  # 直接覆盖也可以
FS = 200
MAX_LAG = 40
BAND = (0.8, 3.0)

def bandpass(x, fs, low, high, order=4):
    nyq = 0.5*fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype="band")
    return signal.filtfilt(b, a, x)

def estimate_lag(ecg_seg, radar_seg):
    e = bandpass(ecg_seg, FS, BAND[0], BAND[1])
    r = bandpass(radar_seg, FS, BAND[0], BAND[1])
    e = (e - e.mean())/(e.std()+1e-6)
    r = (r - r.mean())/(r.std()+1e-6)
    corr = signal.correlate(e, r, mode="full")
    lags = signal.correlation_lags(len(e), len(r), mode="full")
    sel = (lags>=-MAX_LAG) & (lags<=MAX_LAG)
    lags = lags[sel]; corr=corr[sel]
    return int(lags[np.argmax(np.abs(corr))])

def shift_keep_len(x, lag):
    """
    对齐约定（基于 corr = correlate(ecg, radar) 的峰值 lag）：
    - lag < 0  => radar 相对 ecg 延迟 |lag|（radar needs to shift LEFT by |lag|）
    - lag > 0  => radar 相对 ecg 超前 lag（radar needs to shift RIGHT by lag）
    """
    T = len(x)
    if lag == 0:
        return x

    if lag < 0:
        s = abs(lag)
        # radar LEFT shift: drop first s, pad zeros at end
        y = np.pad(x[s:], (0, s), mode="constant")[:T]
    else:
        s = lag
        # radar RIGHT shift: pad zeros at beginning, cut tail
        y = np.pad(x, (s, 0), mode="constant")[:T]
    return y


with h5py.File(IN_H5, "r") as fin:
    radar = fin["radar"][:]      # (N,1,T)
    ecg   = fin["ecg"][:]
    mask  = fin["mask"][:]
    sid   = fin["subject_id"][:] if "subject_id" in fin else None

N, _, T = radar.shape
radar_new = np.empty_like(radar)
lags_out = np.zeros((N,), dtype=np.int32)

for i in tqdm(range(N), desc="Shift radar to align ecg"):
    r = radar[i,0].astype(np.float64)
    e = ecg[i,0].astype(np.float64)
    lag = estimate_lag(e, r)
    lags_out[i] = lag
    radar_new[i,0] = shift_keep_len(r, lag)

with h5py.File(OUT_H5, "w") as fout:
    fout.create_dataset("radar", data=radar_new)
    fout.create_dataset("ecg", data=ecg)
    fout.create_dataset("mask", data=mask)
    fout.create_dataset("xcorr_lag", data=lags_out)
    if sid is not None:
        fout.create_dataset("subject_id", data=sid)

print("Saved:", OUT_H5)
print("Lag median (samples):", float(np.median(lags_out)),
      "IQR:", float(np.percentile(lags_out,75)-np.percentile(lags_out,25)))
