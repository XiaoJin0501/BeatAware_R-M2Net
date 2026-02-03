import h5py, numpy as np
from scipy import signal
from tqdm import tqdm

H5_PATH = "data_preprocessing/processed_to_h5/experiment_A_SubjectIndependent/test.h5"
FS = 200
MAX_LAG = 40  # +/-200ms

def bandpass(x, fs, low=0.8, high=3.0, order=4):
    nyq = 0.5*fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype="band")
    return signal.filtfilt(b, a, x)

def est_lag(ecg_seg, radar_seg):
    e = bandpass(ecg_seg, FS)
    r = bandpass(radar_seg, FS)
    e = (e - e.mean())/(e.std()+1e-6)
    r = (r - r.mean())/(r.std()+1e-6)
    corr = signal.correlate(e, r, mode="full")
    lags = signal.correlation_lags(len(e), len(r), mode="full")
    sel = (lags>=-MAX_LAG) & (lags<=MAX_LAG)
    lags = lags[sel]; corr=corr[sel]
    return int(lags[np.argmax(np.abs(corr))])

with h5py.File(H5_PATH, "r") as f:
    radar = f["radar"][:]  # (N,1,T)
    ecg   = f["ecg"][:]
N = radar.shape[0]

np.random.seed(0)
idxs = np.random.choice(N, size=min(300, N), replace=False)

lags = []
for i in tqdm(idxs):
    r = radar[i,0].astype(np.float64)
    e = ecg[i,0].astype(np.float64)
    try:
        lags.append(est_lag(e, r))
    except Exception:
        pass

lags = np.array(lags, dtype=int)
print("N used:", len(lags))
print("Lag median (samples):", float(np.median(lags)))
print("Lag IQR (samples):", float(np.percentile(lags,75)-np.percentile(lags,25)))
print("Lag median (ms):", float(np.median(lags)*1000/FS))
print("Lag IQR (ms):", float((np.percentile(lags,75)-np.percentile(lags,25))*1000/FS))
print("Zero-lag rate:", float(np.mean(lags==0)))
