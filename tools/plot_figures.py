# tools/plot_figures.py
import os
import json
import glob
import argparse
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


# -----------------------------
# IO helpers
# -----------------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_read_csv(p: str) -> pd.DataFrame:
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)


def _find_one(pattern: str) -> Optional[str]:
    xs = sorted(glob.glob(pattern))
    return xs[0] if len(xs) > 0 else None


# -----------------------------
# Plot helpers
# -----------------------------
def _cdf_xy(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1, dtype=np.float64) / xs.size
    return xs, ys


def _stft_mag_db(x: np.ndarray, fs: int, win_s: float = 2.0, overlap: float = 0.5, nfft: int = 256):
    """
    Protocol STFT:
      - window length: 2s
      - overlap: 50%
      - nfft: 256
    Returns f, t, mag_db
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x)

    nperseg = int(round(win_s * fs))
    noverlap = int(round(overlap * nperseg))
    nperseg = max(nperseg, 8)
    noverlap = min(noverlap, nperseg - 1)

    f, t, Z = signal.stft(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    mag = np.abs(Z)
    mag_db = 20.0 * np.log10(mag + 1e-8)
    return f, t, mag_db


def _plot_spectrogram(ax, f, t, mag_db, title: str, fmax: Optional[float] = None):
    if fmax is not None:
        mask = f <= float(fmax)
        f2 = f[mask]
        mag2 = mag_db[mask, :]
    else:
        f2, mag2 = f, mag_db

    im = ax.imshow(
        mag2,
        origin="lower",
        aspect="auto",
        extent=[t.min() if t.size else 0, t.max() if t.size else 1, f2.min() if f2.size else 0, f2.max() if f2.size else 1],
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return im


# -----------------------------
# Fig.1 – Qualitative (per-subject median cases)
# -----------------------------
def plot_fig1_qualitative(cases_dir: str, out_dir: str, n_subjects: int = 3):
    """
    Fig.1 (optional in some drafts):
      - Select first N subject median cases from cases_dir:
        cases/subject_{sid}_median_*.npz
      - Plot GT vs Pred waveforms.
    """
    pattern = os.path.join(cases_dir, "subject_*_median_*.npz")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print("[Fig.1] No per-subject median case files found. Skip Fig.1.")
        return

    files = files[:max(1, int(n_subjects))]

    fig = plt.figure(figsize=(10, 3.2 * len(files)))
    for i, fp in enumerate(files, start=1):
        data = np.load(fp, allow_pickle=True)
        gt = data["ecg_true"].astype(np.float64)
        pr = data["ecg_pred"].astype(np.float64)
        pcc = float(data["pcc"][0]) if "pcc" in data else np.nan
        sid = int(data["subject_id"][0]) if "subject_id" in data else -1
        seg = int(data["seg_id"][0]) if "seg_id" in data else -1

        ax = fig.add_subplot(len(files), 1, i)
        ax.plot(gt, label="GT ECG")
        ax.plot(pr, label="Pred ECG")
        ax.set_title(f"Subject {sid} | seg {seg} | PCC={pcc:.3f}")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

    plt.tight_layout()
    out_png = os.path.join(out_dir, "Fig1_qualitative.png")
    out_pdf = os.path.join(out_dir, "Fig1_qualitative.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[Fig.1] Saved: {out_png}")
    print(f"[Fig.1] Saved: {out_pdf}")


# -----------------------------
# Fig.2 – Time–Frequency (fixed sample)
# -----------------------------
def plot_fig2_time_frequency(cases_dir: str, meta: Dict[str, Any], out_dir: str,
                             fs: int, win_s: float = 2.0, overlap: float = 0.5, nfft: int = 256,
                             fmax: Optional[float] = 40.0):
    """
    Fig.2 protocol:
      - Use meta.json fig2_subject_id + fig2_seg_id
      - Load cases/fig2_sample_sid{sid}_seg{seg}.npz (recommended)
        otherwise fallback to any fig2_sample*.npz.
      - 4 subplots:
        radar spectrogram / GT ECG spectrogram / Pred ECG spectrogram / |GT-Pred| spectrogram
    """
    sid = int(meta.get("fig2_subject_id", -1))
    seg = int(meta.get("fig2_seg_id", -1))

    preferred = os.path.join(cases_dir, f"fig2_sample_sid{sid}_seg{seg}.npz")
    if os.path.exists(preferred):
        fp = preferred
    else:
        fp = _find_one(os.path.join(cases_dir, "fig2_sample*.npz"))

    if fp is None or (not os.path.exists(fp)):
        print("[Fig.2] Fig2 sample npz not found. Run test.py with --save_cases. Skip Fig.2.")
        return

    data = np.load(fp, allow_pickle=True)
    radar = data["radar"].astype(np.float64)
    gt = data["ecg_true"].astype(np.float64)
    pr = data["ecg_pred"].astype(np.float64)
    diff = np.abs(gt - pr)

    fr, tr, mr = _stft_mag_db(radar, fs=fs, win_s=win_s, overlap=overlap, nfft=nfft)
    fe, te, mg = _stft_mag_db(gt, fs=fs, win_s=win_s, overlap=overlap, nfft=nfft)
    _,  _,  mp = _stft_mag_db(pr, fs=fs, win_s=win_s, overlap=overlap, nfft=nfft)
    _,  _,  md = _stft_mag_db(diff, fs=fs, win_s=win_s, overlap=overlap, nfft=nfft)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = _plot_spectrogram(ax1, fr, tr, mr, "Radar Spectrogram", fmax=fmax)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = _plot_spectrogram(ax2, fe, te, mg, "GT ECG Spectrogram", fmax=fmax)

    ax3 = fig.add_subplot(gs[1, 0])
    im3 = _plot_spectrogram(ax3, fe, te, mp, "Pred ECG Spectrogram", fmax=fmax)

    ax4 = fig.add_subplot(gs[1, 1])
    im4 = _plot_spectrogram(ax4, fe, te, md, "|GT − Pred| Spectrogram", fmax=fmax)

    # colorbars
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # fig.suptitle(f"Fig.2 Time–Frequency Analysis (sid={sid}, seg={seg})", y=1.02)
    plt.tight_layout()

    out_png = os.path.join(out_dir, "Fig2_time_frequency.png")
    out_pdf = os.path.join(out_dir, "Fig2_time_frequency.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig.2] Saved: {out_png}")
    print(f"[Fig.2] Saved: {out_pdf}")


# -----------------------------
# Fig.3 – Subject-wise PCC median + IQR
# -----------------------------
def plot_fig3_subject_pcc(subject_csv: str, out_dir: str):
    df = _safe_read_csv(subject_csv)
    if df.empty:
        print("[Fig.3] subject_summary.csv not found/empty. Skip Fig.3.")
        return

    # required columns
    need = ["subject_id", "pcc_median", "pcc_p25", "pcc_p75"]
    for c in need:
        if c not in df.columns:
            print(f"[Fig.3] Missing column {c} in subject_summary.csv. Skip Fig.3.")
            return

    df = df.sort_values("subject_id").reset_index(drop=True)
    x = np.arange(len(df), dtype=int)
    y = df["pcc_median"].astype(float).values
    y25 = df["pcc_p25"].astype(float).values
    y75 = df["pcc_p75"].astype(float).values
    # error bars around median
    yerr_low = y - y25
    yerr_high = y75 - y
    yerr = np.vstack([yerr_low, yerr_high])

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, y)
    ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(df["subject_id"].astype(int).tolist(), rotation=0)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("PCC (median ± IQR)")
    ax.set_title("Fig.3 Subject-wise PCC (Median with IQR)")
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    out_png = os.path.join(out_dir, "Fig3_subject_pcc.png")
    out_pdf = os.path.join(out_dir, "Fig3_subject_pcc.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[Fig.3] Saved: {out_png}")
    print(f"[Fig.3] Saved: {out_pdf}")


# -----------------------------
# Fig.4 – CDFs (MRE and RR_err)
# -----------------------------
def plot_fig4_cdfs(segment_csv: str, beat_csv: str, out_dir: str):
    df_seg = _safe_read_csv(segment_csv)
    df_beat = _safe_read_csv(beat_csv)

    # Fig4(a) MRE CDF (segment-level)
    figA = plt.figure(figsize=(6, 4))
    axA = figA.add_subplot(1, 1, 1)
    if (not df_seg.empty) and ("mre" in df_seg.columns):
        xs, ys = _cdf_xy(df_seg["mre"].values)
        axA.plot(xs, ys)
    else:
        axA.text(0.5, 0.5, "segment_metrics.csv missing MRE", ha="center", va="center")
    axA.set_xlabel("MRE")
    axA.set_ylabel("CDF")
    axA.set_title("Fig.4(a) CDF of MRE (segment-level)")
    axA.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "Fig4a_cdf_mre.png")
    out_pdf = os.path.join(out_dir, "Fig4a_cdf_mre.pdf")
    figA.savefig(out_png, dpi=300)
    figA.savefig(out_pdf)
    plt.close(figA)
    print(f"[Fig.4a] Saved: {out_png}")
    print(f"[Fig.4a] Saved: {out_pdf}")

    # Fig4(b) RR_err CDF (beat-level)
    figB = plt.figure(figsize=(6, 4))
    axB = figB.add_subplot(1, 1, 1)
    if (not df_beat.empty) and ("rr_err_ms" in df_beat.columns):
        xs, ys = _cdf_xy(df_beat["rr_err_ms"].values)
        axB.plot(xs, ys)
    else:
        axB.text(0.5, 0.5, "beat_metrics.csv missing rr_err_ms", ha="center", va="center")
    axB.set_xlabel("RR interval absolute error (ms)")
    axB.set_ylabel("CDF")
    axB.set_title("Fig.4(b) CDF of RR interval error (beat-level)")
    axB.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "Fig4b_cdf_rrerr.png")
    out_pdf = os.path.join(out_dir, "Fig4b_cdf_rrerr.pdf")
    figB.savefig(out_png, dpi=300)
    figB.savefig(out_pdf)
    plt.close(figB)
    print(f"[Fig.4b] Saved: {out_png}")
    print(f"[Fig.4b] Saved: {out_pdf}")


# -----------------------------
# Main (one-click protocol)
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot Fig.1–Fig.4 from protocol outputs (one-click).")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to the experiment result directory (contains meta.json, *.csv, cases/).")
    parser.add_argument("--n_fig1_subjects", type=int, default=3,
                        help="How many per-subject median cases to show in Fig.1.")
    parser.add_argument("--skip_fig1", action="store_true")
    parser.add_argument("--skip_fig2", action="store_true")
    parser.add_argument("--skip_fig3", action="store_true")
    parser.add_argument("--skip_fig4", action="store_true")
    parser.add_argument("--fs", type=int, default=None,
                        help="Override sampling rate (if meta.json missing).")
    parser.add_argument("--stft_win_s", type=float, default=2.0)
    parser.add_argument("--stft_overlap", type=float, default=0.5)
    parser.add_argument("--stft_nfft", type=int, default=256)
    parser.add_argument("--fmax", type=float, default=40.0,
                        help="Max frequency to display in spectrograms.")
    args = parser.parse_args()

    result_dir = args.result_dir
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"result_dir not found: {result_dir}")

    meta_path = os.path.join(result_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"meta.json not found under result_dir.\n"
            f"Expected: {meta_path}\n"
            f"Please run test.py first."
        )
    meta = _load_json(meta_path)

    cases_dir = os.path.join(result_dir, "cases")
    seg_csv = os.path.join(result_dir, "segment_metrics.csv")
    beat_csv = os.path.join(result_dir, "beat_metrics.csv")
    subject_csv = os.path.join(result_dir, "subject_summary.csv")

    fig_dir = os.path.join(result_dir, "figures")
    _ensure_dir(fig_dir)

    fs = args.fs if args.fs is not None else int(meta.get("fs", 200))

    print("\n========== Plot Protocol ==========")
    print(f"Result dir : {result_dir}")
    print(f"Figures dir: {fig_dir}")
    print(f"FS         : {fs}")
    print(f"STFT       : win={args.stft_win_s}s, overlap={args.stft_overlap*100:.0f}%, nfft={args.stft_nfft}")
    print("===================================\n")

    if (not args.skip_fig1):
        plot_fig1_qualitative(cases_dir, fig_dir, n_subjects=args.n_fig1_subjects)

    if (not args.skip_fig2):
        plot_fig2_time_frequency(
            cases_dir, meta, fig_dir,
            fs=fs,
            win_s=float(args.stft_win_s),
            overlap=float(args.stft_overlap),
            nfft=int(args.stft_nfft),
            fmax=float(args.fmax) if args.fmax is not None else None,
        )

    if (not args.skip_fig3):
        plot_fig3_subject_pcc(subject_csv, fig_dir)

    if (not args.skip_fig4):
        plot_fig4_cdfs(seg_csv, beat_csv, fig_dir)

    print("\n✅ All requested figures generated.")


if __name__ == "__main__":
    main()
