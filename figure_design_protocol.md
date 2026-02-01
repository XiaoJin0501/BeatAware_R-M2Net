# Figure Design and Visualization Protocol

This document specifies the **figure design rules, data sources, and statistical conventions** used throughout the manuscript.
All visualization choices are made to ensure **clarity, fairness, and reviewer robustness**.

---

## Fig. 1 — Qualitative Waveform Reconstruction Comparison  
**(Representative ECG Reconstruction Results)**

### Purpose

To visually demonstrate the **morphological reconstruction quality** of the proposed model by comparing reconstructed ECG waveforms with ground-truth signals.

This figure serves as a **qualitative sanity check**, not a statistical summary.

---

### Figure Structure

- A single figure containing **three subplots**, each corresponding to one test subject:
  - (a) Subject 1 — median-performance sample
  - (b) Subject 2 — median-performance sample
  - (c) Subject 3 — median-performance sample

Each subplot includes:
- Ground-truth ECG waveform (GT ECG)
- Reconstructed ECG waveform (Pred ECG)

---

### Representative Sample Selection (Median-Sample Strategy)

For each test subject:
1. Compute PCC for all test samples belonging to that subject
2. Select the sample whose PCC equals the **median PCC**
3. Use this sample for visualization

This strategy avoids cherry-picking and ensures representativeness.

---

### Data and Axes Definition

**Input data**
- `ECG_gt(t)` : ground-truth ECG waveform  
- `ECG_pred(t)` : reconstructed ECG waveform  
- `fs` : sampling rate (e.g., 200 Hz)

**Axes**
- x-axis: Time (seconds)  
  \[
  t = \frac{0, 1, \dots, N-1}{f_s}
  \]
- y-axis: Amplitude (normalized)

**Time window**
- Duration: 2–3 seconds
- Windows are aligned using **ground-truth R-peaks**
- The same alignment rule is applied to all subjects

---

## Fig. 2 — Time–Frequency Representation Comparison

### Purpose

To compare the **time–frequency characteristics** of:
- radar input,
- ground-truth ECG,
- reconstructed ECG,

and to visualize their discrepancies in the spectral domain.

---

### Figure Structure

Only **one subject** and **one representative (median) sample** are used  
(typically Subject 1, for consistency).

Subplots:
- (a) Radar input spectrogram
- (b) Ground-truth ECG spectrogram
- (c) Reconstructed ECG spectrogram
- (d) Difference map (|GT − Pred|)

---

### STFT Configuration (Must Be Explicitly Stated)

- Window length: 2 s
- Overlap: 50%
- FFT points: 256
- Sampling rate: consistent with ECG sampling rate

**Axes**
- x-axis: Time (s)
- y-axis: Frequency (Hz)
- Colorbar: Magnitude (dB or normalized scale)

---

## Fig. 3 — Subject-wise PCC Bar Plot

### Purpose

To visualize **subject-level reconstruction consistency** across unseen test subjects.

This figure provides **subject-wise aggregation**, complementary to sample-wise statistics.

---

### Plot Design

- Single figure with three bars (one per test subject)
- x-axis: Subject ID
- y-axis: PCC

**Statistics**
- Bar height: median PCC
- Error bar: interquartile range (IQR, 25–75 percentile)

---

## Fig. 4 — CDF Analysis of Reconstruction Errors

### Fig. 4(a): CDF of Mean Relative Error (MRE)

#### Data Source

- All test samples (segments/windows)
- Includes samples from **all three test subjects**
- No subject-wise averaging
- No median aggregation before CDF

Let:
\[
\{ \mathrm{MRE}_i \}_{i=1}^{N}
\]
where \( N \) is the total number of test samples.

---

#### MRE Definition

MRE is computed at the **waveform level**, consistent with Fig. 1 sample selection.

---

#### CDF Computation

1. Sort all MRE values in ascending order  
2. Construct cumulative probability:
\[
\mathrm{CDF}(x) = P(\mathrm{MRE} \le x)
\]

**Axes**
- x-axis: Mean Relative Error (MRE)
- y-axis: CDF

---

### Fig. 4(b): CDF of RR Interval Error

#### Data Source (Different from MRE)

- **Heartbeat-level data**
- For all test ECG segments:
  - Detect R-peaks (using ground-truth or a consistent rule)
  - Compute RR intervals
  - For each interval, compute timing error

Let:
\[
\{ \mathrm{RR\_err}_j \}_{j=1}^{M}
\]
where \( M \) is the total number of successfully detected RR intervals across all test subjects.

A single sample may contribute **multiple RR interval errors**.

---

#### RR Interval Error Definition

A commonly used definition:
\[
\mathrm{RR\_err} = | \mathrm{RR}_{\text{pred}} - \mathrm{RR}_{\text{gt}} |
\]

- Unit: milliseconds (ms)
- Undetected RR intervals may be discarded (must be stated in the paper)

---

#### CDF Computation

Same procedure as MRE:
1. Sort RR errors in ascending order
2. Construct:
\[
\mathrm{CDF}(x) = P(\mathrm{RR\_err} \le x)
\]

---

#### Plot Settings

- x-axis: RR Interval Error (ms)
  - Linear scale
  - Reasonable display range (e.g., 0–300 ms)
- y-axis: CDF

---

## Notes on Consistency

- All figures are generated under the **same subject-independent evaluation protocol**
- Statistical conventions are consistent with:
  - Main performance table
  - Subject-wise analysis
  - Supplementary ablation studies

This document serves as the **single source of truth** for all visualization-related decisions.
