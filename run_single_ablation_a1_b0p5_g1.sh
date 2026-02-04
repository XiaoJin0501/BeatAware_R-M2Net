#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# Single ablation pipeline:
#   alpha=1.0, beta=0.5, gamma=1.0
#   train -> test -> plot (Fig.1–Fig.4)
# ==========================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

# --- fixed hyperparams for this run ---
ALPHA="1.0"
BETA="0.5"
GAMMA="1.0"
EXP_TAG="PaperAblation_v1_Lbeta0p5_gamma1p0"

# --- derived experiment name (must match Config.update_paths naming) ---
EXP_NAME="Exp_a${ALPHA}_b${BETA}_g${GAMMA}_${EXP_TAG}"
EXP_DIR="${ROOT}/experiments/${EXP_NAME}"
RESULT_DIR="${EXP_DIR}/results"

echo "============================================================"
echo "[PIPELINE] ROOT      : ${ROOT}"
echo "[PIPELINE] EXP_TAG   : ${EXP_TAG}"
echo "[PIPELINE] EXP_NAME  : ${EXP_NAME}"
echo "[PIPELINE] EXP_DIR   : ${EXP_DIR}"
echo "[PIPELINE] RESULTDIR : ${RESULT_DIR}"
echo "============================================================"

# 0) basic check
if [[ ! -f "${ROOT}/train.py" || ! -f "${ROOT}/test.py" || ! -f "${ROOT}/tools/plot_figures.py" ]]; then
  echo "[ERROR] Please run this script under the project root (BeatAware_R-M2Net)."
  exit 1
fi

# 1) Train
echo ""
echo "====================== TRAIN ======================"
python train.py \
  --alpha "${ALPHA}" \
  --beta "${BETA}" \
  --gamma "${GAMMA}" \
  --exp_tag "${EXP_TAG}"

# 2) Test (export protocol + save cases for Fig.1 & Fig.2)
echo ""
echo "======================= TEST ======================"
python test.py \
  --exp_tag "${EXP_TAG}" \
  --alpha "${ALPHA}" \
  --beta "${BETA}" \
  --gamma "${GAMMA}" \
  --ckpt best \
  --save_cases

# 3) Plot (Fig.1–Fig.4)
# NOTE: avoid STFT error: ensure nfft >= nperseg = win_s * fs = 2.0 * 200 = 400
# so use nfft=512 (safe)
echo ""
echo "======================= PLOT ======================"
python tools/plot_figures.py \
  --result_dir "${RESULT_DIR}" \
  --n_fig1_subjects 3 \
  --stft_win_s 2.0 \
  --stft_overlap 0.5 \
  --stft_nfft 512 \
  --fmax 40

echo ""
echo "✅ DONE. All outputs are under:"
echo "   ${EXP_DIR}"
echo "   - checkpoints/: *.pth"
echo "   - results/: segment_metrics.csv / beat_metrics.csv / subject_summary.csv / global_summary.json / meta.json"
echo "   - results/figures/: Fig1–Fig4 (png/pdf)"
