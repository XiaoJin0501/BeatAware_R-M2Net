#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# BeatAware R-M2Net: Protocol + Loss Ablation Pipeline (L0-L3)
#   For each variant:
#     1) train
#     2) test (protocol exports + cases)
#     3) plot Fig.1–Fig.4
# ============================================================

# ---------------------------
# User-configurable inputs
# ---------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/Projects/BeatAware_R-M2Net}"

# If you pass a tag prefix, final tags will be:
#   <TAG_PREFIX>_L0_TimeOnly
#   <TAG_PREFIX>_L1_TimeSpectral
#   <TAG_PREFIX>_L2_TimeAnchor
#   <TAG_PREFIX>_L3_Full
TAG_PREFIX="${1:-ProtocolAbl_v1}"

CKPT_MODE="${2:-best}"  # best | last | path
N_FIG1_SUBJECTS="${N_FIG1_SUBJECTS:-3}"

RUN_VERIFY="${RUN_VERIFY:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_PLOT="${RUN_PLOT:-1}"

# ---------------------------
echo "============================================================"
echo "[PIPELINE] TAG_PREFIX      = ${TAG_PREFIX}"
echo "[PIPELINE] CKPT_MODE       = ${CKPT_MODE}"
echo "[PIPELINE] PROJECT_DIR     = ${PROJECT_DIR}"
echo "[PIPELINE] RUN_VERIFY      = ${RUN_VERIFY}"
echo "[PIPELINE] RUN_TRAIN       = ${RUN_TRAIN}"
echo "[PIPELINE] RUN_TEST        = ${RUN_TEST}"
echo "[PIPELINE] RUN_PLOT        = ${RUN_PLOT}"
echo "[PIPELINE] N_FIG1_SUBJECTS = ${N_FIG1_SUBJECTS}"
echo "============================================================"

cd "${PROJECT_DIR}"

python -V
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
PY

# ---------------------------
# Detect whether train.py supports --alpha/--beta/--gamma
# ---------------------------
TRAIN_SUPPORTS_ABLATION=0
if python train.py -h 2>/dev/null | grep -qE "\-\-alpha|\-\-beta|\-\-gamma"; then
  TRAIN_SUPPORTS_ABLATION=1
fi
echo "[CHECK] train.py supports --alpha/--beta/--gamma ? ${TRAIN_SUPPORTS_ABLATION}"

if [[ "${TRAIN_SUPPORTS_ABLATION}" != "1" ]]; then
  echo ""
  echo "❌ [ERROR] Your train.py does NOT expose --alpha/--beta/--gamma."
  echo "    This ablation pipeline requires CLI override of loss weights."
  echo ""
  echo "    Minimal fix (recommended): add CLI args in train.py:"
  echo "      parser.add_argument('--alpha', type=float, default=Config.ALPHA)"
  echo "      parser.add_argument('--beta',  type=float, default=Config.BETA)"
  echo "      parser.add_argument('--gamma', type=float, default=Config.GAMMA)"
  echo "    then after parsing: set Config.ALPHA/BETA/GAMMA = args.*"
  echo ""
  echo "    After you add those, rerun this script."
  exit 1
fi

# ---------------------------
# Read defaults from config.py (for reporting + gamma default)
# ---------------------------
read_defaults() {
python - <<'PY'
from config import Config
print(float(getattr(Config,'ALPHA',1.0)))
print(float(getattr(Config,'BETA',1.0)))
print(float(getattr(Config,'GAMMA',1.0)))
PY
}

DEFAULT_ALPHA="$(read_defaults | sed -n '1p')"
DEFAULT_BETA="$(read_defaults  | sed -n '2p')"
DEFAULT_GAMMA="$(read_defaults | sed -n '3p')"

echo "[DEFAULT] Config.ALPHA=${DEFAULT_ALPHA}, Config.BETA=${DEFAULT_BETA}, Config.GAMMA=${DEFAULT_GAMMA}"

# ---------------------------
# (Optional) Verify alignment on H5
# ---------------------------
if [[ "${RUN_VERIFY}" == "1" ]]; then
  echo ""
  echo "================ [VERIFY] H5 alignment checks ================"
  python tools/verify_radar_ecg_xcorr_lag.py || true
  python tools/verify_mask_axis_on_h5_v2.py || true
  echo "============================================================"
fi

# ---------------------------
# Define ablation variants (L0-L3)
#   L0: Time only        -> alpha=1, beta=0, gamma=0
#   L1: Time + Spectral  -> alpha=1, beta=1, gamma=0
#   L2: Time + Anchor    -> alpha=1, beta=0, gamma=DEFAULT_GAMMA
#   L3: Full (Ours)      -> alpha=1, beta=1, gamma=DEFAULT_GAMMA
# ---------------------------
declare -a VAR_NAMES=("L0_TimeOnly" "L1_TimeSpectral" "L2_TimeAnchor" "L3_Full")
declare -a ALPHAS=(1.0 1.0 1.0 1.0)
declare -a BETAS=(0.0 1.0 0.0 1.0)
declare -a GAMMAS=(0.0 0.0 "${DEFAULT_GAMMA}" "${DEFAULT_GAMMA}")

# ---------------------------
# Helper: locate newest result_dir for a given exp_tag
# ---------------------------
locate_result_dir() {
  local tag="$1"
  local p
  p="$(find "${PROJECT_DIR}" -type f -name meta.json 2>/dev/null | grep "${tag}" | xargs -r ls -t 2>/dev/null | head -n 1 | xargs -r dirname || true)"
  echo "${p}"
}

# ---------------------------
# Run all variants
# ---------------------------
for i in "${!VAR_NAMES[@]}"; do
  VNAME="${VAR_NAMES[$i]}"
  A="${ALPHAS[$i]}"
  B="${BETAS[$i]}"
  G="${GAMMAS[$i]}"
  EXP_TAG="${TAG_PREFIX}_${VNAME}"

  echo ""
  echo "============================================================"
  echo "[RUN] Variant: ${VNAME}"
  echo "[RUN] EXP_TAG : ${EXP_TAG}"
  echo "[RUN] alpha=${A}, beta=${B}, gamma=${G}"
  echo "============================================================"

  # ---- Train ----
  if [[ "${RUN_TRAIN}" == "1" ]]; then
    echo ""
    echo "---------------------- TRAIN: ${EXP_TAG} ---------------------"
    python train.py --exp_tag "${EXP_TAG}" --alpha "${A}" --beta "${B}" --gamma "${G}"
    echo "--------------------------------------------------------------"
  fi

  # ---- Test ----
  if [[ "${RUN_TEST}" == "1" ]]; then
    echo ""
    echo "---------------------- TEST: ${EXP_TAG} ----------------------"
    python test.py \
      --ckpt "${CKPT_MODE}" \
      --exp_tag "${EXP_TAG}" \
      --alpha "${A}" \
      --beta  "${B}" \
      --gamma "${G}" \
      --save_cases

    echo "--------------------------------------------------------------"
  fi

  # ---- Locate result dir ----
  RESULT_DIR="$(locate_result_dir "${EXP_TAG}")"
  if [[ -z "${RESULT_DIR}" ]]; then
    echo "❌ [ERROR] Cannot locate result_dir for EXP_TAG=${EXP_TAG}."
    echo "    Please check whether test.py produced meta.json under experiments/."
    exit 1
  fi
  echo "[OK] RESULT_DIR = ${RESULT_DIR}"

  # ---- Plot ----
  if [[ "${RUN_PLOT}" == "1" ]]; then
    echo ""
    echo "---------------------- PLOT: ${EXP_TAG} ----------------------"
    python tools/plot_figures.py \
      --result_dir "${RESULT_DIR}" \
      --n_fig1_subjects "${N_FIG1_SUBJECTS}"
    echo "--------------------------------------------------------------"
  fi

done

echo ""
echo "✅ ALL ABLATION VARIANTS FINISHED."
echo "   You should now have 4 experiment folders under experiments/ with their own results + figures."
