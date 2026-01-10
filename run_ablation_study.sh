#!/bin/bash
# ==============================================================================
# BeatAware_R-M2Net: Orthogonal Ablation (2^3) + Versioned exp_tag
# Pipeline: train -> test -> analyze_results -> plot_figures
# ==============================================================================

set -euo pipefail

# -----------------------------
# User knobs
# -----------------------------
MIN_FREE_MB="${MIN_FREE_MB:-14500}"     # gpu_waiter min free mem (MB)
DO_POWEROFF="${DO_POWEROFF:-0}"         # 1: poweroff after finishing
RUN_NOTE="${RUN_NOTE:-Ablation2x3}"     # extra tag note (optional)

# Optional: if you want to pin python explicitly
PY="${PY:-python}"

# -----------------------------
# Run identity (versioned exp_tag prefix)
# -----------------------------
RUN_ID="$(date +%Y%m%d_%H%M%S)"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
TAG_PREFIX="V${RUN_ID}_${GIT_SHA}_${RUN_NOTE}"

echo "============================================================"
echo "[RUN] TAG_PREFIX  = ${TAG_PREFIX}"
echo "[RUN] MIN_FREE_MB = ${MIN_FREE_MB}"
echo "[RUN] DO_POWEROFF = ${DO_POWEROFF}"
echo "============================================================"

# -----------------------------
# Orthogonal design: 2^3
# alpha: STFT {0, 0.5}
# beta : Anchor {0, 1.0}
# gamma: Smooth {0, 0.5}
# -----------------------------
experiments=(
  "0.0 0.0 0.25 L1"
  "0.5 0.0 0.25 L1+STFT"
  "0.0 1.0 0.25 L1+Anchor"
  "0.5 1.0 0.25 L1+STFT+Anchor"
  "0.8 0.8 0.25 Full(STFT+Anchor+Smooth)"
)

# -----------------------------
# Helpers
# -----------------------------
run_train () {
  local alpha="$1"; local beta="$2"; local gamma="$3"; local tag="$4"
  echo "----------------------------------------------------------------"
  echo "[TRAIN] ${tag} (alpha=${alpha}, beta=${beta}, gamma=${gamma})"
  echo "----------------------------------------------------------------"

  # IMPORTANT: pass as argv list (do not pack into a single string)
  # gpu_waiter.py will run your command only when GPU has enough free mem.
  ${PY} gpu_waiter.py \
    ${PY} train.py --alpha "${alpha}" --beta "${beta}" --gamma "${gamma}" --exp_tag "${tag}"
}

run_test () {
  local alpha="$1"; local beta="$2"; local gamma="$3"; local tag="$4"
  echo "----------------------------------------------------------------"
  echo "[TEST ] ${tag} (alpha=${alpha}, beta=${beta}, gamma=${gamma})"
  echo "----------------------------------------------------------------"
  ${PY} test.py --alpha "${alpha}" --beta "${beta}" --gamma "${gamma}" --exp_tag "${tag}"
}

# -----------------------------
# Main loop
# -----------------------------
for exp in "${experiments[@]}"; do
  read -r ALPHA BETA GAMMA SHORT <<< "$exp"
  EXP_TAG="${TAG_PREFIX}__${SHORT}"

  run_train "${ALPHA}" "${BETA}" "${GAMMA}" "${EXP_TAG}"
  run_test  "${ALPHA}" "${BETA}" "${GAMMA}" "${EXP_TAG}"

  echo "✅ Finished: ${EXP_TAG}"
done

# -----------------------------
# Analyze across this run only
# -----------------------------
echo "============================================================"
echo "[ANALYZE] Collecting results for this run (pattern=${TAG_PREFIX})"
echo "============================================================"

# analyze_results.py will scan experiments/ and aggregate per-exp global_summary.json + subject_summary.csv
# Use a pattern to only include experiments from this run.
${PY} analyze_results.py \
  --experiments_dir "experiments" \
  --pattern "Exp_a*_b*_g*_${TAG_PREFIX}__*" \
  --out_dir "experiments/_analysis/${TAG_PREFIX}"

# -----------------------------
# Plot figures
# -----------------------------
echo "============================================================"
echo "[PLOT] Generating ablation figures into experiments/_analysis/${TAG_PREFIX}/figures"
echo "============================================================"

${PY} tools/plot_figures.py \
  --exp_root "experiments" \
  --out_dir "experiments/_analysis/${TAG_PREFIX}/figures" \
  --mode "ablation" \
  --do_ba

echo "🎉 All done!"
echo "👉 Analysis output: experiments/_analysis/${TAG_PREFIX}"
echo "👉 Figures output : experiments/_analysis/${TAG_PREFIX}/figures"

# -----------------------------
# Optional poweroff
# -----------------------------
if [[ "${DO_POWEROFF}" == "1" ]]; then
  echo "⚠️ DO_POWEROFF=1 -> powering off now."
  sudo poweroff
fi
