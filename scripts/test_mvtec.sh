#!/bin/bash

# Move to the project code directory
cd "$(dirname "$0")/../code"

# ------------------ USER INPUT -------------------
METHOD=$1  # E.g., 'paladin' or 'paladintopk'
if [ -z "$METHOD" ]; then
  echo "Usage: $0 <method>"
  exit 1
fi

# ------------------ SETUP -----------------------
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/mvtec_${METHOD}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Activate virtual environment if available. Override with PALADIN_VENV=/path/to/env.
VENV_PATH="${PALADIN_VENV:-../.venv}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "[INFO] No virtualenv found at ${VENV_PATH}. Using current shell environment."
fi

DATASET_PATH="${PALADIN_MVTEC_PATH:-./data/mvtec}"

# ------------------ JOB LOOP --------------------
for k_shot in 1 2 4; do
  for round in 1 2 3 4 5; do
    echo "==============================="
    echo "Running METHOD: ${METHOD}"
    echo "k_shot: ${k_shot}, round: ${round}"
    echo "==============================="

    python evaluate_mvtec.py \
        --few_shot True \
        --k_shot ${k_shot} \
        --round ${round} \
        --method ${METHOD} \
        --dataset_path "${DATASET_PATH}" \
        --paladin_ckpt_path ./mvtec_paladin/train_mvtec/epoch_10/pytorch_model.pt
  done
done
