#!/bin/bash

# Move to the project code directory
cd "$(dirname "$0")/../code"

# ------------------ SETUP -----------------------
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_visa_paladin_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Activate virtual environment if available. Override with PALADIN_VENV=/path/to/env.
VENV_PATH="${PALADIN_VENV:-../.venv}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "[INFO] No virtualenv found at ${VENV_PATH}. Using current shell environment."
fi

DATASET_PATH="${PALADIN_VISA_PATH:-./data/visa}"

# ------------------ TRAINING --------------------
# Manually set GPU devices if needed (e.g., for multi-GPU training)
# export CUDA_VISIBLE_DEVICES=0,1

deepspeed --master_port 28400 train_visa.py \
    --model model_paladin \
    --stage 1 \
    --save_path ./visa_paladin/train_visa/ \
    --log_path ./visa_paladin/train_visa/log_rest/ \
    --dataset_path "${DATASET_PATH}"
