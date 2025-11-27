#!/bin/bash
set -e

# 1. Install Dependencies
echo "Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets einops pyyaml hf_transfer

# 2. Unzip Code (if not already unzipped)
# Assuming the user unzips manually or we do it here if the zip exists
if [ -f *.zip ]; then
    echo "Unzipping project..."
    unzip -o *.zip
fi

# 3. Run Training (6 GPUs)
echo "Starting Infinite Proof Training (Gap 35k)..."
# Auto-detect number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs."

PYTHONPATH=. torchrun --nproc_per_node=$NUM_GPUS scripts/stage_a1_ddp.py \
    --model Qwen/Qwen1.5-0.5B-Chat \
    --task kv \
    --gaps 35000 \
    --samples_per_gap 1000 \
    --batch_size 4 \
    --epochs 10 \
    --save_dir ./checkpoints \
    --log_dir ./logs \
    --fast_rank 32 \
    --amp true \
    --chunk_size 512

echo "Training Complete! Download the logs/ folder to see results."
