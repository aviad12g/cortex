#!/bin/bash
# RunPod Setup Script for Cortex Stage A1

set -e

echo "=== Cortex-5 RunPod Setup ==="

# Navigate to workspace
cd /workspace

# Install dependencies
echo "Installing dependencies..."
pip install -q torch transformers>=4.30.0 accelerate sentencepiece pyyaml matplotlib numpy

# Clone or upload your code
echo "Upload your cortex-5 folder to /workspace/cortex-5"
echo "Or run: git clone YOUR_REPO_URL cortex-5"
echo ""
echo "Then run this training command:"
echo ""
echo "cd /workspace/cortex-5"
echo "export PYTHONPATH=/workspace/cortex-5"
echo ""
echo "python scripts/stage_a1_enable_fast.py \\"
echo "    --model mistralai/Mistral-7B-Instruct-v0.1 \\"
echo "    --task kv \\"
echo "    --gaps 512 1024 2048 4096 \\"
echo "    --batch_size 2 \\"
echo "    --epochs 2 \\"
echo "    --save_dir /workspace/checkpoints \\"
echo "    --log_dir /workspace/logs/a1 \\"
echo "    --amp true \\"
echo "    --samples_per_gap 256 \\"
echo "    --seed 42"
echo ""
echo "Training will take ~2-3 hours on RTX 5090!"

