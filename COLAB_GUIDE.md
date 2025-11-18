# Google Colab Training Guide

## Quick Start (Recommended for Memory Issues)

If you're hitting OOM on local hardware, use Google Colab with free GPU.

### Steps:

1. **Prepare your code**
   ```bash
   cd /Users/mazalcohen
   zip -r cortex-4.zip cortex-4/ -x "cortex-4/.git/*" "cortex-4/logs/*" "cortex-4/checkpoints/*"
   ```

2. **Open Google Colab**
   - Go to: https://colab.research.google.com
   - Upload the notebook: `cortex-4/colab_a1_training.ipynb`

3. **Enable GPU**
   - Runtime → Change runtime type → Hardware accelerator → **T4 GPU**
   - Click Save

4. **Upload your code**
   - In Colab, click the folder icon on the left
   - Upload `cortex-4.zip`

5. **Run the cells**
   - Run each cell in order (Shift+Enter or click Play button)
   - The notebook will auto-configure based on GPU memory

### Expected Timeline:

| GPU Type | Memory | Training Time | Config |
|----------|--------|---------------|--------|
| T4 (free)| 16GB   | ~1-2 hours    | 3 gaps, 256 samples |
| V100     | 32GB   | ~45 min       | 4 gaps, 512 samples |
| A100     | 40GB   | ~30 min       | Full config |

### What You'll Get:

- Trained checkpoint files
- Training logs (`probes.jsonl`, `drift.jsonl`)
- Accuracy metrics by gap length
- Fast-weight usage statistics
- Downloadable results ZIP

---

## Alternative: Minimal Local Training (RTX 4060 / M1)

If you want to run locally despite memory constraints:

```bash
cd /Users/mazalcohen/cortex-4
export PYTHONPATH=/Users/mazalcohen/cortex-4

python scripts/stage_a1_enable_fast.py \
    --model Qwen/Qwen1.5-1.8B-Chat \
    --task kv \
    --gaps 256 512 \
    --batch_size 1 \
    --epochs 1 \
    --save_dir ./checkpoints \
    --log_dir ./logs/a1 \
    --amp true \
    --samples_per_gap 32 \
    --seed 42 \
    --fast_rank 8
```

**Memory savings:**
- Shorter gaps (256, 512 instead of 1024+)
- Batch size 1 (no batching)
- Fewer samples (32 instead of 256)
- Smaller rank (8 instead of 16)
- AMP enabled (float16)

This will train but with limited testing of long-range memory.

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `--gaps` further (try just 128 256)
- Set `--fast_rank 4` (minimal rank)
- Reduce `--samples_per_gap 16`

### "No module named 'blocks'"
- Ensure you're in the cortex-4 directory
- Check `PYTHONPATH` is set correctly
- Verify all folders exist (base/, blocks/, mem/, train/)

### "Model download failed"
- Poor internet connection - try again
- Use alternative: `--model Qwen/Qwen1.5-0.5B-Chat` (smaller)

### Colab disconnects
- Free tier has usage limits (~12 hours per day)
- Save checkpoints frequently
- Download results after each epoch

---

## Next Steps After A1

Once A1 completes successfully:

1. **Analyze results**: Check accuracy vs gap length
2. **Stage A2**: Implement segment code training (currently not implemented)
3. **Stage A3**: Add sleep consolidation
4. **Stage A4**: Selective base model unfreezing

See `README.md` for full architecture details.


