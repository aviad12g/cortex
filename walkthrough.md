# Cortex v2 Sleep Phase Debugging & Verification

## Objective
Resolve `NaN` loss issues occurring after the Sleep (Memory Consolidation) phase and initiate a stable full training run for Cortex v2.

## Issues Resolved

### 1. NaN Loss After Sleep
*   **Symptom**: Training loss became `NaN` immediately or shortly after the sleep phase.
*   **Root Cause**: Exploding gradients in the `GatedDeltaCore` or `CortexBlock` during the sleep optimization steps, likely due to random "dream" inputs causing unstable updates to the fast weight matrix `S`.
*   **Fix**:
    *   Implemented **Gradient Clipping** (`torch.nn.utils.clip_grad_norm_`) in `train/sleep.py`.
    *   Added **NaN Loss Detection**: The sleep loop now checks `if torch.isnan(loss):` and skips the update step if detected, preventing corruption of the model parameters.
    *   Ensured `optimizer` and `scaler` are correctly initialized and used within the sleep function.

### 2. `AttributeError` & `NameError` in `HybridCortexBlock`
*   **Symptom**: Crashes due to missing `reset_fast` method and incorrect class ordering.
*   **Fix**:
    *   Implemented `reset_fast(self, batch_size, device)` in `HybridCortexBlock` to correctly clear the state `S`.
    *   Reordered classes in `blocks/hybrid_cortex.py` to ensure `SlidingWindowAttention` and `GatedDeltaCore` are defined before `HybridCortexBlock`.

### 3. Dtype Mismatches
*   **Symptom**: `RuntimeError: mat1 and mat2 must have the same dtype` during evaluation.
*   **Fix**: Wrapped model inference in `evaluate_probe_perplexity` and `run_sleep_phase` with `torch.amp.autocast`.

## Verification Results

### Sleep Phase Stability Test (`sleep_test_retry_8`)
*   **Configuration**: Sleep interval set to 5 steps for rapid testing.
*   **Outcome**:
    *   Sleep phase triggered successfully.
    *   **No NaNs detected** during sleep (Loss ~15.3).
    *   **Post-Sleep Training**: Model resumed training with **valid loss** (e.g., 5.41, 2.10, 0.57) for 30+ steps.
    *   Confirmed that the model parameters remain healthy after consolidation.

### Full Training Run (`a1_kv_Qwen1.5-0_cortex_20251119_104948`)
*   **Status**: **RUNNING**
*   **Command**:
    ```powershell
    python scripts/stage_a1_enable_fast.py --model Qwen/Qwen1.5-0.5B-Chat --task kv --gaps 128 256 512 1024 --batch_size 8 --epochs 1 --save_dir ./checkpoints --log_dir ./logs/full_run_final --samples_per_gap 512 --fast_rank 64 --amp true --chunk_size 512
    ```
*   **Progress**: ~511/2048 samples processed (as of latest check).
*   **Metrics**:
    *   `loss_tok`: Valid and decreasing (e.g., ~2.0 - 8.0 range depending on difficulty).
    *   `m_gate_mean`: Shows `NaN` in logs (benign logging artifact; global controller is unused by `HybridCortexBlock`).
*   **Sleep Config**: Interval reverted to 512 steps (default). Note: With 256 total steps per epoch, sleep may not trigger in this specific 1-epoch run, but the mechanism is verified.

## Next Steps
1.  Monitor the full training run to completion.
2.  Inspect `drift.jsonl` after the run to verify performance on the KV task.
3.  If longer training is desired to utilize Sleep Phase, increase `--epochs` or reduce `--sleep_interval`.
