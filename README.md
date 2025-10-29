# Cortex

Fast-weight memory for LLMs. Biological memory systems inspired this - think hippocampus meets transformer.

## What is this?

Cortex adds trainable "fast weights" to a frozen LLM. These let the model remember things beyond its normal context window. The memory decays over time but can be consolidated during sleep phases.

Main ideas:
- Base model stays frozen
- Only lightweight sidecars get trained
- Learns when to write via neuromodulation
- Sleep-based consolidation for long-term stability

The core trick is using low-rank matrices U and V per attention head. We never form the full product UV (that's expensive). Instead we do efficient factorized reads: `y_fast = V(U^T q)`.

## Quick Start

```bash
pip install torch>=2.0.0 transformers>=4.30.0

from base.hf_wrap import load_qwen_with_cortex, CortexWrapConfig

model = load_qwen_with_cortex(
    "Qwen/Qwen1.5-1.8B-Chat",
    cortex_cfg=CortexWrapConfig(rank_fast=16, decay=0.95)
)

# multi-turn with persistent memory
outputs = model(**inputs, session_id="chat_001", reset_session=True)
```

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Frozen Base LLM                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Transformer Layer 0                     │  │
│  │                                                       │  │
│  │  ┌──────────────┐         ┌──────────────────────┐  │  │
│  │  │  Self-Attn   │ ◄─────► │  CortexBlock         │  │  │
│  │  │  (frozen)    │         │  U[H,d,r], V[H,r,d]  │  │  │
│  │  │              │         │  (trainable)         │  │  │
│  │  └──────────────┘         └──────────────────────┘  │  │
│  │         │                           │                │  │
│  │         └─────── mixed output ──────┘                │  │
│  │                                                       │  │
│  └─────────────────────────────────────────────────────┘  │
│                           ⋮                                │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Transformer Layer N                     │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         ▲                                        │
         │                                        ▼
    ┌────────────┐                        ┌──────────────┐
    │ Controller │                        │ SessionState │
    │    MLP     │                        │  (optional)  │
    └────────────┘                        └──────────────┘
    surprise, phase                       persist U,V
    → plasticity gates                    across turns
```

### Per-Layer Detail

```
                  ┌──────────────────┐
                  │   hidden state   │
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌──────────────────┐    ┌─────────────────────┐
    │   Base Attn      │    │   CortexBlock       │
    │                  │    │                     │
    │  q = W_Q h       │───►│  Read:              │
    │  k = W_K h       │    │  k_u = U^T k        │
    │  v = W_V h       │    │  U += α·k·k_u^T     │
    │                  │    │  V += α·k_u·v^T     │
    │  attn(q,k,v)     │    │                     │
    │      │           │    │  y_fast = V(U^T q)  │
    │      │           │    │      │              │
    └──────┼───────────┘    └──────┼──────────────┘
           │                       │
           └───────┬───────────────┘
                   ▼
          y = σ(g)·y_fast + (1-σ(g))·v
                   │
                   ▼
           ┌───────────────┐
           │  next layer   │
           └───────────────┘
```

### Components

**CortexBlock** - the memory sidecar
- U: `[H, d_head, r]` 
- V: `[H, r, d_head]`
- Typical r=16 gives ~30% overhead
- Never materializes the full `d×d` product

**Controller** - decides when to write
- Tiny MLP: 4 inputs → 2 outputs
- Inputs: surprise, uncertainty, reward, phase
- Outputs: global gate + per-head scales

**SessionState** - cross-turn persistence
- Stores U and V between forward passes
- Optional for multi-turn conversations

## How It Works

### Fast-Weight Update

Each timestep updates the memory:

```python
# projections (shared with base model)
q, k, v = W_Q h, W_K h, W_V h

# decay old memories
U *= 0.95
V *= 0.95

# Hebbian learning
k_u = U.T @ k              # project key to rank-r space
U += alpha * outer(k, k_u)  # strengthen association
V += alpha * outer(k_u, v)  # store value

# anti-Hebbian stabilization
U -= 0.01 * clamp(U, -1, 1)
V -= 0.01 * clamp(V, -1, 1)

# read from memory (the key operation)
y_fast = V @ (U.T @ q)      # two cheap matmuls instead of expensive UV

# mix with slow path
gate = learned_gate(q, k, y_fast)
y = sigmoid(gate) * y_fast + (1 - sigmoid(gate)) * v
```

Why this is fast:
- Two `O(d·r)` operations instead of one `O(d²)`
- For r=16 and d=128 this is ~8x cheaper
- Memory usage stays small

### Controller

Small network that learns when to write:

```python
# compute surprise during training
loss_t = -log p(y_t | x)
ema = 0.99 * ema + 0.01 * loss_t
surprise = loss_t - ema

# also track uncertainty
uncertainty = entropy(p)

# controller decides plasticity
gates = MLP([surprise, uncertainty, reward, phase])
alpha = sigmoid(h) * gates.m_gate * gates.write_scale * 0.05

# safety: don't write when very uncertain
if uncertainty > threshold:
    alpha *= 0.5
```

During inference with no labels we use the model's own predictions to compute surprise - same mechanism works everywhere.

### Consolidation

Every 512 tokens we do a short "sleep" phase:

```python
# detect interesting segments via surprise
if loss - ema > 1.2 * ema:
    # surprise spike - start tracking
    segment_start = (U.clone(), V.clone())

if loss - ema < 1.05 * ema:
    # back to normal - compress and store
    stats = {
        'h_mean': mean(hiddens),
        'h_var': var(hiddens),
        'delta_U': U - segment_start[0],
        'delta_V': V - segment_start[1]
    }
    code = Encoder(stats)  # compress to 128-d vector
    save_for_replay(code, stats)

# during sleep
batch = sample_codes()
targets = load_stats(batch)
reconstructed = Decoder(batch)
loss = mse(reconstructed, targets) + fisher_penalty
# gradient descent on Cortex params only
```

The Fisher penalty keeps parameters from drifting too far from their pre-sleep values.

## Training Stages

We're building this in phases:

| Stage | Goal | Status |
|-------|------|--------|
| A1 | Fast-weight reach extension | Done |
| A2 | Learn segment codes | In progress |
| A3 | Sleep consolidation | Planned |
| A4 | Selective base updates | Planned |

### Synthetic Tasks

Testing memory with four task types:

**KV Binding**
```
HEADER: K1->XBQZ; K2->MFJK; K3->PLWT; ...
DISTRACTOR: (256-4096 tokens of noise)
QUERY: ASK K2?
ANSWER: MFJK
```

**Copy-Reverse**
```
PROMPT: <SEQ> 5 7 2 9 1 3 </SEQ>
DISTRACTOR: (noise)
QUERY: REVERSE <SEQ>
ANSWER: 3 1 9 2 7 5
```

**N-Back**
```
STREAM: A B C D A E [B] C D E ...
QUERY: NBACK 5 B
ANSWER: YES
```

**Long Addition**
```
CALC: + 41592653589793238462 64338327950288419716
DISTRACTOR: (noise)
QUERY: SUM?
ANSWER: 105930981540081658178
```

We vary the gap length and measure accuracy vs distance.

## Implementation

### Files

```
cortex-1/
├── base/
│   └── hf_wrap.py          # wrap HF models with sidecars
├── blocks/
│   ├── cortex_block.py     # U,V matrices + update logic
│   └── controller.py       # plasticity gates
├── mem/
│   ├── session.py          # cross-turn state
│   ├── chunker.py          # segment detection
│   ├── codes.py            # compression encoder/decoder
│   ├── consolidation.py    # sleep coordinator
│   ├── fisher.py           # EWC penalties
│   └── fast_weights.py     # utilities
└── train/
    ├── data_long.py        # synthetic tasks
    └── objectives.py       # loss functions
```

### Integration

We use PyTorch hooks to inject sidecars without modifying the base model:

```python
# attach to each layer
for layer in model.layers:
    cortex = CortexBlock(...)
    cortex.tie_projections(layer.self_attn.q_proj, ...)
    
    # restore U,V before forward
    layer.register_forward_pre_hook(restore_fast_weights)
    
    # override forward to include sidecar
    original_forward = layer.forward
    layer.forward = lambda h: original_forward(h) + cortex(h, gates)
```

The base model never knows Cortex exists. QKV projections are shared so we don't duplicate parameters.

### Cost

For r=16 on Qwen-1.8B:
- Adds 2.5M parameters (0.14% increase)
- Wall-clock ~30% slower
- Memory overhead minimal

Could get to ~20% with fused CUDA kernels.

## Usage

### Training

```bash
python scripts/stage_a1_enable_fast.py \
    --model Qwen/Qwen1.5-1.8B-Chat \
    --task kv \
    --gaps 256,512,1024,2048 \
    --epochs 2 \
    --save_dir ./checkpoints
```

### Loading

```python
from base.hf_wrap import load_qwen_with_cortex, CortexWrapConfig

cortex_cfg = CortexWrapConfig(
    rank_fast=16,
    decay=0.95,
    alpha_max=0.05,
    beta=0.01,
)

model = load_qwen_with_cortex(
    "Qwen/Qwen1.5-1.8B-Chat",
    cortex_cfg=cortex_cfg,
    torch_dtype=torch.float16,
)

# freeze base - train only sidecars
for param in model.base.parameters():
    param.requires_grad = False
```

### Multi-turn Sessions

```python
session_id = "conversation_42"

# turn 1
out1 = model(**inputs1, session_id=session_id, reset_session=True)

# turn 2 - U,V carry forward
out2 = model(**inputs2, session_id=session_id)

# turn 3
out3 = model(**inputs3, session_id=session_id)
```

## Next Steps

### High Priority

**Performance optimization**
- Fused CUDA kernels for update + read
- Profile memory bandwidth
- Try int8 quantization for U,V

**Real benchmarks**
- LongBench (NarrativeQA)
- L-Eval
- PersonaChat with long history
- Compare to RMT and Memorizing Transformers

**Safety features**
- Time-to-live for segments
- PII detection in write gates
- Measure memorization vs baseline

### Medium Term

**Finish training stages**
- A2: segment compression
- A3: sleep consolidation
- A4: selective base unfreezing

**Scale up**
- Test on Llama-70B
- Multi-GPU coordination
- Efficient distributed training

### Long Term

- Adaptive rank per layer
- Multi-timescale decay
- Extension to vision-language models
- Production deployment

## Technical Details

### Math

The key insight is factorization:

```
DON'T DO THIS (slow):
    K_fast = U @ V                    # [d×d] materialization
    y_fast = K_fast @ q               # [d×d] @ [d]
    Cost: O(d²·r) + O(d²) = disaster

DO THIS (fast):
    temp = U.T @ q                    # [r] 
    y_fast = V @ temp                 # [d]
    Cost: O(d·r) + O(d·r) = great
```

Both give the same answer but one is 10x faster.

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| r | 16 | sweet spot for capacity vs speed |
| γ (decay) | 0.95 | balances retention and stability |
| α_max | 0.05 | max learning rate |
| β (anti-Hebb) | 0.01 | prevents saturation |

Going to r=32 helps a bit but costs more. Going to r=8 is too small.

Decay rate matters a lot:
- 0.90: faster forgetting but stable
- 0.95: good balance
- 0.98: remembers longer but can be unstable
- 1.00: no forgetting → saturates and breaks

## References

Biological inspiration:
- McClelland et al. (1995) - complementary learning systems
- O'Reilly & Norman (2002) - hippocampus/cortex

Fast weights:
- Schmidhuber (1992) - original fast weight idea
- Ba et al. (2016) - using fast weights to attend to the recent past
- Schlag et al. (2021) - linear transformers with fast weights

Continual learning:
- Kirkpatrick et al. (2017) - EWC
- Zenke et al. (2017) - synaptic intelligence

Memory networks:
- Graves et al. (2014) - Neural Turing Machines
- Sukhbaatar et al. (2015) - End-to-end memory networks

## License

MIT

---

**Status**: Early research code. Stage A1 working. A2-A4 in development.
