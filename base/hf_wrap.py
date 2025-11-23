"""Wrap HF models with Cortex sidecars. Currently supports Qwen2."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from blocks.cortex_block import CortexBlock, CortexBlockConfig
from blocks.hybrid_cortex import HybridCortexBlock
from blocks.controller import ControllerConfig, CortexController
from mem.fast_weights import allocate_fast_buffers
from mem.session import SessionState

try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
except Exception:
    Qwen2DecoderLayer = None


@dataclass
class CortexWrapConfig:
    rank_fast: int = 64
    decay: float = 0.95
    alpha_max: float = 0.05
    beta: float = 0.01
    code_size: int = 128
    sleep_interval: int = 512
    sleep_interval: int = 512
    sleep_steps: int = 8
    use_hybrid: bool = True
    window_size: int = 128


class CortexWrappedModel(nn.Module):
    """Wraps a base LLM + injects Cortex sidecars via hooks."""

    def __init__(self, base: nn.Module, config: CortexWrapConfig):
        super().__init__()
        self.base = base
        self.config = config
        self.controller = CortexController(
            ControllerConfig(d_model=base.config.hidden_size)
        )
        self._cortex_layers: Sequence[nn.Module] = ()
        self.sessions: Dict[str, SessionState] = {}
        self._active_session: Optional[SessionState] = None
        self._gate_cache = None
        self._controller_inputs_cache = None
        self._last_gates = None
        self.gate_logs = []
        self._default_dtype = next(self.base.parameters()).dtype
        self._mix_mode = "dual"
        self._alpha_override = None
        self._sidecar_enabled = True

    def forward(self, *args, **kwargs):
        session_id = kwargs.pop("session_id", None)
        reset_session = kwargs.pop("reset_session", False)

        session_state: Optional[SessionState] = None
        if session_id is not None:
            session_state = self.sessions.get(session_id)
            if session_state is None:
                session_state = SessionState(session_id=session_id)
                self.sessions[session_id] = session_state
            if reset_session:
                session_state.reset()
            self._active_session = session_state
        else:
            self._active_session = None

        batch_size, seq_len, device = self._extract_batch_seq_device(
            *args, **kwargs
        )
        self.gate_logs.clear()
        self._controller_inputs_cache = None
        self._last_gates = None
        self._prepare_gates(batch_size, seq_len, device)

        outputs = self.base(*args, **kwargs)

        if session_state is not None:
            if hasattr(outputs, "logits") and outputs.logits is not None:
                # Calculate simple metrics for the controller
                # We use detached logits to avoid affecting the computation graph
                logits_det = outputs.logits.detach().float()
                probs = torch.softmax(logits_det, dim=-1)
                log_probs = torch.log_softmax(logits_det, dim=-1)
                # Entropy = -sum(p * log_p)
                entropy = (
                    -(probs * log_probs).sum(dim=-1).mean().item()
                )

                loss_val = None
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss_val = outputs.loss.item()

                session_state.update_metrics(
                    surprise=loss_val, uncertainty=entropy
                )

            self._persist_fast_buffers(session_state)
            session_state.advance(seq_len)

        self._active_session = None
        self._gate_cache = None
        return outputs

    def reset_cortex_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> None:
        allocate_fast_buffers(
            self.base, batch_size, device or next(self.parameters()).device
        )

    def flush_sessions(self) -> None:
        """Clear all stored session states to prevent memory leaks."""
        self.sessions.clear()
        self._active_session = None

    def cortex_parameters(self) -> Iterable[nn.Parameter]:
        # return only Cortex params (not base model)
        seen = set()
        for module in self.modules():
            if getattr(module, "is_cortex_param", False):
                for param in module.parameters():
                    if not param.requires_grad:
                        continue
                    param_id = id(param)
                    if param_id not in seen:
                        seen.add(param_id)
                        yield param

    def set_mix_mode(self, mode: str) -> None:
        self._mix_mode = mode

    def set_alpha_override(self, value: Optional[float]) -> None:
        self._alpha_override = value

    def enable_sidecar(self, enabled: bool) -> None:
        self._sidecar_enabled = enabled

    def _extract_batch_seq_device(
        self, *args, **kwargs
    ) -> Tuple[int, int, torch.device]:
        tensor = None
        if "input_ids" in kwargs and kwargs["input_ids"] is not None:
            tensor = kwargs["input_ids"]
        elif args:
            candidate = args[0]
            if isinstance(candidate, torch.Tensor):
                tensor = candidate
        if tensor is None and kwargs.get("inputs_embeds") is not None:
            tensor = kwargs["inputs_embeds"]
        if tensor is None:
            raise ValueError(
                "CortexWrappedModel requires input_ids or inputs_embeds."
            )

        if tensor.dim() == 3:
            batch_size, seq_len = tensor.shape[0], tensor.shape[1]
        else:
            batch_size, seq_len = tensor.shape[0], tensor.shape[1]
        return batch_size, seq_len, tensor.device

    def _prepare_gates(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> None:
        if seq_len == 0:
            m_gate = torch.zeros(
                batch_size, 0, device=device, dtype=self._default_dtype
            )
            alpha_scale = torch.zeros(
                batch_size,
                0,
                self.base.config.num_attention_heads,
                device=device,
                dtype=self._default_dtype,
            )
            self._gate_cache = (m_gate, alpha_scale)
            return

        session = self._active_session
        if session is None:
            base = torch.zeros(
                batch_size,
                seq_len,
                1,
                device=device,
                dtype=self._default_dtype,
            )
            controller_inputs = {
                "surprise": base,
                "uncertainty": base,
                "reward": base,
                "phase": base,
            }
        else:
            controller_inputs = session.controller_inputs(
                batch_size, seq_len, device
            )

        flattened = {
            key: value.view(-1, 1) for key, value in controller_inputs.items()
        }
        controller_out = self.controller(flattened)
        m_gate = (
            controller_out["m_gate"]
            .view(batch_size, seq_len)
            .to(device=device, dtype=self._default_dtype)
        )
        write_scale = (
            controller_out["write_scale"]
            .view(batch_size, seq_len, 1)
            .to(device=device, dtype=self._default_dtype)
        )
        alpha_scale = write_scale.expand(
            -1, -1, self.base.config.num_attention_heads
        )
        self._gate_cache = (m_gate, alpha_scale)
        self._controller_inputs_cache = controller_inputs

    def _fetch_gates(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._gate_cache is None
            or self._gate_cache[0].shape[1] != seq_len
        ):
            self._prepare_gates(batch_size, seq_len, device)
        m_gate, alpha_scale = self._gate_cache
        m_gate_t = m_gate.to(device)
        alpha_scale_t = alpha_scale.to(device)
        if self._alpha_override is not None:
            alpha_scale_t = torch.full_like(
                alpha_scale_t, self._alpha_override
            )
        self._last_gates = (
            m_gate_t.detach().cpu(),
            alpha_scale_t.detach().cpu(),
        )
        return m_gate_t, alpha_scale_t

    def _persist_fast_buffers(self, session: SessionState) -> None:
        for idx, layer in enumerate(self._cortex_layers):
            # S is the only state now
            session.store_fast_buffer(idx, layer.S, layer.S)

    def _log_gate_stats(
        self, layer_idx: int, m_gate: torch.Tensor, alpha_scale: torch.Tensor
    ) -> None:
        controller_inputs = getattr(self, "_controller_inputs_cache", None)
        surprise_mean = (
            float(controller_inputs["surprise"].mean().item())
            if controller_inputs
            else 0.0
        )
        uncertainty_mean = (
            float(controller_inputs["uncertainty"].mean().item())
            if controller_inputs
            else 0.0
        )
        self.gate_logs.append(
            {
                "layer": layer_idx,
                "m_mean": float(m_gate.mean().item()),
                "m_max": float(m_gate.max().item()),
                "alpha_mean": float(alpha_scale.mean().item()),
                "alpha_max": float(alpha_scale.max().item()),
                "m_mean_batch": m_gate.mean(dim=1).detach().cpu().tolist(),
                "m_max_batch": m_gate.max(dim=1)
                .values.detach()
                .cpu()
                .tolist(),
                "alpha_mean_batch": alpha_scale.mean(dim=1)
                .detach()
                .cpu()
                .tolist(),
                "alpha_max_batch": alpha_scale.max(dim=1)
                .values.detach()
                .cpu()
                .tolist(),
                "surprise_mean": surprise_mean,
                "uncertainty_mean": uncertainty_mean,
            }
        )


def load_qwen_with_cortex(
    model_name: str,
    cortex_cfg: Optional[CortexWrapConfig] = None,
    device_map: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    **from_pretrained_kwargs,
) -> CortexWrappedModel:
    cortex_cfg = cortex_cfg or CortexWrapConfig()
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        **from_pretrained_kwargs,
    )
    wrapped = CortexWrappedModel(base, cortex_cfg)
    attach_cortex_sidecars(wrapped)
    return wrapped


def attach_cortex_sidecars(wrapper: CortexWrappedModel) -> None:
    # inject CortexBlock into each decoder layer
    if Qwen2DecoderLayer is None:
        raise RuntimeError("Qwen2DecoderLayer unavailable.")

    base_model = getattr(wrapper.base, "model", None)
    if base_model is None:
        raise ValueError("Expected base.model attribute.")

    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise ValueError("No .layers found.")

    cfg = wrapper.base.config
    cortex_layers = []
    for idx, layer in enumerate(layers):
        if not isinstance(layer, Qwen2DecoderLayer):
            raise TypeError(
                f"Layer {idx} is {type(layer)}; only Qwen2DecoderLayer supported for now."
            )
        if wrapper.config.use_hybrid:
            cortex_block = HybridCortexBlock(
                dim=cfg.hidden_size,
                head_dim=wrapper.config.rank_fast,
                window_size=wrapper.config.window_size,
            )
            # Hybrid block doesn't support tie_projections yet or needs different handling
        else:
            cortex_block.tie_projections(
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
            )

        layer.cortex_block = cortex_block
        layer.register_forward_pre_hook(_make_restore_hook(wrapper, idx))
        bound_forward = _bind_cortex_forward(wrapper, cortex_block, idx)
        layer.forward = bound_forward.__get__(
            layer, layer.__class__
        )  # type: ignore[assignment]
        cortex_layers.append(cortex_block)

    wrapper._cortex_layers = tuple(cortex_layers)

    # Monkey-patch the final norm to handle potential tuple propagation issues
    if hasattr(wrapper.base.model, "norm"):
        old_norm = wrapper.base.model.norm

        class SafeNorm(nn.Module):
            def __init__(self, original_norm):
                super().__init__()
                self.original_norm = original_norm

            def forward(self, hidden_states):
                while isinstance(hidden_states, (tuple, list)):
                    hidden_states = hidden_states[0]
                return self.original_norm(hidden_states)

        wrapper.base.model.norm = SafeNorm(old_norm)


def _make_restore_hook(wrapper: CortexWrappedModel, layer_idx: int):
    # pre-hook: restore U, V from session if available
    def hook(module: nn.Module, inputs):
        # Unwrap nested tuples until we get to the tensor
        hidden_states = inputs
        while isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        batch = hidden_states.shape[0]
        device = hidden_states.device
        cortex_block: CortexBlock = module.cortex_block
        session = wrapper._active_session
        if session is None:
            cortex_block.reset_fast(batch, device=device)
            return
        snapshot = session.get_fast_buffer(layer_idx)
        if snapshot is None or snapshot.U.shape[0] != batch:
            cortex_block.reset_fast(
                batch, device=device
            ) if hasattr(cortex_block, "reset_fast") else None
            # For HybridBlock, we might need to manually reset S if it's not a buffer
            if isinstance(cortex_block, HybridCortexBlock):
                cortex_block.S = None  # Will be re-init in forward
            return

        if hasattr(cortex_block, "load_fast"):
            cortex_block.load_fast(
                snapshot.U.to(device), snapshot.V.to(device)
            )
        elif isinstance(cortex_block, HybridCortexBlock):
            cortex_block.S = snapshot.U.to(device)

    return hook


def _bind_cortex_forward(
    wrapper: CortexWrappedModel,
    cortex_block: CortexBlock,
    layer_idx: int,
):
    # replace layer forward to add cortex delta
    def forward_with_cortex(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,  # Added for compatibility with gradient checkpointing
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **kwargs,
    ):

        # Unwrap if hidden_states is passed as tuple
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        batch_size, seq_len = normed.size(0), normed.size(1)
        m_gate, alpha_scale = wrapper._fetch_gates(
            batch_size, seq_len, normed.device
        )

        attn_outputs = self.self_attn(
            hidden_states=normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,  # Pass it through
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attn_output = attn_outputs[0]
        mix_mode = (
            wrapper._mix_mode if wrapper._sidecar_enabled else "slow_only"
        )

        if isinstance(cortex_block, HybridCortexBlock):
            # Hybrid block returns (output, state, surprise)
            # We need to handle the state update and surprise logging
            # For now, we just use the output
            cortex_delta, next_state, surprise = cortex_block(
                normed, getattr(cortex_block, "S", None)
            )
            cortex_block.S = next_state  # Update state
            # Log surprise if needed
        else:
            cortex_delta = cortex_block(
                normed, m_gate, alpha_scale, mix_mode=mix_mode
            )

        wrapper._log_gate_stats(layer_idx, m_gate, alpha_scale)
        if wrapper._sidecar_enabled:
            hidden_states = residual + attn_output + cortex_delta
        else:
            hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + mlp_out

        if isinstance(hidden_states, (tuple, list)):
            # print(f"DEBUG: Unwrapping hidden_states from {type(hidden_states)}", file=sys.stderr)
            while isinstance(hidden_states, (tuple, list)):
                hidden_states = hidden_states[0]

        if not isinstance(hidden_states, torch.Tensor):
            # print(f"CRITICAL: hidden_states ended up as {type(hidden_states)}", file=sys.stderr)
            # Emergency fallback if somehow we got a non-tensor
            if hasattr(hidden_states, "tensor"):
                hidden_states = hidden_states.tensor

        # Build outputs tuple matching what Qwen2Model expects
        # Qwen2 expectations: (
        #     hidden_states,
        #     present_key_value,
        #     all_hidden_states,
        #     all_self_attns,
        # )

        # 1. Hidden States (Tensor)
        outputs = (hidden_states,)

        # 2. Present Key Value (Cache) - from attn_outputs[1]
        if len(attn_outputs) > 1:
            outputs += (attn_outputs[1],)

        # 3. Attentions - from attn_outputs[2]
        if len(attn_outputs) > 2:
            outputs += (attn_outputs[2],)

        return outputs

    return forward_with_cortex
