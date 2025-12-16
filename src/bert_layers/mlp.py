# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Tri Dao.

from typing import Optional

import torch
import torch.nn as nn

from .configuration_bert import FlexBertConfig
from .activation import get_act_fn
from .normalization import get_norm_layer
from .initialization import ModuleType, init_weights
try:
    from deepspeed.moe.layer import MoE as DeepSpeedMoELayer
except ImportError:  # pragma: no cover - optional dependency
    DeepSpeedMoELayer = None


class BertResidualGLU(nn.Module):
    """Applies the FFN at the end of each Mosaic BERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality, but
    introduces Gated Linear Units.

    Note: Mosaic BERT adds parameters in order to implement Gated Linear Units. To keep parameter count consistent with that of a
    standard Hugging Face BERT, scale down `config.intermediate_size` by 2/3. For example, a Mosaic BERT constructed with
    `config.intermediate_size=2048` will have the same parameter footprint as its Hugging Face BERT counterpart constructed
    with the `config.intermediate_size=3072`.
    However, in most cases it will not be necessary to adjust `config.intermediate_size` since, despite the increased
    parameter size, Mosaic BERT typically offers a net higher throughput than a Hugging Face BERT built from the same `config`.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.gated_layers = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.act = get_act_fn(config.hidden_act)
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = get_norm_layer(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute new hidden states from current hidden states.

        Args:
            hidden_states (torch.Tensor): The (unpadded) hidden states from
                the attention layer [nnz, dim].
        """
        residual_connection = hidden_states
        # compute the activation
        hidden_states = self.gated_layers(hidden_states)
        gated = hidden_states[:, : self.config.intermediate_size]
        non_gated = hidden_states[:, self.config.intermediate_size :]
        hidden_states = self.act(gated) * non_gated
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + residual_connection)
        return hidden_states


class FlexBertMLPBase(nn.Module):
    """A FlexBERT MLP base class for type hints."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

    def _init_weights(self, reset_params: bool = False):
        raise NotImplementedError("This is a base class and should not be used directly.")

    def reset_parameters(self):
        self._init_weights(reset_params=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is a base class and should not be used directly.")


class FlexBertMLP(FlexBertMLPBase):
    """Applies the MLP at the end of each FlexBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_in_bias)
        self.act = get_act_fn(config.hidden_act)
        self.drop = nn.Dropout(config.mlp_dropout_prob) if config.mlp_dropout_prob > 0.0 else nn.Identity()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wi,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.intermediate_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute new hidden states from current hidden states.

        Args:
            hidden_states (torch.Tensor): The (unpadded) hidden states from
                the attention layer [nnz, dim].
        """
        return self.Wo(self.drop(self.act(self.Wi(hidden_states))))


class FlexBertGLU(FlexBertMLPBase):
    """Applies the GLU at the end of each FlexBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_in_bias)
        self.act = get_act_fn(config.hidden_act)
        self.drop = nn.Dropout(config.mlp_dropout_prob) if config.mlp_dropout_prob > 0.0 else nn.Identity()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wi,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.intermediate_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class FlexBertParallelGLU(FlexBertMLPBase):
    """Applies the GLU at the end of each FlexBERT layer using intermediate_ff computed in parallel of the attention.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.act = get_act_fn(config.hidden_act)
        self.drop = nn.Dropout(config.mlp_dropout_prob) if config.mlp_dropout_prob > 0.0 else nn.Identity()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.intermediate_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(self, intermediate_ff: torch.Tensor) -> torch.Tensor:
        input, gate = intermediate_ff.chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class DeepSpeedGLUExpert(nn.Module):
    """DeepSpeed-compatible GLU expert block."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        dropout_prob: float = 0.0,
        mlp_in_bias: bool = False,
        mlp_out_bias: bool = False,
    ):
        super().__init__()
        self.in_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=mlp_in_bias)
        self.out_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_out_bias)
        self.act = get_act_fn(activation)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_chunk, gate = self.in_proj(hidden_states).chunk(2, dim=-1)
        hidden_states = self.act(input_chunk) * gate
        hidden_states = self.dropout(hidden_states)
        return self.out_proj(hidden_states)



class FlexBertGLUMoE(FlexBertMLPBase):
    """Mixture of Experts with GLU activation for FlexBERT."""
    
    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        
        self.n_exp = getattr(config, 'moe_num_experts', 4)
        self.top_k = getattr(config, 'moe_top_k', 2)
        self.use_noisy_top_k = getattr(config, 'moe_use_noisy_top_k', True)
        self.capacity_factor = getattr(config, 'moe_capacity_factor', 1.25)
        self.moe_intermediate_size = getattr(config, 'moe_intermediate_size', None)
        self.use_loss_free_balance = getattr(config, 'moe_use_loss_free_balance', False)
        self.loss_free_balance_update_rate = getattr(
            config, 'moe_loss_free_balance_update_rate', 1e-5
        )
        self.moe_backend = getattr(config, "moe_backend", "deepspeed")
        if self.moe_backend != "deepspeed":
            raise ValueError("The legacy FlexBERT MoE router has been removed; set `moe_backend: deepspeed`.")

        self._init_deepspeed_moe()

        # Cached metrics for logging
        self.latest_lb_loss = None
        self.latest_router_z_loss = None
        self.latest_aux_loss = None
        self.latest_moe_losses = {}
        self.latest_expert_usage = None
        self.latest_expert_prob = None
        self._ds_gate = None
    
    def _init_weights(self, reset_params: bool = False):
        if DeepSpeedMoELayer is None:
            raise ImportError("DeepSpeed must be installed to use the `deepspeed` MoE backend.")
        # DeepSpeed handles initialization internally.
        return
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._forward_with_deepspeed(hidden_states)

    def _init_deepspeed_moe(self):
        if DeepSpeedMoELayer is None:
            raise ImportError(
                "DeepSpeed is not available. Install `deepspeed` to enable the DeepSpeed MoE backend."
            )

        expert_module = DeepSpeedGLUExpert(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            activation=self.config.hidden_act,
            dropout_prob=self.config.mlp_dropout_prob,
            mlp_in_bias=self.config.mlp_in_bias,
            mlp_out_bias=self.config.mlp_out_bias,
        )
        self._init_ds_expert_weights(expert_module)

        eval_capacity = (
            self.config.moe_eval_capacity_factor
            if self.config.moe_eval_capacity_factor is not None
            else self.capacity_factor
        )
        noisy_policy = "RSample" if self.use_noisy_top_k else "none"
        self.ds_moe = DeepSpeedMoELayer(
            hidden_size=self.config.hidden_size,
            expert=expert_module,
            num_experts=self.n_exp,
            ep_size=getattr(self.config, "moe_expert_parallel_size", 1),
            k=self.top_k,
            capacity_factor=self.capacity_factor,
            eval_capacity_factor=eval_capacity,
            min_capacity=self.config.moe_min_capacity,
            noisy_gate_policy=noisy_policy,
            drop_tokens=True,
            use_residual=False,
        )

    def _init_ds_expert_weights(self, expert_module: DeepSpeedGLUExpert):
        init_weights(
            self.config,
            expert_module.in_proj,
            layer_dim=self.config.hidden_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            expert_module.out_proj,
            layer_dim=self.config.intermediate_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def _forward_with_deepspeed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)

        self.latest_expert_usage = None
        self.latest_expert_prob = None
        self._capture_expert_stats(hidden_states)

        outputs = self.ds_moe(hidden_states)
        if isinstance(outputs, tuple):
            routed_hidden_states = outputs[0]
            ds_aux_loss = outputs[1] if len(outputs) > 1 else None
        else:
            routed_hidden_states = outputs
            ds_aux_loss = None

        self.aux_loss = None
        self.latest_lb_loss = None
        self.latest_router_z_loss = None
        self.latest_aux_loss = None
        self.latest_moe_losses = {}

        if ds_aux_loss is not None:
            self.latest_lb_loss = ds_aux_loss.detach()
            self.latest_aux_loss = self.latest_lb_loss
            self.aux_loss = ds_aux_loss

        losses_dict = getattr(self.ds_moe, "losses_dict", None)
        if not losses_dict:
            losses_dict = getattr(self.ds_moe, "losses", None)
        if isinstance(losses_dict, dict):
            self.latest_moe_losses = {k: v.detach() for k, v in losses_dict.items() if torch.is_tensor(v)}

        return routed_hidden_states.contiguous().view(*original_shape)

    def _resolve_ds_gate(self):
        if self._ds_gate is not None:
            return self._ds_gate
        candidate = getattr(self.ds_moe, "gate", None)
        if candidate is None:
            for module in self.ds_moe.modules():
                if isinstance(module, nn.Linear) and module.out_features == self.n_exp:
                    candidate = module
                    break
        self._ds_gate = candidate
        return self._ds_gate

    def _capture_expert_stats(self, hidden_states: torch.Tensor):
        gate_module = self._resolve_ds_gate()
        if gate_module is None:
            return
        flat = hidden_states
        if flat.dim() > 2:
            flat = flat.view(-1, flat.size(-1))
        if flat.numel() == 0:
            return
        target_dtype = gate_module.weight.dtype
        with torch.no_grad():
            logits = gate_module(flat.to(dtype=target_dtype))
            probs = torch.softmax(logits, dim=-1)
            if probs.numel() == 0:
                return
            self.latest_expert_prob = probs.mean(dim=0).detach()
            topk = torch.topk(logits, k=min(self.top_k, logits.size(-1)), dim=-1).indices
            counts = torch.bincount(topk.view(-1), minlength=self.n_exp).float()
            total = counts.sum()
            if total.item() > 0:
                self.latest_expert_usage = (counts / total).detach()
            self._apply_loss_free_balance(counts)

    def _apply_loss_free_balance(self, counts: torch.Tensor):
        if not self.use_loss_free_balance:
            return
        if counts.numel() == 0:
            return
        gate_module = self._resolve_ds_gate()
        if gate_module is None:
            return
        if getattr(gate_module, "bias", None) is None:
            gate_module.bias = nn.Parameter(torch.zeros(self.n_exp, device=gate_module.weight.device))
        with torch.no_grad():
            avg_count = counts.float().mean()
            if not torch.isfinite(avg_count):
                return
            error = avg_count - counts.float()
            update = self.loss_free_balance_update_rate * torch.sign(error)
            gate_module.bias.data[: self.n_exp] = gate_module.bias.data[: self.n_exp] + update.to(gate_module.bias)


# Update the MLP registry
MLP2CLS = {
    "mlp": FlexBertMLP,
    "glu": FlexBertGLU,
    "parallel_glu": FlexBertParallelGLU,
    "glu_moe": FlexBertGLUMoE,
}


def get_mlp_layer(config: FlexBertConfig, layer_id: Optional[int] = None) -> FlexBertMLPBase:
    try:
        mlp_layer = (
            config.initial_mlp_layer
            if layer_id < config.num_initial_layers and getattr(config, "initial_mlp_layer", None) is not None
            else config.mlp_layer
        )
        return MLP2CLS[mlp_layer](config, layer_id=layer_id)
    except KeyError as e:
        if layer_id < config.num_initial_layers and getattr(config, "initial_mlp_layer", None) is not None:
            raise ValueError(
                f"Invalid MLP layer type: {config.initial_mlp_layer=}, must be one of {MLP2CLS.keys()}. {e}"
            )
        else:
            raise ValueError(f"Invalid MLP layer type: {config.mlp_layer=}, must be one of {MLP2CLS.keys()}. {e}")
