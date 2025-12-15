# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

from collections import defaultdict
from typing import Dict

import torch
from composer.core import Callback, State
from composer.loggers import Logger


class MoEAuxLossMonitor(Callback):
    """Logs Mixture-of-Experts auxiliary losses for each layer.

    The MoE layers cache their latest auxiliary loss components (load balancing
    and router z-loss). This callback collect those values once per batch and
    forwards detached scalars to the Composer logger, which can then be picked
    up by W&B or any other configured logger.
    """

    def __init__(self, log_interval: int = 1):
        self.log_interval = log_interval
        self._batch_idx = 0

    def after_forward(self, state: State, logger: Logger) -> None:
        if self._batch_idx % self.log_interval != 0:
            self._batch_idx += 1
            return

        metrics: Dict[str, torch.Tensor] = {}
        layer_stats = defaultdict(dict)
        for name, module in state.model.named_modules():
            if hasattr(module, "latest_aux_loss") and module.latest_aux_loss is not None:
                layer_stats[name]["moe_aux_loss"] = module.latest_aux_loss
            if hasattr(module, "latest_lb_loss") and module.latest_lb_loss is not None:
                layer_stats[name]["moe_load_balance_loss"] = module.latest_lb_loss
            if hasattr(module, "latest_router_z_loss") and module.latest_router_z_loss is not None:
                layer_stats[name]["moe_router_z_loss"] = module.latest_router_z_loss
            if hasattr(module, "latest_moe_losses") and module.latest_moe_losses:
                for loss_name, tensor in module.latest_moe_losses.items():
                    layer_stats[name][f"moe_ds/{loss_name}"] = tensor

        if layer_stats:
          # initialize on correct device + dtype
             any_tensor = next(iter(next(iter(layer_stats.values())).values()))
             total_aux = any_tensor.new_tensor(0.0)

             for layer_name, values in layer_stats.items():
                prefix = f"moe/{layer_name}"
                for metric_name, tensor_val in values.items():
                   metrics[f"{prefix}/{metric_name}"] = tensor_val.detach().cpu()
                   if metric_name == "moe_aux_loss":
                     total_aux = total_aux + tensor_val

        metrics["moe/total_aux_loss"] = total_aux.detach().cpu()

        logger.log_metrics(metrics)

        self._batch_idx += 1
