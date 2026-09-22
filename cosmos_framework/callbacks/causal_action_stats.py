# SPDX-License-Identifier: OpenMDW-1.1
"""Action mid-training validation metrics with explicit per-source clipping."""

import json

import torch
import torch.distributed as dist

from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback


class CausalActionStats(Callback):
    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration=0):
        norms = {}
        for group in ("state2llm", "action2llm", "llm2action"):
            total = None
            for name, parameter in model.net.named_parameters():
                if group in name and parameter.grad is not None:
                    grad = parameter.grad
                    # DTensor norm handles sharding/replication through its placements.
                    squared = grad.detach().float().square().sum()
                    if hasattr(squared, "full_tensor"):
                        squared = squared.full_tensor()
                    total = squared if total is None else total + squared
            norms[group] = 0.0 if total is None else float(total.sqrt())
        self.grad_norms = norms

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        record = dict(
            iteration=iteration,
            vision_loss=float(output_batch["flow_matching_loss_vision"]),
            action_loss=float(output_batch["flow_matching_loss_action"]),
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            grad_norms=getattr(self, "grad_norms", {}),
        )
        clipping = []
        for plan in data_batch["sequence_plan"]:
            m = plan.causal_action_metadata
            clipping.append(
                dict(
                    source=m.contract.source,
                    block_size=m.block_size,
                    state=m.state_clip_fraction.tolist(),
                    action=m.action_clip_fraction.tolist(),
                )
            )
        record["clipping"] = clipping
        log.info("CAUSAL_ACTION_METRICS " + json.dumps(record))
