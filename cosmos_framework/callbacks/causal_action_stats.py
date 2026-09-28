# SPDX-License-Identifier: OpenMDW-1.1
"""Action mid-training validation metrics with explicit per-source clipping."""

import json

import torch
import torch.distributed as dist
import wandb

from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.causal_action_metrics import mode_metric_means


class CausalActionStats(Callback):
    def __init__(self):
        super().__init__()
        self.mode_metrics = None

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        # 只累计 rank 0 的全部微批次，不做跨卡汇总。
        stats = output_batch.get("causal_action_metrics")
        if stats is not None:
            if self.mode_metrics is None:
                self.mode_metrics = stats.detach().clone()
            else:
                self.mode_metrics += stats.detach()

    def _flush_mode_metrics(self):
        stats = self.mode_metrics
        self.mode_metrics = None
        return {} if stats is None else mode_metric_means(stats)

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
        mode_losses = None
        if self.mode_metrics is not None and iteration % self.config.trainer.logging_iter == 0:
            mode_losses = self._flush_mode_metrics()
        record = dict(
            iteration=iteration,
            modes=list(data_batch["causal_action_mode"]),
            vision_loss=float(output_batch["flow_matching_loss_vision"].detach()),
            action_loss=float(output_batch["flow_matching_loss_action"].detach()),
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
                    action_steps=int(m.action_mask.shape[0]),
                    state_blocks=int(m.anchors.shape[0]),
                    state=m.state_clip_fraction.tolist(),
                    action=m.action_clip_fraction.tolist(),
                )
            )
        record["clipping"] = clipping
        if mode_losses is not None:
            record["mode_losses"] = mode_losses
            if mode_losses and wandb.run is not None:
                wandb.log(mode_losses, step=iteration)
        log.info("CAUSAL_ACTION_METRICS " + json.dumps(record))
