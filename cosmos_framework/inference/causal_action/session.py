# SPDX-License-Identifier: OpenMDW-1.1
"""Complete-block planning with observed visual history and fresh current state."""

from dataclasses import dataclass

import torch


@dataclass
class ObservedBlock:
    observation: torch.Tensor


class CausalActionSession:
    """A caller supplies its model/data-adapter planning callable.

    ``planner`` receives measured state, image and observed visual blocks, and
    returns a [N,D] absolute target tensor in the selected template. Feedback confirms a complete block
    and supplies its observations; no historical action or state is retained.
    The next call always requires a fresh measured state and image.
    """

    def __init__(self, contract, planner, history_blocks=8, *, template):
        if history_blocks < 1:
            raise ValueError("history_blocks must be positive")
        self.contract, self.planner, self.history_blocks = contract, planner, history_blocks
        self.template = template
        self.history, self.pending = [], None
        self.revision = 0

    @classmethod
    def for_model(cls, contract, model, batch_builder, *, template, history_blocks=8, **sampling_options):
        """Connect Cosmos to an execution adapter's batch construction.

        ``batch_builder`` receives measured state/image and ObservedBlock
        history. It must preserve the current measurement and build the same
        template-normalized SequencePlan metadata as training, with visual history
        marked clean. Historical action/state slots may be zero placeholders.
        """
        from cosmos_framework.inference.causal_action.outputs import real_actions

        def planner(**inputs):
            batch = batch_builder(**inputs)
            metadata = batch["sequence_plan"][0].causal_action_metadata
            current = int(batch.get("causal_action_current_block", 0))
            if not 0 <= current < len(metadata.states):
                raise ValueError("Requested block has no measured state")
            batch["causal_action_current_block"] = current
            if metadata.contract != contract:
                raise ValueError("Execution adapter changed the source representation contract")
            if not torch.equal(metadata.anchors[current].cpu(), inputs["state"].cpu()):
                raise ValueError("Batch builder must preserve the current measured state")
            if not torch.equal(metadata.state_mask[current].cpu(), inputs["state_mask"].cpu()):
                raise ValueError("Batch builder must preserve measured state validity")
            batch["causal_action_preview"] = False
            generated = model.generate_samples_from_batch(
                batch,
                causal_block_size=metadata.block_size,
                causal_history_blocks=history_blocks,
                **sampling_options,
            )
            return real_actions(generated, batch)[0]

        return cls(contract, planner, history_blocks=history_blocks, template=template)

    def generate_current_block(self, *, state, state_mask, image, action_mask):
        if self.pending is not None:
            raise RuntimeError("Commit complete execution feedback before planning the next block")
        if state.shape != (self.template.width,):
            raise ValueError("State must match the action template width")
        state_mask = self.template.validate_valid_mask(state_mask).to(state.device)
        action_mask = self.template.validate_valid_mask(action_mask).to(state.device)
        # 绝对目标不依赖同维 state；只要求相对字段具有有效 anchor。
        if (self.template.required_anchor_mask(action_mask) & ~state_mask).any() or not state_mask.any():
            raise ValueError("Current targets require valid measured anchor channels")
        state = self.template.sanitize(state, state_mask)
        history = self.history[-self.history_blocks :]
        # Fresh invocation invalidates all request-local attention/CFG caches.
        proposed = self.planner(
            state=state.clone(),
            state_mask=state_mask.clone(),
            action_mask=action_mask.clone(),
            image=image,
            history=history,
            revision=self.revision,
            contract=self.contract,
        )
        if proposed.ndim != 2 or proposed.shape[1] != self.template.width or not len(proposed):
            raise ValueError("Planner must return a complete real action block [N,template.width]")
        self.pending = len(proposed)
        return proposed.masked_fill(~action_mask.to(proposed.device), 0)

    def commit_execution_feedback(self, *, observation, executed_steps):
        if self.pending is None:
            raise RuntimeError("No planned block awaits feedback")
        if executed_steps != self.pending:
            raise ValueError("Only complete real action blocks can be committed; partial replanning is unsupported")
        self.history.append(ObservedBlock(observation.detach().clone()))
        self.history = self.history[-self.history_blocks :]
        self.pending = None
        self.revision += 1
