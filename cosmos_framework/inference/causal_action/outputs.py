# SPDX-License-Identifier: OpenMDW-1.1
"""Recover absolute physical targets using each action's original block anchor."""

from cosmos_framework.data.generator.action.block_state import absolute_action, prediction_block_ids


def real_actions(result, batch):
    outputs = []
    for i, action in enumerate(result["action"]):
        metadata = batch["sequence_plan"][i].causal_action_metadata
        if metadata is None or metadata.statistics is None:
            raise ValueError("Action export requires its source contract, block anchors and delta statistics")
        mask = metadata.action_mask.to(action.device)
        keep = mask.any(-1)
        if "generated_action_mask" in result:
            keep &= result["generated_action_mask"][i].to(keep.device)
        blocks = prediction_block_ids(metadata.action_frame_ids.to(action.device)[keep], metadata.block_size)
        delta = metadata.statistics.action.denormalize(action[keep], mask[keep])
        outputs.append(absolute_action(delta, metadata.anchors.to(action.device)[blocks], mask[keep]))
    return outputs


def gather_source_slots(actions, mask):
    """Execution adapters select explicit slots, never a contiguous raw-width prefix."""
    if mask.ndim != 1 or mask.shape[0] != 80:
        raise ValueError("Execution slot mapping requires an explicit 80D mask")
    return actions[..., mask.to(actions.device)]
