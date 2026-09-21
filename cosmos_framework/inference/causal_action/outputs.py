# SPDX-License-Identifier: OpenMDW-1.1
"""Recover real actions; state and synthetic first-group slots never escape."""

from cosmos_framework.data.generator.action.action_processing import ActionProcessor, get_action_processing_records
from cosmos_framework.data.generator.sequence_packing.causal_action import action_prefix_length


def real_actions(result, batch):
    records = get_action_processing_records(batch)
    outputs = []
    for i, (vision, action) in enumerate(zip(result["vision"], result["action"], strict=True)):
        prefix = action_prefix_length(action.shape[0], vision.shape[2])
        action = action[prefix:]
        if not records or records[i] is None:
            raise ValueError("Action output requires its preprocessing record for unpadding/denormalization")
        outputs.append(ActionProcessor.postprocess_action(action, records[i]))
    return outputs
