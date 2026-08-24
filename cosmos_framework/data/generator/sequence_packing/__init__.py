# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""High-level entry points for VFM sequence packing."""

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.sequence import (
    PackedSequence,
    SequencePlan,
    build_sequence_plans_from_data_batch,
)
from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    TeacherForcingData,
    TeacherForcingGeometry,
    TeacherForcingLayout,
    build_dense_teacher_forcing_gen_mask,
    build_per_sample_teacher_forcing_gen_masks,
    build_teacher_forcing_frame_block_ids,
    build_teacher_forcing_layout,
    expand_packed_sequence_for_teacher_forcing,
    sample_teacher_forcing_geometry,
    sample_teacher_forcing_parameters,
    select_teacher_forcing_noisy_outputs,
)

__all__ = [
    "ModalityData",
    "PackedSequence",
    "SequencePlan",
    "TeacherForcingData",
    "TeacherForcingGeometry",
    "TeacherForcingLayout",
    "build_dense_teacher_forcing_gen_mask",
    "build_per_sample_teacher_forcing_gen_masks",
    "build_sequence_plans_from_data_batch",
    "build_teacher_forcing_frame_block_ids",
    "build_teacher_forcing_layout",
    "expand_packed_sequence_for_teacher_forcing",
    "pack_input_sequence",
    "sample_teacher_forcing_parameters",
    "sample_teacher_forcing_geometry",
    "select_teacher_forcing_noisy_outputs",
]
