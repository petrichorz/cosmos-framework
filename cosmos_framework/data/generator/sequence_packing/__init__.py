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
    TeacherForcingLayout,
    build_dense_teacher_forcing_gen_mask,
    build_teacher_forcing_layout,
    sample_teacher_forcing_parameters,
)

__all__ = [
    "ModalityData",
    "PackedSequence",
    "SequencePlan",
    "TeacherForcingLayout",
    "build_dense_teacher_forcing_gen_mask",
    "build_sequence_plans_from_data_batch",
    "build_teacher_forcing_layout",
    "pack_input_sequence",
    "sample_teacher_forcing_parameters",
]
