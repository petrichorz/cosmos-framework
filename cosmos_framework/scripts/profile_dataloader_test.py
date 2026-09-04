# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.scripts.profile_dataloader import _advance_training_position


def test_training_iteration_advances_after_gradient_accumulation_window():
    iteration = 7
    grad_accum_index = 0

    iteration, grad_accum_index = _advance_training_position(iteration, grad_accum_index, 2)
    assert (iteration, grad_accum_index) == (7, 1)

    iteration, grad_accum_index = _advance_training_position(iteration, grad_accum_index, 2)
    assert (iteration, grad_accum_index) == (8, 0)


def test_training_loop_fetches_once_after_reaching_max_iteration():
    iteration = 0
    grad_accum_index = 0
    fetch_count = 0
    max_iteration = 3

    while True:
        fetch_count += 1
        if iteration >= max_iteration:
            break
        iteration, grad_accum_index = _advance_training_position(iteration, grad_accum_index, 2)

    assert iteration == max_iteration
    assert fetch_count == max_iteration * 2 + 1
