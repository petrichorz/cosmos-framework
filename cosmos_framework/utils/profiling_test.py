# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.utils.profiling import _profile_schedule_counts


def test_profile_schedule_defaults_to_one_active_step(monkeypatch):
    monkeypatch.delenv("COSMOS_NPU_PROFILE_ACTIVE_STEPS", raising=False)

    assert _profile_schedule_counts(profile_freq=8, warmup=2) == (5, 2, 1)


def test_profile_schedule_supports_three_consecutive_active_steps(monkeypatch):
    monkeypatch.setenv("COSMOS_NPU_PROFILE_ACTIVE_STEPS", "3")

    assert _profile_schedule_counts(profile_freq=10, warmup=2) == (5, 2, 3)


@pytest.mark.parametrize("active", ["0", "-1", "invalid"])
def test_profile_schedule_rejects_invalid_active_steps(monkeypatch, active):
    monkeypatch.setenv("COSMOS_NPU_PROFILE_ACTIVE_STEPS", active)

    with pytest.raises(ValueError, match="COSMOS_NPU_PROFILE_ACTIVE_STEPS"):
        _profile_schedule_counts(profile_freq=10, warmup=2)
