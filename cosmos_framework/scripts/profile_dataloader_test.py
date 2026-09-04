# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os

from cosmos_framework.scripts.profile_dataloader import _configure_gloo_interface


def test_single_node_defaults_gloo_to_loopback(monkeypatch):
    # Register cleanup before the function under test writes this variable.
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "test-placeholder")
    monkeypatch.delenv("GLOO_SOCKET_IFNAME")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")

    assert _configure_gloo_interface(None) == "lo"
    assert os.environ["GLOO_SOCKET_IFNAME"] == "lo"


def test_explicit_gloo_interface_takes_precedence(monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "invalid-existing-value")

    assert _configure_gloo_interface("lo") == "lo"


def test_multi_node_does_not_guess_interface(monkeypatch):
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")

    assert _configure_gloo_interface(None) is None
