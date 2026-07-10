# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from functools import wraps

def with_torch_device(device):
    """
    Decorator factory that wraps a function to execute within a specific torch device context.

    This decorator ensures that all tensor allocations and operations within the decorated
    function use the specified device by default.

    Args:
        device: The torch device to use (e.g. 'cuda', 'cuda:0', 'cpu', or torch.device object).

    Returns:
        A decorator function that wraps the target function with the specified device context.

    Example:
        @with_torch_device('cuda:0')
        def create_tensors():
            x = torch.randn(10, 10)  # Will be created on cuda:0
            return x
    """
    import torch

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with torch.device(device):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
