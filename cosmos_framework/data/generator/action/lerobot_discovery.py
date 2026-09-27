# SPDX-License-Identifier: OpenMDW-1.1
"""Discover LeRobot v3 dataset roots without scanning their data files."""

import json
import os
from pathlib import Path


def discover_dataset_roots(paths):
    """递归发现 v3 数据集；找到根目录后不再遍历它的数据/视频目录，重叠输入去重。"""
    roots = set()
    for path in paths:
        path = Path(path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {path}")
        found = False
        for directory, children, _ in os.walk(path):
            children.sort()
            root = Path(directory)
            info_path = root / "meta/info.json"
            if not info_path.is_file():
                continue
            info = json.loads(info_path.read_text())
            if not str(info.get("codebase_version", "")).startswith("v3."):
                raise ValueError(f"Expected LeRobot v3 metadata: {info_path}")
            if not (root / "data").is_dir():
                raise ValueError(f"Missing data directory: {root}")
            roots.add(root.resolve())
            found = True
            children[:] = []
        if not found:
            raise ValueError(f"No LeRobot v3 datasets found under {path}")
    return sorted(roots)
