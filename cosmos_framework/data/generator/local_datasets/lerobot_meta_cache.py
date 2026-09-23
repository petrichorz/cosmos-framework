# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LeRobot 数据集 metadata 索引缓存。

把「读 episodes parquet + 提取全量 episode 字段」这个耗时、且结果确定的重复计算，
缓存到本地 JSON。第二次启动直接反序列化，省去 parquet 的重复 IO。

关键设计：
- 缓存存【全量原始 episode 字段】，过滤 / 白名单 / split 仍在加载期内存里做，
  因此调整过滤参数不会使缓存失效。
- 缓存文件名固定为 ``index.json``，用「父目录名 + 子数据集相对路径」组织，可读、好找。
- 数据内容指纹（fingerprint）存在 ``index.json`` 内部做新鲜度校验：命中后比对，
  一致才用，不一致就重读 parquet 重建并覆盖。
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

# 缓存结构版本：改动缓存 schema / 提取逻辑时手动 +1，使旧缓存全部失效。
SCHEMA_VERSION = 1

# 环境变量：全局缓存根目录。未设置时回退到 ~/.cache/cosmos_framework/lerobot_meta/。
_CACHE_ROOT_ENV = "LEROBOT_META_CACHE_DIR"


def get_cache_root() -> Path:
    """返回全局缓存根目录。

    优先读环境变量 ``LEROBOT_META_CACHE_DIR``；未设置则落到
    ``~/.cache/cosmos_framework/lerobot_meta/``（兜底，不报错）。
    """
    env = os.environ.get(_CACHE_ROOT_ENV)
    if env:
        return Path(env)
    return Path.home() / ".cache" / "cosmos_framework" / "lerobot_meta"


def resolve_cache_path(lerobot_root: str, root: str) -> Path:
    """计算某个数据集根对应的缓存文件路径。

    规则（全局缓存根下，保留父目录名以区分同名子数据集）::

        缓存根 / <lerobot_root 的目录名> / <root 相对 lerobot_root 的路径> / index.json

    - 单目录模式（root == lerobot_root）：``缓存根/<目录名>/index.json``
    - 父目录模式（root == lerobot_root/aaa）：``缓存根/<目录名>/aaa/index.json``
    """
    top = Path(lerobot_root)
    r = Path(root)
    rel = r.relative_to(top)
    rel_parts = () if rel == Path(".") else rel.parts
    return get_cache_root().joinpath(top.name, *rel_parts) / "index.json"


def compute_fingerprint(
    root: Path,
    video_feature_key: Optional[str],
    caption_key: str,
    video_feature_keywords: Optional[list[str]],
) -> str:
    """计算数据集内容指纹，用于判断缓存是否过期。

    指纹 = sha256(
        schema_version
        + 每个 episodes parquet 的 (相对路径 | size | mtime)
        + info.json 内容
        + 字段选择参数 (video_feature_key / caption_key / video_feature_keywords)
    )

    说明：
    - 只 ``stat`` parquet 元信息 + 读 info.json，不读 parquet 内容，很快。
    - 「字段选择参数」影响缓存里存哪些列（video key / caption 列），故纳入指纹；
      「过滤参数」（min_video_frames 等）不影响缓存内容，不纳入。
    """
    hasher = hashlib.sha256()
    hasher.update(str(SCHEMA_VERSION).encode())

    episodes_dir = root / "meta" / "episodes"
    for parquet_path in sorted(episodes_dir.glob("**/*.parquet")):
        try:
            st = parquet_path.stat()
        except OSError:
            continue
        rel = parquet_path.relative_to(root)
        hasher.update(f"{rel}|{st.st_size}|{st.st_mtime_ns}".encode())

    info_path = root / "meta" / "info.json"
    hasher.update(info_path.read_bytes())

    hasher.update(
        json.dumps(
            {
                "video_feature_key": video_feature_key,
                "caption_key": caption_key,
                # video_feature_keywords 可能来自 Hydra/OmegaConf 的 ListConfig，需先转原生 list 才能 json 序列化
                "video_feature_keywords": (
                    list(video_feature_keywords) if video_feature_keywords is not None else None
                ),
            },
            sort_keys=True,
        ).encode()
    )
    return hasher.hexdigest()


def read_cache(path: Path) -> Optional[dict[str, Any]]:
    """读取缓存文件；不存在或解析失败返回 None（视为 miss）。"""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def write_cache(path: Path, payload: dict[str, Any]) -> bool:
    """原子写缓存：先写临时文件，再 ``os.replace`` 改名，避免并发写坏。

    失败（如缓存目录只读）返回 False，调用方静默降级为「无缓存」。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except OSError:
        return False
