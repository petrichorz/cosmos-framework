# SPDX-License-Identifier: OpenMDW-1.1
"""Read local LeRobot tables without an HF disk cache or a cross-sample LRU."""

from bisect import bisect_right

import pyarrow
import pyarrow.parquet as pq
from datasets.table import table_cast

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import get_hf_features_from_features
from lerobot.datasets.video_utils import get_safe_default_codec


class DirectParquetDataset:
    """只保留文件行号索引；每次取样一次读取各文件的必要列，取样后释放整表。

    视频沿用 LeRobot 的查询函数；不构造 LeRobotDataset，避免其初始化构建 Arrow 缓存。
    """

    def __init__(self, meta, *, tolerance_s, video_backend):
        self.meta = meta
        self.root = meta.root
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend or get_safe_default_codec()
        self.paths = sorted((self.root / "data").glob("*/*.parquet"))
        if not self.paths:
            raise FileNotFoundError(f"No data parquet files in {self.root / 'data'}")
        self.ends = []
        total = 0
        for path in self.paths:
            total += pq.read_metadata(path).num_rows
            self.ends.append(total)
        if total != meta.total_frames:
            raise ValueError(f"{self.root}: parquet rows {total} != meta.total_frames {meta.total_frames}")

    def read_window(self, start: int, stop: int, columns: list[str]) -> dict:
        """返回独立 Python 数据，避免窗口切片引用整文件的 Arrow buffers。

        区间为 [start, stop)，可跨文件；字段只读取一次。使用与 HF 相同的 schema
        转换检查实际维度和类型，忽略未声明的附加列，保留 timestamp 原始精度。
        """
        if not 0 <= start < stop <= self.ends[-1]:
            raise IndexError(f"Invalid row range [{start}, {stop}) for {self.root}")
        columns = list(dict.fromkeys(columns))
        features = get_hf_features_from_features({key: self.meta.features[key] for key in columns})
        if set(features) != set(columns):
            raise ValueError("Only non-video columns can be read from parquet")
        result = {key: [] for key in columns}
        file_index = bisect_right(self.ends, start)
        while start < stop:
            file_start = self.ends[file_index - 1] if file_index else 0
            file_stop = min(stop, self.ends[file_index])
            with pq.ParquetFile(self.paths[file_index]) as parquet:
                table = parquet.read(columns=columns)
            # 先截取样本再做类型转换；只物化样本，避免固定长度转换额外复制整文件。
            table = table_cast(table.slice(start - file_start, file_stop - start), features.arrow_schema)
            values = table.to_pydict()
            for key in columns:
                result[key].extend(values[key])
            del table, values
            start = file_stop
            file_index += 1
        # 样本已转为独立 Python 数据；在当前读取进程归还空闲 Arrow 内存，降低 RSS 驻留。
        # 每个窗口调用一次（跨文件也只调用一次），不降低解压期间的瞬时峰值。
        pyarrow.default_memory_pool().release_unused()
        return result

    def _query_videos(self, query_timestamps, ep_idx):
        return LeRobotDataset._query_videos(self, query_timestamps, ep_idx)
