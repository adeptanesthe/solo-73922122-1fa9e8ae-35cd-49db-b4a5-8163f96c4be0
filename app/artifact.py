"""修正后权重的产物管理。

仅保存“通过整图复查”的最新一份权重，路径固定在服务本地数据目录：
  * publish : 原子替换发布文件（临时文件 + os.replace）；
  * revoke  : 撤下当前发布（无通过复查的权重时网页不再提供下载）；
  * current : 读取当前发布；无则返回 None。

用户上传的原始权重只存在于请求内存中，从不被本模块改写。
"""

from __future__ import annotations

import os
from pathlib import Path

ARTIFACT_FILENAME = "refined_weights.json"


class ArtifactStore:
    def __init__(self, directory: str | os.PathLike[str]):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / ARTIFACT_FILENAME

    def exists(self) -> bool:
        return self.path.exists()

    def current(self) -> bytes | None:
        if not self.path.exists():
            return None
        return self.path.read_bytes()

    def publish(self, payload: bytes) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        # 同目录临时文件 + 原子替换，避免半写文件被下载
        with open(tmp, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def revoke(self) -> bool:
        try:
            self.path.unlink()
            return True
        except FileNotFoundError:
            return False
