"""运行指标采集。

三条底线要求：
1. 只读本地数据，不发起任何网络请求；
2. 无第三方监控依赖（不引入 Prometheus 客户端也可运行，可选导出）；
3. 采集本身的开销可以忽略（纯计数 + 滑动窗口，无锁竞争热点）。
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class StreamMetrics:
    """单路通道运行时指标。"""

    camera_id: str
    started_at: float = field(default_factory=time.time)

    frames_read: int = 0
    frames_dropped: int = 0          # 队列背压丢弃
    frames_enhanced: int = 0         # 真正走过推理
    frames_reused: int = 0           # 静止帧/降帧复用
    frames_written: int = 0
    inference_seconds: float = 0.0

    last_frame_at: float = 0.0
    last_error: str = ""
    reconnects: int = 0

    stage_usage: dict[str, int] = field(default_factory=dict)
    _latencies: deque = field(default_factory=lambda: deque(maxlen=120))
    _plan_reasons: deque = field(default_factory=lambda: deque(maxlen=20))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------ #
    def record_latency(self, seconds: float) -> None:
        with self._lock:
            self._latencies.append(seconds)
            self.inference_seconds += seconds

    def record_plan(self, reason: str, stages: list[str]) -> None:
        with self._lock:
            self._plan_reasons.append(reason)
            for stage in stages:
                self.stage_usage[stage] = self.stage_usage.get(stage, 0) + 1

    # ------------------------------------------------------------------ #
    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies) * 1000.0

    @property
    def p95_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        ordered = sorted(self._latencies)
        index = min(len(ordered) - 1, int(len(ordered) * 0.95))
        return ordered[index] * 1000.0

    @property
    def effective_fps(self) -> float:
        uptime = max(1e-6, time.time() - self.started_at)
        return self.frames_written / uptime

    @property
    def drop_ratio(self) -> float:
        total = self.frames_read
        return self.frames_dropped / total if total else 0.0

    @property
    def reuse_ratio(self) -> float:
        total = self.frames_enhanced + self.frames_reused
        return self.frames_reused / total if total else 0.0

    def snapshot(self) -> dict:
        with self._lock:
            reasons = list(self._plan_reasons)[-1:] or [""]
        return {
            "camera_id": self.camera_id,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "frames_read": self.frames_read,
            "frames_dropped": self.frames_dropped,
            "frames_enhanced": self.frames_enhanced,
            "frames_reused": self.frames_reused,
            "frames_written": self.frames_written,
            "effective_fps": round(self.effective_fps, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "drop_ratio": round(self.drop_ratio, 4),
            "reuse_ratio": round(self.reuse_ratio, 4),
            "reconnects": self.reconnects,
            "stage_usage": dict(self.stage_usage),
            "last_plan": reasons[0],
            "last_error": self.last_error,
        }


# --------------------------------------------------------------------------- #
# GPU 采集
# --------------------------------------------------------------------------- #
_GPU_LOCK = threading.Lock()
_GPU_AVAILABLE = True


def read_gpu_util(device_id: int = 0) -> dict | None:
    """读取 GPU 利用率与显存。

    优先使用 ``nvidia-smi``（CUDA 镜像里必然存在，无需额外 Python 依赖）；
    不可用时返回 None，由上层按「无观测」处理，绝不因此报错或阻塞出流。
    """
    global _GPU_AVAILABLE
    if not _GPU_AVAILABLE:
        return None
    with _GPU_LOCK:
        if shutil.which("nvidia-smi") is None:
            _GPU_AVAILABLE = False
            return None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device_id}",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            _GPU_AVAILABLE = False
            return None
        if result.returncode != 0:
            _GPU_AVAILABLE = False
            return None
        line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) < 4:
            return None
        try:
            used, total = float(parts[1]), float(parts[2])
            return {
                "utilization": float(parts[0]),
                "memory_used_mb": used,
                "memory_total_mb": total,
                "memory_ratio": round(used / total, 4) if total else 0.0,
                "temperature_c": float(parts[3]),
            }
        except (ValueError, IndexError):
            return None
