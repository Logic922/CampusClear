"""多路调度监督器。

职责边界：
- 按数据库中的通道配置创建/销毁 ``StreamPipeline``；
- 周期性采集 GPU 利用率与队列压力，驱动全局负载档位；
- 通道级别的健康兜底（进程活着但画面冻结、回推断开等）。

刻意不做的两件事：
1. 不做跨通道的帧搬运（零拷贝只发生在单路内部）。跨路复用会引入复杂的
   锁竞争，收益不如直接做 ROI；
2. 不在监督器里做任何推理。监督器必须保持轻量，即使 GPU 卡死也能正常
   响应管理接口，否则运维会失去唯一的观测窗口。
"""

from __future__ import annotations

import logging
import threading
import time

from ..models.registry import ModelRegistry
from .camera import CameraConfig
from .enhancer import ClassicalEnhancer, OnnxEnhancer
from .metrics import read_gpu_util
from .pipeline import StreamPipeline
from .planner import LoadGovernor
from .store import CameraStore

logger = logging.getLogger(__name__)


def build_enhancer(settings, registry: ModelRegistry):
    """按可用性选择推理实现：有权重且装得上 onnxruntime 走 GPU，否则经典兜底。"""
    enhancer = OnnxEnhancer(
        registry,
        use_fp16=settings.use_fp16,
        tile_max_pixels=settings.tile_max_pixels,
        tile_overlap=settings.tile_overlap,
        device_id=settings.device_id,
    )
    if enhancer.available():
        logger.info("推理链路: ONNX（阶段 %s）", ",".join(registry.available_stages()))
        return enhancer
    logger.warning("未检测到可用离线权重，推理链路退化为经典 CPU 算法")
    return ClassicalEnhancer()


class StreamSupervisor:
    """所有视频通道的总控。"""

    def __init__(self, settings, store: CameraStore, registry: ModelRegistry | None = None) -> None:
        self.settings = settings
        self.store = store
        self.registry = registry or ModelRegistry(
            settings.models_dir, verify_hash=settings.verify_model_hash
        )
        self.enhancer = build_enhancer(settings, self.registry)
        self.governor = LoadGovernor(
            high_watermark=settings.gpu_high_watermark,
            low_watermark=settings.gpu_low_watermark,
        )

        self._pipelines: dict[str, StreamPipeline] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._monitor: threading.Thread | None = None
        self.last_gpu: dict | None = None
        self.started_at = time.time()

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self.reload()
        self._monitor = threading.Thread(target=self._monitor_loop, name="cc-monitor", daemon=True)
        self._monitor.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._monitor is not None:
            self._monitor.join(timeout=5)
        with self._lock:
            for pipeline in self._pipelines.values():
                pipeline.stop()
            self._pipelines.clear()
        logger.info("调度器已关闭")

    # ------------------------------------------------------------------ #
    # 通道管理
    # ------------------------------------------------------------------ #
    def reload(self) -> dict:
        """按数据库配置重建通道集合。

        采用「先停后建」而不是热更新：通道配置变更（如分辨率、倍率）会改变
        编码器参数与内部缓冲尺寸，热更新的复杂度远高于重建带来的几秒中断。
        内网运维场景下，几秒中断完全可接受。
        """
        cameras = self.store.list_cameras()
        desired = {cam.id: cam for cam in cameras if cam.enabled}

        with self._lock:
            for camera_id in list(self._pipelines):
                if camera_id not in desired:
                    self._pipelines.pop(camera_id).stop()
                    logger.info("通道 %s 已移除或停用", camera_id)

            for camera_id, camera in desired.items():
                if camera_id in self._pipelines:
                    self._pipelines.pop(camera_id).stop()
                if len(self._pipelines) >= self.settings.max_streams:
                    logger.warning(
                        "已达最大并发通道数 %d，跳过 %s", self.settings.max_streams, camera_id
                    )
                    continue
                pipeline = StreamPipeline(
                    camera.resolve(self.settings), self.settings, self.enhancer, self.governor
                )
                try:
                    pipeline.start()
                    self._pipelines[camera_id] = pipeline
                except Exception as exc:  # noqa: BLE001 - 单路失败不影响其余通道
                    logger.error("通道 %s 启动失败: %s", camera_id, exc)
                    pipeline.stop()

        return self.snapshot()

    def start_camera(self, camera: CameraConfig) -> bool:
        with self._lock:
            if len(self._pipelines) >= self.settings.max_streams and camera.id not in self._pipelines:
                raise RuntimeError(f"已达最大并发通道数 {self.settings.max_streams}")
            if camera.id in self._pipelines:
                self._pipelines.pop(camera.id).stop()
            pipeline = StreamPipeline(
                camera.resolve(self.settings), self.settings, self.enhancer, self.governor
            )
            try:
                pipeline.start()
            except Exception as exc:  # noqa: BLE001 - 一律转成可读错误，配置已入库不回滚
                pipeline.stop()
                raise RuntimeError(f"通道 {camera.id} 启动失败: {exc}") from exc
            self._pipelines[camera.id] = pipeline
        return True

    def stop_camera(self, camera_id: str) -> bool:
        with self._lock:
            pipeline = self._pipelines.pop(camera_id, None)
        if pipeline is None:
            return False
        pipeline.stop()
        return True

    def restart_camera(self, camera_id: str) -> bool:
        with self._lock:
            pipeline = self._pipelines.get(camera_id)
            if pipeline is None:
                return False
            pipeline.restart()
        return True

    def get_pipeline(self, camera_id: str) -> StreamPipeline | None:
        with self._lock:
            return self._pipelines.get(camera_id)

    # ------------------------------------------------------------------ #
    # 监控循环
    # ------------------------------------------------------------------ #
    def _monitor_loop(self) -> None:
        interval = max(5, self.settings.health_interval_seconds)
        while not self._stop_event.wait(interval):
            try:
                self._tick()
            except Exception:  # noqa: BLE001 - 监控线程绝不能死
                logger.exception("监控循环异常")

    def _tick(self) -> None:
        gpu = read_gpu_util(self.settings.device_id)
        self.last_gpu = gpu

        with self._lock:
            pipelines = list(self._pipelines.values())

        # 队列压力：取所有通道中最深的队列占队列上限的比例
        pressure = 0.0
        for pipeline in pipelines:
            depth = pipeline._queue.qsize()  # noqa: SLF001 - 同包内访问，属内部协作
            pressure = max(pressure, depth / max(1, self.settings.queue_size))

        util = gpu["utilization"] if gpu else None
        previous = self.governor.level
        level = self.governor.observe(util, queue_pressure=pressure)
        if level is not previous:
            logger.warning("负载档位变更: %s → %s（%s）", previous.value, level.value, self.governor.last_reason)

        # 通道健康兜底：进程还在但指标长期不动，说明线程可能已死
        for pipeline in pipelines:
            stats = pipeline.metrics
            age = time.time() - (stats.last_frame_at or 0)
            if stats.last_frame_at and age > 120:
                logger.warning(
                    "通道 %s 已 %.0f 秒无新帧，触发重启", pipeline.camera.id, age
                )
                try:
                    pipeline.restart()
                except Exception:  # noqa: BLE001
                    logger.exception("通道 %s 重启失败", pipeline.camera.id)

    # ------------------------------------------------------------------ #
    # 快照
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        with self._lock:
            rows = [pipeline.snapshot() for pipeline in self._pipelines.values()]
        return {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "stream_count": len(rows),
            "max_streams": self.settings.max_streams,
            "load": self.governor.snapshot(),
            "gpu": self.last_gpu,
            "enhancer": getattr(self.enhancer, "name", "unknown"),
            "streams": rows,
        }

    def metrics_snapshot(self) -> dict:
        with self._lock:
            pipelines = list(self._pipelines.values())
        return {
            "load": self.governor.snapshot(),
            "gpu": self.last_gpu,
            "streams": [pipeline.metrics.snapshot() for pipeline in pipelines],
            "stage_latency": (
                self.enhancer.stats_snapshot()
                if isinstance(self.enhancer, OnnxEnhancer)
                else {}
            ),
        }
