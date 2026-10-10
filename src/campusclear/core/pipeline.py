"""单路视频处理管线。

线程模型（每路两条线程，刻意保持简单）：

    [FFmpeg 解码进程] → reader 线程 → 有界帧队列 → worker 线程 → [FFmpeg 编码进程]

背压与丢帧策略：
- 帧队列有界（默认 8 帧）。队列满时**丢最旧的一帧**而不是等新帧。
  实时监控里，「延迟 3 秒的完整画面」价值远低于「延迟 200ms 的当前画面」；
- 丢弃在 reader 端完成，worker 端永远不会因为读队而阻塞，保证编码端节奏稳定。

推理帧率与输出帧率解耦：
- 源流 25fps，但推理只需要 8-12fps 就能覆盖安防取证需求；
- 未到推理时机的帧直接复用上一轮增强结果，输出仍是 25fps 的连续流，
  既保持安防平台侧「有流不断」的预期，又把 GPU 负载压到 1/2 ~ 1/3。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import CameraConfig, ResolvedCamera
from .ingest import RtspReader
from .metrics import StreamMetrics
from .output import RtspPusher
from .planner import EnhancePlan, LoadGovernor, PlannerPolicy, build_plan
from .quality import QualityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    running: bool = False
    connected: bool = False
    output_ready: bool = False


class StreamPipeline:
    """一路摄像头的完整处理管线。"""

    def __init__(
        self,
        resolved: ResolvedCamera,
        settings,
        enhancer,
        governor: LoadGovernor,
        *,
        policy: PlannerPolicy | None = None,
    ) -> None:
        self.resolved = resolved
        self.camera: CameraConfig = resolved.camera
        self.settings = settings
        self.enhancer = enhancer
        self.governor = governor
        self.policy = policy or PlannerPolicy(
            enable_lowlight=resolved.enable_lowlight,
            enable_denoise=resolved.enable_denoise,
            enable_deblur=resolved.enable_deblur,
            enable_sr=resolved.enable_sr,
            base_scale=resolved.scale,
            enable_static_reuse=settings.enable_static_reuse,
        )

        self.analyzer = QualityAnalyzer(
            low_light_threshold=settings.low_light_threshold,
            noise_threshold=settings.noise_threshold,
            blur_threshold=settings.blur_threshold,
            motion_threshold=settings.motion_threshold,
            reuse_max_noise=settings.static_reuse_max_noise,
        )
        self.metrics = StreamMetrics(camera_id=self.camera.id)
        self.state = PipelineState()

        self.reader = RtspReader(
            self.camera.source_rtsp,
            ffmpeg_bin=settings.ffmpeg_bin,
            timeout_us=settings.rtsp_timeout_us,
            width=self.camera.width,
            height=self.camera.height,
            fps=self.camera.fps,
            reconnect_backoff=tuple(settings.reconnect_backoff_seconds),
        )
        self.pusher: RtspPusher | None = None

        self._queue: queue.Queue = queue.Queue(maxsize=settings.queue_size)
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

        self._last_output: np.ndarray | None = None
        self._last_raw: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._last_infer_at = 0.0
        self._infer_interval = 1.0 / max(1, settings.infer_fps)
        self._frozen_counter = 0
        self._last_signature: bytes | None = None

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self.state.running:
            return
        self._stop_event.clear()

        probe = self.reader.probe()
        if probe is not None:
            self.reader.configure(probe)
            logger.info(
                "通道 %s 探测到 %dx%d@%.2f (%s)",
                self.camera.id, probe.width, probe.height, probe.fps, probe.codec,
            )
        self._configure_output()

        self.reader.open()
        self.state.connected = True

        self._threads = [
            threading.Thread(target=self._reader_loop, name=f"cc-read-{self.camera.id}", daemon=True),
            threading.Thread(target=self._worker_loop, name=f"cc-work-{self.camera.id}", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self.state.running = True
        logger.info("通道 %s 已启动", self.camera.id)

    def stop(self, timeout: float = 5.0) -> None:
        if not self.state.running:
            return
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()
        self.reader.close()
        if self.pusher is not None:
            self.pusher.close()
        self.state.running = False
        self.state.connected = False
        self.state.output_ready = False
        with self._frame_lock:
            self._last_output = None
            self._last_raw = None
        logger.info("通道 %s 已停止", self.camera.id)

    def restart(self) -> None:
        logger.info("通道 %s 重启", self.camera.id)
        self.stop()
        self.analyzer.reset()
        self._last_signature = None
        self._frozen_counter = 0
        self.start()

    # ------------------------------------------------------------------ #
    # 输出配置
    # ------------------------------------------------------------------ #
    def _configure_output(self) -> None:
        scale = self.resolved.scale if self.resolved.enable_sr else 1
        out_width = self.reader.width * scale
        out_height = self.reader.height * scale
        target = f"{self.settings.rtsp_output_base}/{self.camera.output_name}"
        self.pusher = RtspPusher(
            target,
            width=out_width,
            height=out_height,
            fps=self.resolved.fps,
            bitrate_kbps=self._suggest_bitrate(out_width, out_height),
            ffmpeg_bin=self.settings.ffmpeg_bin,
            username=self.settings.rtsp_output_user,
            password=self.settings.rtsp_output_password,
        )
        try:
            self.pusher.start()
            self.state.output_ready = True
        except OSError as exc:
            self.state.output_ready = False
            self.metrics.last_error = f"回推启动失败: {exc}"
            logger.error("通道 %s 回推启动失败: %s", self.camera.id, exc)

    @staticmethod
    def _suggest_bitrate(width: int, height: int) -> int:
        """按分辨率线性给码率：1080p 约 4Mbps，2K 约 8Mbps。

        内网带宽通常充裕，但摄像机侧解码与存储 NVR 的写盘能力有限，
        盲目提高码率反而会让 NVR 掉帧，因此不做激进设置。
        """
        pixels = width * height
        base = 4_000_000 / (1920 * 1080)
        return int(max(2000, min(20000, pixels * base / 1000)))

    # ------------------------------------------------------------------ #
    # reader 线程
    # ------------------------------------------------------------------ #
    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self.reader.alive:
                self.state.connected = False
                self.metrics.reconnects += 1
                if not self.reader.reconnect():
                    continue
                self.state.connected = True
                self.analyzer.reset()

            try:
                frame = self.reader.read(timeout=3.0)
            except Exception as exc:  # noqa: BLE001 - 单路异常不能拖垮整机
                self.metrics.last_error = f"读取异常: {exc}"
                frame = None

            if frame is None:
                self.metrics.last_error = self.reader.last_error or "读取失败"
                self.state.connected = False
                self._safe_reconnect()
                continue

            self.metrics.frames_read += 1
            self.metrics.last_frame_at = time.time()
            self._offer(frame)

    def _safe_reconnect(self) -> None:
        self.reader.close()
        time.sleep(1.0)
        try:
            self.reader.open()
            self.state.connected = True
            self.analyzer.reset()
        except OSError as exc:
            self.metrics.last_error = f"重连失败: {exc}"

    def _offer(self, frame: np.ndarray) -> None:
        """把帧放进队列；满则丢弃最旧的一帧。

        这就是「保实时不保完整」的具体实现。丢帧计数会暴露在指标页面上，
        让值班人员能看出「画面在动但服务器已经跟不上了」。
        """
        try:
            self._queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
            self.metrics.frames_dropped += 1
        except queue.Empty:  # pragma: no cover - 竞态下的兜底
            pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:  # pragma: no cover
            self.metrics.frames_dropped += 1

    # ------------------------------------------------------------------ #
    # worker 线程
    # ------------------------------------------------------------------ #
    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                output = self._process_frame(frame)
            except Exception as exc:  # noqa: BLE001
                self.metrics.last_error = f"处理异常: {exc}"
                logger.exception("通道 %s 处理异常", self.camera.id)
                output = frame

            if output is None:
                continue

            with self._frame_lock:
                self._last_output = output

            if self.pusher is None or not self.pusher.alive:
                if self.pusher is not None and not self.pusher.restart():
                    self.metrics.last_error = "回推不可用"
                    continue
                self._configure_output()

            if not self.pusher.write(output):
                self.metrics.last_error = "写帧失败，重建回推"
                self.pusher.restart()
                continue
            self.metrics.frames_written += 1

    # ------------------------------------------------------------------ #
    def _process_frame(self, frame: np.ndarray) -> np.ndarray | None:
        # 留一份未增强的原始帧，供客户端「原图 / 增强」对比使用
        with self._frame_lock:
            self._last_raw = frame
        quality = self.analyzer.analyze(frame)

        # ---- 画面冻结检测 ----
        # 老旧摄像头偶发「输出静止帧」故障（NVR 侧看起来在线，实际画面卡死）。
        # 用 8x8 缩略图的字节序列做指纹，连续多次相同即判定冻结。
        signature = cv2.resize(frame, (8, 8), interpolation=cv2.INTER_AREA).tobytes()
        if signature == self._last_signature:
            self._frozen_counter += 1
            if self._frozen_counter >= self.settings.frozen_frame_restart_times:
                self.metrics.last_error = "检测到画面冻结，触发重连"
                self._frozen_counter = 0
                self.analyzer.reset()
                self._safe_reconnect()
        else:
            self._frozen_counter = 0
        self._last_signature = signature

        level = self.governor.level
        plan = build_plan(quality, self.policy, level)
        self.metrics.record_plan(plan.reason, plan.stages())

        # ---- 复用判定 ----
        reuse = self._should_reuse(plan)
        if reuse and self._last_output is not None:
            self.metrics.frames_reused += 1
            return self._last_output

        roi = self._extract_roi(frame, quality) if self.camera.roi_enabled else None

        start = time.perf_counter()
        output = self.enhancer.enhance(frame, plan, roi)
        self.metrics.record_latency(time.perf_counter() - start)

        self.metrics.frames_enhanced += 1
        # 兜底实现可能返回同尺寸结果，保证后续编码参数与实际帧一致
        if output.shape[0] == frame.shape[0] and self.resolved.scale > 1 and plan.run_sr:
            output = cv2.resize(
                output,
                (frame.shape[1] * self.resolved.scale, frame.shape[0] * self.resolved.scale),
                interpolation=cv2.INTER_CUBIC,
            )
        return output

    def _should_reuse(self, plan: EnhancePlan) -> bool:
        """是否跳过本轮推理。"""
        if plan.reuse_previous:
            return True
        # 推理降频：不到间隔就复用上一结果
        now = time.monotonic()
        if now - self._last_infer_at < self._infer_interval:
            return True
        self._last_infer_at = now
        return False

    # ------------------------------------------------------------------ #
    def _extract_roi(self, frame: np.ndarray, quality) -> list[tuple] | None:
        """从运动掩膜中提取 ROI 外接矩形。

        ROI 的价值在于把有限算力压在「有东西在动」的区域。
        校园监控里绝大多数时间画面是空走廊、空操场，全帧超分纯属浪费。
        """
        try:
            mask = self.analyzer.motion_mask(frame)
        except Exception:  # noqa: BLE001
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        height, width = frame.shape[:2]
        frame_area = float(height * width)
        padding = self.settings.roi_padding

        boxes: list[tuple[int, int, int, int, float]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(w * h)
            # 太小的连通域多半是噪点，过滤掉
            if area < frame_area * 0.0005:
                continue
            boxes.append(
                (
                    max(0, x - padding),
                    max(0, y - padding),
                    min(width, x + w + padding),
                    min(height, y + h + padding),
                    area,
                )
            )

        if not boxes:
            return None

        boxes.sort(key=lambda item: item[4], reverse=True)
        boxes = boxes[: self.settings.roi_max_boxes]

        # 合并相近的框，减少条数也减少拼接开销
        boxes = _merge_boxes([(b[0], b[1], b[2], b[3]) for b in boxes])

        covered = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / frame_area
        if covered > self.policy.roi_max_area_ratio or covered < self.policy.roi_min_area_ratio:
            # 覆盖太少说明运动是零星噪点，覆盖太多说明不如整帧处理
            return None
        return boxes

    # ------------------------------------------------------------------ #
    # 对外
    # ------------------------------------------------------------------ #
    def snapshot_jpeg(self, quality: int = 85) -> bytes | None:
        with self._frame_lock:
            frame = self._last_output
        if frame is None:
            return None
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buffer.tobytes() if ok else None

    def snapshot_raw_jpeg(self, quality: int = 85) -> bytes | None:
        """未增强的原始帧（对比用）。"""
        with self._frame_lock:
            frame = self._last_raw
        if frame is None:
            return None
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buffer.tobytes() if ok else None

    def snapshot(self) -> dict:
        data = self.metrics.snapshot()
        data.update(
            {
                "name": self.camera.name,
                "location": self.camera.location,
                "enabled": self.camera.enabled,
                "running": self.state.running,
                "connected": self.state.connected,
                "output_ready": self.state.output_ready,
                "output_url": f"{self.settings.rtsp_output_base}/{self.camera.output_name}",
                "source_host": _host_of(self.camera.source_rtsp),
                "plan": self.metrics.snapshot().get("last_plan", ""),
                "queue_depth": self._queue.qsize(),
            }
        )
        if self.pusher is not None:
            data["encoder"] = self.pusher.stats()
        return data


# --------------------------------------------------------------------------- #
# 几何工具
# --------------------------------------------------------------------------- #
def _merge_boxes(boxes: list[tuple[int, int, int, int]], gap: int = 32) -> list[tuple[int, int, int, int]]:
    """把距离小于 ``gap`` 的矩形合并，直到不再变化。

    朴素 N² 实现即可：ROI 数量上限只有 6，性能完全不是瓶颈。
    """
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        result: list[tuple[int, int, int, int]] = []
        for box in merged:
            absorbed = False
            for index, existing in enumerate(result):
                if _near(existing, box, gap):
                    result[index] = (
                        min(existing[0], box[0]),
                        min(existing[1], box[1]),
                        max(existing[2], box[2]),
                        max(existing[3], box[3]),
                    )
                    absorbed = True
                    changed = True
                    break
            if not absorbed:
                result.append(box)
        merged = result
    return merged


def _near(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def _host_of(url: str) -> str:
    """只保留主机部分，接口返回时不要把源流完整地址和凭证暴露出去。"""
    if "://" not in url:
        return url
    _, _, rest = url.partition("://")
    if "@" in rest:
        _, _, rest = rest.partition("@")
    return rest.split("/")[0]
