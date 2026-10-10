"""RTSP 拉流。

为什么不用 ``cv2.VideoCapture``？

1. 无法精细控制 RTSP 传输方式与超时，内网交换机抖动时会静默卡死而不报错；
2. 内置缓冲不可控，最长可积压数秒，对「实时清晰化」是致命的；
3. 出错后的重连语义不明确，长时间运行容易出现「进程还活着但画面冻住」。

改用 FFmpeg 子进程 + ``rawvideo`` 管道：
- 参数完全可控（TCP 传输、低延迟、超时自恢复、丢弃损坏帧）；
- 一帧固定 ``H*W*3`` 字节，读满即一帧，边界清晰，不存在「半个包」问题；
- 进程死了就是死了，可被监督器明确感知并重建。

代价是每路多一个 FFmpeg 进程（约 20-40MB 内存），在校园内网服务器上完全可接受。
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np

from .sources import build_input_args

logger = logging.getLogger(__name__)


@dataclass
class StreamProbe:
    """探测到的流参数。"""

    width: int
    height: int
    fps: float
    codec: str = "unknown"


class RtspSourceError(RuntimeError):
    """拉流不可恢复错误。"""


class RtspReader:
    """单路 RTSP 读取器（自带重连）。"""

    def __init__(
        self,
        url: str,
        *,
        ffmpeg_bin: str = "ffmpeg",
        timeout_us: int = 5_000_000,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        transport: str = "tcp",
        reconnect_backoff: tuple[int, int] = (1, 30),
    ) -> None:
        self.url = url
        self.ffmpeg_bin = ffmpeg_bin
        self.timeout_us = timeout_us
        self.transport = transport
        self.reconnect_backoff = reconnect_backoff

        # 目标输出尺寸。源流尺寸未知时先用 1080p 占位，probe 后修正。
        self.width = width or 1920
        self.height = height or 1080
        self.fps = fps or 25

        self._process: subprocess.Popen | None = None
        self._frame_bytes = self.width * self.height * 3
        self._lock = threading.Lock()
        self.reconnects = 0
        self.last_error: str = ""

    # ------------------------------------------------------------------ #
    def configure(self, probe: StreamProbe) -> None:
        """按探测结果设定实际读取尺寸（只在未显式指定时生效）。"""
        with self._lock:
            if self.width == 1920 and self.height == 1080:
                self.width = probe.width
                self.height = probe.height
            self.fps = probe.fps or self.fps
            self._frame_bytes = self.width * self.height * 3

    def probe(self, timeout: float = 10.0) -> StreamProbe | None:
        """用 ffprobe 探测流参数。ffprobe 不在场时返回 None，由调用方按默认值处理。"""
        ffprobe = self.ffmpeg_bin.replace("ffmpeg", "ffprobe")
        command = [
            ffprobe,
            "-v", "error",
            "-rtsp_transport", self.transport,
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,codec_name",
            "-of", "default=noprint_wrappers=1",
            self.url,
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("ffprobe 不可用或超时: %s", exc)
            return None
        if result.returncode != 0:
            logger.warning("ffprobe 探测失败: %s", result.stderr.strip()[:200])
            return None

        values: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()

        try:
            width = int(values.get("width", 0)) or self.width
            height = int(values.get("height", 0)) or self.height
        except ValueError:
            return None

        fps = self.fps
        raw_rate = values.get("avg_frame_rate", "")
        if "/" in raw_rate:
            num, _, den = raw_rate.partition("/")
            try:
                if float(den) > 0:
                    fps = round(float(num) / float(den), 2)
            except ValueError:
                pass
        return StreamProbe(width=width, height=height, fps=fps, codec=values.get("codec_name", "unknown"))

    # ------------------------------------------------------------------ #
    def open(self) -> None:
        """启动 FFmpeg 解码进程。

        输入参数按 URL 协议自适应（rtsp / rtmp / http-flv / hls / rtp / 本地文件），
        见 ``core/sources.py``——不再只认 RTSP。
        """
        command = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "error",
            *build_input_args(
                self.url, rtsp_transport=self.transport, timeout_us=self.timeout_us
            ),
            "-i", self.url,
            "-an",                      # 丢弃音频：安防回传只需画面
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vsync", "0",              # 不做帧率重采样，保持源时序
            "-",
        ]
        logger.info("启动拉流: %s", _mask_url(self.url))
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._frame_bytes * 2,
        )

    def read(self, timeout: float = 5.0) -> np.ndarray | None:
        """读取一帧。返回 ``None`` 表示本路当前不可用，调用方应触发重连。"""
        process = self._process
        if process is None or process.stdout is None:
            return None
        deadline = time.monotonic() + timeout
        buffer = bytearray()
        while len(buffer) < self._frame_bytes:
            if time.monotonic() > deadline:
                self.last_error = "读取超时"
                return None
            chunk = process.stdout.read(self._frame_bytes - len(buffer))
            if not chunk:
                self.last_error = "解码进程已退出（流中断或源不可达）"
                return None
            buffer.extend(chunk)
        frame = np.frombuffer(bytes(buffer), dtype=np.uint8)
        return frame.reshape((self.height, self.width, 3))

    def reconnect(self) -> bool:
        """按指数退避重建拉流连接。返回是否成功打开。"""
        self.close()
        self.reconnects += 1
        base, maximum = self.reconnect_backoff
        delay = min(maximum, base * (2 ** min(self.reconnects - 1, 5)))
        logger.warning(
            "第 %d 次重连 %s，等待 %ds（原因：%s）",
            self.reconnects, _mask_url(self.url), delay, self.last_error or "未知",
        )
        time.sleep(delay)
        try:
            self.open()
            return True
        except OSError as exc:
            self.last_error = f"重连失败: {exc}"
            return False

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdout:
                process.stdout.close()
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        except OSError:  # pragma: no cover
            pass

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None


def _mask_url(url: str) -> str:
    """日志脱敏：抹掉 RTSP URL 中的账号密码。"""
    if "@" not in url or "://" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.partition("@")
    return f"{scheme}://***@{host}"
