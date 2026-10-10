"""全局配置。

设计原则：
1. 所有可调参数集中在本文件，便于现场部署时按硬件条件裁剪；
2. 每个参数都可通过 ``CAMPUSCLEAR_`` 前缀的环境变量覆盖，无需改代码；
3. 默认值面向「纯内网 + 可物理断网 + 零公网依赖」场景，
   不包含任何需要联网回调、遥测上报、在线拉取模型的默认行为。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# 环境变量读取助手
# --------------------------------------------------------------------------- #
def _str(key: str, default: str) -> str:
    raw = os.environ.get(key)
    return default if raw is None or raw.strip() == "" else raw.strip()


def _int(key: str, default: int) -> int:
    try:
        return int(_str(key, str(default)))
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(_str(key, str(default)))
    except ValueError:
        return default


def _bool(key: str, default: bool) -> bool:
    return _str(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _providers(key: str, default: list[str]) -> list[str]:
    raw = _str(key, ",".join(default))
    return [item.strip() for item in raw.split(",") if item.strip()]


DEFAULT_PROVIDERS = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


@dataclass
class Settings:
    """运行期配置快照。实例化时机决定读取到的环境变量值。"""

    # ---------------- 服务 ----------------
    app_name: str = field(default_factory=lambda: _str("CAMPUSCLEAR_APP_NAME", "CampusClear"))
    api_prefix: str = field(default_factory=lambda: _str("CAMPUSCLEAR_API_PREFIX", "/api/v1"))
    host: str = field(default_factory=lambda: _str("CAMPUSCLEAR_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _int("CAMPUSCLEAR_PORT", 8000))
    tls_cert: str = field(default_factory=lambda: _str("CAMPUSCLEAR_TLS_CERT", ""))
    tls_key: str = field(default_factory=lambda: _str("CAMPUSCLEAR_TLS_KEY", ""))
    # 管理面是否强制走 HTTPS（内网自签 CA）。生产环境建议开启。
    require_tls: bool = field(default_factory=lambda: _bool("CAMPUSCLEAR_REQUIRE_TLS", False))

    # ---------------- 安全 ----------------
    jwt_secret: str = field(default_factory=lambda: _str("CAMPUSCLEAR_JWT_SECRET", ""))
    jwt_algorithm: str = field(default_factory=lambda: _str("CAMPUSCLEAR_JWT_ALG", "HS256"))
    jwt_expire_minutes: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_JWT_EXPIRE_MINUTES", 480)
    )
    bootstrap_admin: str = field(
        default_factory=lambda: _str("CAMPUSCLEAR_BOOTSTRAP_ADMIN", "admin")
    )
    bootstrap_password: str = field(
        default_factory=lambda: _str("CAMPUSCLEAR_BOOTSTRAP_PASSWORD", "")
    )
    # 登录失败锁定：连续失败达到阈值即锁定该账号，抵御内网暴力破解
    login_max_failures: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_LOGIN_MAX_FAILURES", 5)
    )
    login_lock_seconds: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_LOGIN_LOCK_SECONDS", 300)
    )

    # ---------------- 目录 ----------------
    data_dir: Path = field(
        default_factory=lambda: Path(_str("CAMPUSCLEAR_DATA_DIR", "./data")).resolve()
    )
    models_dir: Path = field(
        default_factory=lambda: Path(_str("CAMPUSCLEAR_MODELS_DIR", "./models")).resolve()
    )
    log_dir: Path = field(
        default_factory=lambda: Path(_str("CAMPUSCLEAR_LOG_DIR", "./data/logs")).resolve()
    )
    snapshots_dir: Path = field(
        default_factory=lambda: Path(
            _str("CAMPUSCLEAR_SNAPSHOT_DIR", "./data/snapshots")
        ).resolve()
    )

    # ---------------- 推理 ----------------
    providers: list[str] = field(default_factory=lambda: _providers("CAMPUSCLEAR_PROVIDERS", DEFAULT_PROVIDERS))
    device_id: int = field(default_factory=lambda: _int("CAMPUSCLEAR_DEVICE_ID", 0))
    use_fp16: bool = field(default_factory=lambda: _bool("CAMPUSCLEAR_USE_FP16", True))
    tile_max_pixels: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_TILE_MAX_PIXELS", 1920 * 1080 // 2)
    )
    tile_overlap: int = field(default_factory=lambda: _int("CAMPUSCLEAR_TILE_OVERLAP", 16))
    # 首次加载后缓存 TensorRT engine 的目录（离线预构建，避免启动期重复编译）
    trt_engine_cache_dir: Path = field(
        default_factory=lambda: Path(
            _str("CAMPUSCLEAR_TRT_CACHE", "./data/trt_cache")
        ).resolve()
    )
    # 模型清单与权重文件必须校验 SHA256，防止内网传输/拷贝过程被篡改
    verify_model_hash: bool = field(
        default_factory=lambda: _bool("CAMPUSCLEAR_VERIFY_MODEL_HASH", True)
    )

    # ---------------- 流与调度 ----------------
    max_streams: int = field(default_factory=lambda: _int("CAMPUSCLEAR_MAX_STREAMS", 8))
    queue_size: int = field(default_factory=lambda: _int("CAMPUSCLEAR_QUEUE_SIZE", 8))
    infer_fps: int = field(default_factory=lambda: _int("CAMPUSCLEAR_INFER_FPS", 12))
    output_fps: int = field(default_factory=lambda: _int("CAMPUSCLEAR_OUTPUT_FPS", 25))
    rtsp_output_base: str = field(
        default_factory=lambda: _str("CAMPUSCLEAR_RTSP_OUTPUT_BASE", "rtsp://127.0.0.1:8554")
    )
    rtsp_output_user: str = field(
        default_factory=lambda: _str("CAMPUSCLEAR_RTSP_OUTPUT_USER", "campusclear")
    )
    rtsp_output_password: str = field(
        default_factory=lambda: _str("CAMPUSCLEAR_RTSP_OUTPUT_PASSWORD", "")
    )
    ffmpeg_bin: str = field(default_factory=lambda: _str("CAMPUSCLEAR_FFMPEG", "ffmpeg"))
    # RTSP 接收超时（微秒）。内网交换机瞬断时靠它触发重连。
    rtsp_timeout_us: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_RTSP_TIMEOUT_US", 5_000_000)
    )
    reconnect_backoff_seconds: list[int] = field(
        default_factory=lambda: [
            _int("CAMPUSCLEAR_RECONNECT_BASE_SECONDS", 1),
            _int("CAMPUSCLEAR_RECONNECT_MAX_SECONDS", 30),
        ]
    )

    # ---------------- 增强策略 ----------------
    enable_lowlight: bool = field(default_factory=lambda: _bool("CAMPUSCLEAR_ENABLE_LOWLIGHT", True))
    enable_denoise: bool = field(default_factory=lambda: _bool("CAMPUSCLEAR_ENABLE_DENOISE", True))
    enable_deblur: bool = field(default_factory=lambda: _bool("CAMPUSCLEAR_ENABLE_DEBLUR", True))
    enable_sr: bool = field(default_factory=lambda: _bool("CAMPUSCLEAR_ENABLE_SR", True))
    sr_scale: int = field(default_factory=lambda: _int("CAMPUSCLEAR_SR_SCALE", 2))
    enable_static_reuse: bool = field(
        default_factory=lambda: _bool("CAMPUSCLEAR_ENABLE_STATIC_REUSE", True)
    )

    # ---------------- 质量判据阈值 ----------------
    low_light_threshold: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_LOW_LIGHT_THRESHOLD", 60.0)
    )
    noise_threshold: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_NOISE_THRESHOLD", 6.0)
    )
    blur_threshold: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_BLUR_THRESHOLD", 90.0)
    )
    motion_threshold: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_MOTION_THRESHOLD", 0.004)
    )
    # 降采样后的噪声超过此值时，认为背景建模不可信，关闭静止帧复用。
    # 代价是算力（所有帧都真实处理），收益是不出现「画面在动却复用旧帧」的卡顿拖影。
    static_reuse_max_noise: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_STATIC_REUSE_MAX_NOISE", 3.0)
    )
    roi_padding: int = field(default_factory=lambda: _int("CAMPUSCLEAR_ROI_PADDING", 24))
    roi_max_boxes: int = field(default_factory=lambda: _int("CAMPUSCLEAR_ROI_MAX_BOXES", 6))

    # ---------------- 负载治理 ----------------
    gpu_high_watermark: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_GPU_HIGH_WATERMARK", 85.0)
    )
    gpu_low_watermark: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_GPU_LOW_WATERMARK", 60.0)
    )

    # ---------------- 健康检查 ----------------
    health_interval_seconds: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_HEALTH_INTERVAL", 20)
    )
    # 连续 N 次判定为「画面冻结」即触发重启
    frozen_frame_restart_times: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_FROZEN_RESTART_TIMES", 3)
    )

    def __post_init__(self) -> None:
        for path in (
            self.data_dir,
            self.models_dir,
            self.log_dir,
            self.snapshots_dir,
            self.trt_engine_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # ---------------- 派生属性 ----------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "campusclear.db"

    @property
    def secret_path(self) -> Path:
        """JWT 密钥落盘位置。落盘后即使重启也保持同一密钥，避免已签发 token 全部失效。"""
        return self.data_dir / "secret.key"

    @property
    def tls_enabled(self) -> bool:
        return bool(self.tls_cert and self.tls_key)

    def resolve_jwt_secret(self) -> str:
        """解析 JWT 密钥：优先环境变量，其次本地文件，最后随机生成并持久化。

        之所以落盘而不是每次随机，是为了让内网服务重启后已签发的令牌仍然有效；
        文件权限设为 0600，仅服务账户可读。
        """
        if self.jwt_secret:
            return self.jwt_secret
        if self.secret_path.exists():
            return self.secret_path.read_text(encoding="utf-8").strip()
        import secrets

        generated = secrets.token_urlsafe(48)
        self.secret_path.write_text(generated, encoding="utf-8")
        try:
            os.chmod(self.secret_path, 0o600)
        except OSError:
            # Windows 上 chmod 语义有限，忽略即可
            pass
        return generated

    def provider_profile(self) -> dict:
        """给 /system/health 用的脱敏摘要。"""
        return {
            "providers": self.providers,
            "device_id": self.device_id,
            "fp16": self.use_fp16,
            "max_streams": self.max_streams,
            "sr_scale": self.sr_scale,
            "tls_enabled": self.tls_enabled,
            "offline_mode": True,
        }


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """进程内单例。``refresh=True`` 供测试重置使用。"""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
