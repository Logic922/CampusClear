"""明眸客户端 · 桌面启动器。

这是打包成「一个程序」后的入口。它做三件事：

1. 在**本机**（默认 127.0.0.1）启动一个内置服务——**只给本机浏览器/窗口看**，
   不对外监听，符合「纯客户端、本地处理」的定位；
2. 数据目录放在用户可写的位置（打包后不能再依赖源码目录）；
3. 打开界面：有内嵌浏览器内核（pywebview）就开独立窗口，否则用系统默认浏览器。

用法:
    python -m campusclear.app                 # 启动并打开界面
    python -m campusclear.app --port 9000     # 指定端口
    python -m campusclear.app --no-window     # 只起服务，不自动开界面
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _user_data_dir() -> Path:
    """打包后数据目录：Windows 用 %LOCALAPPDATA%，其它平台用 ~/.campusclear。"""
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "CampusClear"
    return Path.home() / ".campusclear"


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) != 0


def _pick_port(host: str, want: int) -> int:
    """优先用指定端口；若被占用（可能是上一个没退干净的实例）则自动换一个空闲端口。

    否则会出现：新实例绑不上 8000，而旧实例还在 8000 上监听，
    启动器误判「服务已就绪」，于是把浏览器开到旧实例上——用户看到的就是旧界面。
    """
    if want and _port_free(host, want):
        return want
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _wait_ready(host: str, port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.15)
    return False


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="明眸客户端（本地处理，界面只在本机）")
    ap.add_argument("--host", default="127.0.0.1", help="仅本机使用，勿改为对外地址")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", default=None, help="录像/输出/数据库目录")
    ap.add_argument("--models-dir", default=None, help="ONNX 权重目录（重量增强用）")
    ap.add_argument("--no-window", action="store_true", help="不自动打开界面")
    ap.add_argument("--url-only", action="store_true", help="只打印地址后退出（调试用）")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    data_dir = Path(args.data_dir).resolve() if args.data_dir else _user_data_dir()
    models_dir = Path(args.models_dir).resolve() if args.models_dir else (data_dir / "models")
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "recordings").mkdir(exist_ok=True)

    # 在导入应用之前把配置写进环境：config 是惰性读取环境变量的
    os.environ.setdefault("CAMPUSCLEAR_DATA_DIR", str(data_dir))
    os.environ.setdefault("CAMPUSCLEAR_MODELS_DIR", str(models_dir))
    os.environ.setdefault("CAMPUSCLEAR_LOG_DIR", str(data_dir / "logs"))
    os.environ.setdefault("CAMPUSCLEAR_HOST", args.host)
    # 客户端形态：无权重时不该拒绝启动，校验哈希放宽（权重到位后可在配置里打开）
    os.environ.setdefault("CAMPUSCLEAR_VERIFY_MODEL_HASH", "false")

    port = _pick_port(args.host, args.port)
    url = f"http://{args.host}:{port}/client"

    if args.url_only:
        print(url)
        return 0

    import uvicorn

    try:
        from .main import app
    except ImportError:
        # PyInstaller 打包后以「脚本」方式启动入口，包上下文丢失，需要绝对导入
        from campusclear.main import app

    config = uvicorn.Config(app, host=args.host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, name="cc-app", daemon=True)
    thread.start()

    # 只认「我们自己这个 uvicorn 是否真的起来了」，而不是「端口是否有人应答」——
    # 后者在旧实例占用端口时会误判。
    deadline = time.time() + 20
    while time.time() < deadline and not getattr(server, "started", False):
        if not thread.is_alive():
            break
        time.sleep(0.1)
    if not getattr(server, "started", False):
        print("服务启动失败（端口被占用或启动异常）", file=sys.stderr)
        return 1
    print(f"明眸客户端已就绪： {url}")
    print(f"数据目录： {data_dir}")

    if args.no_window:
        thread.join()
        return 0

    if not _open_window(url, data_dir):
        # 窗口内核不可用时退回默认浏览器
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
        thread.join()
    return 0


def _open_window(url: str, data_dir: Path) -> bool:
    """尝试用 pywebview 开独立窗口；不可用返回 False 交给浏览器兜底。"""
    try:
        import webview  # type: ignore
    except Exception:  # noqa: BLE001
        return False
    try:
        webview.create_window("明眸 · 校园监控增强客户端", url, width=1280, height=800)
        webview.start()
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
