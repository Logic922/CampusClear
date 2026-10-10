"""老师专用客户端接口。

定位：这是「学生处老师」用的客户端，不是运维面板。
它只做两件事——**实时看**（轻量增强）与**回放查**（重量增强）。

关键约定（应你的要求）：**所有流都由服务器提供，客户端只消费**。
- 实时：服务器把增强帧编码成 MJPEG（``multipart/x-mixed-replace``），
  客户端用普通 ``<img>`` 就能看，浏览器原生支持、零依赖、离线可用；
- 回放：老师选一段录像 → 服务器起一个增强任务（重量档，慢慢跑）→
  产出 mp4，客户端用 ``<video>`` 播放。

认证：这是「老师自己电脑上的单机客户端」——服务只绑定 127.0.0.1，
访问者只能是本机，因此**本组接口不做登录**（单用户本地工具无需账号体系）。
运维面板 ``/api/v1`` 的 JWT/RBAC 鉴权保持原样，与客户端互不影响。

安全：录像名一律做 basename 化并校验落在录像目录内，杜绝路径穿越。
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse

from ..core.camera import CameraConfig
from ..core.demo_source import DemoScene
from ..core.modes import EnhanceMode, build_for_mode, policy_for
from ..core.planner import build_plan
from ..core.quality import QualityAnalyzer
from ..core.snapshot import enhance_still, list_presets
from ..core.sources import describe_methods, detect_scheme, vendor_live_url

logger = logging.getLogger(__name__)

router = APIRouter()

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".ts"}

# 演示用合成源（无摄像头时让客户端也有画面）。仅用于演示，不能当画质依据。
_demo_scene = DemoScene()
_demo_enhancer = None
_demo_policy = policy_for(EnhanceMode.REALTIME)


# --------------------------------------------------------------------------- #
# 依赖与工具
# --------------------------------------------------------------------------- #
def _settings(request: Request):
    return request.app.state.settings


def _supervisor(request: Request):
    return request.app.state.supervisor


def local_user(request: Request) -> dict:
    """本机客户端：**不做登录**。

    这是「老师自己电脑上的单机程序」，服务只绑定 127.0.0.1，访问者只能是本机。
    对单用户本地工具来说，登录界面只是负担，因此客户端这组接口不做鉴权。
    运维面板 /api/v1 的鉴权（JWT/RBAC）保持原样，不受影响。
    """
    return {
        "username": "local",
        "role": "admin",
        "client_ip": request.client.host if request.client else "",
    }


def _recordings_dir(settings) -> Path:
    path = Path(settings.data_dir) / "recordings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _out_dir(settings) -> Path:
    path = Path(settings.data_dir) / "client_out"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_recording(settings, name: str) -> Path:
    """把录像名解析为目录内的真实路径，防止路径穿越。"""
    base = _recordings_dir(settings).resolve()
    candidate = (base / Path(name).name).resolve()
    if candidate.parent != base or not candidate.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"录像不存在: {name}")
    return candidate


# --------------------------------------------------------------------------- #
# 实时
# --------------------------------------------------------------------------- #
@router.get("/client/streams", tags=["client"])
def list_streams(request: Request, user: dict = Depends(local_user)) -> dict:
    """返回客户端可看的实时源：真实通道 + 一个演示源。"""
    settings = _settings(request)
    supervisor = _supervisor(request)
    streams = []
    for cam in request.app.state.store.list_cameras():
        pipeline = supervisor.get_pipeline(cam.id)
        streams.append({
            "id": cam.id,
            "name": cam.name,
            "location": cam.location,
            "kind": "camera",
            "online": bool(pipeline and pipeline.state.connected and pipeline.state.running),
            "url": f"{settings.api_prefix}/client/live/{cam.id}.mjpeg",
        })
    # 演示源永远在线，保证没有摄像头时客户端不是一片空白
    streams.append({
        "id": "demo",
        "name": "演示画面（合成）",
        "location": "本机合成",
        "kind": "demo",
        "online": True,
        "url": f"{settings.api_prefix}/client/live/demo.mjpeg",
    })
    return {"streams": streams, "fps": 15}


def _mjpeg_demo(enhance: bool = True):
    """合成源 →（可选）实时轻量增强 → MJPEG。``enhance=False`` 输出未增强原图。"""
    global _demo_enhancer
    if enhance and _demo_enhancer is None:
        _demo_enhancer, _, _ = build_for_mode(EnhanceMode.REALTIME)
    analyzer = QualityAnalyzer()
    t0 = time.time()
    interval = 1.0 / 15
    while True:
        tick = time.time()
        frame = _demo_scene.frame(time.time() - t0)
        out = frame
        if enhance:
            plan = build_plan(analyzer.analyze(frame), _demo_policy)
            out = _demo_enhancer.enhance(frame, plan)
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(max(0.0, interval - (time.time() - tick)))


def _mjpeg_camera(pipeline, raw: bool = False):
    """真实通道 → 该通道的最新帧（增强或原图）→ MJPEG。"""
    while True:
        jpeg = pipeline.snapshot_raw_jpeg() if raw else pipeline.snapshot_jpeg()
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1.0 / 15)


def _live_response(request: Request, stream_id: str, raw: bool) -> StreamingResponse:
    headers = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
    if stream_id == "demo":
        gen = _mjpeg_demo(enhance=not raw)
    else:
        pipeline = _supervisor(request).get_pipeline(stream_id)
        if pipeline is None or not pipeline.state.running:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "该通道未在运行")
        gen = _mjpeg_camera(pipeline, raw=raw)
    return StreamingResponse(
        gen, media_type="multipart/x-mixed-replace; boundary=frame", headers=headers
    )


@router.get("/client/live/{stream_id}.mjpeg", tags=["client"])
def live_mjpeg(stream_id: str, request: Request) -> StreamingResponse:
    """增强后的实时画面。"""
    return _live_response(request, stream_id, raw=False)


@router.get("/client/live/{stream_id}/raw.mjpeg", tags=["client"])
def live_raw_mjpeg(stream_id: str, request: Request) -> StreamingResponse:
    """未增强的原始画面（供「原图/增强」对比）。"""
    return _live_response(request, stream_id, raw=True)


# --------------------------------------------------------------------------- #
# 数据源管理（连到学校监控服务器）
# --------------------------------------------------------------------------- #
@router.get("/client/methods", tags=["client"])
def list_methods(user: dict = Depends(local_user)) -> dict:
    """返回支持的取流连接方式，供界面下拉选择。"""
    return {"methods": describe_methods()}


@router.post("/client/sources", status_code=status.HTTP_201_CREATED, tags=["client"])
def add_source(request: Request, payload: dict, user: dict = Depends(local_user)) -> dict:
    """新增一个数据源：直接给 URL，或按厂商构造实时取流地址。"""
    store = request.app.state.store
    supervisor = _supervisor(request)

    if payload.get("brand"):
        try:
            url = vendor_live_url(
                payload["brand"],
                payload["host"],
                channel=int(payload.get("channel", 1)),
                sub=bool(payload.get("sub")),
                username=payload.get("username", ""),
                password=payload.get("password", ""),
                port=int(payload.get("port", 554)),
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"构造取流地址失败: {exc}") from exc
    else:
        url = (payload.get("url") or "").strip()

    if not url:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "请提供取流地址，或选择厂商并填写主机")

    cid = payload.get("id") or f"src_{uuid.uuid4().hex[:6]}"
    camera = CameraConfig(
        id=cid,
        name=(payload.get("name") or "").strip() or cid,
        location=(payload.get("location") or "").strip(),
        source_rtsp=url,
        enabled=True,
    )
    store.upsert_camera(camera)
    store.add_audit(
        "client_add_source", username=user["username"], target=cid,
        detail=f"scheme={detect_scheme(url) or 'file'}", client_ip=user.get("client_ip", ""),
    )

    started, error = True, ""
    try:
        supervisor.start_camera(camera)
    except RuntimeError as exc:  # 已入库但启动失败：界面可提示，改地址后重试
        started, error = False, str(exc)

    return {"id": cid, "name": camera.name, "started": started, "error": error}


@router.delete("/client/sources/{source_id}", tags=["client"])
def remove_source(source_id: str, request: Request, user: dict = Depends(local_user)) -> dict:
    store = request.app.state.store
    _supervisor(request).stop_camera(source_id)
    if not store.delete_camera(source_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "数据源不存在")
    store.add_audit(
        "client_remove_source", username=user["username"], target=source_id,
        client_ip=user.get("client_ip", ""),
    )
    return {"removed": source_id}


# --------------------------------------------------------------------------- #
# 截图增强（全屏时「截图并增强」）
# --------------------------------------------------------------------------- #
def _snapshots_dir(settings) -> Path:
    path = Path(settings.data_dir) / "snapshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _grab_one_frame(request: Request, stream_id: str) -> np.ndarray | None:
    """抓取某路**当前的原图帧**（截图后才做增强）。"""
    if stream_id == "demo":
        return _demo_scene.frame(time.time())
    pipeline = _supervisor(request).get_pipeline(stream_id)
    if pipeline is None:
        return None
    jpeg = pipeline.snapshot_raw_jpeg()
    if not jpeg:
        return None
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


@router.get("/client/presets", tags=["client"])
def snapshot_presets(user: dict = Depends(local_user)) -> dict:
    """截图增强可选的预设标准。"""
    return {"presets": list_presets()}


@router.post("/client/snapshot", status_code=status.HTTP_201_CREATED, tags=["client"])
def take_snapshot(request: Request, payload: dict, user: dict = Depends(local_user)) -> dict:
    """抓当前帧 → 按预设增强 → 存盘，返回可访问地址。"""
    settings = _settings(request)
    stream_id = payload.get("stream_id") or "demo"
    frame = _grab_one_frame(request, stream_id)
    if frame is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "抓不到该点位的画面")

    output, preset = enhance_still(
        frame, payload.get("preset") or "standard", models_dir=settings.models_dir
    )
    ok, buf = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "编码失败")
    name = f"{preset.key}-{int(time.time())}-{uuid.uuid4().hex[:6]}.jpg"
    (_snapshots_dir(settings) / name).write_bytes(buf.tobytes())
    request.app.state.store.add_audit(
        "client_snapshot", username=user["username"], target=stream_id,
        detail=preset.key, client_ip=user.get("client_ip", ""),
    )
    return {
        "id": name,
        "url": f"{settings.api_prefix}/client/snapshots/{name}",
        "preset": preset.key,
        "label": preset.label,
        "width": output.shape[1],
        "height": output.shape[0],
        "source_size": f"{frame.shape[1]}x{frame.shape[0]}",
    }


@router.get("/client/snapshots", tags=["client"])
def list_snapshots(request: Request, user: dict = Depends(local_user)) -> dict:
    settings = _settings(request)
    base = _snapshots_dir(settings)
    items = [
        {
            "name": p.name,
            "url": f"{settings.api_prefix}/client/snapshots/{p.name}",
            "mtime": int(p.stat().st_mtime),
        }
        for p in sorted(base.glob("*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)
    ]
    return {"snapshots": items}


@router.get("/client/snapshots/{name}", tags=["client"])
def get_snapshot(name: str, request: Request) -> FileResponse:
    base = _snapshots_dir(_settings(request)).resolve()
    path = (base / Path(name).name).resolve()
    if path.parent != base or not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "截图不存在")
    return FileResponse(str(path), media_type="image/jpeg")


# --------------------------------------------------------------------------- #
# 回放
# --------------------------------------------------------------------------- #
@router.get("/client/recordings", tags=["client"])
def list_recordings(request: Request, user: dict = Depends(local_user)) -> dict:
    settings = _settings(request)
    base = _recordings_dir(settings)
    items = []
    for path in sorted(base.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXT:
            items.append({
                "name": path.name,
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                "mtime": int(path.stat().st_mtime),
            })
    return {"recordings": items, "dir": str(base)}


class _Jobs:
    """极简回放任务管理：单线程排队执行，避免离开时把机器压垮。"""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cc-playback")

    def submit(self, settings, payload: dict) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "state": "queued",
            "created_at": time.time(),
            "params": payload,
            "progress_frames": 0,
            "result": None,
            "error": None,
            "output": str(_out_dir(settings) / f"{job_id}.mp4"),
        }
        with self._lock:
            self._jobs[job_id] = job
        self._pool.submit(self._run, settings, job)
        return job

    def _run(self, settings, job: dict) -> None:
        from ..core.playback import enhance_clip

        job["state"] = "running"
        params = job["params"]
        try:
            source = _safe_recording(settings, params["recording"])
            result = enhance_clip(
                source,
                job["output"],
                mode=params.get("mode", "forensic"),
                scale=params.get("scale"),
                region=params.get("region"),
                start=params.get("start"),
                end=params.get("end"),
                progress_cb=lambda n: job.__setitem__("progress_frames", n),
            )
            job["result"] = result.as_dict()
            job["state"] = "done"
        except Exception as exc:  # noqa: BLE001 - 任务失败要留在状态里供前端显示
            logger.exception("回放增强任务失败")
            job["error"] = str(exc)
            job["state"] = "failed"

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j["created_at"], reverse=True)


jobs = _Jobs()


@router.post("/client/jobs", status_code=status.HTTP_201_CREATED, tags=["client"])
def create_job(request: Request, payload: dict, user: dict = Depends(local_user)) -> dict:
    settings = _settings(request)
    _safe_recording(settings, payload.get("recording", ""))  # 校验存在且不含穿越
    mode = payload.get("mode", "forensic")
    if mode not in {m.value for m in EnhanceMode}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"未知档位: {mode}")
    job = jobs.submit(settings, payload)
    request.app.state.store.add_audit(
        "client_playback", username=user["username"], target=payload.get("recording", ""),
        detail=f"mode={mode}", client_ip=user.get("client_ip", ""),
    )
    return {"id": job["id"], "state": job["state"]}


@router.get("/client/jobs", tags=["client"])
def list_jobs(request: Request, user: dict = Depends(local_user)) -> dict:
    return {"jobs": jobs.list()}


@router.get("/client/jobs/{job_id}", tags=["client"])
def job_status(job_id: str, request: Request, user: dict = Depends(local_user)) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")
    return {k: v for k, v in job.items() if k != "output"}


@router.get("/client/jobs/{job_id}/result", tags=["client"])
def job_result(job_id: str, request: Request) -> FileResponse:
    job = jobs.get(job_id)
    if job is None or job["state"] != "done":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "结果不存在或尚未完成")
    return FileResponse(job["output"], media_type="video/mp4", filename=f"{job_id}.mp4")
