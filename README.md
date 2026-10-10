# CampusClear 明眸 · 校园监控离线增强客户端

面向学校用户的**单机客户端**：从监控服务器（摄像头 / NVR / 平台）取流，
在本机做画质增强，再显示出来。**完全离线、数据不出内网、可物理断网运行。**

- **实时看**：拉流 → 轻量增强 → 浏览器直接看（多协议自适应）。
- **回放查**：选一段录像 → 重量增强 → 播放（慢一点没关系）。
- **不用登录**：服务只绑 `127.0.0.1`，打开即用。

---

## 主要能力

| 能力 | 说明 |
| --- | --- |
| 免登录单机运行 | 只绑定本机，无账号体系；运维面板 `/api/v1` 的鉴权不受影响 |
| 多协议取流 | RTSP / rtsps / RTMP / HTTP-FLV / HLS / HTTP(MJPEG) / RTP·UDP / 本地文件 |
| 按厂商接入 | 海康 / 大华 / 宇视：实时（主/子码流）与按时间段回放地址自动构造 |
| 三级增强 | `realtime` 轻量（实时）/ `balanced`（2×）/ `forensic` 取证放大（慢、最优） |
| 离线 AI 权重 | 已接入 Real-ESRGAN general-x4v3（4× 超分）；**缺失阶段自动回退经典算法** |
| 对比与全屏 | 点画面全屏；每路「对比」看 *原图 vs 增强* |
| 系统内网安全 | Argon2 + JWT + RBAC + 审计（运维面板）；输出流鉴权 |

## 目录结构

```
src/campusclear/     核心代码
  app.py             桌面启动器（也是打包入口）
  main.py            FastAPI 入口
  api/               client.py（客户端接口）、routes.py（运维接口）
  core/              取流/分析/决策/增强/回放/来源/调度...
  static/            client.html（老师客户端，纯本地零 CDN）
tests/               测试（含客户端接口、取流来源、增强器）
tools/               探测、试机、取证增强、权重校验、打包
models/              离线权重 + models_manifest.json
packaging/           AppImage 构建（spec + 脚本）
pylibs/              exfat 上用的依赖（pip --target，无符号链接）
CampusClear-x86_64.AppImage   打包好的可执行程序
```

## 快速开始

**直接运行（推荐）**
```bash
./CampusClear-x86_64.AppImage          # 双击亦可，自动打开界面
```

**从源码运行**（本目录在 exfat 上，用 pylibs 而非 venv）
```bash
./run_client.sh          # 启动客户端
./run_tests.sh           # 跑测试
./run_client.sh --port 9000 --no-window
```

**打包**
```bash
bash packaging/build.sh   # 需在装有 pyinstaller 的环境；见 packaging/campusclear.spec
```

## 离线权重

权重放在 `models/`，清单记 `文件名 + SHA256 + 倍率`；启动时校验，不匹配拒绝启动。
当前已接入：

| 阶段 | 模型 | 说明 |
| --- | --- | --- |
| `sr` | Real-ESRGAN general-x4v3（4.65 MB） | 4× 超分，动态尺寸 |
| lowlight / denoise / deblur | — | 未接入，**自动回退经典算法** |

```bash
python tools/verify_models.py --models-dir models                 # 校验
python tools/verify_models.py --models-dir models --write-manifest # 回填哈希
```

## 取流与探测

```bash
python tools/probe_source.py --methods
python tools/probe_source.py --url "rtsp://user:pass@10.0.0.5:554/Streaming/Channels/101"
python tools/probe_source.py --brand hikvision --host 10.0.0.5 --user admin --password xxx --sub
```

## 测试

```bash
./run_tests.sh            # 全绿（客户端接口 / 取流来源 / 增强器 / 规划 / 存储 / 认证 …）
```

## 说明与边界

- 只做**画质增强**，不做人脸识别、不采集生物特征。
- 不支持 RTSP/ONVIF、只有云端 SDK 的设备不适用。
- 厂商回放地址、通道编号各固件不同，**探测成功再写死**。
- CPU 上整帧重模型很慢——重量档本就是「离线按需」；可用 ROI/分块/降采样提速。
