# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：把「明眸客户端」打成可执行程序。

构建（在项目根目录执行）：
    pyinstaller packaging/campusclear.spec --noconfirm --clean

产物：
    dist/CampusClear/            （onedir，双击 CampusClear 启动）
    dist/CampusClear/CampusClear （Linux 可执行；Windows 上为 CampusClear.exe）

说明：
- uvicorn / onnxruntime 是**运行时才导入**的，PyInstaller 静态分析发现不了，
  因此这里显式收集它们的全部子模块；
- 界面静态文件（static/*.html）要作为数据一并打包，否则打包后 /client 打不开；
- 体积主要由 onnxruntime + opencv 决定；若不打算用重量 AI，可把 onnxruntime
  从 collect_submodules 里去掉以显著瘦身。
"""
import os

from PyInstaller.utils.hooks import collect_submodules

SRC = os.path.join(SPECPATH, "..", "src")

hiddenimports = []
for _pkg in ("uvicorn", "onnxruntime"):
    hiddenimports += collect_submodules(_pkg)
hiddenimports += [
    "uvicorn", "onnxruntime", "cv2", "numpy",
    "fastapi", "starlette", "pydantic", "argon2", "jwt",
    "campusclear.app", "campusclear.main",
]

datas = [
    (os.path.join(SRC, "campusclear", "static"), "campusclear/static"),
]

a = Analysis(
    [os.path.join(SRC, "campusclear", "app.py")],
    pathex=[SRC],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pytest", "matplotlib"],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CampusClear",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="CampusClear",
)
