#!/usr/bin/env bash
# 在 MOKOUSPACE（exfat）上跑测试。
# 依赖在 pylibs/（pip --target 安装），不依赖 venv。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE/src:$HERE/pylibs${PYTHONPATH:+:$PYTHONPATH}"
cd "$HERE"
exec python3 -m pytest "$@"
