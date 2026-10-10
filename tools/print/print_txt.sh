#!/usr/bin/env bash
# 一键把纯文本打印到 HP LaserJet P1007（无需系统驱动）。
#
# 用法:
#   ./print_txt.sh <文件.txt> [打印机名]
# 默认打印机名 HP_P1007。
#
# 原理: 文本 -> PIL 渲染成 PBM 裸位图 -> foo2xqx 转成打印机原生 ZJS
#       -> 通过 raw 队列发给 USB 打印机。全程不需要装内核驱动/root。
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)"
TXT="${1:?用法: print_txt.sh <文件.txt> [打印机名]}"
PRINTER="${2:-HP_P1007}"
VENV_PY="/home/mokou/项目/CampusClear/.venv/bin/python"
OUT="/tmp/opencode/print"

mkdir -p "$OUT"
"$VENV_PY" "$SELF/render_pbm.py" "$TXT" "$OUT"

shopt -s nullglob
for bin in "$OUT"/page*.bin; do
    name="$(basename "$bin" .bin)"
    pbm="$OUT/$name.pbm"; xqx="$OUT/$name.xqx"
    { printf 'P4\n4960 7016\n'; cat "$bin"; } > "$pbm"
    "$SELF/foo2xqx" -r600x600 -g4960x7016 -p9 -m1 -s7 -d1 -n1 -T3 -D0 < "$pbm" > "$xqx"
    lp -d "$PRINTER" -o raw "$xqx"
    echo "已发送 $name  ($(stat -c%s "$xqx") 字节)"
done

echo "全部发送完毕。查看队列: lpstat -o"
