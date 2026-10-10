#!/usr/bin/env python3
"""把纯文本渲染成 foo2xqx 需要的 PBM 图（A4 @ 600x600, 1=黑）。

用法: render_pbm.py <input.txt> <outdir>
输出: <outdir>/page1.bin page2.bin ...（裸位图，随后由 print_txt.sh 加 P4 头）
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
W, H = 4960, 7016          # A4 @ 600x600 dpi
MARGIN = 300

f_title = ImageFont.truetype(FONT, 128, index=0)
f_head = ImageFont.truetype(FONT, 92, index=0)
f_body = ImageFont.truetype(FONT, 76, index=0)


def is_head(s):
    return s[:2] in ("一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、") or s.startswith("【")


def is_rule(s):
    t = s.strip()
    return bool(t) and set(t) <= set("=-─—")


def newpage():
    return Image.new("L", (W, H), 0), MARGIN   # 黑底，白字=着墨


def main():
    txt, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        if f.startswith("page") and f.endswith(".bin"):
            os.remove(os.path.join(outdir, f))

    img, y = newpage()
    d = ImageDraw.Draw(img)
    maxw = W - 2 * MARGIN
    pages = []
    first = True

    for raw in open(txt, encoding="utf-8").read().split("\n"):
        s = raw.rstrip("\n")
        if not s:
            y += 46
            continue
        if is_rule(s):
            d.line([MARGIN, y, W - MARGIN, y], fill=255, width=6)
            y += 46
            continue
        if first:
            font, gap = f_title, 40
            first = False
        elif is_head(s):
            font, gap = f_head, 26
        else:
            font, gap = f_body, 22
        cur = ""
        for ch in s:
            if d.textlength(cur + ch, font=font) > maxw and cur:
                d.text((MARGIN, y), cur, font=font, fill=255)
                y += font.size + gap
                cur = ch
                if y > H - MARGIN:
                    pages.append(img); img, y = newpage(); d = ImageDraw.Draw(img)
            else:
                cur += ch
        if cur:
            if y > H - MARGIN:
                pages.append(img); img, y = newpage(); d = ImageDraw.Draw(img)
            d.text((MARGIN, y), cur, font=font, fill=255)
            y += font.size + gap

    pages.append(img)
    for i, pg in enumerate(pages, 1):
        bw = pg.point(lambda v: 255 if v > 128 else 0).convert("1")
        with open(os.path.join(outdir, f"page{i}.bin"), "wb") as f:
            f.write(bw.tobytes())
    print(f"渲染完成：{len(pages)} 页 -> {outdir}")


if __name__ == "__main__":
    main()
