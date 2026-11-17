# -*- coding: utf-8 -*-
"""生成 AQuant 应用图标 assets/app_icon.ico（Pillow，离线，无外链）。

风格：深色底 + 网格 + 递增柱体 + 上行折线，256 画布，导出多尺寸 ICO。
仅研究用途，不构成投资建议。
"""

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ICON_PATH = ASSETS / "app_icon.ico"
SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def build_icon(size=256):
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    scale = size / 256.0

    def box(value):
        return tuple(int(round(v * scale)) for v in value)

    # 深色圆角底
    left, top, right, bottom = box((6, 6, 250, 250))
    draw.rounded_rectangle((left, top, right, bottom), radius=int(52 * scale),
                           fill=(14, 26, 43, 255))
    # 细网格
    for step in range(1, 4):
        x = int(round(6 + (244 / 4) * step * scale))
        y = int(round(6 + (244 / 4) * step * scale))
        draw.line([(x, top), (x, bottom)],
                  fill=(28, 51, 80, 255), width=max(1, int(2 * scale)))
        draw.line([(left, y), (right, y)],
                  fill=(28, 51, 80, 255), width=max(1, int(2 * scale)))

    # 递增柱体（低波低估值风格的抽象表达）
    bars = [
        (52, 178, 86, 226, (46, 196, 182, 255)),
        (104, 148, 138, 226, (64, 214, 194, 255)),
        (156, 112, 190, 226, (244, 162, 97, 255)),
    ]
    for left, top, right, bottom, color in bars:
        draw.rounded_rectangle(box((left, top, right, bottom)),
                               radius=int(8 * scale), fill=color)

    # 上行折线
    points = [(44, 150), (92, 122), (140, 96), (188, 70), (214, 48)]
    draw.line([box((x, y)) for x, y in points], fill=(246, 246, 246, 255),
              width=max(2, int(10 * scale)), joint="curve")
    for x, y in points[1:]:
        radius = 9 * scale
        draw.ellipse(box((x - radius, y - radius, x + radius, y + radius)),
                     fill=(246, 246, 246, 255))
    return canvas


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    icon = build_icon(256)
    icon.save(ICON_PATH, format="ICO", sizes=SIZES)
    print("图标已生成：", ICON_PATH)
    print("尺寸：", ", ".join("%dx%d" % s for s in SIZES))


if __name__ == "__main__":
    main()
