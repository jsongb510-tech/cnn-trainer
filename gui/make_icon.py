"""Generate the window icons:
mnist_gui.ico   - a pixelated handwritten-style "7" on a blue rounded square
cifar10_gui.ico - a 3x3 grid of colorful image tiles on a teal rounded square
"""
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
DIGIT = [  # 7 x 9 pixel "7"
    ".######",
    ".######",
    "......#",
    ".....##",
    "....##.",
    "...##..",
    "...#...",
    "..##...",
    "..#....",
]
TILE_COLORS = [
    (125, 211, 252), (74, 222, 128), (253, 224, 71),
    (96, 165, 250), (34, 197, 94), (251, 146, 60),
    (244, 114, 182), (255, 255, 255), (167, 139, 250),
]


def rounded_base(color):
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, SIZE - 8, SIZE - 8), radius=52, fill=color)
    return img, d


def mnist_icon():
    img, d = rounded_base((37, 99, 235, 255))
    rows, cols = len(DIGIT), len(DIGIT[0])
    cell = 20
    x0 = (SIZE - cols * cell) // 2 - 8
    y0 = (SIZE - rows * cell) // 2
    for r, line in enumerate(DIGIT):
        for c, ch in enumerate(line):
            if ch == "#":
                x, y = x0 + c * cell, y0 + r * cell
                d.rounded_rectangle((x + 1, y + 1, x + cell - 1, y + cell - 1), radius=4, fill=(255, 255, 255, 255))
    return img


def cifar10_icon():
    img, d = rounded_base((13, 148, 136, 255))
    tile, gap = 52, 12
    x0 = y0 = (SIZE - 3 * tile - 2 * gap) // 2
    for k, color in enumerate(TILE_COLORS):
        r, c = divmod(k, 3)
        x, y = x0 + c * (tile + gap), y0 + r * (tile + gap)
        d.rounded_rectangle((x, y, x + tile, y + tile), radius=10, fill=color + (255,))
    return img


def main():
    here = Path(__file__).resolve().parent
    for name, make in (("mnist_gui", mnist_icon), ("cifar10_gui", cifar10_icon)):
        out = here / f"{name}.ico"
        make().save(out, sizes=ICO_SIZES)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
