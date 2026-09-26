"""Generate mnist_gui.ico: a pixelated handwritten-style "7" on a blue rounded square."""
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
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


def main():
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, SIZE - 8, SIZE - 8), radius=52, fill=(37, 99, 235, 255))

    rows, cols = len(DIGIT), len(DIGIT[0])
    cell = 20
    x0 = (SIZE - cols * cell) // 2 - 8
    y0 = (SIZE - rows * cell) // 2
    for r, line in enumerate(DIGIT):
        for c, ch in enumerate(line):
            if ch == "#":
                x, y = x0 + c * cell, y0 + r * cell
                d.rounded_rectangle((x + 1, y + 1, x + cell - 1, y + cell - 1), radius=4, fill=(255, 255, 255, 255))

    out = Path(__file__).resolve().with_name("mnist_gui.ico")
    img.save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    img.save(out.with_suffix(".png"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
