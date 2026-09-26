"""Trainer window: pick options, train a script in WSL on the GPU, show accuracy, confusion matrix and mistakes.

Runs with a Windows Python (tkinter + Pillow); training itself happens inside WSL via wsl.exe.
Each dataset is a profile in PROFILES; mnist_gui.pyw / cifar10_gui.pyw just call main("<profile>").
Launch from the repo's WSL path (e.g. \\\\wsl$\\Ubuntu-24.04\\home\\<user>\\cnn-trainer\\gui\\mnist_gui.pyw)
and the distro and project folder are picked up from that path automatically.

Training scripts talk to this window with "@@{json}" lines (see ../jsonlog.py).
"""
import base64
import ctypes
import io
import json
import math
import queue
import re
import shlex
import subprocess
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

HERE = Path(__file__).resolve()
# Used when the script isn't being run from a \\wsl$ path.
DEFAULT_DISTRO = "Ubuntu-24.04"
DEFAULT_PROJECT = "~/cnn_examples"


def wsl_location(script):
    """(distro, Linux repo dir) for a script under \\\\wsl$\\<distro>\\... or \\\\wsl.localhost\\<distro>\\..."""
    host, _, distro = script.drive.lstrip("\\").partition("\\")
    if host.lower() in ("wsl$", "wsl.localhost") and distro:
        return distro, "/" + "/".join(script.parts[1:-2])  # repo = parent of gui/
    return DEFAULT_DISTRO, DEFAULT_PROJECT


DISTRO, PROJECT = wsl_location(HERE)
PROJECT_SHOWN = re.sub(r"^/home/[^/]+", "~", PROJECT)  # for display: don't show the Linux user name
CREATE_NO_WINDOW = 0x08000000
TF_ENV = "source .venv-tf/bin/activate && KERAS_BACKEND=tensorflow TF_CPP_MIN_LOG_LEVEL=2 python"
LOG_NOISE = ("oneDNN custom operations", "absl::InitializeLog", "cpu_feature_guard", "rebuild TensorFlow",
             "To enable the following instructions")
BATCH_SIZES = ["32", "64", "128", "256", "512"]
LEARNING_RATES = ["0.0001", "0.0003", "0.001", "0.003", "0.01"]


@dataclass
class Opt:
    """One training option in the side panel, passed to the script as a command-line flag."""
    key: str
    label: str
    kind: str  # "int", "float", "choice" or "check"
    default: str | bool
    flag: str  # for "check": passed (without a value) when the box differs from its default
    lo: float = 0
    hi: float = 0
    step: float = 1
    values: list = field(default_factory=list)


@dataclass
class Profile:
    title: str
    app_id: str
    icon: str
    noun: str  # what a class is called in the UI, e.g. "숫자"
    labels: list  # class names shown in the UI
    image: tuple  # (PIL mode, side length) of the raw images sent by the script, or ("JPEG", side)
    backends: dict  # UI name -> (script file, shell command template with {args})
    options: list
    thumb: int = 100  # displayed size of a mistake thumbnail, in 96-dpi pixels


PROFILES = {
    "mnist": Profile(
        title="MNIST 학습기", app_id="mnist_trainer.gui", icon="mnist_gui.ico",
        noun="숫자", labels=[str(d) for d in range(10)], image=("L", 28),
        backends={
            "Keras · PyTorch 엔진": ("mnist2.py", "KERAS_BACKEND=torch .venv/bin/python -u mnist2.py --json {args}"),
            "Keras · TensorFlow 엔진": ("mnist2.py", TF_ENV + " -u mnist2.py --json {args}"),
        },
        options=[
            Opt("epochs", "Epoch 수", "int", "5", "--epochs", lo=1, hi=100),
            Opt("batch", "배치 크기", "choice", "128", "--batch-size", values=BATCH_SIZES),
            Opt("lr", "학습률", "choice", "0.001", "--lr", values=LEARNING_RATES),
            Opt("dropout", "Dropout", "float", "0.25", "--dropout", lo=0.0, hi=0.8, step=0.05),
        ]),
    "cifar10": Profile(
        title="CIFAR-10 학습기", app_id="cifar10_trainer.gui", icon="cifar10_gui.ico",
        noun="클래스", labels=["비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭"],
        image=("RGB", 32),
        backends={"PyTorch": ("cifar10_torch.py", ".venv/bin/python -u cifar10_torch.py --json {args}")},
        options=[
            Opt("epochs", "Epoch 수", "int", "30", "--epochs", lo=1, hi=200),
            Opt("batch", "배치 크기", "choice", "128", "--batch-size", values=BATCH_SIZES),
            Opt("lr", "학습률", "choice", "0.001", "--lr", values=LEARNING_RATES),
            Opt("wd", "Weight decay", "choice", "0.0005", "--weight-decay",
                values=["0", "0.0001", "0.0005", "0.001", "0.005"]),
            Opt("augment", "데이터 증강 (반전·이동)", "check", True, "--no-augment"),
        ]),
    "imagenette": Profile(
        title="Imagenette 학습기", app_id="imagenette_trainer.gui", icon="imagenette_gui.ico",
        noun="클래스",
        labels=["물고기", "스패니얼", "카세트", "전기톱", "교회", "호른", "쓰레기차", "주유기", "골프공", "낙하산"],
        image=("JPEG", 128), thumb=150,
        backends={"PyTorch": ("imagenette_torch.py", ".venv/bin/python -u imagenette_torch.py --json {args}")},
        options=[
            Opt("model", "모델", "choice", "resnet18", "--model", values=["resnet18", "resnet50"]),
            Opt("pretrained", "전이학습 (ImageNet 사전학습 가중치)", "check", False, "--pretrained"),
            Opt("epochs", "Epoch 수", "int", "20", "--epochs", lo=1, hi=100),
            Opt("batch", "배치 크기", "choice", "64", "--batch-size", values=["16", "32", "64", "128"]),
            Opt("lr", "학습률", "choice", "0.001", "--lr", values=LEARNING_RATES),
            Opt("size", "이미지 크기 (px)", "choice", "224", "--img-size", values=["128", "160", "224", "288"]),
        ]),
    "imagewoof": Profile(
        title="Imagewoof 학습기 (개 품종)", app_id="imagewoof_trainer.gui", icon="imagewoof_gui.ico",
        noun="품종",
        labels=["시추", "리지백", "비글", "폭스하운드", "보더테리어", "호주테리어", "골든리트리버", "쉽독",
                "사모예드", "딩고"],
        image=("JPEG", 128), thumb=150,
        backends={"PyTorch": ("imagenette_torch.py",
                              ".venv/bin/python -u imagenette_torch.py --dataset imagewoof --json {args}")},
        options=[
            Opt("model", "모델", "choice", "resnet18", "--model", values=["resnet18", "resnet50"]),
            Opt("pretrained", "전이학습 (ImageNet 사전학습 가중치)", "check", False, "--pretrained"),
            Opt("epochs", "Epoch 수", "int", "20", "--epochs", lo=1, hi=100),
            Opt("batch", "배치 크기", "choice", "64", "--batch-size", values=["16", "32", "64", "128"]),
            Opt("lr", "학습률", "choice", "0.001", "--lr", values=LEARNING_RATES),
            Opt("size", "이미지 크기 (px)", "choice", "224", "--img-size", values=["128", "160", "224", "288"]),
        ]),
}

COLORS = {
    "panel": "#ffffff", "border": "#d0d7de", "text": "#1f2328", "muted": "#6b7280", "grid": "#e5e7eb",
    "train": "#2563eb", "val": "#ea580c", "diag": "#2563eb", "err": "#dc2626", "empty": "#f6f8fa",
    "ok": "#15803d", "bad": "#b91c1c",
}
FONT = ("Malgun Gothic", 10)
FONT_S = ("Malgun Gothic", 9)
FONT_B = ("Malgun Gothic", 10, "bold")
FONT_BIG = ("Malgun Gothic", 26, "bold")


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


class Chart(tk.Canvas):
    title = ""

    def __init__(self, master, scale):
        super().__init__(master, bg=COLORS["panel"], highlightthickness=1,
                         highlightbackground=COLORS["border"])
        self.s = scale
        self.bind("<Configure>", lambda e: self.redraw())

    def px(self, v):
        return int(round(v * self.s))

    def draw_frame(self):
        self.delete("all")
        self.create_text(self.px(12), self.px(12), text=self.title, anchor="nw", font=FONT_B, fill=COLORS["text"])

    def placeholder(self, w, h):
        self.create_text(w / 2, h / 2, text="학습을 시작하면 여기에 표시됩니다", font=FONT_S, fill=COLORS["muted"])


class LearningCurve(Chart):
    title = "학습 곡선 (정확도)"
    SERIES = (("accuracy", "학습 데이터", COLORS["train"]), ("val_accuracy", "테스트 데이터", COLORS["val"]))

    def __init__(self, master, scale):
        super().__init__(master, scale)
        self.epochs, self.history = 5, []

    def reset(self, epochs):
        self.epochs, self.history = epochs, []
        self.redraw()

    def add(self, logs):
        self.history.append(logs)
        self.redraw()

    def redraw(self):
        self.draw_frame()
        w, h = self.winfo_width(), self.winfo_height()
        if w < 120 or h < 120:
            return
        px = self.px
        x_end = w - px(14)
        for key, label, color in reversed(self.SERIES):  # legend, right-aligned
            tid = self.create_text(x_end, px(21), text=label, anchor="e", font=FONT_S, fill=COLORS["muted"])
            x0 = self.bbox(tid)[0] - px(6)
            self.create_line(x0 - px(18), px(21), x0, px(21), fill=color, width=px(3), capstyle="round")
            x_end = x0 - px(30)
        if not self.history:
            self.placeholder(w, h)
            return

        left, right, top, bottom = px(56), px(20), px(46), px(56)
        vals = [e[k] for e in self.history for k, _, _ in self.SERIES if k in e]
        lo = max(0.0, math.floor((min(vals) - 0.002) * 100) / 100)
        hi = 1.0
        step = next(s for s in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if (hi - lo) / s <= 6)
        n = max(self.epochs, 1)

        def X(epoch):
            return left + (w - left - right) * ((epoch - 1) / (n - 1) if n > 1 else 0.5)

        def Y(v):
            return top + (h - top - bottom) * (hi - v) / (hi - lo)

        v = math.ceil(lo / step - 1e-9) * step
        while v <= hi + 1e-9:
            y = Y(v)
            self.create_line(left, y, w - right, y, fill=COLORS["grid"])
            self.create_text(left - px(8), y, text=f"{v * 100:.{1 if step < 0.01 else 0}f}%", anchor="e",
                             font=FONT_S, fill=COLORS["muted"])
            v += step
        every = max(1, math.ceil(n / 10))
        for e in range(1, n + 1):
            if (e - 1) % every == 0 or e == n:
                self.create_text(X(e), h - bottom + px(8), text=str(e), anchor="n", font=FONT_S, fill=COLORS["muted"])
        self.create_text((left + w - right) / 2, h - px(6), text="epoch", anchor="s", font=FONT_S, fill=COLORS["muted"])

        last = self.history[-1]
        top_key = max((k for k, _, _ in self.SERIES if k in last), key=lambda k: last[k])
        r = px(3.5) if n <= 40 else px(2)
        for key, _, color in self.SERIES:
            pts = [(X(i + 1), Y(e[key])) for i, e in enumerate(self.history) if key in e]
            if len(pts) > 1:
                self.create_line(*[c for p in pts for c in p], fill=color, width=px(2.5), smooth=False)
            for x, y in pts:
                self.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=COLORS["panel"], width=px(1.5))
            if pts:  # value label: higher series above its point, lower series below, so they don't collide
                x, y = pts[-1]
                dy = -px(12) if key == top_key else px(12)
                self.create_text(x + (px(8) if len(pts) < n else -px(8)), y + dy, text=f"{last[key] * 100:.2f}%",
                                 anchor="w" if len(pts) < n else "e", font=FONT_S, fill=color)


class ConfusionMatrix(Chart):
    title = "혼동 행렬 (테스트 데이터)"

    def __init__(self, master, scale, labels, noun, on_cell=None):
        super().__init__(master, scale)
        self.labels, self.noun = labels, noun
        self.cm, self.on_cell, self.geom = None, on_cell, None
        self.bind("<Button-1>", self.click)
        self.bind("<Motion>", lambda e: self.configure(cursor="hand2" if self.cell_at(e.x, e.y) else ""))

    def set(self, cm):
        self.cm = cm
        total = sum(map(sum, cm)) if cm else 0
        self.title = f"혼동 행렬 (테스트 데이터 {total:,}장)" if cm else "혼동 행렬 (테스트 데이터)"
        self.redraw()

    def cell_at(self, x, y):
        """(true, pred) of a clickable off-diagonal cell under (x, y), else None."""
        if self.cm is None or self.geom is None:
            return None
        gx, gy, cell, n = self.geom
        i, j = math.floor((y - gy) / cell), math.floor((x - gx) / cell)
        if 0 <= i < n and 0 <= j < n and i != j and self.cm[i][j]:
            return i, j
        return None

    def click(self, e):
        c = self.cell_at(e.x, e.y)
        if c and self.on_cell:
            self.on_cell(*c)

    def redraw(self):
        self.draw_frame()
        self.geom = None
        w, h = self.winfo_width(), self.winfo_height()
        if w < 120 or h < 120:
            return
        if self.cm is None:
            self.placeholder(w, h)
            return
        px, n, cm = self.px, len(self.cm), self.cm
        long_labels = max(len(l) for l in self.labels) > 1
        tick_font = FONT_S if long_labels else FONT_B
        label_w = max(tkfont.Font(font=tick_font).measure(l) for l in self.labels)
        left = px(34) + label_w
        # Column names wider than a cell are staggered over up to 3 rows so neighbours don't overlap.
        rows, top = 1, px(70)
        for _ in range(2):  # more label rows -> taller header -> smaller cells -> maybe more rows
            cell = max(8, min((w - left - px(14)) // n, (h - top - px(36)) // n))
            rows = min(3, max(1, math.ceil(label_w / (cell * 0.9))))
            top = px(70) + (rows - 1) * px(16)
        cell = max(8, min((w - left - px(14)) // n, (h - top - px(36)) // n))
        gx = left + (w - left - px(14) - cell * n) // 2
        gy = top
        self.geom = (gx, gy, cell, n)
        self.create_text(w / 2, h - px(8), text="빨간 칸을 클릭하면 그 경우의 틀린 이미지를 볼 수 있어요",
                         anchor="s", font=FONT_S, fill=COLORS["muted"])
        diag_max = max(cm[i][i] for i in range(n)) or 1
        off_max = max((cm[i][j] for i in range(n) for j in range(n) if i != j), default=0) or 1
        num_font = ("Malgun Gothic", -max(7, int(cell * 0.3)))

        self.create_text(gx + cell * n / 2, gy - px(28) - (rows - 1) * px(16), text=f"예측한 {self.noun} →",
                         font=FONT_S, fill=COLORS["muted"])
        self.create_text(gx - label_w - px(22), gy + cell * n / 2, text=f"← 실제 {self.noun}", angle=90,
                         font=FONT_S, fill=COLORS["muted"])
        for k, label in enumerate(self.labels):
            y = gy - px(10) - (k % rows) * px(16)
            self.create_text(gx + cell * (k + 0.5), y, text=label, font=tick_font, fill=COLORS["text"])
            self.create_text(gx - px(8), gy + cell * (k + 0.5), text=label, anchor="e", font=tick_font,
                             fill=COLORS["text"])
        for i in range(n):
            for j in range(n):
                v = cm[i][j]
                if i == j:
                    t = 0.35 + 0.65 * v / diag_max
                    fill = mix(COLORS["panel"], COLORS["diag"], t)
                elif v:
                    t = 0.15 + 0.85 * v / off_max
                    fill = mix(COLORS["panel"], COLORS["err"], t)
                else:
                    t, fill = 0, COLORS["empty"]
                x0, y0 = gx + j * cell, gy + i * cell
                self.create_rectangle(x0, y0, x0 + cell, y0 + cell, fill=fill, outline=COLORS["panel"], width=px(1))
                if cell >= px(18) and v:
                    self.create_text(x0 + cell / 2, y0 + cell / 2, text=str(v), font=num_font,
                                     fill="#ffffff" if t > 0.5 else COLORS["text"])


class MistakesView(ttk.Frame):
    """Scrollable grid of misclassified test images with true/predicted labels, shown PAGE tiles at a time."""
    PAGE = 200

    def __init__(self, master, scale, profile, on_count):
        super().__init__(master, padding=int(10 * scale))
        self.s, self.on_count = scale, on_count
        self.labels, self.noun = profile.labels, profile.noun
        self.mode, self.side = profile.image
        self.items, self.total, self.filter = [], None, None
        self.current, self.shown = [], 0  # filtered + sorted items, and how many have tiles
        self.tiles, self.photos, self.cols = [], [], 0
        self.disp = self.px(profile.thumb)  # thumbnail side on screen
        # Tiny images (MNIST, CIFAR) are enlarged with visible pixels; photos are resampled smoothly.
        self.resample = Image.NEAREST if self.side * 2 <= self.disp else Image.LANCZOS
        self.stacked = max(len(l) for l in self.labels) > 1  # long class names: true/pred on separate lines
        self.tile_w = max(self.disp, self.px(130)) + self.px(16)
        self.sorts = {
            "확신도 높은 순": lambda m: -m["conf"],
            f"실제 {self.noun} 순": lambda m: (m["true"], m["pred"], -m["conf"]),
            f"예측 {self.noun} 순": lambda m: (m["pred"], m["true"], -m["conf"]),
        }

        head = ttk.Frame(self)
        head.pack(fill="x")
        self.summary = ttk.Label(head, font=FONT_B)
        self.summary.pack(side="left")
        self.show_all = ttk.Button(head, text="전체 보기", command=self.clear_filter)
        self.sort = tk.StringVar(value="확신도 높은 순")
        sort_box = ttk.Combobox(head, values=list(self.sorts), textvariable=self.sort, state="readonly", width=14)
        sort_box.pack(side="right")
        sort_box.bind("<<ComboboxSelected>>", lambda e: self.render())
        ttk.Label(head, text="정렬").pack(side="right", padx=(0, self.px(6)))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=(self.px(8), 0))
        self.canvas = tk.Canvas(body, bg=COLORS["panel"], highlightthickness=1,
                                highlightbackground=COLORS["border"], yscrollincrement=self.px(24))
        sb = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=COLORS["panel"], padx=self.px(6), pady=self.px(6))
        self.more_btn = ttk.Button(self.inner, command=self.add_page)
        self.win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self.on_wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))
        self.render()

    def px(self, v):
        return int(round(v * self.s))

    def on_wheel(self, e):
        self.canvas.yview_scroll(-int(e.delta / 120) * 3, "units")

    def on_resize(self, e):
        self.canvas.itemconfigure(self.win, width=e.width)
        cols = max(1, (e.width - self.px(12)) // self.tile_w)
        if cols != self.cols:
            self.cols = cols
            self.layout()

    # ---------- data ----------
    def clear(self):
        self.items, self.total, self.filter = [], None, None
        self.render()
        self.on_count(None)

    def set_items(self, total, items):
        self.items, self.total, self.filter = items, total, None
        self.render()
        self.on_count(total)

    def set_filter(self, true, pred):
        self.filter = (true, pred)
        self.render()

    def clear_filter(self):
        self.filter = None
        self.render()

    # ---------- drawing ----------
    def render(self):
        for t in self.tiles:
            t.destroy()
        self.tiles, self.photos, self.current, self.shown = [], [], [], 0
        self.show_all.pack_forget()

        if self.total is None:
            self.summary.configure(text="학습이 끝나면 테스트 데이터에서 틀린 이미지를 보여줘요.")
            self.layout()
            return
        capped = self.total > len(self.items)  # the script only sends the most confident MAX_MISTAKES
        self.current = [m for m in self.items if self.filter is None or (m["true"], m["pred"]) == self.filter]
        self.current.sort(key=self.sorts[self.sort.get()])
        if self.filter:
            t, p = self.filter
            note = f" (받은 {len(self.items):,}장 중에서)" if capped else ""
            self.summary.configure(
                text=f"실제 {self.labels[t]} → 예측 {self.labels[p]} 로 틀린 이미지 {len(self.current):,}장{note}")
            self.show_all.pack(side="left", padx=(self.px(12), 0))
        elif self.total == 0:
            self.summary.configure(text="틀린 이미지가 하나도 없어요!")
        else:
            note = f" (확신도 높은 {len(self.items):,}장까지 표시)" if capped else ""
            self.summary.configure(text=f"틀린 이미지 {self.total:,}장{note}")
        self.add_page()
        self.canvas.yview_moveto(0)

    def add_page(self):
        end = min(self.shown + self.PAGE, len(self.current))
        for m in self.current[self.shown:end]:
            self.tiles.append(self.make_tile(m))
        self.shown = end
        self.layout()

    def make_tile(self, m):
        bg = COLORS["panel"]
        f = tk.Frame(self.inner, bg=bg, padx=self.px(8), pady=self.px(8))
        data = base64.b64decode(m["img"])
        if m.get("fmt") == "jpeg":
            img = Image.open(io.BytesIO(data)).convert("RGB")
        else:
            img = Image.frombytes(self.mode, (self.side, self.side), data)
        photo = ImageTk.PhotoImage(img.resize((self.disp, self.disp), self.resample))
        self.photos.append(photo)
        tk.Label(f, image=photo, bd=0).pack()
        true_txt, pred_txt = f"실제 {self.labels[m['true']]}", f"예측 {self.labels[m['pred']]}"
        if self.stacked:
            tk.Label(f, text=true_txt, font=FONT_B, fg=COLORS["diag"], bg=bg).pack(pady=(self.px(5), 0))
            tk.Label(f, text=pred_txt, font=FONT_B, fg=COLORS["err"], bg=bg).pack()
        else:
            cap = tk.Frame(f, bg=bg)
            cap.pack(pady=(self.px(5), 0))
            tk.Label(cap, text=true_txt, font=FONT_B, fg=COLORS["diag"], bg=bg).pack(side="left")
            tk.Label(cap, text="→", font=FONT, fg=COLORS["muted"], bg=bg).pack(side="left", padx=self.px(3))
            tk.Label(cap, text=pred_txt, font=FONT_B, fg=COLORS["err"], bg=bg).pack(side="left")
        tk.Label(f, text=f"확신도 {m['conf'] * 100:.0f}% · #{m['index']}", font=FONT_S,
                 fg=COLORS["muted"], bg=bg).pack()
        return f

    def layout(self):
        cols = max(self.cols, 1)
        for c in range(cols):
            self.inner.columnconfigure(c, minsize=self.tile_w, uniform="tile")
        for k, t in enumerate(self.tiles):
            t.grid(row=k // cols, column=k % cols, sticky="n")
        remaining = len(self.current) - self.shown
        if remaining > 0:
            self.more_btn.configure(text=f"더 보기  (+{min(remaining, self.PAGE)}장, {remaining:,}장 남음)")
            self.more_btn.grid(row=len(self.tiles) // cols + 1, column=0, columnspan=cols, pady=self.px(12))
        else:
            self.more_btn.grid_forget()


class App:
    def __init__(self, root, profile):
        self.root = root
        self.p = profile
        self.s = root.winfo_fpixels("1i") / 96
        self.proc = None
        self.script = None
        self.q = queue.Queue()
        self.t_start = None
        self.total_steps = None
        self.got_result = False
        self.stopping = False

        root.title(profile.title)
        root.geometry(f"{self.px(1180)}x{self.px(780)}")
        root.minsize(self.px(960), self.px(640))
        icon = HERE.with_name(profile.icon)
        if icon.exists():
            root.iconbitmap(default=str(icon))
        ttk.Style().configure(".", font=FONT)
        ttk.Style().configure("Run.TButton", font=FONT_B)

        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        self.build_options(root)
        self.build_results(root)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(50, self.poll)

    def px(self, v):
        return int(round(v * self.s))

    # ---------- layout ----------
    def build_options(self, root):
        side = ttk.Frame(root, padding=self.px(14))
        side.grid(row=0, column=0, sticky="ns")

        opts = ttk.LabelFrame(side, text=" 학습 옵션 ", padding=self.px(12))
        opts.pack(fill="x")
        self.inputs, self.vars = [], {}
        row = 0

        names = list(self.p.backends)
        self.backend = tk.StringVar(value=names[0])
        ttk.Label(opts, text="프레임워크").grid(row=row, column=0, sticky="w", pady=(0, self.px(4)))
        row += 1
        if len(names) == 1:
            ttk.Label(opts, text=names[0], font=FONT_B).grid(row=row, column=0, columnspan=2, sticky="w",
                                                             pady=(0, self.px(10)))
        else:
            rb = ttk.Frame(opts)
            rb.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, self.px(10)))
            for name in names:
                b = ttk.Radiobutton(rb, text=name, value=name, variable=self.backend)
                b.pack(anchor="w", pady=self.px(1))
                self.inputs.append(b)
        row += 1

        for o in self.p.options:
            if o.kind == "check":
                var = tk.BooleanVar(value=o.default)
                widget = ttk.Checkbutton(opts, text=o.label, variable=var)
                widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=(self.px(8), self.px(4)))
            else:
                var = tk.StringVar(value=o.default)
                if o.kind == "choice":
                    widget = ttk.Combobox(opts, values=o.values, textvariable=var, width=8, state="readonly")
                elif o.kind == "int":
                    widget = ttk.Spinbox(opts, from_=o.lo, to=o.hi, increment=1, textvariable=var, width=8)
                else:
                    widget = ttk.Spinbox(opts, from_=o.lo, to=o.hi, increment=o.step, textvariable=var, width=8,
                                         format="%.2f")
                ttk.Label(opts, text=o.label).grid(row=row, column=0, sticky="w", pady=self.px(4),
                                                   padx=(0, self.px(16)))
                widget.grid(row=row, column=1, sticky="e", pady=self.px(4))
            self.vars[o.key] = var
            self.inputs.append(widget)
            row += 1

        btns = ttk.Frame(side, padding=(0, self.px(12), 0, 0))
        btns.pack(fill="x")
        self.run_btn = ttk.Button(btns, text="▶  학습 시작", style="Run.TButton", command=self.start)
        self.run_btn.pack(fill="x", ipady=self.px(6))
        self.stop_btn = ttk.Button(btns, text="■  중지", command=self.stop, state="disabled")
        self.stop_btn.pack(fill="x", pady=(self.px(6), 0), ipady=self.px(2))

        info = ttk.LabelFrame(side, text=" 실행 환경 ", padding=self.px(12))
        info.pack(fill="x", pady=(self.px(14), 0))
        self.env_label = ttk.Label(info, text=f"WSL: {DISTRO}\n폴더: {PROJECT_SHOWN}\n장치: 학습 시작 시 확인",
                                   foreground=COLORS["muted"], justify="left",
                                   wraplength=self.px(220))
        self.env_label.pack(anchor="w")

    def build_results(self, root):
        main = ttk.Frame(root, padding=(0, self.px(14), self.px(14), self.px(14)))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        head = ttk.Frame(main)
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(1, weight=1)
        acc_box = ttk.Frame(head)
        acc_box.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, self.px(24)))
        ttk.Label(acc_box, text="테스트 정확도", foreground=COLORS["muted"]).pack(anchor="w")
        self.acc_label = ttk.Label(acc_box, text="—", font=FONT_BIG)
        self.acc_label.pack(anchor="w")
        self.acc_sub = ttk.Label(acc_box, text=" ", foreground=COLORS["muted"])
        self.acc_sub.pack(anchor="w")

        self.status = ttk.Label(head, text="준비됨. 옵션을 고르고 [학습 시작]을 누르세요.")
        self.status.grid(row=0, column=1, sticky="sw")
        prog_row = ttk.Frame(head)
        prog_row.grid(row=1, column=1, sticky="ew", pady=(self.px(6), 0))
        prog_row.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(prog_row, mode="determinate", maximum=1.0)
        self.progress.grid(row=0, column=0, sticky="ew")
        self.elapsed = ttk.Label(prog_row, text="", width=10, anchor="e", foreground=COLORS["muted"])
        self.elapsed.grid(row=0, column=1, padx=(self.px(8), 0))

        ttk.Style().configure("TNotebook.Tab", font=FONT_B, padding=(self.px(14), self.px(4)))
        self.tabs = ttk.Notebook(main)
        self.tabs.grid(row=1, column=0, sticky="nsew", pady=(self.px(14), self.px(10)))
        charts = ttk.Frame(self.tabs, padding=self.px(10))
        charts.columnconfigure(0, weight=1, uniform="c")
        charts.columnconfigure(1, weight=1, uniform="c")
        charts.rowconfigure(0, weight=1)
        self.curve = LearningCurve(charts, self.s)
        self.curve.grid(row=0, column=0, sticky="nsew", padx=(0, self.px(7)))
        self.matrix = ConfusionMatrix(charts, self.s, self.p.labels, self.p.noun, on_cell=self.show_mistakes)
        self.matrix.grid(row=0, column=1, sticky="nsew", padx=(self.px(7), 0))
        self.tabs.add(charts, text="결과 요약")
        self.mistakes = MistakesView(self.tabs, self.s, self.p, on_count=self.set_mistake_count)
        self.tabs.add(self.mistakes, text="틀린 이미지")

        log_frame = ttk.LabelFrame(main, text=" 실행 로그 ", padding=self.px(6))
        log_frame.grid(row=2, column=0, sticky="ew")
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=7, font=("Consolas", 9), wrap="none", relief="flat",
                           bg=COLORS["empty"], fg=COLORS["text"], state="disabled")
        self.log.grid(row=0, column=0, sticky="ew")
        sb = ttk.Scrollbar(log_frame, command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=sb.set)

    def set_mistake_count(self, n):
        self.tabs.tab(self.mistakes, text="틀린 이미지" if n is None else f"틀린 이미지 ({n:,})")

    def show_mistakes(self, true, pred):
        self.mistakes.set_filter(true, pred)
        self.tabs.select(self.mistakes)

    # ---------- actions ----------
    def build_args(self):
        """Command-line flags from the option widgets; raises ValueError with a message for bad input."""
        args, epochs = [], None
        for o in self.p.options:
            value = self.vars[o.key].get()
            if o.kind == "check":
                if value != o.default:
                    args.append(o.flag)
                continue
            if o.kind in ("int", "float"):
                try:
                    num = int(value) if o.kind == "int" else float(value)
                except ValueError:
                    raise ValueError(f"{o.label}은(는) {'정수' if o.kind == 'int' else '숫자'}로 입력하세요.")
                if not o.lo <= num <= o.hi:
                    raise ValueError(f"{o.label}은(는) {o.lo:g}~{o.hi:g} 사이로 입력하세요.")
                value = f"{num:g}" if o.kind == "float" else str(num)
                if o.key == "epochs":
                    epochs = num
            args.append(f"{o.flag} {value}")
        return " ".join(args), epochs

    def start(self):
        try:
            args, epochs = self.build_args()
        except ValueError as e:
            messagebox.showwarning("옵션 확인", str(e), parent=self.root)
            return
        self.script, template = self.p.backends[self.backend.get()]
        project = PROJECT if PROJECT.startswith("~") else shlex.quote(PROJECT)
        cmd = f"cd {project} && " + template.format(args=args)

        self.curve.reset(epochs)
        self.matrix.set(None)
        self.mistakes.clear()
        self.tabs.select(0)
        self.acc_label.configure(text="—", foreground=COLORS["text"])
        self.acc_sub.configure(text=" ")
        self.progress.configure(value=0)
        self.log_clear()
        self.log_write("$ " + cmd.replace(project, PROJECT_SHOWN, 1))
        self.total_steps, self.got_result, self.stopping, self.out_of_memory = None, False, False, False
        self.set_running(True)
        self.set_status("WSL과 모델을 준비하는 중… (처음 실행은 10초 정도 걸릴 수 있어요)")
        self.t_start = time.time()
        self.tick()

        try:
            self.proc = subprocess.Popen(["wsl.exe", "-d", DISTRO, "--exec", "bash", "-lc", cmd],
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        except OSError as e:
            self.set_running(False)
            self.set_status("실행 실패")
            messagebox.showerror("실행 실패", f"wsl.exe를 실행하지 못했어요.\n{e}", parent=self.root)
            return
        threading.Thread(target=self.reader, args=(self.proc,), daemon=True).start()

    def reader(self, proc):
        for raw in proc.stdout:
            self.q.put(("line", raw.decode("utf-8", "replace").rstrip()))
        self.q.put(("exit", proc.wait()))

    def stop(self):
        if not self.proc or self.proc.poll() is not None:
            return
        self.stopping = True
        self.set_status("중지하는 중…")
        self.proc.terminate()
        # Make sure the Linux side is gone too, not just the wsl.exe client.
        subprocess.run(["wsl.exe", "-d", DISTRO, "--exec", "pkill", "-f", f"{self.script} .*--json"],
                       creationflags=CREATE_NO_WINDOW, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("종료", "학습이 진행 중이에요. 중지하고 닫을까요?", parent=self.root):
                return
            self.stop()
        self.root.destroy()

    # ---------- process output ----------
    def poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "line":
                    self.handle_line(payload)
                else:
                    self.handle_exit(payload)
        except queue.Empty:
            pass
        self.root.after(50, self.poll)

    def handle_line(self, line):
        if not line.startswith("@@"):
            if "OutOfMemoryError" in line or "out of memory" in line:
                self.out_of_memory = True
            if line.strip() and not any(s in line for s in LOG_NOISE):
                self.log_write(line)
            return
        try:
            msg = json.loads(line[2:])
        except json.JSONDecodeError:
            self.log_write(line)
            return
        t = msg["type"]
        if t == "start":
            self.env_label.configure(
                text=f"WSL: {DISTRO}\n폴더: {PROJECT_SHOWN}\n{msg['framework']}\n장치: {msg['device']}")
            self.log_write(f"{msg['framework']} · 장치 {msg['device']}")
            self.set_status("데이터를 불러오는 중…")
        elif t == "batch":
            if self.total_steps is None:
                self.total_steps = msg["steps"] * self.curve.epochs
            done = (msg["epoch"] - 1) * msg["steps"] + msg["batch"]
            self.progress.configure(value=done / self.total_steps)
            self.set_status(f"학습 중… epoch {msg['epoch']}/{self.curve.epochs}  "
                            f"(배치 {msg['batch']}/{msg['steps']})")
        elif t == "epoch":
            self.curve.add(msg)
            self.log_write(f"epoch {msg['epoch']}/{msg['epochs']}  loss {msg['loss']:.4f}  "
                           f"학습 정확도 {msg['accuracy'] * 100:.2f}%  테스트 정확도 {msg['val_accuracy'] * 100:.2f}%  "
                           f"({msg['seconds']:.1f}초)")
            if msg["epoch"] == msg["epochs"]:
                self.set_status("테스트 데이터로 평가하는 중…")
        elif t == "mistakes":
            self.mistakes.set_items(msg["total"], msg["items"])
            self.log_write(f"틀린 이미지 {msg['total']:,}장 → [틀린 이미지] 탭에서 볼 수 있어요")
        elif t == "result":
            self.got_result = True
            cm = msg["confusion"]
            self.matrix.set(cm)
            self.acc_label.configure(text=f"{msg['accuracy'] * 100:.2f}%", foreground=COLORS["ok"])
            self.acc_sub.configure(text=f"{msg['correct']:,} / {msg['total']:,}장 정답 · "
                                        f"학습 {msg['train_seconds']:.1f}초")
            per_class = "  ".join(f"{self.p.labels[d]}:{cm[d][d] / sum(cm[d]) * 100:.1f}%" for d in range(len(cm)))
            self.log_write(f"테스트 정확도 {msg['accuracy'] * 100:.2f}% ({msg['correct']}/{msg['total']})")
            self.log_write(f"{self.p.noun}별 정확도  {per_class}")
            self.log_write(f"모델 저장: {PROJECT_SHOWN}/{msg['model_path']}")

    def handle_exit(self, code):
        self.proc = None
        self.set_running(False)
        total = time.time() - self.t_start
        if self.got_result:
            self.progress.configure(value=1.0)
            self.set_status(f"완료! 전체 {total:.0f}초 걸렸어요.")
        elif self.stopping:
            self.set_status("중지했어요.")
            self.acc_label.configure(text="—")
        elif self.out_of_memory:
            self.set_status("GPU 메모리가 부족해서 멈췄어요. 배치 크기나 이미지 크기를 줄여 보세요.")
            self.acc_label.configure(text="오류", foreground=COLORS["bad"])
            messagebox.showerror("GPU 메모리 부족",
                                 "GPU 메모리(12GB)가 부족해서 학습이 멈췄어요.\n"
                                 "배치 크기나 이미지 크기를 줄이거나, 더 작은 모델을 골라 다시 시도해 보세요.",
                                 parent=self.root)
        else:
            self.set_status(f"오류로 종료됐어요 (코드 {code}). 아래 로그를 확인하세요.")
            self.acc_label.configure(text="오류", foreground=COLORS["bad"])
            messagebox.showerror("학습 실패", "학습이 중간에 끝났어요. 실행 로그의 마지막 부분을 확인하세요.",
                                 parent=self.root)

    # ---------- helpers ----------
    def tick(self):
        if self.proc is not None or self.t_start and time.time() - self.t_start < 1:
            secs = int(time.time() - self.t_start)
            self.elapsed.configure(text=f"{secs // 60}:{secs % 60:02d}")
            self.root.after(500, self.tick)

    def set_running(self, running):
        for w in self.inputs:
            w.configure(state="disabled" if running else ("readonly" if isinstance(w, ttk.Combobox) else "normal"))
        self.run_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")

    def set_status(self, text):
        self.status.configure(text=text)

    def log_clear(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def log_write(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


def main(profile_name):
    profile = PROFILES[profile_name]
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text on high-DPI screens
    except (AttributeError, OSError):
        pass
    try:
        # Own taskbar identity, so the taskbar shows this app's icon instead of Python's.
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(profile.app_id)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    App(root, profile)
    root.mainloop()
