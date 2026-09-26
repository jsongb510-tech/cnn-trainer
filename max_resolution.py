"""Find the largest square input resolution each model can handle on this GPU.

Uses random images (memory use doesn't depend on pixel values), fp16 autocast and
binary search over resolutions that are multiples of 32.
"""
import time

import torch
import torch.nn.functional as F
from torchvision import models

STEP = 32
START = 224
NUM_CLASSES = 1000
MODELS = {
    "resnet50": models.resnet50,
    "convnext_tiny": models.convnext_tiny,
}
MODES = [  # (name, train, batch size)
    ("inference bs=1", False, 1),
    ("train bs=1", True, 1),
    ("train bs=8", True, 8),
]


class TooBig(Exception):
    pass


def run_step(model, opt, res, train, bs):
    """One forward (+backward) pass at res x res. Returns (seconds, peak GiB)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        x = torch.randn(bs, 3, res, res, device="cuda").to(memory_format=torch.channels_last)
        y = torch.randint(0, NUM_CLASSES, (bs,), device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            if train:
                loss = F.cross_entropy(model(x), y)
            else:
                with torch.no_grad():
                    model(x)
        if train:
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return time.time() - t0, torch.cuda.max_memory_allocated() / 2**30
    except torch.cuda.OutOfMemoryError:
        raise TooBig("out of GPU memory")
    except RuntimeError as e:  # e.g. kernels that can't index tensors with > 2^31 elements
        raise TooBig(str(e).splitlines()[0][:80])
    finally:
        x = y = loss = None
        if train:
            opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def fits(model, opt, res, train, bs):
    try:
        run_step(model, opt, res, train, bs)
        return True, None
    except TooBig as e:
        return False, str(e)


def max_resolution(model, opt, train, bs):
    # Double until failure, then binary search between the last success and the failure.
    lo, hi, reason = None, START, None
    while True:
        ok, why = fits(model, opt, hi, train, bs)
        if not ok:
            reason = why
            break
        lo, hi = hi, hi * 2
    if lo is None:
        return None, reason
    while hi - lo > STEP:
        mid = (lo + hi) // 2 // STEP * STEP
        ok, why = fits(model, opt, mid, train, bs)
        if ok:
            lo = mid
        else:
            hi, reason = mid, why
    return lo, reason


def main():
    torch.backends.cudnn.benchmark = False
    free, total = torch.cuda.mem_get_info()
    # Keep the allocator inside physical VRAM so the Windows driver can't silently spill to system RAM.
    torch.cuda.set_per_process_memory_fraction(free / total * 0.95)
    print(f"GPU: {torch.cuda.get_device_name(0)} | total {total / 2**30:.1f} GiB | "
          f"usable here {free / 2**30 * 0.95:.1f} GiB | PyTorch {torch.__version__}\n")

    rows = []
    for name, ctor in MODELS.items():
        model = ctor(num_classes=NUM_CLASSES).cuda().to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for mode, train, bs in MODES:
            model.train(train)
            res, reason = max_resolution(model, opt, train, bs)
            secs, peak = run_step(model, opt, res, train, bs) if res else (float("nan"), float("nan"))
            rows.append((name, mode, res, peak, secs, reason))
            print(f"{name:14} {mode:15} max {res or 0:>6}x{res or 0:<6} peak {peak:5.1f} GiB  "
                  f"{secs:6.2f} s/step   (next size failed: {reason})", flush=True)
        del model, opt
        torch.cuda.empty_cache()

    print("\nSummary (max square resolution, multiple of 32):")
    print(f"{'model':14} " + "".join(f"{m:>16}" for m, _, _ in MODES))
    for name in MODELS:
        vals = [r[2] for r in rows if r[0] == name]
        print(f"{name:14} " + "".join(f"{(str(v) + 'px') if v else '-':>16}" for v in vals))


if __name__ == "__main__":
    main()
