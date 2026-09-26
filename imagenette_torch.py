"""Imagenette (10 easy ImageNet classes, full-size photos) classification with ResNet in PyTorch.

Train from scratch, or fine-tune ImageNet-pretrained weights with --pretrained (transfer learning).
Note: Imagenette's classes are a subset of ImageNet, so pretrained weights have already seen these
kinds of photos - that's why transfer learning reaches ~99% here within a few epochs.
"""
import argparse
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from jsonlog import emit, mistakes

CLASSES = ("tench", "English springer", "cassette player", "chain saw", "church",
           "French horn", "garbage truck", "gas pump", "golf ball", "parachute")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics
DATA = Path("data")
THUMB = 128  # side of the JPEG thumbnails of misclassified images sent to the GUI


def parse_args():
    p = argparse.ArgumentParser(description="ResNet on Imagenette")
    p.add_argument("--model", choices=("resnet18", "resnet50"), default="resnet18")
    p.add_argument("--pretrained", action="store_true", help="start from ImageNet weights (transfer learning)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="learning rate (with --pretrained: for the new head; the backbone gets lr/10)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--img-size", type=int, default=224, help="training / test image side in pixels")
    p.add_argument("--workers", type=int, default=12, help="DataLoader worker processes (JPEG decoding)")
    p.add_argument("--json", action="store_true",
                   help="print machine-readable @@{...} progress lines instead of text logs (used by the GUI)")
    return p.parse_args()


def build_model(name, pretrained):
    weights = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
               "resnet50": models.ResNet50_Weights.IMAGENET1K_V2}[name] if pretrained else None
    model = getattr(models, name)(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))  # new 10-class head
    return model


def load_data(img_size):
    """ImageFolder datasets for train/val (downloads Imagenette 320px on first use, ~330 MB)."""
    root = DATA / "imagenette2-320"
    if not root.exists():
        datasets.Imagenette(str(DATA), split="train", size="320px", download=True)
    resize = round(img_size * 256 / 224)  # usual ImageNet eval: resize, then center crop
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(resize), transforms.CenterCrop(img_size), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return (datasets.ImageFolder(str(root / "train"), train_tf),
            datasets.ImageFolder(str(root / "val"), test_tf))


def thumbnail_jpeg(path):
    """The center crop the model saw at test time, as a THUMB x THUMB JPEG."""
    img = Image.open(path).convert("RGB")
    img = transforms.CenterCrop(THUMB)(transforms.Resize(round(THUMB * 256 / 224))(img))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def print_confusion_matrix(cm):
    short = [c.split()[-1][:6] for c in CLASSES]
    print("rows = true class, cols = predicted class")
    print(" " * 7 + "".join(f"{c:>7}" for c in short))
    for name, row in zip(short, cm.tolist()):
        print(f"{name:>6} " + "".join(f"{v:>7}" for v in row))


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = f"cuda ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "cpu"
    mode = "전이학습 (ImageNet 가중치)" if args.pretrained else "처음부터 학습"
    if args.json:
        emit("start", framework=f"PyTorch {torch.__version__} · {args.model} · {mode}", device=device_name,
             **{k: v for k, v in vars(args).items() if k != "json"})
    else:
        print(f"PyTorch {torch.__version__} | device: {device_name} | {args.model}, "
              f"{'pretrained' if args.pretrained else 'from scratch'}, {args.img_size}px")

    train_ds, test_ds = load_data(args.img_size)
    loader_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    test_dl = DataLoader(test_ds, batch_size=128, **loader_kw)

    model = build_model(args.model, args.pretrained).to(device, memory_format=torch.channels_last)
    if args.pretrained:  # discriminative learning rates: gentle on pretrained layers, full speed on the new head
        head = list(model.fc.parameters())
        backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
        groups = [{"params": backbone, "lr": args.lr / 10}, {"params": head, "lr": args.lr}]
    else:
        groups = [{"params": model.parameters(), "lr": args.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps = len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=args.epochs * steps, pct_start=0.15)

    def evaluate():
        """(true labels, softmax probabilities) on the validation set."""
        model.eval()
        probs, labels = [], []
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            for x, y in test_dl:
                x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
                probs.append(model(x).float().softmax(1).cpu())
                labels.append(y)
        return torch.cat(labels), torch.cat(probs)

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, correct, seen = 0.0, 0, 0
        for step, (x, y) in enumerate(train_dl, start=1):
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out, y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)
            if args.json and (step % 10 == 1 or step == steps):
                emit("batch", epoch=epoch, batch=step, steps=steps)

        y_true, probs = evaluate()
        test_acc = (y_true == probs.argmax(1)).float().mean().item()
        if args.json:
            emit("epoch", epoch=epoch, epochs=args.epochs, seconds=time.time() - t0, loss=total_loss / seen,
                 accuracy=correct / seen, val_accuracy=test_acc)
        else:
            print(f"epoch {epoch:>2}/{args.epochs}  loss {total_loss / seen:.4f}  "
                  f"train acc {100 * correct / seen:.2f}%  test acc {100 * test_acc:.2f}%  "
                  f"({time.time() - t0:.1f}s)", flush=True)
    train_seconds = time.time() - t_start

    y_true, probs = evaluate()
    y_pred = probs.argmax(1)
    n = len(CLASSES)
    cm = torch.bincount(y_true * n + y_pred, minlength=n * n).reshape(n, n)
    torch.save(model.state_dict(), "imagenette_resnet.pt")
    correct, total = int(cm.trace()), int(cm.sum())
    if args.json:
        paths = [p for p, _ in test_ds.samples]
        n_wrong, items = mistakes(None, y_true.numpy(), probs.numpy(), encode=lambda i: thumbnail_jpeg(paths[i]))
        emit("mistakes", total=n_wrong, items=items)
        emit("result", accuracy=correct / total, correct=correct, total=total, confusion=cm.tolist(),
             train_seconds=train_seconds, model_path="imagenette_resnet.pt")
        return

    print(f"\nTraining time: {train_seconds:.0f}s")
    print("\nConfusion matrix (validation set):")
    print_confusion_matrix(cm)
    print("\nPer-class accuracy:")
    for i, name in enumerate(CLASSES):
        print(f"  {name:<16} {100 * cm[i, i] / cm[i].sum():6.2f}%")
    print(f"\nTest accuracy: {100 * correct / total:.2f}% ({correct}/{total})")
    print("saved model to imagenette_resnet.pt")


if __name__ == "__main__":
    main()
