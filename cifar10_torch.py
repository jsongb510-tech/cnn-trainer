"""CIFAR-10 classification with a small VGG-style CNN in plain PyTorch."""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from jsonlog import emit, mistakes

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
SHORT = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def parse_args():
    p = argparse.ArgumentParser(description="PyTorch CNN on CIFAR-10")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=8, help="DataLoader worker processes")
    p.add_argument("--no-augment", action="store_true", help="turn off random crop / horizontal flip")
    p.add_argument("--json", action="store_true",
                   help="print machine-readable @@{...} progress lines instead of text logs (used by the GUI)")
    return p.parse_args()


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 64), conv_block(64, 128), conv_block(128, 256))  # 32->16->8->4
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 10))

    def forward(self, x):
        return self.head(self.features(x))


def print_confusion_matrix(cm):
    print("rows = true class, cols = predicted class")
    print(" " * 7 + "".join(f"{c:>7}" for c in SHORT))
    for name, row in zip(SHORT, cm.tolist()):
        print(f"{name:>6} " + "".join(f"{v:>7}" for v in row))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = f"cuda ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "cpu"
    if args.json:
        emit("start", framework=f"PyTorch {torch.__version__}", device=device_name, epochs=args.epochs,
             batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, augment=not args.no_augment)
    else:
        print(f"PyTorch {torch.__version__} | device: {device_name}")

    augment = [] if args.no_augment else [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    train_tf = transforms.Compose(augment + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10("data", train=False, download=True, transform=test_tf)
    loader_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    test_dl = DataLoader(test_ds, batch_size=1000, **loader_kw)

    model = Net().to(device, memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))

    def evaluate():
        """(true labels, softmax probabilities) on the test set."""
        model.eval()
        probs, labels = [], []
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            for x, y in test_dl:
                x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
                probs.append(model(x).float().softmax(1).cpu())
                labels.append(y)
        return torch.cat(labels), torch.cat(probs)

    steps = len(train_dl)
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
                loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)
            if args.json and (step % 20 == 1 or step == steps):
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
    cm = torch.bincount(y_true * 10 + y_pred, minlength=100).reshape(10, 10)
    torch.save(model.state_dict(), "cifar10_cnn.pt")
    if args.json:
        n_wrong, items = mistakes(test_ds.data, np.array(test_ds.targets), probs.numpy())
        emit("mistakes", total=n_wrong, items=items)
        correct, total = int(cm.trace()), int(cm.sum())
        emit("result", accuracy=correct / total, correct=correct, total=total, confusion=cm.tolist(),
             train_seconds=train_seconds, model_path="cifar10_cnn.pt")
        return

    print(f"\nTraining time: {train_seconds:.0f}s")
    print("\nConfusion matrix (test set):")
    print_confusion_matrix(cm)
    print("\nPer-class accuracy:")
    for name, row in zip(CLASSES, cm.tolist()):
        print(f"  {name:<10} {100 * row[CLASSES.index(name)] / sum(row):6.2f}%")
    correct = int(cm.trace())
    print(f"\nTest accuracy: {100 * correct / cm.sum().item():.2f}% ({correct}/{cm.sum().item()})")
    print("saved model to cifar10_cnn.pt")


if __name__ == "__main__":
    main()
