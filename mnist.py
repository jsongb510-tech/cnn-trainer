import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 14 -> 7
        x = self.dropout(torch.flatten(x, 1))
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} | device: {device}", end="")
    print(f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=1000, pin_memory=True)

    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()

        print(f"epoch {epoch}/{EPOCHS}  loss {total_loss / len(train_ds):.4f}  "
              f"test acc {100 * correct / len(test_ds):.2f}%  ({time.time() - t0:.1f}s)")

    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("saved model to mnist_cnn.pt")


if __name__ == "__main__":
    main()
