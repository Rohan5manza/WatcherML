import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import watcherml as watcher

device = "mps" if torch.backends.mps.is_available() else "cpu"

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        return self.fc(x)

config = {"model": "small_cnn", "lr": 1e-3, "batch_size": 64, "epochs": 50, "device": device}

with watcher.init(project="fashion-mnist", config=config) as run:
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    val_data = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    run.set_dataset("./data")  # real dataset fingerprint, not a placeholder

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256)

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        run.log({"train_loss": total_loss / len(train_loader), "val_accuracy": val_acc}, step=epoch)
        print(f"epoch {epoch}: loss={total_loss / len(train_loader):.4f}  val_acc={val_acc:.4f}")

print(f"\nDone. Run ID: {run.run_id}")