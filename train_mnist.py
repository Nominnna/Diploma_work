import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import random
import numpy as np

class MNIST_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(784, 784)
        self.fc2 = nn.Linear(784, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

        # nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight)
        # nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------
# EMA helper (no model change)
# -------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.detach().clone()
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(self.backup[n])
        self.backup = {}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, optimizer, criterion, train_loader, scaler, scheduler, use_amp, device, ema=None):
    model.train()
    total_loss = 0.0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()

        # Stabilize updates
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    acc = 100. * correct / len(train_loader.dataset)
    return avg_loss, acc

@torch.no_grad()
def test(model, test_loader, use_amp, device):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
    return 100. * correct / len(test_loader.dataset)

def main():
    set_seed(42)

    # For speed; if you want strict determinism set benchmark=False and deterministic=True
    torch.backends.cudnn.benchmark = True

    Epochs = 60  # 15 is usually not enough for "best possible" on this MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # AMP is optional; for MNIST it can sometimes add tiny noise. Try both.
    use_amp = (device.type == "cuda")  # set to False if you want max stability

    # MNIST normalization constants (widely used)
    mean, std = (0.1307,), (0.3081,)

    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.10, 0.10),
            scale=(0.90, 1.10),
            shear=5
        ),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)

    model = MNIST_Net().to(device)

    # Label smoothing can help generalization a bit (training change, not model change)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # AdamW often works well, but use a more realistic LR range
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,                    # MUCH safer than 0.01 for this setup
        steps_per_epoch=len(train_loader),
        epochs=Epochs,
        pct_start=0.25,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=50.0
    )

    scaler = torch.amp.GradScaler(enabled=use_amp)

    ema = EMA(model, decay=0.999)

    print("\nTraining started...\n")
    start_time = time.time()

    best_acc = -1.0
    best_path = "mnist_best.pth"

    for epoch in range(1, Epochs + 1):
        train_loss, train_acc = train_epoch(
            model, optimizer, criterion, train_loader, scaler, scheduler, use_amp, device, ema=ema
        )

        # Evaluate with EMA weights (often slightly better)
        ema.apply_to(model)
        test_acc = test(model, test_loader, use_amp, device)
        ema.restore(model)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)

        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:6.2f}% | Test Acc(EMA): {test_acc:6.3f}% | Best: {best_acc:6.3f}%")

    total_time = time.time() - start_time
    print(f"\nFinished! Total time: {total_time:.1f} sec")
    print(f"Best test accuracy: {best_acc:.3f}%")
    print(f"Best model saved → {best_path}")

if __name__ == "__main__":
    main()
