import os, time, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Kullanılan cihaz: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

BATCH_SIZE      = 64
EPOCHS_FROZEN   = 5
EPOCHS_FINETUNE = 10
LR_FROZEN       = 1e-3
LR_FINETUNE     = 1e-4
NUM_CLASSES     = 6
IMG_SIZE        = 224
NUM_WORKERS     = 4

DATA_DIR  = Path("data")
TRAIN_DIR = DATA_DIR / "seg_train" / "seg_train"
TEST_DIR  = DATA_DIR / "seg_test"  / "seg_test"
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR  = Path("outputs/plots"); PLOT_DIR.mkdir(parents=True, exist_ok=True)

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

full_train = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
test_ds    = datasets.ImageFolder(TEST_DIR,  transform=test_transforms)

val_size   = int(0.2 * len(full_train))
train_size = len(full_train) - val_size
train_ds, val_ds = torch.utils.data.random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_ds.dataset = copy.deepcopy(full_train)
val_ds.dataset.transform = test_transforms

loaders = {
    "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True),
    "val":   DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
    "test":  DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
}

CLASS_NAMES = full_train.classes
print(f"\n📂  Sınıflar: {CLASS_NAMES}")
print(f"   Train: {train_size} | Val: {val_size} | Test: {len(test_ds)}")

def build_model(freeze_all: bool = True) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if freeze_all:
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    return model.to(DEVICE)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = running_correct = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss    += loss.item() * imgs.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
    return running_loss / len(loader.dataset), running_correct / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = running_correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss    += loss.item() * imgs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
    return running_loss / len(loader.dataset), running_correct / len(loader.dataset)

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0.0

def run_training(model, epochs, lr, phase_name):
    global best_val_acc
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n{'='*55}")
    print(f"  {phase_name}  ({epochs} epoch, lr={lr})")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, loaders["val"],   criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
            tag = "  ✅ kaydedildi"
        else:
            tag = ""

        elapsed = time.time() - t0
        print(f"  Epoch [{epoch:02d}/{epochs}] "
              f"Loss: {tr_loss:.4f}/{vl_loss:.4f}  "
              f"Acc: {tr_acc*100:.2f}%/{vl_acc*100:.2f}%  "
              f"({elapsed:.0f}s){tag}")


def main():
    
    model = build_model(freeze_all=True)
    run_training(model, EPOCHS_FROZEN, LR_FROZEN, "AŞAMA 1 — Feature Extraction")

    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True

    run_training(model, EPOCHS_FINETUNE, LR_FINETUNE, "AŞAMA 2 — Fine-Tuning")

    print(f"\n🏆  En iyi Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"   Model kaydedildi: models/best_model.pth")

    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=DEVICE))
    test_loss, test_acc = evaluate(model, loaders["test"], nn.CrossEntropyLoss())
    print(f"\n📊  Test Accuracy: {test_acc*100:.2f}%  |  Test Loss: {test_loss:.4f}")

    total_epochs = EPOCHS_FROZEN + EPOCHS_FINETUNE
    ep = range(1, total_epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#16213e")
        for spine in ax.spines.values(): spine.set_edgecolor("#444")
        ax.tick_params(colors="white")

    axes[0].plot(ep, history["train_loss"], color="#4E79A7", lw=2, label="Train Loss")
    axes[0].plot(ep, history["val_loss"],   color="#F28E2B", lw=2, label="Val Loss", linestyle="--")
    axes[0].axvline(x=EPOCHS_FROZEN + 0.5, color="gray", linestyle=":", alpha=0.7, label="Fine-Tune Başlangıcı")
    axes[0].set_title("Loss Eğrisi", color="white", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", color="white"); axes[0].set_ylabel("Loss", color="white")
    axes[0].legend(facecolor="#222", labelcolor="white")

    axes[1].plot(ep, [a*100 for a in history["train_acc"]], color="#59A14F", lw=2, label="Train Accuracy")
    axes[1].plot(ep, [a*100 for a in history["val_acc"]],   color="#E15759", lw=2, label="Val Accuracy", linestyle="--")
    axes[1].axvline(x=EPOCHS_FROZEN + 0.5, color="gray", linestyle=":", alpha=0.7, label="Fine-Tune Başlangıcı")
    axes[1].set_title("Accuracy Eğrisi", color="white", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", color="white"); axes[1].set_ylabel("Accuracy (%)", color="white")
    axes[1].legend(facecolor="#222", labelcolor="white")
    axes[1].set_ylim(0, 105)

    plt.suptitle("ResNet50 — Eğitim Eğrileri", color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_curves.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("\n✅  Kaydedildi: outputs/plots/training_curves.png")
    plt.close()

    print("\n🎉  Eğitim tamamlandı!")

if __name__ == "__main__":
    main()
