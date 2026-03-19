"""
4_evaluate.py
Intel Image Classification — Test Seti Değerlendirme
Confusion Matrix + Classification Report üretir.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)
from pathlib import Path

# ─── Ayarlar ────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
IMG_SIZE    = 224
BATCH_SIZE  = 64
NUM_WORKERS = 0
CLASSES     = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

DATA_DIR  = Path("data") / "seg_test" / "seg_test"
MODEL_DIR = Path("models") / "best_model.pth"
PLOT_DIR  = Path("outputs/plots"); PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Veri ───────────────────────────────────────────────────────────────────
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_ds = datasets.ImageFolder(DATA_DIR, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
print(f"📂  Test: {len(test_ds)} görüntü, {len(test_loader)} batch")

# ─── Model ──────────────────────────────────────────────────────────────────
def load_model():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    model.load_state_dict(torch.load(MODEL_DIR, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

model = load_model()
print(f"✅  Model yüklendi: {MODEL_DIR}")

# ─── Tahmin ─────────────────────────────────────────────────────────────────
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ─── Metrikler ──────────────────────────────────────────────────────────────
acc = (all_preds == all_labels).mean() * 100
print(f"\n📊  Test Accuracy: {acc:.2f}%")

report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)
print("\n" + report)

# ─── Confusion Matrix (Görsel) ───────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, linecolor="#333",
            annot_kws={"size": 12, "weight": "bold"},
            ax=ax, cbar_kws={"shrink": 0.8})

ax.set_xlabel("Tahmin", color="white", fontsize=12)
ax.set_ylabel("Gerçek",  color="white", fontsize=12)
ax.set_title(f"Confusion Matrix  (Test Accuracy: {acc:.2f}%)",
             color="white", fontsize=14, fontweight="bold", pad=15)
ax.tick_params(colors="white")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", color="white")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="white")
plt.colorbar(ax.collections[0]).ax.yaxis.set_tick_params(color="white")

plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("✅  Kaydedildi: outputs/plots/confusion_matrix.png")
plt.close()

# ─── Per-Class F1 Bar Chart ──────────────────────────────────────────────────
from sklearn.metrics import f1_score, precision_score, recall_score
f1s  = f1_score(all_labels, all_preds, average=None)
prec = precision_score(all_labels, all_preds, average=None)
rec  = recall_score(all_labels, all_preds, average=None)

COLORS = ["#4E79A7", "#59A14F", "#76B7B2", "#9C755F", "#EDC948", "#B07AA1"]
x = np.arange(len(CLASSES)); w = 0.25

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#16213e")
ax.bar(x - w,   prec, w, label="Precision", color="#4E79A7", alpha=0.9)
ax.bar(x,        rec, w, label="Recall",    color="#59A14F", alpha=0.9)
ax.bar(x + w,    f1s, w, label="F1-Score",  color="#F28E2B", alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(CLASSES, color="white", rotation=20, ha="right")
ax.tick_params(colors="white"); ax.set_ylim(0, 1.1)
ax.set_ylabel("Score", color="white"); ax.set_xlabel("Sınıf", color="white")
ax.set_title("Per-Class Precision / Recall / F1-Score",
             color="white", fontsize=13, fontweight="bold")
ax.legend(facecolor="#222", labelcolor="white")
for spine in ax.spines.values(): spine.set_edgecolor("#444")

# Makro ortalamayı yazdır
macro_f1 = f1_score(all_labels, all_preds, average="macro")
ax.text(0.98, 0.97, f"Macro F1: {macro_f1:.4f}",
        transform=ax.transAxes, ha="right", va="top",
        color="white", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#333", alpha=0.8))

plt.tight_layout()
plt.savefig(PLOT_DIR / "classification_report.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("✅  Kaydedildi: outputs/plots/classification_report.png")
plt.close()

print(f"\n🎉  Değerlendirme tamamlandı! Test Acc: {acc:.2f}%  Macro F1: {macro_f1:.4f}")
