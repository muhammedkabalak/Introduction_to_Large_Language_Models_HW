"""
1_data_exploration.py
Intel Image Classification — Veri Keşfi (EDA)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

# ─── Ayarlar ───────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
TRAIN_DIR  = DATA_DIR / "seg_train" / "seg_train"
TEST_DIR   = DATA_DIR / "seg_test"  / "seg_test"
OUT_DIR    = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
COLORS  = ["#4E79A7", "#59A14F", "#76B7B2", "#9C755F", "#EDC948", "#B07AA1"]

# ─── 1. Sınıf Dağılımı ─────────────────────────────────────────────────────
def count_images(root: Path):
    counts = {}
    for cls in CLASSES:
        p = root / cls
        counts[cls] = len(list(p.glob("*.jpg"))) + len(list(p.glob("*.png"))) if p.exists() else 0
    return counts

train_counts = count_images(TRAIN_DIR)
test_counts  = count_images(TEST_DIR)

print("\n📊  Dataset İstatistikleri")
print(f"{'Sınıf':<12} {'Train':>8} {'Test':>8}")
print("-" * 30)
total_train = total_test = 0
for cls in CLASSES:
    t, v = train_counts[cls], test_counts[cls]
    total_train += t; total_test += v
    print(f"{cls:<12} {t:>8} {v:>8}")
print("-" * 30)
print(f"{'TOPLAM':<12} {total_train:>8} {total_test:>8}")

# ─── 2. Bar Chart ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#1a1a2e")

for ax, counts, title in zip(axes,
                               [train_counts, test_counts],
                               ["Eğitim Seti", "Test Seti"]):
    bars = ax.bar(CLASSES, [counts[c] for c in CLASSES], color=COLORS, edgecolor="white", linewidth=0.5)
    ax.set_facecolor("#16213e")
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors="white"); ax.set_ylabel("Görüntü Sayısı", color="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")
    for bar, cnt in zip(bars, [counts[c] for c in CLASSES]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(cnt), ha="center", va="bottom", color="white", fontsize=9)
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", color="white", fontsize=9)

plt.suptitle("Intel Image Classification — Sınıf Dağılımı",
             fontsize=16, fontweight="bold", color="white", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "class_distribution.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\n✅  Kaydedildi: outputs/plots/class_distribution.png")
plt.close()

# ─── 3. Örnek Görüntü Izgara ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 6, figsize=(18, 7))
fig.patch.set_facecolor("#1a1a2e")
fig.suptitle("Her Sınıftan Örnek Görüntüler", color="white", fontsize=16, fontweight="bold")

for col, cls in enumerate(CLASSES):
    cls_dir = TRAIN_DIR / cls
    imgs = sorted(cls_dir.glob("*.jpg"))[:2] if cls_dir.exists() else []
    for row in range(2):
        ax = axes[row, col]
        ax.set_facecolor("#16213e")
        if row < len(imgs):
            img = mpimg.imread(str(imgs[row]))
            ax.imshow(img)
        ax.axis("off")
        if row == 0:
            ax.set_title(cls.capitalize(), color=COLORS[col], fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / "sample_images.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("✅  Kaydedildi: outputs/plots/sample_images.png")
plt.close()

print("\n🎉  Veri keşfi tamamlandı!")
