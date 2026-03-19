"""
3_gradcam.py
Intel Image Classification — Grad-CAM Görselleştirme
Her sınıftan 1 örnek görüntü için ısı haritası üretir.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import models, transforms, datasets
from pathlib import Path
from PIL import Image

# ─── Cihaz ──────────────────────────────────────────────────────────────────
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE  = 224
NUM_CLASSES = 6
CLASSES   = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
COLORS    = ["#4E79A7", "#59A14F", "#76B7B2", "#9C755F", "#EDC948", "#B07AA1"]

DATA_DIR  = Path("data") / "seg_test" / "seg_test"
MODEL_DIR = Path("models") / "best_model.pth"
OUT_DIR   = Path("outputs/gradcam"); OUT_DIR.mkdir(parents=True, exist_ok=True)

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

# ─── Grad-CAM ───────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, img_tensor, class_idx=None):
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        logits = self.model(img_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global Avg Pool
        cam    = (weights * self.activations).sum(dim=1).squeeze()
        cam    = torch.relu(cam).cpu().numpy()
        cam    = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam    = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

# ─── Dönüşüm ────────────────────────────────────────────────────────────────
normalize = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = img * std + mean
    return np.clip(img, 0, 1)

# ─── Overlay Fonksiyonu ──────────────────────────────────────────────────────
def overlay_cam(img_np, cam, alpha=0.45):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay  = alpha * heatmap + (1 - alpha) * img_np
    return np.clip(overlay, 0, 1)

# ─── Ana Akış ────────────────────────────────────────────────────────────────
model  = load_model()
gradcam = GradCAM(model, model.layer4[-1])

fig, axes = plt.subplots(2, 6, figsize=(22, 8))
fig.patch.set_facecolor("#1a1a2e")
fig.suptitle("Grad-CAM — Model Neye Bakıyor?",
             color="white", fontsize=16, fontweight="bold")

for col, cls in enumerate(CLASSES):
    cls_dir = DATA_DIR / cls
    imgs    = sorted(cls_dir.glob("*.jpg"))
    if not imgs:
        continue
    img_path = imgs[0]
    pil_img  = Image.open(img_path).convert("RGB")
    tensor   = normalize(pil_img)
    cam, pred_idx = gradcam(tensor)
    img_np  = denormalize(tensor)
    overlay = overlay_cam(img_np, cam)

    # Orijinal
    axes[0, col].imshow(img_np)
    axes[0, col].set_title(f"{cls}\n(Gerçek)", color=COLORS[col],
                           fontsize=10, fontweight="bold")
    axes[0, col].axis("off")

    # Grad-CAM
    axes[1, col].imshow(overlay)
    pred_label = CLASSES[pred_idx]
    match_color = "#59A14F" if pred_idx == col else "#E15759"
    axes[1, col].set_title(f"Tahmin: {pred_label}",
                           color=match_color, fontsize=10, fontweight="bold")
    axes[1, col].axis("off")

    # Her görsel için ayrı dosya kaydet
    fig_single, ax_single = plt.subplots(1, 2, figsize=(8, 4))
    fig_single.patch.set_facecolor("#1a1a2e")
    ax_single[0].imshow(img_np); ax_single[0].set_title("Orijinal", color="white"); ax_single[0].axis("off")
    ax_single[1].imshow(overlay); ax_single[1].set_title(f"Grad-CAM → {pred_label}", color=match_color); ax_single[1].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"gradcam_{cls}.png", dpi=130, bbox_inches="tight",
                facecolor=fig_single.get_facecolor())
    plt.close(fig_single)
    print(f"  ✅ outputs/gradcam/gradcam_{cls}.png")

plt.tight_layout()
plt.savefig(OUT_DIR / "gradcam_all.png", dpi=130, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\n✅  Kaydedildi: outputs/gradcam/gradcam_all.png")
plt.close()
print("\n🎉  Grad-CAM görselleştirme tamamlandı!")
