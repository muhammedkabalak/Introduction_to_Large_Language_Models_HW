"""
5_app.py
Intel Image Classification — Gradio Demo Uygulaması
Kullanıcı görüntü yükler → Sınıf tahmini + confidence + Grad-CAM heatmap
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import io

# ─── Ayarlar ────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
IMG_SIZE    = 224
MODEL_PATH  = Path("models/best_model.pth")

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
EMOJIS  = ["🏢",        "🌲",     "🏔️",     "⛰️",      "🌊",  "🛣️"]
COLORS  = ["#4E79A7", "#59A14F", "#76B7B2", "#9C755F", "#EDC948", "#B07AA1"]

CLASS_DESCS = {
    "buildings": "Binalar ve kentsel yapılar",
    "forest":    "Ormanlık ve doğal alanlar",
    "glacier":   "Buzullar ve buz kütleleri",
    "mountain":  "Dağlar ve yüksek zirveler",
    "sea":       "Deniz ve su yüzeyleri",
    "street":    "Sokaklar ve kent yolları",
}

# ─── Model ──────────────────────────────────────────────────────────────────
def load_model():
    model = models.resnet50(weights=None)
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(in_feat, 256),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

model = load_model()
print(f"✅  Model yüklendi | Cihaz: {DEVICE}")

# ─── Dönüşümler ─────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * std + mean, 0, 1)

# ─── Grad-CAM ───────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.grads = None; self.acts = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o.detach()))
        target_layer.register_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0].detach()))

    def __call__(self, tensor, class_idx=None):
        t = tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
        out = self.model(t)
        if class_idx is None: class_idx = out.argmax(1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()
        w   = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(dim=1)).squeeze().cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

gradcam = GradCAM(model, model.layer4[-1])

# ─── Tahmin + Grad-CAM Fonksiyonu ────────────────────────────────────────────
def predict_and_explain(pil_img):
    if pil_img is None:
        return "Lütfen bir görüntü yükleyin.", None

    pil_img = pil_img.convert("RGB")
    tensor  = preprocess(pil_img)

    # Tahmin
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(DEVICE))
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx   = probs.argmax()
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx] * 100

    # Grad-CAM
    cam, _ = gradcam(tensor, class_idx=pred_idx)
    img_np = denormalize(tensor)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay  = np.clip(0.45 * heatmap + 0.55 * img_np, 0, 1)

    # Sonuç metni
    result_text = (
        f"**{EMOJIS[pred_idx]}  Tahmin: {pred_class.upper()}**\n\n"
        f"**Güven:** {confidence:.1f}%\n\n"
        f"**Açıklama:** {CLASS_DESCS[pred_class]}\n\n"
        "---\n\n**Tüm Olasılıklar:**\n\n" +
        "\n".join(
            f"{'▶' if i == pred_idx else ' '} {EMOJIS[i]} `{CLASSES[i]:<12}` "
            f"{'█' * int(probs[i]*20):<20} {probs[i]*100:5.1f}%"
            for i in range(NUM_CLASSES)
        )
    )

    # Görsel oluştur: 1x2 (orijinal | gradcam)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes: ax.set_facecolor("#16213e")

    axes[0].imshow(img_np)
    axes[0].set_title("Yüklenen Görüntü", color="white", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM — Model Neye Bakıyor?",
                      color=COLORS[pred_idx], fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.suptitle(f"Tahmin: {EMOJIS[pred_idx]} {pred_class.upper()}  ({confidence:.1f}%)",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close()
    result_img = Image.open(buf)

    return result_text, result_img

# ─── Örnek Görseller ────────────────────────────────────────────────────────
EXAMPLE_DIR = Path("data/seg_test/seg_test")
examples = []
for cls in CLASSES:
    imgs = sorted((EXAMPLE_DIR / cls).glob("*.jpg"))
    if imgs:
        examples.append([str(imgs[0])])

# ─── Gradio Arayüzü ─────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    title="Intel Image Classifier",
    css="""
    body { background: #1a1a2e !important; }
    .gradio-container { max-width: 960px; margin: auto; }
    .result-text { font-family: monospace; font-size: 0.9em; }
    """
) as demo:
    gr.Markdown(
        """
        # 🌍 Intel Image Classification
        ### ResNet50 Transfer Learning + Grad-CAM Görselleştirme
        Bir doğa veya şehir fotoğrafı yükleyin — model sınıfı tahmin edip **neye baktığını** göstersin!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="📸 Görüntü Yükle", height=280)
            predict_btn = gr.Button("🔍 Tahmin Et", variant="primary", size="lg")
            gr.Markdown("**Örnek Görseller:**")
            if examples:
                gr.Examples(examples=examples, inputs=img_input,
                            label="Hazır örneklerden seç")

        with gr.Column(scale=1):
            result_text = gr.Markdown(label="📊 Tahmin Sonucu", elem_classes=["result-text"])

    gradcam_img = gr.Image(label="🗺️ Grad-CAM Isı Haritası", height=350)

    predict_btn.click(
        fn=predict_and_explain,
        inputs=[img_input],
        outputs=[result_text, gradcam_img],
    )
    img_input.change(
        fn=predict_and_explain,
        inputs=[img_input],
        outputs=[result_text, gradcam_img],
    )

    gr.Markdown(
        """
        ---
        **Sınıflar:** 🏢 Buildings · 🌲 Forest · 🏔️ Glacier · ⛰️ Mountain · 🌊 Sea · 🛣️ Street
        
        > Model: **ResNet50** (ImageNet pretrained → fine-tuned on Intel Dataset) | Yazılım: PyTorch + Gradio
        """
    )

if __name__ == "__main__":
    print("🚀  Gradio demo başlatılıyor...")
    demo.launch(share=False, server_port=7860)
