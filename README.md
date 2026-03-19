# 🌍 Intel Image Classification
### Track 2: Computer Vision | ResNet50 Transfer Learning + Grad-CAM + Gradio

---

## 📁 Proje Yapısı

```
ödev2/
├── data/                     ← Kaggle'dan indirilen dataset buraya
│   ├── seg_train/seg_train/  ← buildings/ forest/ glacier/ mountain/ sea/ street/
│   └── seg_test/seg_test/
├── outputs/
│   ├── plots/                ← Eğitim eğrileri, confusion matrix vb.
│   └── gradcam/              ← Grad-CAM ısı haritaları
├── models/
│   └── best_model.pth        ← Eğitim sonrası kaydedilen model
├── 1_data_exploration.py
├── 2_train.py
├── 3_gradcam.py
├── 4_evaluate.py
├── 5_app.py
└── requirements.txt
```

---

## 🚀 Kurulum

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset İndirme (Manuel)

1. Kaggle'a git: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
2. **Download** butonuna tıkla → `archive.zip` indir
3. ZIP dosyasını `ödev2/data/` klasörüne çıkart
4. Klasör yapısının şu şekilde olduğunu doğrula:
   ```
   data/seg_train/seg_train/buildings/
   data/seg_train/seg_train/forest/
   ...
   data/seg_test/seg_test/buildings/
   ...
   ```

---

## ▶️ Çalıştırma Adımları (Sırayla)

### 1️⃣ Veri Keşfi
```bash
python 1_data_exploration.py
```
→ `outputs/plots/class_distribution.png` ve `sample_images.png` oluşur

### 2️⃣ Model Eğitimi (~30-45 dk GPU)
```bash
python 2_train.py
```
→ `models/best_model.pth` ve `outputs/plots/training_curves.png` oluşur

### 3️⃣ Grad-CAM Görselleştirme
```bash
python 3_gradcam.py
```
→ `outputs/gradcam/` altında her sınıf için heatmap PNG oluşur

### 4️⃣ Değerlendirme
```bash
python 4_evaluate.py
```
→ `outputs/plots/confusion_matrix.png` ve `classification_report.png` oluşur

### 5️⃣ Gradio Demo Başlat
```bash
python 5_app.py
```
→ Tarayıcıda: **http://127.0.0.1:7860**

---

## 🏗️ Model Mimarisi

```
ResNet50 (ImageNet pretrained)
    │
    ├── Aşama 1 (Feature Extraction): Tüm backbone frozen, sadece FC eğitildi (5 epoch)
    └── Aşama 2 (Fine-Tuning):       layer3 + layer4 + FC açıldı (10 epoch)
        │
        └── FC Head:
            Dropout(0.4) → Linear(2048→256) → ReLU → Dropout(0.3) → Linear(256→6)

Optimizer : AdamW  | weight_decay=1e-4
Scheduler : CosineAnnealingLR
Loss      : CrossEntropyLoss (label_smoothing=0.1)
```

---

## 📊 Beklenen Sonuçlar

| Metrik       | Beklenen |
|--------------|----------|
| Test Accuracy | %92–96   |
| Macro F1     | %91–95   |

---

## 🗺️ Grad-CAM Hakkında

Grad-CAM (Gradient-weighted Class Activation Mapping), modelin hangi piksellere bakarak karar verdiğini görselleştirir. Bu projede ResNet50'nin son konvolüsyon katmanı (`layer4[-1]`) üzerinde uygulanmakta ve sonuç orijinal görüntü üzerine ısı haritası olarak bindirilmektedir.
