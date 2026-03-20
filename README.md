# 🌍 Intel Image Classification via Deep Transfer Learning

[cite_start]This repository contains a full-stack **Intelligent Application** developed for the **Building an Intelligent Application** course[cite: 2, 6]. [cite_start]The project utilizes **ResNet50** and **Transfer Learning** to classify natural and urban scenes with high accuracy and explainability[cite: 1, 15, 37].

## 🚀 Key Features
* [cite_start]**Two-Phase Transfer Learning:** Sequential training starting with **Feature Extraction** (frozen backbone) followed by deep **Fine-Tuning** of the last two residual blocks[cite: 48, 50].
* [cite_start]**Explainable AI (XAI):** Integrated **Grad-CAM** visualizations to interpret exactly where the model focuses its visual attention[cite: 169, 172].
* [cite_start]**Interactive Demo:** A functional web interface powered by **Gradio** for real-time inference and heat-map generation[cite: 176, 177].
* [cite_start]**Advanced Optimization:** Utilizes **AdamW** optimizer with **Cosine Annealing** and **Label Smoothing** ($\alpha=0.1$) for robust generalization[cite: 53, 57, 59].

## 📊 Dataset Statistics
[cite_start]The project utilizes the **Intel Image Classification** dataset[cite: 17]:
* [cite_start]**Total Images:** 17,034 images[cite: 18].
* [cite_start]**Split:** 14,034 for training/validation and 3,000 for testing[cite: 18].
* [cite_start]**Categories:** 6 balanced classes: Buildings, Forest, Glacier, Mountain, Sea, and Street[cite: 14, 19].
* [cite_start]**Resolution:** Uniformly sized at $150 \times 150$ pixels, upscaled to $224 \times 224$ for ResNet50 input[cite: 18, 28].

## 🛠️ Project Structure
* `1_data_exploration.py`: Exploratory Data Analysis (EDA) and class distribution visualization.
* `2_train.py`: Core training pipeline including data augmentation and two-phase backpropagation.
* `3_gradcam.py`: Script for generating and saving Grad-CAM heatmaps for model transparency.
* `4_evaluate.py`: Evaluation suite producing Confusion Matrices and per-class F1-Score reports.
* `5_app.py`: The final interactive Gradio-based web application.

## 📈 Final Performance
[cite_start]The model achieved an outstanding **94.03% Accuracy** on the strictly unseen test set[cite: 67, 180].
* [cite_start]**Macro F1-Score:** 0.9416[cite: 67].
* [cite_start]**Top Performers:** Forest and Sea classes achieved near-perfect F1-scores[cite: 97].
* [cite_start]**Key Insight:** Minor confusion was observed only between highly similar visual concepts like Glaciers and Mountains[cite: 100, 101].

## 🔗 Project Links
* [cite_start]**Video Presentation:** [Insert YouTube / Google Drive Video Link Here] [cite: 10]
* [cite_start]**Code Repository:** [GitHUB Link] [cite: 9]
* **Full Report:** [Insert PDF Link Here]

---
[cite_start]Developed by **Muhammed Kabalak**[cite: 3].
