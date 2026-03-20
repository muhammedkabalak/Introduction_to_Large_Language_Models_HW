# 🌍 Intel Image Classification via Deep Transfer Learning

This repository contains a full-stack **Intelligent Application** developed for the **Building an Intelligent Application** course. The project utilizes **ResNet50** and **Transfer Learning** to classify natural and urban scenes with high accuracy and explainability.

## 🚀 Key Features
* **Two-Phase Transfer Learning:** Sequential training starting with **Feature Extraction** (frozen backbone) followed by deep **Fine-Tuning** of the last two residual blocks.
* **Explainable AI (XAI):** Integrated **Grad-CAM** visualizations to interpret exactly where the model focuses its visual attention.
* **Interactive Demo:** A functional web interface powered by **Gradio** for real-time inference and heat-map generation.
* **Advanced Optimization:** Utilizes **AdamW** optimizer with **Cosine Annealing** and **Label Smoothing** ($\alpha=0.1$) for robust generalization.

## 📊 Dataset Statistics
The project utilizes the **Intel Image Classification** dataset:
* **Total Images:** 17,034 images.
* **Split:** 14,034 for training/validation and 3,000 for testing.
* **Categories:** 6 balanced classes: Buildings, Forest, Glacier, Mountain, Sea, and Street.
* **Resolution:** Uniformly sized at 150x150 pixels, upscaled to 224x224 for ResNet50 input.

## 🛠️ Project Structure
* `1_data_exploration.py`: Exploratory Data Analysis (EDA) and class distribution visualization.
* `2_train.py`: Core training pipeline including data augmentation and İki aşamalı backpropagation.
* `3_gradcam.py`: Script for generating and saving Grad-CAM heatmaps for model transparency.
* `4_evaluate.py`: Evaluation suite producing Confusion Matrices and per-class F1-Score reports.
* `5_app.py`: The final interactive Gradio-based web application.

## 📈 Final Performance
The model achieved an outstanding **94.03% Accuracy** on the strictly unseen test set.
* **Macro F1-Score:** 0.9416.
* **Top Performers:** Forest and Sea classes achieved near-perfect F1-scores.
* **Key Insight:** Minor confusion was observed only between highly similar visual concepts like Glaciers and Mountains.

## 🔗 Project Links
* **Video Presentation:** [Insert YouTube / Google Drive Video Link Here]
* **Code Repository:** [GitHUB Link]
* **Full Report:** [Insert PDF Link Here]

---
Developed by **Muhammed Kabalak**.
