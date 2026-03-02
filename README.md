# 👁️ Retinal AI: Lesion Detector

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io/)

A professional-grade, deep-learning based diagnostic laboratory for **Multi-Label Retinal Disease Detection**. This system compares **EfficientNet-B3** and **ResNet-50** side-by-side to provide a clinical consensus for four major retinal pathologies.

---

## 🚀 Key Features

*   **Dual-Model Diagnostic Engine**: Simultaneous analysis using two state-of-the-art architectures (EfficientNet & ResNet).
*   **Grad-CAM Heatmaps (Explainability)**: Visualizes the AI's "attention zones" with high-contrast heatmaps and target markers (🎯) pinpointing lesion sites.
*   **Clinical Calibration Boost**: Adaptive sensitivity controls for detecting early-stage or subtle pathologies that standard models often miss.
*   **Multi-Disease Coverage**: Detects **Diabetic Retinopathy (DR)**, **Glaucoma**, **Cataract**, and **Age-related Macular Degeneration (AMD)**.
*   **Lesion Identification**: Detailed clinical interpretation of highlighted regions (e.g., identifying microaneurysms, macular drusen, or optic disc cupping).
*   **Performance Reliability Hub**: Real-time system benchmarks using animated gauges for Accuracy, F1-Score, and Recall.

---

## 📂 Project Structure

*   `app.py`: The main **Streamlit Interactive Dashboard**.
*   `Retinal_AI_Final.ipynb`: Research-oriented **Jupyter Notebook** with detailed visualizations.
*   `data/`: Localized clinical dataset (~5,000 fundus images).
*   `models/`: Directory for pre-trained weights (`best_efficientnet_b3.pth`, `best_resnet50.pth`).
*   `final_dataset_5000.csv`: Unified metadata for the clinical image library.

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Vamsikrishna-post/Retinal-AI-Lesion-Detector
   cd "Retinal-AI-Lesion-Detector"
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision numpy pandas opencv-python Pillow streamlit matplotlib tqdm
   ```

3. **Ensure models are present**:
   Place your trained weights in the `models/` folder as `best_efficientnet_b3.pth` and `best_resnet50.pth`.

---

## 🖥️ How to Use

### 1. Launch the Live Clinic (Streamlit)
To start the interactive web application, run:
```powershell
python -m streamlit run app.py
```
*   **Upload** a high-resolution fundus image (.jpg or .png).
*   Use the **Sidebar** to adjust clinical sensitivity or apply high-contrast calibration.
*   Toggle between **Tabs** to view scores, heatmaps, and lesion analysis.

### 2. Research & Analysis (Jupyter Notebook)
Open `Retinal_AI_Final.ipynb` in your preferred editor (VS Code or Jupyter Lab) to perform batch analysis and generate high-quality research visualizations.

---

## 🔬 Clinical Methodology

The system utilizes **Grad-CAM (Gradient-weighted Class Activation Mapping)** to ensure medical transparency. It identifies regions of interest (ROI) such as the **Macula**, **Optic Nerve**, or **Peripheral Retina** and applies noise-suppression algorithms to provide clean, lesion-focused visualizations.

---

## ⚠️ Disclaimer
*This application is a research-oriented AI screening tool. It is designed to assist clinicians in identifying structural and textural pathologies in fundus scans but should not be used as a standalone diagnostic system without professional ophthalmological verification.*

---

**Built with ❤️ for Clinical AI Research.** 👁️🛡️🔬⚖️
