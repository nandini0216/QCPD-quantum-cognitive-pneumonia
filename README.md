# Quantum-Cognitive Pneumonia Detection: Hybrid QSNN-QLSTM Architecture

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum%20ML-000000)
![License](https://img.shields.io/badge/License-MIT-green)

A novel hybrid classical-quantum machine learning model designed to detect pneumonia from chest X-ray images. This architecture combines a classical ResNet18 feature extractor with a biologically-inspired quantum neural network bridge, achieving high accuracy while offering robust Explainable AI (XAI) visualizations.

## 🧠 Architecture Overview

![Hybrid Quantum-Cognitive Architecture](Quantum%20Architecture%20Diagram%20QSNN%20+%20QLSTM.png)

This project mimics biological brain functions using quantum circuits to process complex medical imaging data:
* **Classical Backbone:** A pretrained ResNet18 model extracts high-level features from chest X-rays, which are then compressed into a 4-qubit quantum state.
* **QSNN (Hypothalamus):** A Quantum Spiking Neural Network utilizing RX encoding and a "Data Re-uploading" mechanism to simulate temporal summation in biological neurons.
* **QLSTM (Hippocampus):** A Quantum Long Short-Term Memory network utilizing RY encoding and CRX/CRY controlled gating to mimic memory and forget/input gates.

## ✨ Key Features

* **Advanced Data Augmentation:** Implements Contrast Limited Adaptive Histogram Equalization (CLAHE) for local contrast enhancement, alongside random affine transformations to prevent overfitting.
* **Class Imbalance Handling:** Utilizes PyTorch's `WeightedRandomSampler` to effectively manage the inherent imbalances in medical datasets.
* **Explainable AI (XAI) Suite:** Includes custom visualizers to interpret the model's decision-making process:
  * **Classical Grad-CAM:** Highlights classical feature attention.
  * **Landmark Analysis:** Extracts precise Regions of Interest (ROI) via bounding boxes.
  * **Quantum Attention:** Generates smoothed saliency maps to visualize the quantum layer's focus.

## 📊 Model Performance

The model was evaluated on the chest-xray-pneumonia dataset and achieved the following metrics on the test set (Threshold: 0.97):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 91.83% |
| **Recall (Sensitivity)** | 95.38% |
| **Precision** | 91.85% |
| **F1 Score** | 0.9358 |
| **Specificity** | 85.90% |
| **ROC AUC** | 0.9224 |

*Estimated Inference Latency: ~109.82 ms / image (CPU)*

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision, MONAI
* **Quantum Computing:** PennyLane (`default.qubit`)
* **Data Processing & Vision:** OpenCV, PIL, NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Metrics:** Scikit-learn

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies using pip:

```bash
pip install torch torchvision pennylane opencv-python numpy pandas matplotlib seaborn scikit-learn monai kagglehub
