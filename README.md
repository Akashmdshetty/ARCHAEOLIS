# 🏛️ AI-Driven Archaeological Site Mapping

A production-ready deep learning platform designed to analyze satellite and drone imagery to identify archaeological terrain patterns and ancient structures using **Self-Supervised Learning (SSL)** and **Multi-Task Analytics**.

## 🌟 Overview

This system uses **Bootstrap Your Own Latent (BYOL)** with a **ResNet18 backbone** to learn robust geospatial representations. It features:
- **Terrain Classification**: High-accuracy identification of mounds, settlements, and natural features.
- **Advanced Analytics**: Automated masks for **Ruins (Red)**, **Vegetation (Green)**, and **Erosion Risk (Yellow Heatmap)**.
- **Artifact Detection**: localized **Blue Bounding Boxes** for archaeological artifacts.

## 🚀 Execution Steps

### Step 1: Dataset Setup
Place your dataset at:
`C:\Users\aakas\Downloads\Val_norm.zip`

### Step 2: Dataset Preparation
Automatically extract, validate, and resize images to 224x224:
```bash
python data/prepare_dataset.py
```

### Step 3: SSL Model Training (Unlabeled)
Train the encoder on the processed imagery:
```bash
python ssl_training/train_byol.py
```

### Step 4: Feature Extraction & Discovery
Extract feature vectors and run clustering to discover new site patterns:
```bash
python feature_extraction/extract_embeddings.py
python clustering/discover_sites.py
```

### Step 5: Linear Classifier Training
Fine-tune the linear head for specific site types:
```bash
python classification/train_classifier.py
```

### Step 6: Launch Dashboard
Interactive UI for multi-layered archaeological analysis:
```bash
streamlit run dashboard/streamlit_app.py
```

## 🛠️ Configuration
Modify hyperparameters in `configs/config.yaml`.

## 📜 Project Structure
- `data/prepare_dataset.py`: Extraction and cleaning.
- `ssl_training/`: BYOL implementation and training logic.
- `models/analysis_heads.py`: Multi-task decoder for segmentation and detection.
- `utils/visualization_utils.py`: Rendering masks, boxes, and heatmaps.
- `dashboard/`: Streamlit interactive app with Advanced Analytics tab.
