# 🌊 Water Quality Classification & Monitoring System

This project implements a sophisticated **Water Analysis & Quality Monitoring system** using Deep Learning (Computer Vision) and Statistical Machine Learning techniques. The system is designed to analyze water images for type classification and process sensor data for forecasting and anomaly detection, helping environmentalists and researchers monitor water health in real-time.

## Technical Implementation

### Computer Vision Components
**Image Preprocessing:**
*   **Normalization:** Utilizes ImageNet-standard mean and standard deviation for robust feature scaling.
*   **Transformation:** Implements `torchvision.transforms` for consistent resizing (256px), center cropping (224px), and tensor conversion.
*   **Augmentation:** Ensures input consistency across multiple deep learning architectures.

### Deep Learning Architecture
**Stacking Ensemble Model:**
*   **Base Learners:** Combines the predictive power of **ResNet-50**, **EfficientNet-B3**, and **MobileNetV3-Large** trained on diverse water datasets.
*   **Probability Feature Extraction:** Transforms raw images into a high-dimensional probability space by concatenating output logs from all three base models.
*   **Meta-Learner:** Uses a refined **Logistic Regression** model as a meta-classifier to make the final prediction based on ensemble consensus, significantly increasing accuracy.

### Environmental Monitoring
*   **Forecasting Model:** Uses Linear Regression to predict parameters like **pH**, **Temperature**, and **Dissolved Oxygen**.
*   **Anomaly Detection:** Implements a **Z-Score analysis engine** to flag dangerous contamination levels based on standard deviations from the historical mean.
*   **Serialized Models:** Saves and loads all PyTorch (`.pth`) and Scikit-learn (`.pkl`) models using `joblib` for high-performance inference.

## Features
*   **Interactive Web Interface:** Built with React and Vite for a lightning-fast, premium user experience.
*   **Real-time Image Analysis:** Instant classification of water types (Clean, Muddy, Industrial, etc.) via API.
*   **Automated Forecasting:** Visualizes water quality trends for the next 24 hours or 7 days.
*   **Batch Anomaly Reporting:** Scans historical sensor data to rank and identify critical water quality deviations.
*   **Modern Aesthetics:** Dark-mode interface with glassmorphism and responsive design elements.

## Dataset
The system utilizes two primary data sources:
1.  **Image Library:** Thousands of classified water samples across 8 distinct categories.
2.  **Sensor Data:** A comprehensive historical dataset (`water_quality_with_timestamp.csv`) containing real-world water parameters.

## How to Use
1.  Access the application through the **Vercel-hosted web interface**.
2.  **Classification:** Upload a photo of a water body to receive an instant AI analysis and top-3 confidence scores.
3.  **Forecasting:** Select a parameter (e.g., pH) and a time horizon to see predicted environmental trends.
4.  **Anomaly:** Set a Z-score threshold (e.g., 2.5) to filter and inspect statistical outliers in the water data.

## Technical Requirements
*   **Python 3.11+**
*   **FastAPI / Uvicorn**
*   **PyTorch / Torchvision**
*   **React 18+ / Vite**
*   **Scikit-learn**
*   **Pandas / Numpy**

## Project Structure
```text
├── main.py                 # FastAPI production backend
├── models/                 
│   ├── image_classification/ # Stacking Ensemble models (.pth & .pkl)
│   └── forecast_regression/  # Parameter regression models (.pkl)
├── frontend/               
│   ├── src/                  # React components and dashboard logic
│   └── vercel.json           # Frontend deployment configuration
├── water_quality_with_timestamp.csv  # Historical sensor dataset
├── Dockerfile              # HuggingFace Spaces configuration
└── requirements-fastapi.txt # Backend dependencies
```
