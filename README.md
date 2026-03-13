# Multi-View Visual Geolocation 
**Dual-Task US State Classification & GPS Regression**

Achieved a **0.943 Kaggle score** by engineering a sophisticated computer vision pipeline that predicts geographic locations from panoramic imagery.

## Project Overview
This system processes four directional 256×256 images (North, East, South, West) to predict the **US State** and **Precise GPS Coordinates**. The challenge lies in extracting environmental cues like architectural styles, vegetation, and road textures to geolocate images without metadata.

## Technical Architecture
### 1. Multi-View Fusion Architecture
Instead of processing images independently, the model treats the four directions as a unified panoramic context:
* **Backbone Encoders:** Utilized **DINOv2 (ViT-L/16)** and **ConvNeXt V2-Large** (pre-trained on LVD-142M) for robust feature extraction.
* **Transformer Fusion:** A 2-layer Transformer Encoder uses **self-attention** to correlate features across the four views, creating a global spatial descriptor.
* **Dual-Head Output:** * **State Head:** Multi-class classification for 33 US states (70% score weight).
    * **GPS Head:** Continuous regression for Latitude/Longitude (30% score weight).

### 2. Optimization & Mathematical Constraints
* **Haversine Distance Loss:** Optimized GPS regression using the Haversine formula to account for the Earth's curvature, minimizing actual kilometer-distance error rather than Euclidean distance.
* **Differential Learning Rates:** Backbones were fine-tuned at $1 \times 10^{-5}$ while prediction heads were trained at $2 \times 10^{-4}$ to preserve pre-trained feature integrity.

### 3. "Grandmaster" Inference Strategies
* **Test-Time Augmentation (TTA):** Implemented 5-crop TTA and horizontal flips to reduce prediction variance.
* **Ensemble Blending:** Weighted averaging (70/30) of DINOv2 and ConvNeXt models for balanced classification and spatial stability.
* **Hard Grid Snapping:** Utilized **K-Nearest Neighbors (KNN)** to snap predicted coordinates to known road networks from the training database, eliminating "off-road" impossibilities.

## Performance
* **Kaggle Score:** 0.943
* **Competition Timeline:** Dec 1, 2025 – Dec 20, 2025
* **Dataset:** 65,980 Training samples (263,920 total images)

## Repository Structure
* `street-view.ipynb`: End-to-end pipeline (Data loading, Architecture, Training, Inference).
* `requirements.txt`: Environment dependencies.
