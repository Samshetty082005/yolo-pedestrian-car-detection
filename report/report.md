# YOLO Pedestrian & Car Detection — Project Report

## 1. Approach Overview

This project implements a complete object detection pipeline for identifying **pedestrians** and **cars** in images using the YOLOv8 architecture. The workflow covers dataset acquisition and preparation, model training, inference with heatmap visualization, quantitative evaluation, and qualitative failure analysis.

The pipeline follows a modular design:
- **Dataset Builder** — Downloads COCO 2017 validation images and converts annotations to YOLO format
- **Training** — Fine-tunes YOLOv8n on the custom 2-class dataset
- **Inference** — Runs detection on test images with confidence-based heatmap overlays
- **Evaluation** — Computes precision, recall, and mAP metrics per class
- **Visualization** — Generates failure analysis grids, comparison images, and confidence histograms

---

## 2. Dataset Description

| Property | Value |
|----------|-------|
| **Source** | COCO 2017 Validation Set |
| **Total Images** | 100 (filtered for person/car annotations) |
| **Train Split** | 70 images (70%) |
| **Validation Split** | 15 images (15%) |
| **Test Split** | 15 images (15%) |
| **Classes** | 2 — `pedestrian` (COCO "person"), `car` (COCO "car") |
| **Annotation Format** | YOLO (normalized cx, cy, w, h) |

### Class Distribution
- **Pedestrian** annotations dominate the dataset (~84 instances in test set, 70 with GT labels)
- **Car** annotations are less frequent (~14 instances in test set)
- This class imbalance affects model performance, particularly recall for cars

The dataset was sourced via FiftyOne's COCO downloader (with a manual fallback due to MongoDB unavailability). Bounding boxes were converted from COCO absolute `[x, y, w, h]` to YOLO relative `[cx, cy, w, h]` format with clamping to `[0, 1]`.

---

## 3. Model Architecture

| Property | Value |
|----------|-------|
| **Model** | YOLOv8n (nano) |
| **Parameters** | 3,006,038 |
| **GFLOPs** | 8.1 |
| **Layers** | 73 (fused) |

**Why YOLOv8n?**
- Lightweight enough for rapid prototyping on a small dataset
- Fast inference (~3-5ms per image on RTX 3050 Ti)
- Strong baseline performance even with limited training data
- Built-in data augmentation (mosaic, flip, scale, etc.)

---

## 4. Training Setup

| Parameter | Value |
|-----------|-------|
| **Epochs** | 50 |
| **Image Size** | 640×640 |
| **Batch Size** | 16 |
| **Learning Rate (lr0)** | 0.001 |
| **Optimizer** | SGD (Ultralytics default) |
| **GPU** | NVIDIA GeForce RTX 3050 Ti (4GB VRAM) |
| **Training Time** | ~1 minute (50 epochs) |
| **Augmentations** | Mosaic (first 40 epochs), horizontal flip, scale, HSV jitter |

Training used mosaic augmentation for the first 40 epochs (automatically disabled for the final 10), which helps the model learn to detect objects at various scales and in cluttered scenes.

---

## 5. Results

### Test Set Metrics

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|---------------|
| **Pedestrian** | 0.4617 | 0.4429 | 0.4095 | 0.2610 |
| **Car** | 0.5673 | 0.2143 | 0.3136 | 0.1920 |
| **Overall** | 0.5145 | 0.3286 | 0.3615 | 0.2267 |

### Validation Set Metrics (Best Epoch)

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|---------------|
| **Pedestrian** | 0.850 | 0.533 | 0.551 | 0.272 |
| **Car** | 0.964 | 0.429 | 0.503 | 0.309 |
| **Overall** | 0.907 | 0.481 | 0.527 | 0.290 |

The validation set performance (mAP50=0.527) is noticeably higher than the test set (mAP50=0.362), suggesting some overfitting or distribution variance across the small splits.

---

## 6. Unique Features Implemented

### 6.1 Confidence Heatmap Overlay (`src/detect.py`)
The detection script overlays a JET colormap-based heatmap on predicted bounding boxes, where intensity corresponds to confidence score. This provides an intuitive visual indicator of where the model is most and least certain.

### 6.2 Failure Case Auto-Detection Grid (`src/visualize.py`)
Automatically identifies predictions with confidence below 0.4, crops them with red borders, and arranges them in a diagnostic grid. **8 failure cases** were detected in the test set, helping identify challenging scenarios (small objects, partial occlusions).

### 6.3 Side-by-Side Prediction Comparison (`src/visualize.py`)
Generates a 3-row grid showing original test images alongside their predicted annotations, making it easy to visually assess detection coverage and accuracy.

### 6.4 Confidence Score Histogram (`src/visualize.py`)
Visualizes the distribution of confidence scores across both classes, revealing that the model tends to make predictions with moderate confidence (0.4–0.7 range) with few highly confident detections.

---

## 7. Failure Cases & Issues

### False Positives Observed
- Some background objects (poles, fire hydrants) occasionally misclassified as pedestrians at low confidence
- Car-like objects (vans, trucks) sometimes detected as cars with moderate confidence

### False Negatives Observed
- **Small/distant pedestrians** frequently missed — the model struggles with objects occupying less than ~5% of the image
- **Heavily occluded pedestrians** in crowded scenes are often not detected
- **Cars in unusual orientations** (aerial view, partially hidden) missed entirely
- 3 test images produced **zero detections**, indicating complete failure on certain scene types

### Difficult Scenarios
- **Crowded scenes**: Multiple overlapping pedestrians cause missed detections
- **Low contrast/lighting**: Dim or backlit scenes reduce detection confidence
- **Small objects at distance**: The 640px input resolution limits detection of far-away subjects
- **Class imbalance**: With only ~14 car instances in the test set, car recall (0.21) is notably lower than pedestrian recall (0.44)

---

## 8. Improvements Attempted

1. **Bug Fix — Learning Rate Parameter**: The original `train.py` passed `lr=` to Ultralytics, which expects `lr0=`. Fixed to enable successful training.

2. **Dataset Fallback**: When FiftyOne's MongoDB dependency failed, implemented a direct COCO JSON parser to extract and convert annotations without the full FiftyOne pipeline.

### Potential Future Improvements
- **More training data**: 70 training images is very limited; scaling to 500+ would significantly boost performance
- **Larger model**: Switching from YOLOv8n to YOLOv8s or YOLOv8m for better accuracy
- **More epochs**: Training for 100-200 epochs with learning rate scheduling
- **Class-balanced sampling**: Address the pedestrian/car imbalance with oversampling
- **Test-time augmentation (TTA)**: Multi-scale inference for better small object detection
- **Transfer learning**: Start from COCO-pretrained weights (already partially done via YOLOv8n defaults)

---

## 9. Conclusion

This project successfully demonstrates an end-to-end YOLO-based object detection pipeline for pedestrian and car detection. While the overall mAP@0.5 of **0.362** on the test set indicates room for improvement, the model achieves reasonable precision (0.515) and shows clear learning from the limited 70-image training set.

Key takeaways:
- **YOLOv8n** provides a solid foundation for rapid prototyping with minimal compute
- **Data quantity** is the primary bottleneck — 70 training images is insufficient for robust generalization
- **Class imbalance** (pedestrians vs cars) directly impacts per-class performance
- The **visualization tools** (heatmap overlays, failure analysis, confidence histograms) provide valuable diagnostic capabilities beyond raw metrics

The pipeline is fully automated and reproducible: dataset download → training → inference → evaluation → visualization → reporting, all executable with simple Python commands.
