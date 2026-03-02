# Antigravity Tasks

## Branch: `feature/dataset-eval-report`

## Your Responsibilities:
1. Create/download dataset (pedestrians + cars)
2. Create YOLO annotations
3. Write evaluation script
4. Generate visualizations
5. Write final report
6. Do final integration testing

## Steps:

### Step 1 - Setup
```bash
git checkout -b feature/dataset-eval-report
```

### Step 2 - Create Dataset
Use this script to download from Open Images / COCO subset:
```python
# scripts/download_dataset.py
# Download ~200 images: 100 pedestrian, 100 car scenes
# from COCO dataset using fiftyone or manual download
# Convert annotations to YOLO format: class_id cx cy w h (normalized)
```

YOLO label format per line:
```
0 0.5 0.4 0.3 0.6   # class 0 = pedestrian
1 0.7 0.5 0.2 0.4   # class 1 = car
```

### Step 3 - Create `src/evaluate.py`
- Load model from `outputs/weights/best.pt`
- Run on test set
- Compute: Precision, Recall, mAP@0.5, mAP@0.5:0.95
- Save metrics to `outputs/metrics/results.json`
- Generate confusion matrix plot
- Generate PR curve plot

### Step 4 - Create `src/visualize.py`
UNIQUE FEATURES to add:
- Side-by-side grid: original vs predicted
- Failure case detector (highlight low-confidence detections)
- Class distribution chart
- Confidence score histogram

### Step 5 - Create `report/report.md`
Sections:
1. Approach Overview
2. Dataset Description (# images, class balance)
3. Model Architecture (YOLOv8n specs)
4. Training Setup (epochs, img size, augmentations)
5. Results Table (Precision/Recall/mAP per class)
6. Failure Cases Analysis
7. Improvements Attempted
8. Conclusion

### Step 6 - Push dataset branch first
```bash
git add dataset/ config/ src/evaluate.py src/visualize.py report/
git commit -m "feat: dataset, annotations, evaluation and report"
git push origin feature/dataset-eval-report
```

### Step 7 - Create PR to main, merge it

### Step 8 (AFTER Copilot merges their branch) - Final Testing
```bash
git checkout main
git pull origin main
python src/train.py
python src/detect.py
python src/evaluate.py
python src/visualize.py
```

Verify:
- [ ] Training completes without errors
- [ ] outputs/predictions/ has annotated images
- [ ] outputs/metrics/results.json exists
- [ ] mAP > 0.3 (acceptable for small dataset)
- [ ] Report is complete