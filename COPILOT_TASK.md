# GitHub Copilot Tasks

## Branch: `feature/model-pipeline`

## Your Responsibilities:
1. Setup YOLOv8 training pipeline
2. Write training script
3. Write detection/inference script
4. Write model config

## Steps:

### Step 1 - Setup
```bash
git checkout -b feature/model-pipeline
pip install ultralytics torch torchvision opencv-python
```

### Step 2 - Create `requirements.txt`
```
ultralytics>=8.0.0
torch>=2.0.0
torchvision
opencv-python
Pillow
matplotlib
pandas
seaborn
PyYAML
```

### Step 3 - Create `config/dataset.yaml`
```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test

nc: 2
names: ['pedestrian', 'car']
```

### Step 4 - Create `src/train.py`
- Load YOLOv8n (nano) model
- Train on dataset.yaml for 50 epochs
- Save best weights to `outputs/weights/best.pt`
- Log metrics to `outputs/metrics/`

### Step 5 - Create `src/detect.py`
- Load trained model
- Run inference on test images
- Draw bounding boxes with confidence scores
- Save annotated images to `outputs/predictions/`
- Add UNIQUE FEATURE: Real-time confidence heatmap overlay

### Step 6 - Push
```bash
git add .
git commit -m "feat: complete model training and detection pipeline"
git push origin feature/model-pipeline
```

## WAIT for Antigravity to merge dataset branch first before training!