#!/usr/bin/env python3
"""
Generate visualization outputs for the YOLO pedestrian + car detection project.

Outputs:
  - outputs/metrics/failure_cases.png       — grid of low-confidence / failed predictions
  - outputs/metrics/comparison_grid.png     — 3x3 side-by-side original vs predicted
  - outputs/metrics/confidence_histogram.png — confidence score distribution by class
"""

import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from ultralytics import YOLO


# ── Configuration ───────────────────────────────────────────────
MODEL_PATH = "outputs/weights/best.pt"
TEST_DIR = "dataset/images/test"
PREDICTIONS_DIR = "outputs/predictions"
METRICS_DIR = "outputs/metrics"
CLASS_NAMES = ["pedestrian", "car"]
CONF_THRESHOLD = 0.4   # below this = failure case


def load_test_images():
    """Return sorted list of test image paths."""
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(TEST_DIR, pat)))
    return sorted(paths)


def run_predictions(model, image_paths):
    """Run model on images and return list of (img_path, img, results) tuples."""
    all_results = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        results = model(img, verbose=False)
        all_results.append((img_path, img, results[0]))
    return all_results


# ═══════════════════════════════════════════════════════════════
# 1. Failure Case Grid
# ═══════════════════════════════════════════════════════════════
def generate_failure_cases(all_results, save_path):
    """Find low-confidence predictions and display them in a grid with red border."""
    print("\n🔍  Generating failure case grid...")

    failure_crops = []  # list of (crop_img, label_text)

    for img_path, img, res in all_results:
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, clss):
            if conf < CONF_THRESHOLD:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Pad the crop a bit
                h, w = img.shape[:2]
                pad = 15
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(w, x2 + pad)
                cy2 = min(h, y2 + pad)
                crop = img[cy1:cy2, cx1:cx2].copy()
                if crop.size == 0:
                    continue

                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
                reason = f"Low conf: {conf:.2f}"
                label = f"{cls_name}\n{reason}"
                failure_crops.append((crop, label, conf))

    if not failure_crops:
        # If no low-conf predictions, show images with no detections as failures
        for img_path, img, res in all_results:
            if len(res.boxes) == 0:
                crop = cv2.resize(img, (200, 200))
                failure_crops.append((crop, "No detections\n(False negative)", 0.0))

    if not failure_crops:
        print("  ⚠️  No failure cases found — model performed well!")
        # Create a simple "no failures" image
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, "No failure cases detected!\nAll predictions above confidence threshold.",
                ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return

    # Arrange in grid
    n = len(failure_crops)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    rows = min(rows, 4)  # max 4 rows
    display_n = rows * cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("Failure Cases (conf < 0.4)", fontsize=16, fontweight="bold", y=0.98)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        if idx < len(failure_crops):
            crop, label, conf = failure_crops[idx]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ax.imshow(crop_rgb)
            # Red border
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)
            ax.set_title(label, fontsize=10, color="red", fontweight="bold")
        else:
            ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅  Failure cases saved to {save_path} ({len(failure_crops)} cases)")


# ═══════════════════════════════════════════════════════════════
# 2. Side-by-Side Comparison Grid
# ═══════════════════════════════════════════════════════════════
def generate_comparison_grid(all_results, save_path):
    """Pick up to 9 test images, show original | predicted side by side in 3x3."""
    print("\n📸  Generating comparison grid...")

    # Select up to 9 images
    selected = all_results[:9]
    n = len(selected)
    rows = min(3, n)

    fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 5))
    if rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Original vs Predicted (Side-by-Side)", fontsize=16, fontweight="bold", y=0.98)

    for idx in range(rows):
        if idx < n:
            img_path, original, res = selected[idx]

            # Original image
            orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            axes[idx, 0].imshow(orig_rgb)
            axes[idx, 0].set_title(f"Original — {os.path.basename(img_path)}", fontsize=10)

            # Predicted image — draw boxes on copy
            predicted = original.copy()
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)

            colors_map = {0: (0, 255, 0), 1: (255, 165, 0)}  # green for pedestrian, orange for car
            for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, clss):
                color = colors_map.get(cls_id, (0, 255, 0))
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
                cv2.rectangle(predicted, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(predicted, f"{cls_name} {conf:.2f}",
                            (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)

            pred_rgb = cv2.cvtColor(predicted, cv2.COLOR_BGR2RGB)
            axes[idx, 1].imshow(pred_rgb)
            n_dets = len(boxes)
            axes[idx, 1].set_title(f"Predicted — {n_dets} detection(s)", fontsize=10)

        for j in range(2):
            axes[idx, j].set_xticks([])
            axes[idx, j].set_yticks([])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅  Comparison grid saved to {save_path}")


# ═══════════════════════════════════════════════════════════════
# 3. Confidence Score Histogram
# ═══════════════════════════════════════════════════════════════
def generate_confidence_histogram(all_results, save_path):
    """Bar chart of confidence score distribution, colored by class."""
    print("\n📊  Generating confidence histogram...")

    class_confs = {cls_name: [] for cls_name in CLASS_NAMES}

    for img_path, img, res in all_results:
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)

        for conf, cls_id in zip(confs, clss):
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
            if cls_name in class_confs:
                class_confs[cls_name].append(float(conf))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bins = np.linspace(0, 1, 21)
    colors = {"pedestrian": "#2196F3", "car": "#FF5722"}
    bar_width = (bins[1] - bins[0]) / (len(CLASS_NAMES) + 0.5)

    for i, (cls_name, confs) in enumerate(class_confs.items()):
        if confs:
            counts, _ = np.histogram(confs, bins=bins)
            positions = bins[:-1] + i * bar_width
            ax.bar(positions, counts, width=bar_width * 0.9,
                   color=colors.get(cls_name, "#9E9E9E"),
                   label=f"{cls_name} (n={len(confs)})", alpha=0.85,
                   edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Confidence Score", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Confidence Score Distribution by Class", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    total = sum(len(v) for v in class_confs.values())
    print(f"  ✅  Confidence histogram saved to {save_path} ({total} total detections)")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    os.makedirs(METRICS_DIR, exist_ok=True)

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Trained weights not found: {MODEL_PATH}")

    print("=" * 60)
    print("  YOLO Visualization Suite")
    print("=" * 60)

    model = YOLO(MODEL_PATH)

    # Load test images and run predictions
    image_paths = load_test_images()
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {TEST_DIR}")
    print(f"\n📷  Found {len(image_paths)} test images")

    all_results = run_predictions(model, image_paths)

    # Generate all three visualizations
    generate_failure_cases(all_results, os.path.join(METRICS_DIR, "failure_cases.png"))
    generate_comparison_grid(all_results, os.path.join(METRICS_DIR, "comparison_grid.png"))
    generate_confidence_histogram(all_results, os.path.join(METRICS_DIR, "confidence_histogram.png"))

    print(f"\n🎉  All visualizations saved to {METRICS_DIR}/")


if __name__ == "__main__":
    main()
