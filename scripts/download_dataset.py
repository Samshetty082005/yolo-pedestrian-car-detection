#!/usr/bin/env python3
"""
Download ~200 images (pedestrians + cars) from COCO via FiftyOne
and convert annotations to YOLO format.

Class mapping:
  0 = pedestrian (COCO: "person")
  1 = car        (COCO: "car")
"""

import os
import shutil
import random
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image

# ── Configuration ───────────────────────────────────────────────
NUM_PEDESTRIAN = 100
NUM_CAR = 100
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
COCO_TO_YOLO = {"person": 0, "car": 1}
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")

random.seed(42)


# ── Helpers ─────────────────────────────────────────────────────
def coco_to_yolo_bbox(bbox):
    """Convert COCO [x, y, w, h] (relative) → YOLO [cx, cy, w, h] (relative)."""
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    return cx, cy, w, h


def ensure_dirs():
    """Create dataset directory tree."""
    for split in SPLIT_RATIOS:
        os.makedirs(os.path.join(BASE_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, "labels", split), exist_ok=True)


def download_samples(label: str, max_samples: int, dataset_name: str):
    """Download COCO validation samples containing a given label."""
    print(f"⏳  Downloading up to {max_samples} COCO samples with '{label}'…")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=[label],
        max_samples=max_samples,
        dataset_name=dataset_name,
        shuffle=True,
    )
    return dataset


def export_sample(sample, split, idx, seen_files):
    """Copy image and write YOLO label for one FiftyOne sample."""
    src_path = sample.filepath
    if src_path in seen_files:
        return False
    seen_files.add(src_path)

    ext = os.path.splitext(src_path)[1]
    dst_img = os.path.join(BASE_DIR, "images", split, f"{split}_{idx:04d}{ext}")
    dst_lbl = os.path.join(BASE_DIR, "labels", split, f"{split}_{idx:04d}.txt")

    shutil.copy2(src_path, dst_img)

    # Validate image
    try:
        with Image.open(dst_img) as img:
            img.verify()
    except Exception:
        os.remove(dst_img)
        return False

    # Write YOLO labels
    lines = []
    if sample.ground_truth and sample.ground_truth.detections:
        for det in sample.ground_truth.detections:
            if det.label in COCO_TO_YOLO:
                cls_id = COCO_TO_YOLO[det.label]
                cx, cy, w, h = coco_to_yolo_bbox(det.bounding_box)
                # Clamp values to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w  = max(0.0, min(1.0, w))
                h  = max(0.0, min(1.0, h))
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if not lines:
        os.remove(dst_img)
        return False

    with open(dst_lbl, "w") as f:
        f.write("\n".join(lines) + "\n")

    return True


# ── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  COCO → YOLO Dataset Builder")
    print("=" * 60)

    ensure_dirs()

    # Download person and car samples separately
    ped_ds = download_samples("person", NUM_PEDESTRIAN, "coco_pedestrians_tmp")
    car_ds = download_samples("car", NUM_CAR, "coco_cars_tmp")

    # Merge into a single sample list (deduplicate by filepath)
    all_samples = []
    seen_fp = set()
    for ds in [ped_ds, car_ds]:
        for sample in ds:
            if sample.filepath not in seen_fp:
                seen_fp.add(sample.filepath)
                all_samples.append(sample)

    random.shuffle(all_samples)
    total = len(all_samples)
    print(f"\n📦  Total unique samples: {total}")

    # Split
    n_train = int(total * SPLIT_RATIOS["train"])
    n_val = int(total * SPLIT_RATIOS["val"])
    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train : n_train + n_val],
        "test": all_samples[n_train + n_val :],
    }

    # Export
    seen_files = set()
    stats = {}
    for split_name, samples in splits.items():
        idx = 0
        for sample in samples:
            if export_sample(sample, split_name, idx, seen_files):
                idx += 1
        stats[split_name] = idx
        print(f"  ✅  {split_name}: {idx} images")

    # Cleanup FiftyOne datasets
    for name in ["coco_pedestrians_tmp", "coco_cars_tmp"]:
        if fo.dataset_exists(name):
            fo.delete_dataset(name)

    print("\n" + "=" * 60)
    print("  Dataset ready at:", BASE_DIR)
    print("  Splits:", stats)
    print("=" * 60)


if __name__ == "__main__":
    main()
