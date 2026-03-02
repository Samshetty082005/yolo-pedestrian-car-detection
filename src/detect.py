import os
import glob
import cv2
import numpy as np

from ultralytics import YOLO


def overlay_heatmap(image, detections):
    """Return image with a confidence-based heatmap overlay.

    detections is a list of [x1,y1,x2,y2,conf,cls]
    """
    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for x1, y1, x2, y2, conf, _cls in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # fill region with maximum confidence value
        heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], conf)
    # normalize to 0-255
    if heatmap.max() > 0:
        norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        norm = heatmap.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.7, heatmap_color, 0.3, 0)
    return overlay


def main():
    model_path = "outputs/weights/best.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained weights not found: {model_path}")
    model = YOLO(model_path)

    os.makedirs("outputs/predictions", exist_ok=True)

    test_folder = "dataset/images/test"
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(test_folder, pat)))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        results = model(img)
        # ultralytics returns a Results object, handle first
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)

        annotated = img.copy()
        detections = []
        for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            detections.append([x1, y1, x2, y2, conf, cls])

        if detections:
            annotated = overlay_heatmap(annotated, detections)

        out_path = os.path.join("outputs/predictions", os.path.basename(img_path))
        cv2.imwrite(out_path, annotated)
        print(f"Saved annotated image to {out_path}")


if __name__ == "__main__":
    main()
