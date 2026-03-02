import os
import glob
import shutil
import yaml

from ultralytics import YOLO


def main():
    # load model config
    with open("config/model_config.yaml") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.get("model", "yolov8n")
    print(f"Loading model {model_name}")
    model = YOLO(model_name)

    save_dir = cfg.get("save_dir", "outputs/weights")
    metrics_dir = cfg.get("metrics_dir", "outputs/metrics")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print("Starting training...")
    # run training; ultralytics will output to runs/train/exp*
    model.train(
        data="config/dataset.yaml",
        epochs=cfg.get("epochs", 50),
        imgsz=cfg.get("img_size", 640),
        batch=cfg.get("batch", 16),
        lr0=cfg.get("lr", 0.001),
        project="runs/train",
        name="exp",
        exist_ok=True,
        save=True,
    )

    # locate best weights and metrics from run directory
    # Ultralytics may nest output under runs/detect/ or runs/train/
    run_dirs = sorted(glob.glob("runs/train/exp*"))
    if not run_dirs:
        run_dirs = sorted(glob.glob("runs/**/train/exp*", recursive=True))
    if run_dirs:
        run_dir = run_dirs[-1]
        best_src = os.path.join(run_dir, "weights", "best.pt")
        if os.path.isfile(best_src):
            shutil.copy(best_src, os.path.join(save_dir, "best.pt"))
            print(f"Copied best weights to {os.path.join(save_dir, 'best.pt')}")
        # copy results.csv if available
        results_csv = os.path.join(run_dir, "results.csv")
        if os.path.isfile(results_csv):
            shutil.copy(results_csv, os.path.join(metrics_dir, "results.csv"))
            print(f"Copied metrics to {os.path.join(metrics_dir, 'results.csv')}")
    else:
        print("No run directories found, training may have failed.")


if __name__ == "__main__":
    main()
