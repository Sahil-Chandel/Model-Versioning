import mlflow
import os
import yaml
from ultralytics import YOLO
import shutil
from datetime import datetime
import subprocess

# ===============================
# LOAD PARAMS
# ===============================
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

MODEL_PATH = params["train"]["model_path"]
DATA_YAML = params["train"]["data_yaml"]
EPOCHS = params["train"]["epochs"]
IMGSZ = params["train"]["imgsz"]

SAVE_DIR = "models"
RUN_NAME = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ===============================
# DVC + GIT TAG FUNCTION
# ===============================
def dvc_commit_and_tag(run_id, tag_prefix="train"):
    try:
        subprocess.run(["dvc", "add", "models"], check=True)
        subprocess.run(["git", "add", "models.dvc"], check=True)
        subprocess.run(["git", "commit", "-m", f"Add {tag_prefix} model version {run_id}"], check=True)
        tag_name = f"{tag_prefix}_{run_id[:8]}"  # short tag
        subprocess.run(["git", "tag", tag_name], check=True)
        print(f"✅ Model committed & tagged as {tag_name}")
    except Exception as e:
        print("⚠️ DVC commit/tag failed:", e)

# ===============================
# TRAIN FUNCTION
# ===============================
def train_and_log():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("YOLO_Experiments")

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params(params["train"])

        model = YOLO(MODEL_PATH)
        results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ)

        version_dir = os.path.join(SAVE_DIR, run.info.run_id)
        os.makedirs(version_dir, exist_ok=True)

        best_model_path = results.save_dir + "/weights/best.pt"
        final_path = os.path.join(version_dir, "best.pt")
        shutil.copy(best_model_path, final_path)

        metrics = results.results_dict
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.log_artifact(final_path)

        print(f"✅ Model saved at {final_path}")
        print(f"📊 Run logged in MLflow: {run.info.run_id}")

        dvc_commit_and_tag(run.info.run_id, tag_prefix="train")


if __name__ == "__main__":
    train_and_log()
