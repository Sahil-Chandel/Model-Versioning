import mlflow
from ultralytics import YOLO
import os
import shutil
import json
from datetime import datetime
import subprocess
import yaml

# ===============================
# LOAD PARAMS
# ===============================
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

MODEL_PATH = params["predict"]["model_path"]
CONF_THRESHOLD = params["predict"]["conf_threshold"]
IMGSZ = params["predict"]["imgsz"]
SOURCE = params["predict"]["source"]

RUN_NAME = f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ===============================
# DVC + GIT TAG FUNCTION
# ===============================
def dvc_commit_and_tag(run_id, tag_prefix="predict"):
    try:
        subprocess.run(["dvc", "add", "Predict"], check=True)
        subprocess.run(["git", "add", "Predict.dvc"], check=True)
        subprocess.run(["git", "commit", "-m", f"Add {tag_prefix} run {run_id}"], check=True)
        tag_name = f"{tag_prefix}_{run_id[:8]}"  # short tag
        subprocess.run(["git", "tag", tag_name], check=True)
        print(f"✅ Predictions committed & tagged as {tag_name}")
    except Exception as e:
        print("⚠️ DVC commit/tag failed:", e)

# ===============================
# PREDICTION FUNCTION
# ===============================
def run_prediction():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("YOLO_Predictions")

    model = YOLO(MODEL_PATH)
    model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]

    base_predict_dir = "Predict"
    os.makedirs(base_predict_dir, exist_ok=True)

    output_dir = os.path.join(base_predict_dir, f"PREDICT-{model_name}_CONFIDENCE-{CONF_THRESHOLD}")
    os.makedirs(output_dir, exist_ok=True)

    predicted_dir = os.path.join(output_dir, "Predicted_images")
    non_predicted_dir = os.path.join(output_dir, "NonPredicted_images")
    os.makedirs(predicted_dir, exist_ok=True)
    os.makedirs(non_predicted_dir, exist_ok=True)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params(params["predict"])

        results = model.predict(
            source=SOURCE,
            device=0,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            save=True
        )

        predictions_json = []
        default_run_dir = results[0].save_dir

        if os.path.exists(default_run_dir):
            for r in results:
                img_name = os.path.basename(r.path)
                pred_img_path = os.path.join(default_run_dir, img_name)

                for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
                    candidate = os.path.splitext(pred_img_path)[0] + ext
                    if os.path.exists(candidate):
                        pred_img_path = candidate
                        break

                if len(r.boxes) > 0:
                    dst_file = os.path.join(predicted_dir, os.path.basename(pred_img_path))
                    result_dict = json.loads(r.to_json())
                    predictions_json.append({
                        "image_name": img_name,
                        "detections": result_dict
                    })
                else:
                    dst_file = os.path.join(non_predicted_dir, os.path.basename(pred_img_path))

                if os.path.exists(pred_img_path):
                    shutil.move(pred_img_path, dst_file)

            json_path = os.path.join(output_dir, "predicted_results.json")
            with open(json_path, "w") as f:
                json.dump(predictions_json, f, indent=4)

            mlflow.log_artifact(json_path)
            mlflow.log_artifact(predicted_dir)
            mlflow.log_artifact(non_predicted_dir)

            shutil.rmtree(default_run_dir, ignore_errors=True)

            print(f"✅ Predictions organized inside: {output_dir}")
            print(f"📝 JSON results saved at: {json_path}")
            print("🧹 Cleaned up default YOLO runs/detect folder.")
        else:
            print("❌ No prediction folder found!")

        dvc_commit_and_tag(run.info.run_id, tag_prefix="predict")


if __name__ == '__main__':
    run_prediction()
