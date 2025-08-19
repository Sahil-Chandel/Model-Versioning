#!/bin/bash
set -e  # stop if any command fails

echo "🚀 Starting YOLO training + prediction pipeline..."

# ===============================
# STEP 0: Parse CLI Arguments
# ===============================
EPOCHS=""
IMGSZ=""
CONF=""
SOURCE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift ;;
        --imgsz) IMGSZ="$2"; shift ;;
        --conf) CONF="$2"; shift ;;
        --source) SOURCE="$2"; shift ;;
        *) echo "❌ Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# ===============================
# STEP 1: Interactive fallback
# ===============================
if [ -z "$EPOCHS" ]; then
  read -p "Enter number of training epochs [default=10]: " EPOCHS
  EPOCHS=${EPOCHS:-10}
fi

if [ -z "$IMGSZ" ]; then
  read -p "Enter training image size [default=640]: " IMGSZ
  IMGSZ=${IMGSZ:-640}
fi

if [ -z "$CONF" ]; then
  read -p "Enter confidence threshold for predictions [default=0.25]: " CONF
  CONF=${CONF:-0.25}
fi

if [ -z "$SOURCE" ]; then
  read -p "Enter prediction source path [default=data/test/images]: " SOURCE
  SOURCE=${SOURCE:-data/test/images}
fi

# ===============================
# STEP 2: Update params.yaml
# ===============================
echo "⚙️ Updating params.yaml..."
cat > params.yaml <<EOL
train:
  model_path: "yolov8n.pt"
  data_yaml: "data.yaml"
  epochs: ${EPOCHS}
  imgsz: ${IMGSZ}

predict:
  model_path: "models/latest/best.pt"
  conf_threshold: ${CONF}
  imgsz: ${IMGSZ}
  source: "${SOURCE}"
EOL

echo "✅ params.yaml updated:"
cat params.yaml

# ===============================
# STEP 3: Train
# ===============================
echo "📦 Running training..."
python train.py

# ===============================
# STEP 4: Predict
# ===============================
echo "🔎 Running predictions..."
python predict.py

# ===============================
# STEP 5: DVC Push (if remote configured)
# ===============================
if dvc remote list | grep -q .; then
    echo "☁️ Pushing data to DVC remote..."
    dvc push
else
    echo "⚠️ No DVC remote configured. Skipping push."
fi

# ===============================
# STEP 6: Git Push
# ===============================
echo "⬆️ Pushing Git commits + tags..."
git push
git push --tags

echo "✅ Pipeline finished successfully!"
