## If you want to change model then goto 
In params.yaml under train.model_path, just replace the base model checkpoint

## If you want to start from a pretrained YOLOv11 checkpoint →  use "yolov11n.pt", "yolov11s.pt", etc.

## If you already trained and saved your own YOLOv11 weights → point to them, e.g.:
train:
  model_path: "models/custom_yolov11/best.pt"


## 2. Change dataset path for training

In params.yaml → update train.data_yaml to point to your dataset YAML:

train:
  model_path: "yolov11n.pt"
  data_yaml: "datasets/my_project/data.yaml"   # <-- your dataset config
  epochs: 50
  imgsz: 640

And inside your dataset’s data.yaml (this is the standard YOLO dataset file), define the image folders:

train: datasets/my_project/images/train
val: datasets/my_project/images/val
test: datasets/my_project/images/test   # optional

nc: 3
names: ["car", "truck", "bus"]


## This can be:

a single image → "data/test/images/sample1.jpg"
a folder of images → "data/test/images"
a video → "videos/highway.mp4"

## 4. How the scripts pick up these changes

train.py reads params.yaml → train.model_path and train.data_yaml
predict.py reads params.yaml → predict.model_path and predict.source
Your run_pipeline.sh rewrites params.yaml each time you run interactively / non-interactively
So the only place you really need to change manually is params.yaml (unless you want run_pipeline.sh to override it).