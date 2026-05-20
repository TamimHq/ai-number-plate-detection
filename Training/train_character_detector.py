from ultralytics import YOLO

MODEL = "yolo11n.pt"

DATA_YAML = "datasets/Number Plate Detection pr-400.v2i.yolov8/data.yaml"

model = YOLO(MODEL)

model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=64,
    device=0,
    project="models/character",
    name="bangla_character_detector"
)