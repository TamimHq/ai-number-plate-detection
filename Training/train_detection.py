from ultralytics import YOLO

MODEL = "yolo11n.pt"

DATA_YAML = "datasets/My Custom Data.v4i.yolov8/data.yaml"

model = YOLO(MODEL)

model.train(
    data=DATA_YAML,
    epochs=300,
    imgsz=640,
    batch=64,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,
    device=0,
    project="models/detection",
    name="vehicle_plate_detector"
)