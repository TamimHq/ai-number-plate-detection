from ultralytics import YOLO

MODEL = "yolo11n-cls.pt"

DATASET = "datasets/vehicle_colors_dataset"

model = YOLO(MODEL)

model.train(
    data=DATASET,
    epochs=100,
    imgsz=224,
    batch=256,
    device=0,
    project="models/color_classifier",
    name="vehicle_color_model"
)