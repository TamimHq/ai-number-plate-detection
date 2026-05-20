import cv2
import os

from ultralytics import YOLO

from utils.preprocessing import preprocess_plate
from utils.ocr_utils import run_ocr
from utils.visualization import draw_vehicle

# CONFIGURATION
DETECT_MODEL_PATH = "models/detection/vehicle_plate_detector/weights/best.pt"
COLOR_MODEL_PATH = "models/classification/color_classifier/weights/best.pt"

IMAGE_PATH = "data/images/test.jpg"
OUTPUT_IMAGE = "EnglishPlateDetect.jpg"

DETECT_CONF = 0.10
OCR_CONF = 0.45

VEHICLE_CLASSES = {
    "bike",
    "bus",
    "car",
    "truck",
    "cng"
}

# LOAD MODELS
print("Loading models...")

detect_model = YOLO(DETECT_MODEL_PATH)
color_model = YOLO(COLOR_MODEL_PATH)

print("Models loaded.")

# LOAD IMAGE
if not os.path.exists(IMAGE_PATH):

    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

frame = cv2.imread(
    IMAGE_PATH
)

if frame is None:

    raise ValueError(
        "Failed to load image."
    )

debug = frame.copy()

h, w = frame.shape[:2]

print(
    f"\nProcessing Image: {w}x{h}"
)

# DETECTION
results = detect_model(

    frame,
    conf=DETECT_CONF,
    imgsz=1920,
    verbose=False

)[0]

vehicles = []
plates = []

print("\n----- DETECTIONS -----")

if results.boxes is not None:

    for box in results.boxes:

        cls_id = int(
            box.cls[0]
        )

        conf = float(
            box.conf[0]
        )

        cls_name = detect_model.names[
            cls_id
        ]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        print(
            f"{cls_name} {conf:.2f}"
        )

        if cls_name in VEHICLE_CLASSES:

            vehicles.append({

                "type": cls_name,
                "coords": (x1, y1, x2, y2)

            })

        elif cls_name == "License_Plate":

            plates.append({

                "coords": (x1, y1, x2, y2)

            })

print(
    f"\nVehicles Detected: {len(vehicles)}"
)

print(
    f"Plates Detected: {len(plates)}"
)

# PROCESS PLATES
for i, p in enumerate(plates):

    px1, py1, px2, py2 = p["coords"]

    plate_crop = frame[
        py1:py2,
        px1:px2
    ]

    if plate_crop.size == 0:
        continue

    processed = preprocess_plate(
        plate_crop
    )

    plate_text, ocr_conf = run_ocr(
        processed
    )

    box_color = (0, 255, 0)

    if plate_text == "No Plate":

        box_color = (0, 0, 255)

    cv2.rectangle(

        debug,
        (px1, py1),
        (px2, py2),
        box_color,
        3

    )

    cv2.putText(

        debug,
        f"{plate_text} ({ocr_conf:.2f})",
        (px1, py1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        box_color,
        2

    )

    print("\n===================")
    print(f"Plate #{i+1}")
    print(f"Text: {plate_text}")
    print(f"Confidence: {round(ocr_conf,3)}")

    if ocr_conf >= OCR_CONF:

        print("Status: VALID")

    else:

        print("Status: LOW CONF")

# VEHICLE COLOR CLASSIFICATION

for vehicle in vehicles:

    x1, y1, x2, y2 = vehicle["coords"]

    vehicle_crop = frame[
        y1:y2,
        x1:x2
    ]

    if vehicle_crop.size == 0:
        continue

    try:

        color_result = color_model(

            vehicle_crop,
            verbose=False

        )[0]

        vehicle_color = color_model.names[
            color_result.probs.top1
        ]

    except:

        vehicle_color = "Unknown"

    label = (
        f"{vehicle_color} "
        f"{vehicle['type']}"
    )

    draw_vehicle(

        debug,
        vehicle["coords"],
        label

    )

# SAVE OUTPUT
os.makedirs(
    "outputs",
    exist_ok=True
)

cv2.imwrite(

    OUTPUT_IMAGE,
    debug

)

print(
    f"\nSaved Output: {OUTPUT_IMAGE}"
)

# DISPLAY OUTPUT
try:

    from google.colab.patches import cv2_imshow
    from google.colab import files

    cv2_imshow(

        cv2.resize(

            debug,
            None,
            fx=0.7,
            fy=0.7

        )

    )

    files.download(
        OUTPUT_IMAGE
    )

except:

    cv2.imshow(
        "Output",
        debug
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()