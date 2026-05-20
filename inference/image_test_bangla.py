from ultralytics import YOLO

import cv2
import os

from utils.bangla_utils import *
from utils.visualization import draw_vehicle
from utils.tracking_utils import assign_plate

# CONFIG

DETECT_MODEL_PATH = "models/detection/vehicle_plate_detector/weights/best.pt"

COLOR_MODEL_PATH = "models/color_classifier/vehicle_color_model/weights/best.pt"

CHAR_MODEL_PATH = "models/character/bangla_character_detector/weights/best.pt"

IMAGE_PATH = "test_images/test.jpg"

OUTPUT_IMAGE = "outputs/EnglishPlateDetect.jpg"

DETECT_CONF = 0.15
CHAR_CONF = 0.50
CHAR_IOU = 0.30

MIN_PLATE_LEN = 4
MIN_SCORE = 0.50

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

char_model = YOLO(CHAR_MODEL_PATH)

print("Models loaded.")

# LOAD IMAGE
if not os.path.exists(IMAGE_PATH):

    raise FileNotFoundError(IMAGE_PATH)

frame = cv2.imread(IMAGE_PATH)

debug = frame.copy()

print(
    f"Processing {frame.shape[1]}x{frame.shape[0]}"
)

# DETECTION
result = detect_model(

    frame,

    conf=DETECT_CONF,

    imgsz=1280,

    verbose=False

)[0]

vehicles = []
plates = []

for box in result.boxes:

    cls = detect_model.names[
        int(box.cls[0])
    ]

    conf = float(
        box.conf[0]
    )

    coords = tuple(
        map(
            int,
            box.xyxy[0]
        )
    )

    if cls in VEHICLE_CLASSES:

        vehicles.append({

            "type":cls,
            "coords":coords,
            "conf":conf

        })

    elif cls == "License_Plate":

        plates.append({

            "coords":coords,
            "conf":conf

        })

print(f"Vehicles = {len(vehicles)}")
print(f"Plates = {len(plates)}")

# PROCESS PLATES
for p in plates:

    px1,py1,px2,py2 = p["coords"]

    plate_crop = frame[
        py1:py2,
        px1:px2
    ]

    if plate_crop.size == 0:
        continue

    h,w = plate_crop.shape[:2]

    if h < 10 or w < 30:
        continue

    matched_vehicle = assign_plate(
        p["coords"],
        vehicles
    )

    vehicle_color = "Unknown"

    if matched_vehicle:

        vx1,vy1,vx2,vy2 = matched_vehicle["coords"]

        vh = vy2 - vy1
        vw = vx2 - vx1

        color_crop = frame[
            vy1 + int(vh*0.4):vy1 + int(vh*0.8),
            vx1 + int(vw*0.1):vx1 + int(vw*0.9)
        ]

        try:

            if color_crop.size > 0:

                color_result = color_model(
                    color_crop,
                    verbose=False
                )[0]

                vehicle_color = color_model.names[
                    color_result.probs.top1
                ]

        except:
            pass

        draw_vehicle(

            debug,

            matched_vehicle["coords"],

            f"{vehicle_color} {matched_vehicle['type']}",

            color=(0,165,255),
            thickness=3

        )

    # CHARACTER DETECTION
    scale = 3

    plate_large = cv2.resize(

        plate_crop,

        None,

        fx=scale,
        fy=scale,

        interpolation=cv2.INTER_CUBIC

    )

    char_result = char_model(

        plate_large,

        conf=CHAR_CONF,

        iou=CHAR_IOU,

        verbose=False

    )[0]

    plate_text,score,_,_ = reconstruct_plate(

        char_result.boxes,

        char_model.names,

        plate_large.shape[0]

    )

    # DRAW PLATE
    color = (0,255,0)

    if score < MIN_SCORE:

        color = (0,0,255)

    cv2.rectangle(

        debug,

        (px1,py1),
        (px2,py2),

        color,

        3

    )

    cv2.putText(

        debug,

        f"{plate_text} ({score:.2f})",

        (px1,py1-10),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        color,

        2

    )

    print("\n===================")

    print("Plate:",plate_text)

    print(
        "Confidence:",
        round(score,3)
    )

    if (

        len(plate_text) >= MIN_PLATE_LEN
        and score >= MIN_SCORE

    ):

        print("VALID")

    else:

        print("LOW CONF")

# SAVE
os.makedirs(
    "outputs",
    exist_ok=True
)

cv2.imwrite(
    OUTPUT_IMAGE,
    debug
)

print(f"\nSaved: {OUTPUT_IMAGE}")

# DISPLAY

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

    files.download(OUTPUT_IMAGE)

except:

    cv2.imshow(
        "Debug",
        debug
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()