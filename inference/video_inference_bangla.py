import os
import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime

from utils.bangla_utils import (
    BANGLA_MAP,
    assign_plate,
    read_plate,
    plate_quality
)

from utils.preprocessing import preprocess_plate
from utils.visualization import draw_vehicle

# CONFIGURATION

DETECT_MODEL_PATH = "models/detection/vehicle_plate_detector/weights/best.pt"

COLOR_MODEL_PATH = "models/color_classifier/vehicle_color_model/weights/best.pt"

CHAR_MODEL_PATH = "models/character/bangla_character_detector/weights/best.pt"


VIDEO_PATH = "data/videos/test.mp4"

OUTPUT_VIDEO = "outputs/videos/BanglaPlateDetect.mp4"
EXCEL_FILE = "outputs/vehicle_log.xlsx"

DETECT_CONF = 0.25
CHAR_CONF = 0.25

PROCESS_EVERY_N = 1

DETECT_IMGSZ = 1280
CHAR_IMGSZ = 640

MIN_PLATE_LEN = 4
MIN_SCORE = 0.50

VOTE_HISTORY = 10
MIN_VOTES = 5

VEHICLE_CLASSES = {
    "bike",
    "bus",
    "car",
    "truck",
    "cng"
}

# LOAD MODELS
print("Loading models...")

detect_model = YOLO(
    DETECT_MODEL_PATH
)

color_model = YOLO(
    COLOR_MODEL_PATH
)

char_model = YOLO(
    CHAR_MODEL_PATH
)

print("Models loaded.")

# OUTPUT DIRECTORY
os.makedirs(
    "outputs",
    exist_ok=True
)

# MEMORY STORAGE
history = defaultdict(list)

database = {}

seen_plates = set()


# VIDEO SETUP
cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

fps = cap.get(
    cv2.CAP_PROP_FPS
)

w = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

h = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

writer = cv2.VideoWriter(

    OUTPUT_VIDEO,

    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),

    fps,

    (w, h)

)

frame_count = 0

print("Processing video...")

# MAIN LOOP
while cap.isOpened():

    ok, frame = cap.read()

    if not ok:
        break

    frame_count += 1

    # Skip frames if needed

    if frame_count % PROCESS_EVERY_N != 0:

        writer.write(frame)

        continue

    # VEHICLE + PLATE DETECTION
    results = detect_model.track(

        frame,

        persist=True,

        tracker="botsort.yaml",

        conf=DETECT_CONF,

        imgsz=DETECT_IMGSZ,

        verbose=False

    )[0]

    vehicles = []
    plates = []

    if results.boxes is not None:

        for box in results.boxes:

            cls = detect_model.names[
                int(box.cls[0])
            ]

            coords = tuple(
                map(
                    int,
                    box.xyxy[0]
                )
            )

            track_id = -1

            if box.id is not None:

                track_id = int(
                    box.id[0]
                )

            if cls in VEHICLE_CLASSES:

                vehicles.append({

                    "coords": coords,
                    "track": track_id,
                    "type": cls

                })

            elif cls == "License_Plate":

                plates.append({

                    "coords": coords

                })

    # PROCESS VEHICLES
    for vehicle in vehicles:

        vx1, vy1, vx2, vy2 = vehicle["coords"]

        track_id = vehicle["track"]

        if track_id == -1:
            continue

        matched_plate = assign_plate(

            vehicle["coords"],
            plates

        )

        plate_text = "No Plate"

        plate_conf = 0

        # CHARACTER RECOGNITION
        if matched_plate:

            px1, py1, px2, py2 = matched_plate["coords"]

            plate_crop = frame[
                py1:py2,
                px1:px2
            ]

            if plate_crop.size > 0:

                target_height = 192

                scale = (
                    target_height /
                    plate_crop.shape[0]
                )

                plate_crop = cv2.resize(

                    plate_crop,

                    None,

                    fx=scale,
                    fy=scale,

                    interpolation=cv2.INTER_CUBIC

                )

                processed = preprocess_plate(
                    plate_crop
                )

                char_results = char_model(

                    processed,

                    imgsz=CHAR_IMGSZ,

                    conf=CHAR_CONF,

                    iou=0.30,

                    verbose=False

                )[0]

                plate_text, plate_conf = read_plate(

                    char_results.boxes,

                    char_model.names,

                    processed.shape[0]

                )

                # Draw plate box

                cv2.rectangle(

                    frame,

                    (px1, py1),

                    (px2, py2),

                    (0, 255, 255),

                    2

                )

                cv2.putText(

                    frame,

                    f"{plate_text} ({plate_conf:.2f})",

                    (px1, py1 - 10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (0, 255, 255),

                    2

                )

    
        # COLOR CLASSIFICATION
        vehicle_color = "Unknown"

        vehicle_crop = frame[
            vy1:vy2,
            vx1:vx2
        ]

        if vehicle_crop.size > 0:

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

        # TEMPORAL VOTING
        if plate_text != "No Plate":

            history[track_id].append(

                (plate_text, plate_conf)

            )

        history[track_id] = history[
            track_id
        ][-VOTE_HISTORY:]

        # FINAL PLATE DECISION
        if len(history[track_id]) >= MIN_VOTES:

            candidates = {}

            for text, score in history[track_id]:

                quality = plate_quality(

                    text,
                    score

                )

                if text not in candidates:

                    candidates[text] = []

                candidates[text].append(
                    quality
                )

            final_plate = max(

                candidates,

                key=lambda x:
                np.mean(
                    candidates[x]
                )

            )

            final_score = np.mean(

                candidates[
                    final_plate
                ]

            )

            if (

                final_plate not in seen_plates
                and len(final_plate) >= MIN_PLATE_LEN
                and final_score >= MIN_SCORE

            ):

                database[track_id] = {

                    "Time":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                    "Track_ID":
                    track_id,

                    "Color":
                    vehicle_color,

                    "Vehicle_Type":
                    vehicle["type"],

                    "Plate_Number":
                    final_plate,

                    "Confidence":
                    round(
                        final_score,
                        3
                    )

                }

                seen_plates.add(
                    final_plate
                )

                print(

                    f"Logged: "
                    f"{vehicle_color} "
                    f"{vehicle['type']} | "
                    f"{final_plate} "
                    f"({final_score:.2f})"

                )

        # DRAW VEHICLE
        label = (

            f"ID:{track_id} "
            f"{vehicle_color} "
            f"{vehicle['type']}"

        )

        draw_vehicle(

            frame,

            vehicle["coords"],

            label

        )

    # WRITE FRAME
    writer.write(
        frame
    )

# RELEASE
cap.release()

writer.release()


# SAVE EXCEL
if len(database) > 0:

    df = pd.DataFrame(

        list(
            database.values()
        )

    )

    df.to_excel(

        EXCEL_FILE,

        index=False

    )

    print(
        f"Excel saved: {EXCEL_FILE}"
    )

print(
    f"Video saved: {OUTPUT_VIDEO}"
)

print("Finished.")