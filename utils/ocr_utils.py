import easyocr
import numpy as np
import torch

from utils.preprocessing import preprocess_plate
from utils.plate_utils import valid_plate

USE_GPU = torch.cuda.is_available()

reader = easyocr.Reader(
    ['en'],
    gpu=USE_GPU
)

def normalize_plate(txt):

    txt = txt.replace(" ","")
    txt = txt.replace("-","")

    return "".join(
        c for c in txt
        if c.isalnum()
    ).upper()


def run_ocr(img):

    img = preprocess_plate(img)

    result = reader.readtext(img)

    if len(result) == 0:
        return "No Plate",0

    texts = [x[1] for x in result]
    confs = [x[2] for x in result]

    text = normalize_plate(
        "".join(texts)
    )

    conf = np.mean(confs)

    if not valid_plate(text):
        return "No Plate",0

    return text,conf