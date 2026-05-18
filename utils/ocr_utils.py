import easyocr
import numpy as np

reader = easyocr.Reader(
    ['en'],
    gpu=True
)

def normalize_plate(txt):

    txt = txt.replace(" ","")
    txt = txt.replace("-","")

    return "".join(
        c for c in txt
        if c.isalnum()
    ).upper()


def run_ocr(img):

    result = reader.readtext(img)

    if len(result) == 0:
        return "No Plate",0

    texts = [x[1] for x in result]
    confs = [x[2] for x in result]

    text = normalize_plate(
        "".join(texts)
    )

    conf = np.mean(confs)

    return text,conf