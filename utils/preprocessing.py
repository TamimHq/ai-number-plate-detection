import cv2
import numpy as np

def sharpen(img):

    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    return cv2.filter2D(img,-1,kernel)


def preprocess_plate(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.fastNlMeansDenoising(
        gray,
        h=10
    )

    clahe = cv2.createCLAHE(
        clipLimit=3,
        tileGridSize=(4,4)
    )

    gray = clahe.apply(gray)

    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    sharp = cv2.filter2D(
        gray,
        -1,
        kernel
    )

    _,binary = cv2.threshold(
        sharp,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return cv2.cvtColor(
        binary,
        cv2.COLOR_GRAY2BGR
    )