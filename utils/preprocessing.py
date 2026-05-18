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

    return gray