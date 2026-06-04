import cv2
import numpy as np


def threshold_manual(image, thresh_value=150):
    """
    Binerisasi manual dengan nilai ambang thresh_value (0-255).
    Untuk citra berwarna, dikonversi ke grayscale terlebih dahulu.
    Hasil: citra hitam putih (0 dan 255).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Terapkan threshold
    _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    return binary


def threshold_otsu(image):
    """
    Binerisasi dengan metode Otsu (threshold otomatis).
    Untuk citra berwarna, dikonversi ke grayscale terlebih dahulu.
    Hasil: citra hitam putih (0 dan 255).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary