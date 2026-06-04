import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_histogram_gray(image):
    """
    Menghitung histogram untuk citra grayscale.
    Jika citra berwarna, akan dikonversi ke grayscale terlebih dahulu.
    Mengembalikan array histogram (256,).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist.flatten()


def calculate_histogram_rgb(image):
    """
    Menghitung histogram untuk setiap kanal RGB.
    Mengembalikan tuple (hist_b, hist_g, hist_r) masing-masing array 256.
    """
    # BGR -> RGB
    b, g, r = cv2.split(image)
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
    return hist_b, hist_g, hist_r


def equalize_histogram(image):
    """
    Ekualisasi histogram untuk citra BGR.
    Menggunakan ruang warna YCrCb, hanya kanal Y yang diekualisasi,
    kemudian dikonversi kembali ke BGR.
    Cocok untuk citra berwarna, jika citra sudah grayscale akan langsung diekualisasi.
    """
    if len(image.shape) == 2:
        # Citra grayscale
        return cv2.equalizeHist(image)
    else:
        # Citra BGR: konversi ke YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        result = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
        return result