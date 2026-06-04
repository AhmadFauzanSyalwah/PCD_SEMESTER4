import cv2
import numpy as np


def adjust_brightness(image, value=50):
    """
    Menambah atau mengurangi kecerahan gambar.
    value > 0: menambah kecerahan
    value < 0: mengurangi kecerahan
    Mendukung citra grayscale maupun BGR.
    """
    # Konversi ke int16 agar aman dari overflow, lalu clip ke 0-255
    result = image.astype(np.int16) + value
    result = np.clip(result, 0,    255).astype(np.uint8)
    return result


def adjust_contrast(image, alpha=1.5):
    """
    Mengubah kontras gambar.
    alpha > 1: meningkatkan kontras
    0 < alpha < 1: menurunkan kontras
    alpha = 1: tidak ada perubahan
    """
    # cv2.convertScaleAbs melakukan: dst = alpha * src + beta
    # Di sini beta = 0 agar kecerahan rata-rata tidak berubah.
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


def negative(image):
    """
    Menghasilkan citra negatif (inversi intensitas).
    """
    # cv2.bitwise_not bekerja untuk semua channel
    return cv2.bitwise_not(image)