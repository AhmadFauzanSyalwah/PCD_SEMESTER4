import cv2
import numpy as np

def convert_to_grayscale(image):
    """Mengubah citra BGR menjadi grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def red_channel(image):
    """Mengambil saluran merah (red channel)."""
    result = np.zeros_like(image)
    result[:, :, 2] = image[:, :, 2]
    return result

def green_channel(image):
    """Mengambil saluran hijau (green channel)."""
    result = np.zeros_like(image)
    result[:, :, 1] = image[:, :, 1]
    return result

def blue_channel(image):
    """Mengambil saluran biru (blue channel)."""
    result = np.zeros_like(image)
    result[:, :, 0] = image[:, :, 0]
    return result