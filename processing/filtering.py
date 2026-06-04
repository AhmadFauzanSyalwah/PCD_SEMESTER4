import cv2
import numpy as np

def mean_filter(image, kernel_size=15):
    """
    Mean filter dengan kernel besar (default 15) agar blur terlihat jelas.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.blur(image, (kernel_size, kernel_size))

def gaussian_filter(image, kernel_size=15, sigma=5.0):
    """
    Gaussian filter dengan kernel besar dan sigma tinggi.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def sharpen(image, strength=3.0):
    """
    Penajaman dengan kernel Laplacian yang diperkuat.
    strength=3.0 memberikan efek tajam yang sangat terlihat.
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]]) * strength
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def unsharp_mask(image, strength=2.0, radius=5):
    """
    Unsharp masking: sharpening dengan mengurangi versi blur.
    Lebih alami dan bisa dikontrol.
    """
    blurred = cv2.GaussianBlur(image, (radius, radius), 0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def sobel_edge_detection(image):
    """
    Deteksi tepi dengan Sobel (tetap seperti semula, sudah jelas terlihat).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    return magnitude