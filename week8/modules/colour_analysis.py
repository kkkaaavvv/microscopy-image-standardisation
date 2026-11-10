import cv2
import numpy as np


def convert_colour_spaces(rgb_image):
    """
    Convert an RGB microscopy image into HSV and LAB colour spaces.
    """

    if rgb_image is None:
        raise ValueError("Input image is empty.")

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected an RGB image with three channels.")

    if rgb_image.dtype != np.uint8:
        raise TypeError("Expected an 8-bit RGB image.")

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)

    print("Colour space conversion completed.")
    print("RGB shape:", rgb_image.shape)
    print("HSV shape:", hsv_image.shape)
    print("LAB shape:", lab_image.shape)

    return hsv_image, lab_image
    