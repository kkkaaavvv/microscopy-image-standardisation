import cv2
import numpy as np
from pathlib import Path


def save_image(rgb_image, output_path):
    """
    Save an RGB image to the specified location.
    OpenCV requires BGR format when saving.
    """

    if rgb_image is None:
        raise ValueError("Cannot save an empty image.")

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected an RGB image with three channels.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB back to BGR for OpenCV
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(output_path), bgr_image)

    if not success:
        raise IOError(f"Unable to save image: {output_path}")

    print("Image saved successfully:", output_path)