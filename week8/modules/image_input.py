import cv2
import numpy as np
from pathlib import Path


def load_image(image_path):
    """
    Load a microscopy image and convert it from BGR to RGB.

    Returns:
        RGB image as a NumPy array.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Validate image
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have three colour channels.")

    if image.dtype != np.uint8:
        raise TypeError("Expected an 8-bit image.")

    print("Image loaded successfully.")
    print("Image dimensions:", image.shape)
    print("Data type:", image.dtype)

    return image