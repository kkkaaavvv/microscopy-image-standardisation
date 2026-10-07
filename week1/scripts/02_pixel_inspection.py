import cv2
import numpy as np


# Load image
image_path = "week1/images/microscopy_sample.png"
image = cv2.imread(image_path)


# Check image
if image is None:
    print("Error: Could not load image.")
else:
    print("Image loaded successfully!")

    # Basic information
    print("Shape:", image.shape)
    print("Data type:", image.dtype)
    print("Minimum pixel value:", image.min())
    print("Maximum pixel value:", image.max())

    # Inspect one pixel
    pixel = image[400, 400]

    print("Pixel at (400, 400):", pixel)

    # Inspect a small region
    region = image[400:405, 400:405]

    print("\n5 × 5 pixel region:")
    print(region)