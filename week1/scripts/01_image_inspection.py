import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Load image
image_path = "images/microscopy_sample.png"

image = cv2.imread(image_path)

# Check whether image loaded successfully
if image is None:
    print("Error: Could not load image.")
else:
    print("Image loaded successfully!")

    # Image dimensions
    height, width, channels = image.shape

    print("Width:", width)
    print("Height:", height)
    print("Channels:", channels)

    # Display image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.title("Original Microscopy Image")
    plt.axis("off")
    plt.show()