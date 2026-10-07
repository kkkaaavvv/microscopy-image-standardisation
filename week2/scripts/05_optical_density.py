import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "week1/images/microscopy_sample.png"

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
else:
    print("Image loaded successfully!")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to floating point
    image_float = image_rgb.astype(np.float32)

    # Reference intensity
    I0 = 255.0

    # Avoid log(0)
    image_float = np.clip(image_float, 1, 255)

    # Calculate Optical Density
    optical_density = -np.log(image_float / I0)

    print("RGB minimum:", image_rgb.min())
    print("RGB maximum:", image_rgb.max())

    print("OD minimum:", optical_density.min())
    print("OD maximum:", optical_density.max())

    # Display original and OD representation
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original RGB Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(optical_density)
    plt.title("Optical Density Representation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()