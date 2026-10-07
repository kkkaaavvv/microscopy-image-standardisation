import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms


image_path = "week1/images/microscopy_sample.png"

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
else:
    print("Source image loaded successfully!")

    # Convert BGR to RGB
    source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a controlled reference image
    reference = source.copy().astype(np.float32)

    # Apply a small colour transformation
    reference[:, :, 0] = np.clip(reference[:, :, 0] * 0.90 + 10, 0, 255)
    reference[:, :, 1] = np.clip(reference[:, :, 1] * 1.05, 0, 255)
    reference[:, :, 2] = np.clip(reference[:, :, 2] * 1.10, 0, 255)

    reference = reference.astype(np.uint8)

    # Match source histogram to reference histogram
    matched = match_histograms(
        source,
        reference,
        channel_axis=-1
    )

    matched = np.clip(matched, 0, 255).astype(np.uint8)

    # Print statistics
    print("\nSource image:")
    print("Mean RGB:", source.mean(axis=(0, 1)))

    print("\nReference image:")
    print("Mean RGB:", reference.mean(axis=(0, 1)))

    print("\nMatched image:")
    print("Mean RGB:", matched.mean(axis=(0, 1)))

    # Display images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(source)
    plt.title("Source Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(reference)
    plt.title("Reference Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(matched)
    plt.title("Histogram-Matched Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()