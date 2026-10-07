import cv2
import matplotlib.pyplot as plt

image_path = "week1/images/microscopy_sample.png"

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
else:
    print("Image loaded successfully!")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Separate RGB channels
    red, green, blue = cv2.split(image_rgb)

    # Print basic statistics
    print("Red channel mean:", red.mean())
    print("Green channel mean:", green.mean())
    print("Blue channel mean:", blue.mean())

    print("Red channel range:", red.min(), "-", red.max())
    print("Green channel range:", green.min(), "-", green.max())
    print("Blue channel range:", blue.min(), "-", blue.max())

    # Create figure
    plt.figure(figsize=(12, 8))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Microscopy Image")
    plt.axis("off")

    # Red histogram
    plt.subplot(2, 2, 2)
    plt.hist(red.ravel(), bins=256, range=(0, 256))
    plt.title("Red Channel Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

    # Green histogram
    plt.subplot(2, 2, 3)
    plt.hist(green.ravel(), bins=256, range=(0, 256))
    plt.title("Green Channel Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

    # Blue histogram
    plt.subplot(2, 2, 4)
    plt.hist(blue.ravel(), bins=256, range=(0, 256))
    plt.title("Blue Channel Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()