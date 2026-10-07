import cv2
import matplotlib.pyplot as plt


# Load image
image_path = "week1/images/microscopy_sample.png"
image = cv2.imread(image_path)


if image is None:
    print("Error: Could not load image.")

else:
    print("Image loaded successfully!")

    # Convert BGR image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Original shape:", image.shape)
    print("Grayscale shape:", gray.shape)

    # Display grayscale image
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    # Display histogram
    plt.subplot(1, 2, 2)
    plt.hist(gray.ravel(), bins=256, range=(0, 256))
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()