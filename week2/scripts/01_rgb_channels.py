import cv2
import matplotlib.pyplot as plt

image_path = "week1/images/microscopy_sample.png"

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
else:
    print("Image loaded successfully!")
    print("Image shape:", image.shape)

    # OpenCV loads images as BGR
    blue, green, red = cv2.split(image)

    # Convert original image from BGR to RGB for displaying with Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(red, cmap="gray")
    plt.title("Red Channel")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(green, cmap="gray")
    plt.title("Green Channel")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(blue, cmap="gray")
    plt.title("Blue Channel")
    plt.axis("off")

    plt.tight_layout()
    plt.show()