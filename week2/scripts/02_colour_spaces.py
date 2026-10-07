import cv2
import matplotlib.pyplot as plt

image_path = "week1/images/microscopy_sample.png"

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
else:
    print("Image loaded successfully!")
    print("Original shape:", image.shape)

    # Convert from OpenCV's BGR format to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Convert to LAB
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split channels
    r, g, b = cv2.split(image_rgb)
    h, s, v = cv2.split(image_hsv)
    l, a, lab_b = cv2.split(image_lab)

    print("RGB shape:", image_rgb.shape)
    print("HSV shape:", image_hsv.shape)
    print("LAB shape:", image_lab.shape)

    plt.figure(figsize=(12, 10))

    # RGB
    plt.subplot(3, 3, 1)
    plt.imshow(image_rgb)
    plt.title("RGB - Original")
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.imshow(r, cmap="gray")
    plt.title("R - Red")
    plt.axis("off")

    plt.subplot(3, 3, 3)
    plt.imshow(g, cmap="gray")
    plt.title("G - Green")
    plt.axis("off")

    # HSV
    plt.subplot(3, 3, 4)
    plt.imshow(h, cmap="gray")
    plt.title("HSV - Hue")
    plt.axis("off")

    plt.subplot(3, 3, 5)
    plt.imshow(s, cmap="gray")
    plt.title("HSV - Saturation")
    plt.axis("off")

    plt.subplot(3, 3, 6)
    plt.imshow(v, cmap="gray")
    plt.title("HSV - Value")
    plt.axis("off")

    # LAB
    plt.subplot(3, 3, 7)
    plt.imshow(l, cmap="gray")
    plt.title("LAB - Lightness")
    plt.axis("off")

    plt.subplot(3, 3, 8)
    plt.imshow(a, cmap="gray")
    plt.title("LAB - a*")
    plt.axis("off")

    plt.subplot(3, 3, 9)
    plt.imshow(lab_b, cmap="gray")
    plt.title("LAB - b*")
    plt.axis("off")

    plt.tight_layout()
    plt.show()