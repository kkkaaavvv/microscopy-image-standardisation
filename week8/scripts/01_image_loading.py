
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# WEEK 8 - TASK 1: IMAGE LOADING AND DISPLAY
# ==========================================

def load_microscopy_image(image_path):
    """
    Load a microscopy image from disk using OpenCV.
    Convert the image from BGR to RGB for display.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image using OpenCV
    bgr_image = cv2.imread(str(image_path))

    if bgr_image is None:
        raise ValueError("Unable to read the image.")

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    print("Image loaded successfully!")
    print("Image path:", image_path)
    print("Image dimensions:", rgb_image.shape)
    print("Image data type:", rgb_image.dtype)

    return rgb_image


def display_microscopy_image(image):
    """
    Display the loaded microscopy image.
    """

    plt.figure(figsize=(8, 8))

    plt.imshow(image)
    plt.title("Original Microscopy Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    image_path = "week8/images/microscopy_sample.png"

    microscopy_image = load_microscopy_image(image_path)

    display_microscopy_image(microscopy_image)
