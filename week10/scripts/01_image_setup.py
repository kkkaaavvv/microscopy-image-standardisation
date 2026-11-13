
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# WEEK 10 - TASK 1
# REFERENCE AND SOURCE IMAGE SETUP
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "week10" / "images"
RESULT_DIR = PROJECT_ROOT / "results" / "week10"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
def load_image(image_path):
    """
    Load microscopy image and convert BGR to RGB.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr_image = cv2.imread(str(image_path))

    if bgr_image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    print(f"\nLoaded: {image_path.name}")
    print("Dimensions:", rgb_image.shape)
    print("Data type:", rgb_image.dtype)

    return rgb_image


def display_images(source, reference):
    """
    Display source and reference images side by side.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(source)
    axes[0].set_title("Source Image")
    axes[0].axis("off")

    axes[1].imshow(reference)
    axes[1].set_title("Reference Image")
    axes[1].axis("off")

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "source_reference_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def generate_rgb_histogram(image, image_name):
    """
    Generate RGB channel distribution histograms.
    """

    channels = [
        ("Red", 0, "red"),
        ("Green", 1, "green"),
        ("Blue", 2, "blue")
    ]

    plt.figure(figsize=(10, 5))

    for name, index, colour in channels:

        histogram = cv2.calcHist(
            [image], [index], None, [256], [0, 256]
        )

        plt.plot(histogram, color=colour, label=name)

    plt.title(f"{image_name} RGB Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Pixel Frequency")
    plt.xlim([0, 256])
    plt.legend()
    plt.grid(alpha=0.2)

    plt.savefig(
        RESULT_DIR / f"{image_name.lower()}_histogram.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def calculate_lab_statistics(image, image_name):
    """
    Calculate mean and standard deviation
    of LAB colour channels.
    """

    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    channel_names = ["L", "A", "B"]

    print(f"\n{image_name} LAB STATISTICS")
    print("-" * 40)

    for index, name in enumerate(channel_names):

        channel = lab_image[:, :, index]

        mean_value = np.mean(channel)
        std_value = np.std(channel)

        print(f"{name} Channel")
        print(f"Mean: {mean_value:.4f}")
        print(f"Standard Deviation: {std_value:.4f}")
        print()


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    source_image = load_image(
        IMAGE_DIR / "source_image.png"
    )

    reference_image = load_image(
        IMAGE_DIR / "reference_image.png"
    )

    # Display images
    display_images(source_image, reference_image)

    # Generate RGB histograms
    generate_rgb_histogram(source_image, "Source")
    generate_rgb_histogram(reference_image, "Reference")

    # Calculate LAB statistics
    calculate_lab_statistics(source_image, "SOURCE IMAGE")
    calculate_lab_statistics(reference_image, "REFERENCE IMAGE")

    print("\n Completed successfully!")
    
