import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def generate_rgb_histograms(image):
    """
    Generate separate histograms for Red, Green and Blue channels.
    """

    if image is None:
        raise ValueError("Input image is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel RGB image.")

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

    plt.title("RGB Colour Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.xlim([0, 256])
    plt.legend()
    plt.grid(alpha=0.2)

    output_path = Path("results/week8/rgb_histogram.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"RGB histogram saved: {output_path}")