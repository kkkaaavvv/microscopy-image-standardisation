
import cv2
import numpy as np
from pathlib import Path


# ==========================================
# WEEK 8 - TASK 3: BASIC COLOUR STATISTICS
# ==========================================

def calculate_colour_statistics(image):
    """
    Calculate mean and standard deviation
    for each RGB colour channel.
    """

    if image is None:
        raise ValueError("Input image is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel RGB image.")

    channel_names = ["Red", "Green", "Blue"]

    print("\nMICROSCOPY IMAGE COLOUR STATISTICS")
    print("=" * 45)

    statistics = {}

    for index, channel_name in enumerate(channel_names):

        channel = image[:, :, index]

        mean_value = np.mean(channel)
        std_value = np.std(channel)

        statistics[channel_name] = {
            "Mean": mean_value,
            "Standard Deviation": std_value
        }

        print(f"\n{channel_name} Channel")
        print(f"Mean Intensity: {mean_value:.4f}")
        print(f"Standard Deviation: {std_value:.4f}")

    print("\n" + "=" * 45)
    print("Statistical analysis completed.")

    return statistics


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    image_path = Path("week8/images/microscopy_sample.png")

    # Load image
    bgr_image = cv2.imread(str(image_path))

    if bgr_image is None:
        raise FileNotFoundError(image_path)

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Calculate colour statistics
    results = calculate_colour_statistics(rgb_image)
