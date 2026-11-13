
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from week8.modules.colour_analysis import convert_colour_spaces

# ==========================================
# WEEK 8 - TASK 2: COLOUR SPACE CONVERSION
# ==========================================

def display_colour_channels(image, channel_names, title):
    """
    Display the individual channels of a colour space.
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for index, name in enumerate(channel_names):
        axes[index].imshow(image[:, :, index], cmap="gray")
        axes[index].set_title(name)
        axes[index].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    # Save output
    output_dir = Path("results/week8")
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")

    output_path = Path("week8/evidence/01_image_loading_output.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Image display saved:", output_path)

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    image_path = "week8/images/microscopy_sample.png"

    # Load original image
    bgr_image = cv2.imread(image_path)

    if bgr_image is None:
        raise FileNotFoundError(image_path)

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Convert RGB to HSV and LAB
    hsv_image, lab_image = convert_colour_spaces(rgb_image)

    # Display HSV channels
    display_colour_channels(
        hsv_image,
        ["Hue", "Saturation", "Value"],
        "HSV Colour Space"
    )

    # Display LAB channels
    display_colour_channels(
        lab_image,
        ["Lightness (L)", "A Channel", "B Channel"],
        "LAB Colour Space"
    )
