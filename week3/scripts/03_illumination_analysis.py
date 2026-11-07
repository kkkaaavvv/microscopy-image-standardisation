
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = PROJECT_ROOT / "week1" / "images" / "microscopy_sample.png"

RESULTS_DIR = PROJECT_ROOT / "results" / "week3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load microscopy image
# --------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --------------------------------------------------
# 3. Estimate background illumination
# --------------------------------------------------

# Apply Gaussian blur to estimate large-scale brightness
# variations across the image.

illumination_map = cv2.GaussianBlur(
    gray_image,
    (101, 101),
    0
)

# --------------------------------------------------
# 4. Calculate illumination statistics
# --------------------------------------------------

mean_illumination = np.mean(illumination_map)
std_illumination = np.std(illumination_map)

min_illumination = np.min(illumination_map)
max_illumination = np.max(illumination_map)

illumination_range = (
    max_illumination - min_illumination
)

# Calculate regional brightness

height, width = gray_image.shape

regions = {
    "Top Left": gray_image[:height//2, :width//2],
    "Top Right": gray_image[:height//2, width//2:],
    "Bottom Left": gray_image[height//2:, :width//2],
    "Bottom Right": gray_image[height//2:, width//2:]
}

# --------------------------------------------------
# 5. Display numerical results
# --------------------------------------------------

print("=" * 50)
print("MICROSCOPY IMAGE ILLUMINATION ANALYSIS")
print("=" * 50)

print("\nImage Information")
print("Image shape:", image_rgb.shape)

print("\nIllumination Statistics")
print("Mean illumination:", round(mean_illumination, 2))
print("Standard deviation:", round(std_illumination, 2))
print("Minimum illumination:", min_illumination)
print("Maximum illumination:", max_illumination)
print("Illumination range:", illumination_range)

print("\nRegional Brightness Analysis")

for region_name, region in regions.items():
    print(
        f"{region_name}: "
        f"Mean brightness = {np.mean(region):.2f}"
    )

# --------------------------------------------------
# 6. Calculate illumination variation
# --------------------------------------------------

if mean_illumination > 0:
    variation_percentage = (
        std_illumination / mean_illumination
    ) * 100
else:
    variation_percentage = 0

print("\nIllumination Variation")
print(
    "Relative standard deviation:",
    round(variation_percentage, 2),
    "%"
)

print("\nNote: These measurements are exploratory.")
print("They do not independently confirm uneven lighting.")

# --------------------------------------------------
# 7. Visualise illumination
# --------------------------------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Microscopy Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(illumination_map, cmap="gray")
plt.title("Estimated Illumination Map")
plt.axis("off")

plt.subplot(1, 3, 3)

# Calculate difference from estimated illumination
illumination_difference = (
    gray_image.astype(np.float32)
    - illumination_map.astype(np.float32)
)

plt.imshow(illumination_difference, cmap="gray")
plt.title("Local Intensity Difference")
plt.axis("off")

plt.tight_layout()

# Save visualisation
output_path = RESULTS_DIR / "03_illumination_analysis.png"

plt.savefig(output_path, dpi=300)
plt.close()

# --------------------------------------------------
# 8. Save analysis report
# --------------------------------------------------

report_path = RESULTS_DIR / "03_illumination_report.txt"

with open(report_path, "w") as file:

    file.write("MICROSCOPY IMAGE ILLUMINATION ANALYSIS\n")
    file.write("=" * 45 + "\n")

    file.write(f"Image shape: {image_rgb.shape}\n")

    file.write(
        f"Mean illumination: {mean_illumination:.2f}\n"
    )

    file.write(
        f"Standard deviation: {std_illumination:.2f}\n"
    )

    file.write(
        f"Minimum illumination: {min_illumination}\n"
    )

    file.write(
        f"Maximum illumination: {max_illumination}\n"
    )

    file.write(
        f"Illumination range: {illumination_range}\n"
    )

    file.write(
        f"Relative variation: {variation_percentage:.2f}%\n"
    )

    file.write("\nRegional Brightness:\n")

    for region_name, region in regions.items():
        file.write(
            f"{region_name}: {np.mean(region):.2f}\n"
        )

    file.write(
        "\nNote: Gaussian smoothing provides an approximate "
        "large-scale illumination estimate. It may also "
        "include contributions from tissue and cell structures.\n"
    )

print("\nAnalysis completed successfully!")
print("Visualisation saved to:", output_path)
print("Report saved to:", report_path)
