
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

# Convert RGB image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# --------------------------------------------------
# 3. Calculate intensity statistics
# --------------------------------------------------

mean_intensity = np.mean(gray_image)
median_intensity = np.median(gray_image)
std_intensity = np.std(gray_image)

min_intensity = np.min(gray_image)
max_intensity = np.max(gray_image)

# Calculate intensity percentiles
p5 = np.percentile(gray_image, 5)
p25 = np.percentile(gray_image, 25)
p75 = np.percentile(gray_image, 75)
p95 = np.percentile(gray_image, 95)

# --------------------------------------------------
# 4. Assess exposure
# --------------------------------------------------

total_pixels = gray_image.size

# Count very dark and very bright pixels
dark_pixels = np.sum(gray_image <= 5)
bright_pixels = np.sum(gray_image >= 250)

# Calculate percentage of dark and bright pixels
dark_percentage = (dark_pixels / total_pixels) * 100
bright_percentage = (bright_pixels / total_pixels) * 100

# --------------------------------------------------
# 5. Display numerical results
# --------------------------------------------------

print("=" * 50)
print("MICROSCOPY IMAGE INTENSITY ANALYSIS")
print("=" * 50)

print("\nImage Information")
print("Image shape:", image_rgb.shape)
print("Data type:", image_rgb.dtype)

print("\nIntensity Statistics")
print("Mean intensity:", round(mean_intensity, 2))
print("Median intensity:", round(median_intensity, 2))
print("Standard deviation:", round(std_intensity, 2))
print("Minimum intensity:", min_intensity)
print("Maximum intensity:", max_intensity)

print("\nIntensity Percentiles")
print("5th percentile:", round(p5, 2))
print("25th percentile:", round(p25, 2))
print("75th percentile:", round(p75, 2))
print("95th percentile:", round(p95, 2))

print("\nExposure Assessment")
print("Dark pixels (<= 5):", dark_pixels)
print("Dark pixel percentage:", round(dark_percentage, 2), "%")
print("Bright pixels (>= 250):", bright_pixels)
print("Bright pixel percentage:", round(bright_percentage, 2), "%")

print("\nNote: These are preliminary exposure indicators.")
print("Thresholds require validation across multiple images.")

# --------------------------------------------------
# 6. Analyse individual RGB channels
# --------------------------------------------------

print("\nRGB Channel Statistics")

for i, channel_name in enumerate(["Red", "Green", "Blue"]):
    channel = image_rgb[:, :, i]

    print(
        f"{channel_name}: "
        f"Mean = {np.mean(channel):.2f}, "
        f"Std Dev = {np.std(channel):.2f}"
    )

# --------------------------------------------------
# 7. Generate intensity histogram
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    gray_image.ravel(),
    bins=256,
    range=(0, 256),
    color="gray"
)

plt.title("Grayscale Intensity Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")

plt.xlim(0, 255)
plt.grid(alpha=0.2)

plt.tight_layout()

# Save histogram
histogram_path = RESULTS_DIR / "02_intensity_histogram.png"

plt.savefig(histogram_path, dpi=300)
plt.close()

# --------------------------------------------------
# 8. Save analysis report
# --------------------------------------------------

report_path = RESULTS_DIR / "02_intensity_report.txt"

with open(report_path, "w") as file:
    file.write("MICROSCOPY IMAGE INTENSITY ANALYSIS\n")
    file.write("=" * 40 + "\n")

    file.write(f"Image shape: {image_rgb.shape}\n")
    file.write(f"Mean intensity: {mean_intensity:.2f}\n")
    file.write(f"Median intensity: {median_intensity:.2f}\n")
    file.write(f"Standard deviation: {std_intensity:.2f}\n")
    file.write(f"Minimum intensity: {min_intensity}\n")
    file.write(f"Maximum intensity: {max_intensity}\n")

    file.write(f"5th percentile: {p5:.2f}\n")
    file.write(f"25th percentile: {p25:.2f}\n")
    file.write(f"75th percentile: {p75:.2f}\n")
    file.write(f"95th percentile: {p95:.2f}\n")

    file.write(f"Dark pixel percentage: {dark_percentage:.2f}%\n")
    file.write(f"Bright pixel percentage: {bright_percentage:.2f}%\n")

print("\nAnalysis completed successfully!")
print("Histogram saved to:", histogram_path)
print("Report saved to:", report_path)
