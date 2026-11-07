
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
# 3. Apply Laplacian operator
# --------------------------------------------------

laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

# Calculate variance of Laplacian
blur_score = laplacian.var()

# --------------------------------------------------
# 4. Display numerical results
# --------------------------------------------------

print("=" * 45)
print("MICROSCOPY IMAGE BLUR DETECTION")
print("=" * 45)

print("\nImage Information")
print("Image shape:", image_rgb.shape)
print("Data type:", image_rgb.dtype)

print("\nBlur Analysis")
print("Variance of Laplacian:", round(blur_score, 2))

# Preliminary interpretation
print("\nImage Assessment")

if blur_score < 50:
    print("Low edge detail detected. Possible blur.")
elif blur_score < 150:
    print("Moderate edge detail detected.")
else:
    print("Strong edge detail detected.")

print("\nNote: Thresholds are preliminary.")
print("They must be validated using multiple images.")

# --------------------------------------------------
# 5. Visualise original and Laplacian images
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Microscopy Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian Edge Response")
plt.axis("off")

plt.tight_layout()

# Save visualisation
output_path = RESULTS_DIR / "01_blur_detection.png"

plt.savefig(output_path, dpi=300)
plt.close()

# --------------------------------------------------
# 6. Save numerical report
# --------------------------------------------------

report_path = RESULTS_DIR / "01_blur_report.txt"

with open(report_path, "w") as file:
    file.write("MICROSCOPY IMAGE BLUR DETECTION\n")
    file.write("=" * 40 + "\n")

    file.write(f"Image shape: {image_rgb.shape}\n")
    file.write(f"Image data type: {image_rgb.dtype}\n")
    file.write(f"Variance of Laplacian: {blur_score:.2f}\n")

    file.write("\nNote: Blur thresholds are preliminary.\n")

print("\nAnalysis completed successfully!")
print("Visualisation saved to:", output_path)
print("Report saved to:", report_path)
