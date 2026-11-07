
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# EXPERIMENT 8: CONTRAST-AWARE EDGE PRESERVATION ANALYSIS
# ==========================================================

print("=" * 65)
print("CONTRAST-AWARE EDGE PRESERVATION ANALYSIS")
print("=" * 65)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    PROJECT_ROOT / "week1" / "images" / "microscopy_sample.png"
)

NORMALISED_PATH = (
    PROJECT_ROOT
    / "results"
    / "week4"
    / "stain_components"
    / "normalised_image.png"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "week4"
    / "contrast_aware_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# LOAD IMAGES
# ==========================================================

def load_image(path):

    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


source = load_image(SOURCE_PATH)
normalised = load_image(NORMALISED_PATH)

if source.shape != normalised.shape:
    raise ValueError("Source and normalised images must have matching sizes.")

source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
normalised_gray = cv2.cvtColor(normalised, cv2.COLOR_RGB2GRAY)

print("\nImages loaded successfully!")
print("Image dimensions:", source.shape)


# ==========================================================
# METHOD 1: FIXED CANNY THRESHOLDS
# ==========================================================

print("\nMethod 1: Fixed Canny thresholds")

source_fixed_edges = cv2.Canny(source_gray, 100, 200)
normalised_fixed_edges = cv2.Canny(normalised_gray, 100, 200)


# ==========================================================
# METHOD 2: ADAPTIVE CANNY THRESHOLDS
# ==========================================================

def adaptive_canny(image):

    median = np.median(image)

    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))

    if upper <= lower:
        upper = min(255, lower + 1)

    edges = cv2.Canny(image, lower, upper)

    return edges, lower, upper


print("\nMethod 2: Adaptive Canny thresholds")

source_adaptive_edges, source_lower, source_upper = adaptive_canny(
    source_gray
)

normalised_adaptive_edges, normalised_lower, normalised_upper = adaptive_canny(
    normalised_gray
)

print(
    f"Source thresholds: lower={source_lower}, upper={source_upper}"
)

print(
    f"Normalised thresholds: lower={normalised_lower}, upper={normalised_upper}"
)


# ==========================================================
# METHOD 3: CLAHE + ADAPTIVE CANNY
# ==========================================================

print("\nMethod 3: CLAHE-enhanced adaptive edge detection")

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

source_clahe = clahe.apply(source_gray)
normalised_clahe = clahe.apply(normalised_gray)

source_clahe_edges, source_clahe_lower, source_clahe_upper = adaptive_canny(
    source_clahe
)

normalised_clahe_edges, normalised_clahe_lower, normalised_clahe_upper = adaptive_canny(
    normalised_clahe
)

print("CLAHE enhancement completed.")


# ==========================================================
# EDGE DENSITY CALCULATION
# ==========================================================

def calculate_edge_density(edge_image):

    edge_pixels = np.count_nonzero(edge_image)

    total_pixels = edge_image.shape[0] * edge_image.shape[1]

    return edge_pixels / total_pixels


results = {
    "Source_Fixed_Canny": calculate_edge_density(source_fixed_edges),
    "Normalised_Fixed_Canny": calculate_edge_density(normalised_fixed_edges),

    "Source_Adaptive_Canny": calculate_edge_density(source_adaptive_edges),
    "Normalised_Adaptive_Canny": calculate_edge_density(normalised_adaptive_edges),

    "Source_CLAHE_Adaptive": calculate_edge_density(source_clahe_edges),
    "Normalised_CLAHE_Adaptive": calculate_edge_density(normalised_clahe_edges)
}


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print("\n" + "=" * 65)
print("EDGE DENSITY COMPARISON")
print("=" * 65)

for name, value in results.items():

    print(f"{name}: {value:.6f} ({value * 100:.2f}%)")


# ==========================================================
# SAVE CSV
# ==========================================================

csv_path = OUTPUT_DIR / "contrast_aware_edge_results.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["Method", "Image", "Edge_Density"])

    for method, source_key, normalised_key in [
        (
            "Fixed Canny",
            "Source_Fixed_Canny",
            "Normalised_Fixed_Canny"
        ),
        (
            "Adaptive Canny",
            "Source_Adaptive_Canny",
            "Normalised_Adaptive_Canny"
        ),
        (
            "CLAHE + Adaptive Canny",
            "Source_CLAHE_Adaptive",
            "Normalised_CLAHE_Adaptive"
        )
    ]:

        writer.writerow([
            method,
            "Source",
            results[source_key]
        ])

        writer.writerow([
            method,
            "Normalised",
            results[normalised_key]
        ])

print("\nCSV saved:", csv_path)


# ==========================================================
# VISUAL COMPARISON
# ==========================================================

fig, axes = plt.subplots(3, 3, figsize=(15, 13))

# Original grayscale images
axes[0, 0].imshow(source_gray, cmap="gray")
axes[0, 0].set_title("Source Grayscale")

axes[0, 1].imshow(normalised_gray, cmap="gray")
axes[0, 1].set_title("Normalised Grayscale")

axes[0, 2].axis("off")
axes[0, 2].text(
    0.1, 0.5,
    "Original grayscale images",
    fontsize=12
)

# Fixed Canny
axes[1, 0].imshow(source_fixed_edges, cmap="gray")
axes[1, 0].set_title("Source - Fixed Canny")

axes[1, 1].imshow(normalised_fixed_edges, cmap="gray")
axes[1, 1].set_title("Normalised - Fixed Canny")

axes[1, 2].axis("off")
axes[1, 2].text(
    0.1, 0.5,
    "Fixed thresholds: 100, 200",
    fontsize=12
)

# Adaptive Canny
axes[2, 0].imshow(source_adaptive_edges, cmap="gray")
axes[2, 0].set_title("Source - Adaptive Canny")

axes[2, 1].imshow(normalised_adaptive_edges, cmap="gray")
axes[2, 1].set_title("Normalised - Adaptive Canny")

axes[2, 2].axis("off")
axes[2, 2].text(
    0.05, 0.7,
    "CLAHE-enhanced results\nare saved separately.",
    fontsize=11
)

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

comparison_path = OUTPUT_DIR / "edge_detection_comparison.png"

plt.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# CLAHE COMPARISON FIGURE
# ==========================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].imshow(source_clahe, cmap="gray")
axes[0, 0].set_title("Source + CLAHE")

axes[0, 1].imshow(normalised_clahe, cmap="gray")
axes[0, 1].set_title("Normalised + CLAHE")

axes[0, 2].imshow(source_clahe_edges, cmap="gray")
axes[0, 2].set_title("Source CLAHE Edges")

axes[1, 0].imshow(normalised_clahe_edges, cmap="gray")
axes[1, 0].set_title("Normalised CLAHE Edges")

axes[1, 1].imshow(source_adaptive_edges, cmap="gray")
axes[1, 1].set_title("Source Adaptive Edges")

axes[1, 2].imshow(normalised_adaptive_edges, cmap="gray")
axes[1, 2].set_title("Normalised Adaptive Edges")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

clahe_path = OUTPUT_DIR / "clahe_edge_comparison.png"

plt.savefig(
    clahe_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# COMPLETION
# ==========================================================

print("\n" + "=" * 65)
print("EXPERIMENT 8 COMPLETED")
print("=" * 65)

print("Results directory:", OUTPUT_DIR)
print("CSV:", csv_path)
print("Edge comparison:", comparison_path)
print("CLAHE comparison:", clahe_path)
