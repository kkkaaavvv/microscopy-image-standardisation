
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# EXPERIMENT 7: STRUCTURAL PRESERVATION ASSESSMENT
# ==========================================================

print("=" * 60)
print("STRUCTURAL PRESERVATION AND IMAGE QUALITY ASSESSMENT")
print("=" * 60)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    PROJECT_ROOT
    / "week1"
    / "images"
    / "microscopy_sample.png"
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
    / "structural_evaluation"
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
    raise ValueError(
        "Source and normalised images must have matching dimensions."
    )

print("\nImages loaded successfully!")
print("Source shape:", source.shape)
print("Normalised shape:", normalised.shape)


# ==========================================================
# CONVERT TO GRAYSCALE
# ==========================================================

source_gray = cv2.cvtColor(
    source,
    cv2.COLOR_RGB2GRAY
)

normalised_gray = cv2.cvtColor(
    normalised,
    cv2.COLOR_RGB2GRAY
)


# ==========================================================
# STRUCTURAL SIMILARITY (SSIM)
# ==========================================================

def calculate_ssim(image1, image2):

    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    # Local means
    mu1 = cv2.GaussianBlur(image1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(image2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # Local variances and covariance
    sigma1_sq = (
        cv2.GaussianBlur(image1 * image1, (11, 11), 1.5)
        - mu1_sq
    )

    sigma2_sq = (
        cv2.GaussianBlur(image2 * image2, (11, 11), 1.5)
        - mu2_sq
    )

    sigma12 = (
        cv2.GaussianBlur(image1 * image2, (11, 11), 1.5)
        - mu1_mu2
    )

    # SSIM constants for 8-bit images
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    numerator = (
        (2 * mu1_mu2 + C1)
        * (2 * sigma12 + C2)
    )

    denominator = (
        (mu1_sq + mu2_sq + C1)
        * (sigma1_sq + sigma2_sq + C2)
    )

    ssim_map = numerator / (denominator + 1e-10)

    return float(np.mean(ssim_map))


# ==========================================================
# LAPLACIAN VARIANCE
# ==========================================================

def calculate_laplacian_variance(image):

    return float(
        cv2.Laplacian(
            image,
            cv2.CV_64F
        ).var()
    )


# ==========================================================
# EDGE DENSITY
# ==========================================================

def calculate_edge_density(image):

    # Use the same thresholds for both images
    edges = cv2.Canny(
        image,
        100,
        200
    )

    edge_pixels = np.count_nonzero(edges)

    total_pixels = image.shape[0] * image.shape[1]

    edge_density = edge_pixels / total_pixels

    return float(edge_density), edges


# ==========================================================
# GRADIENT MAGNITUDE
# ==========================================================

def calculate_gradient_magnitude(image):

    sobel_x = cv2.Sobel(
        image,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        image,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitude = np.sqrt(
        sobel_x ** 2 + sobel_y ** 2
    )

    return float(np.mean(magnitude))


# ==========================================================
# CALCULATE METRICS
# ==========================================================

print("\nCalculating structural metrics...")

ssim_score = calculate_ssim(
    source_gray,
    normalised_gray
)

source_laplacian = calculate_laplacian_variance(
    source_gray
)

normalised_laplacian = calculate_laplacian_variance(
    normalised_gray
)

source_edge_density, source_edges = calculate_edge_density(
    source_gray
)

normalised_edge_density, normalised_edges = calculate_edge_density(
    normalised_gray
)

source_gradient = calculate_gradient_magnitude(
    source_gray
)

normalised_gradient = calculate_gradient_magnitude(
    normalised_gray
)


# ==========================================================
# PRINT RESULTS
# ==========================================================

results = {
    "SSIM_Source_vs_Normalised": ssim_score,
    "Source_Laplacian_Variance": source_laplacian,
    "Normalised_Laplacian_Variance": normalised_laplacian,
    "Source_Edge_Density": source_edge_density,
    "Normalised_Edge_Density": normalised_edge_density,
    "Source_Gradient_Magnitude": source_gradient,
    "Normalised_Gradient_Magnitude": normalised_gradient
}

print("\n" + "=" * 60)
print("STRUCTURAL EVALUATION RESULTS")
print("=" * 60)

for metric, value in results.items():
    print(f"{metric}: {value:.6f}")


# ==========================================================
# SAVE CSV
# ==========================================================

csv_path = OUTPUT_DIR / "structural_evaluation.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["Metric", "Value"])

    for metric, value in results.items():
        writer.writerow([metric, value])

print("\nStatistics saved:", csv_path)


# ==========================================================
# GENERATE VISUAL COMPARISON
# ==========================================================

difference_map = cv2.absdiff(
    source_gray,
    normalised_gray
)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(source)
axes[0, 0].set_title("Original Source Image")

axes[0, 1].imshow(normalised)
axes[0, 1].set_title("Normalised Image")

axes[0, 2].imshow(
    difference_map,
    cmap="gray"
)
axes[0, 2].set_title("Absolute Grayscale Difference")

axes[1, 0].imshow(
    source_edges,
    cmap="gray"
)
axes[1, 0].set_title("Source Edge Map")

axes[1, 1].imshow(
    normalised_edges,
    cmap="gray"
)
axes[1, 1].set_title("Normalised Edge Map")

axes[1, 2].imshow(
    source_gray,
    cmap="gray"
)
axes[1, 2].set_title("Source Grayscale")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

comparison_path = OUTPUT_DIR / "structural_comparison.png"

plt.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Visual comparison saved:", comparison_path)


# ==========================================================
# COMPLETION
# ==========================================================

print("\n" + "=" * 60)
print("EXPERIMENT 7 COMPLETED")
print("=" * 60)

print("Output directory:", OUTPUT_DIR)
