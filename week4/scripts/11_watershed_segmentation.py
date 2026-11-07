
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# EXPERIMENT 11: WATERSHED-BASED REGION SEPARATION
# ==========================================================

print("=" * 65)
print("WATERSHED-BASED REGION SEPARATION")
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
    / "watershed_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_REGION_AREA = 20


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
    raise ValueError("Images must have matching dimensions.")

print("\nImages loaded successfully!")
print("Source shape:", source.shape)
print("Normalised shape:", normalised.shape)


# ==========================================================
# WATERSHED SEGMENTATION FUNCTION
# ==========================================================

def apply_watershed(image, image_name):

    print(f"\nProcessing {image_name}...")

    # Step 1: Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 2: Gaussian blur
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Step 3: Otsu thresholding
    threshold_value, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 4: Morphological opening
    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    # Step 5: Estimate sure background
    sure_background = cv2.dilate(
        opening,
        kernel,
        iterations=3
    )

    # Step 6: Distance transform
    distance = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    # Step 7: Identify sure foreground
    # 35% of the maximum distance is used as a baseline
    _, sure_foreground = cv2.threshold(
        distance,
        0.35 * distance.max(),
        255,
        0
    )

    sure_foreground = np.uint8(sure_foreground)

    # Step 8: Identify unknown region
    unknown = cv2.subtract(
        sure_background,
        sure_foreground
    )

    # Step 9: Generate markers
    number_labels, markers = cv2.connectedComponents(
        sure_foreground
    )

    # Shift markers so background is 1 instead of 0
    markers = markers + 1

    # Unknown region receives marker value 0
    markers[unknown == 255] = 0

    # Step 10: Apply watershed
    watershed_image = image.copy()

    markers = cv2.watershed(
        watershed_image,
        markers
    )

    # Step 11: Extract segmented regions
    region_mask = np.zeros(gray.shape, dtype=np.uint8)

    region_areas = []

    for label in range(2, int(markers.max()) + 1):

        component_mask = (markers == label)

        area = int(np.count_nonzero(component_mask))

        if area >= MIN_REGION_AREA:

            region_mask[component_mask] = 255
            region_areas.append(area)

    # Step 12: Extract watershed boundaries
    boundary_mask = np.zeros(gray.shape, dtype=np.uint8)
    boundary_mask[markers == -1] = 255

    # Step 13: Calculate statistics
    foreground_pixels = np.count_nonzero(region_mask)

    foreground_percentage = (
        foreground_pixels / region_mask.size
    ) * 100

    if region_areas:

        mean_area = float(np.mean(region_areas))
        median_area = float(np.median(region_areas))
        max_area = int(np.max(region_areas))
        min_area = int(np.min(region_areas))
        std_area = float(np.std(region_areas))

    else:

        mean_area = 0
        median_area = 0
        max_area = 0
        min_area = 0
        std_area = 0

    boundary_pixels = np.count_nonzero(boundary_mask)

    results = {
        "Image": image_name,
        "Otsu_Threshold": float(threshold_value),
        "Watershed_Region_Count": len(region_areas),
        "Foreground_Pixels": int(foreground_pixels),
        "Foreground_Percentage": float(foreground_percentage),
        "Mean_Region_Area": mean_area,
        "Median_Region_Area": median_area,
        "Minimum_Region_Area": min_area,
        "Maximum_Region_Area": max_area,
        "Region_Area_Std": std_area,
        "Boundary_Pixels": int(boundary_pixels),
        "Boundary_Percentage": float(
            boundary_pixels / boundary_mask.size * 100
        )
    }

    print("Otsu threshold:", threshold_value)
    print("Initial foreground markers:", number_labels - 1)
    print("Watershed regions:", len(region_areas))
    print(f"Foreground coverage: {foreground_percentage:.2f}%")
    print(f"Mean region area: {mean_area:.2f}")
    print(f"Boundary coverage: {results['Boundary_Percentage']:.2f}%")

    return {
        "gray": gray,
        "binary": binary,
        "opening": opening,
        "distance": distance,
        "sure_foreground": sure_foreground,
        "sure_background": sure_background,
        "unknown": unknown,
        "markers": markers,
        "region_mask": region_mask,
        "boundary_mask": boundary_mask,
        "region_areas": region_areas,
        "results": results
    }


# ==========================================================
# PROCESS BOTH IMAGES
# ==========================================================

source_data = apply_watershed(
    source,
    "Source"
)

normalised_data = apply_watershed(
    normalised,
    "Normalised"
)


# ==========================================================
# SAVE SUMMARY CSV
# ==========================================================

summary_path = OUTPUT_DIR / "watershed_summary.csv"

with open(summary_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(source_data["results"].keys())
    )

    writer.writeheader()
    writer.writerow(source_data["results"])
    writer.writerow(normalised_data["results"])

print("\nSummary CSV saved:", summary_path)


# ==========================================================
# SAVE REGION AREA DETAILS
# ==========================================================

area_path = OUTPUT_DIR / "watershed_region_areas.csv"

with open(area_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["Image", "Region_ID", "Area_Pixels"])

    for image_name, data in [
        ("Source", source_data),
        ("Normalised", normalised_data)
    ]:

        for index, area in enumerate(data["region_areas"], start=1):

            writer.writerow([
                image_name,
                index,
                area
            ])

print("Region area CSV saved:", area_path)


# ==========================================================
# CREATE BOUNDARY OVERLAYS
# ==========================================================

def create_boundary_overlay(image, boundary_mask):

    overlay = image.copy()

    overlay[boundary_mask > 0] = [255, 0, 0]

    return overlay


source_overlay = create_boundary_overlay(
    source,
    source_data["boundary_mask"]
)

normalised_overlay = create_boundary_overlay(
    normalised,
    normalised_data["boundary_mask"]
)


# ==========================================================
# VISUALISE SEGMENTATION RESULTS
# ==========================================================

fig, axes = plt.subplots(2, 4, figsize=(18, 10))

axes[0, 0].imshow(source)
axes[0, 0].set_title("Source Image")

axes[0, 1].imshow(
    source_data["distance"],
    cmap="viridis"
)
axes[0, 1].set_title("Source Distance Transform")

axes[0, 2].imshow(
    source_data["region_mask"],
    cmap="gray"
)
axes[0, 2].set_title("Source Watershed Regions")

axes[0, 3].imshow(source_overlay)
axes[0, 3].set_title("Source Boundary Overlay")

axes[1, 0].imshow(normalised)
axes[1, 0].set_title("Normalised Image")

axes[1, 1].imshow(
    normalised_data["distance"],
    cmap="viridis"
)
axes[1, 1].set_title("Normalised Distance Transform")

axes[1, 2].imshow(
    normalised_data["region_mask"],
    cmap="gray"
)
axes[1, 2].set_title("Normalised Watershed Regions")

axes[1, 3].imshow(normalised_overlay)
axes[1, 3].set_title("Normalised Boundary Overlay")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

comparison_path = OUTPUT_DIR / "watershed_comparison.png"

plt.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# REGION AREA DISTRIBUTION
# ==========================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(
    source_data["region_areas"],
    bins=30
)

axes[0].set_title("Source Watershed Region Areas")
axes[0].set_xlabel("Region Area (pixels)")
axes[0].set_ylabel("Frequency")

axes[1].hist(
    normalised_data["region_areas"],
    bins=30
)

axes[1].set_title("Normalised Watershed Region Areas")
axes[1].set_xlabel("Region Area (pixels)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()

histogram_path = OUTPUT_DIR / "watershed_area_distribution.png"

plt.savefig(
    histogram_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# SAVE INDIVIDUAL MASKS
# ==========================================================

cv2.imwrite(
    str(OUTPUT_DIR / "source_watershed_mask.png"),
    source_data["region_mask"]
)

cv2.imwrite(
    str(OUTPUT_DIR / "normalised_watershed_mask.png"),
    normalised_data["region_mask"]
)

cv2.imwrite(
    str(OUTPUT_DIR / "source_watershed_boundaries.png"),
    source_data["boundary_mask"]
)

cv2.imwrite(
    str(OUTPUT_DIR / "normalised_watershed_boundaries.png"),
    normalised_data["boundary_mask"]
)


# ==========================================================
# COMPLETION
# ==========================================================

print("\n" + "=" * 65)
print("EXPERIMENT 11 COMPLETED")
print("=" * 65)

print("Summary:", summary_path)
print("Region areas:", area_path)
print("Visual comparison:", comparison_path)
print("Area distribution:", histogram_path)
print("Output directory:", OUTPUT_DIR)
