
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# EXPERIMENT 10: CELL SEGMENTATION AND COMPONENT ANALYSIS
# ==========================================================

print("=" * 65)
print("CELL SEGMENTATION AND CONNECTED-COMPONENT ANALYSIS")
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
    / "segmentation_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_COMPONENT_AREA = 20


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
    raise ValueError("Source and normalised images must match in size.")

print("\nImages loaded successfully!")
print("Source shape:", source.shape)
print("Normalised shape:", normalised.shape)


# ==========================================================
# SEGMENTATION FUNCTION
# ==========================================================

def segment_image(image, image_name):

    print(f"\nProcessing {image_name}...")

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 2: Apply Gaussian blur to reduce small intensity noise
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

    # Step 4: Morphological opening to remove small noise
    opening_kernel = np.ones((3, 3), np.uint8)

    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        opening_kernel
    )

    # Step 5: Morphological closing to connect small gaps
    closing_kernel = np.ones((5, 5), np.uint8)

    cleaned = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        closing_kernel
    )

    # Step 6: Connected-component labelling
    number_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            cleaned,
            connectivity=8
        )
    )

    # Label 0 represents the background
    components = []

    filtered_mask = np.zeros_like(cleaned)

    for label in range(1, number_labels):

        area = int(stats[label, cv2.CC_STAT_AREA])

        if area >= MIN_COMPONENT_AREA:

            components.append({
                "label": label,
                "area": area,
                "width": int(stats[label, cv2.CC_STAT_WIDTH]),
                "height": int(stats[label, cv2.CC_STAT_HEIGHT]),
                "centroid_x": float(centroids[label][0]),
                "centroid_y": float(centroids[label][1])
            })

            filtered_mask[labels == label] = 255

    # Calculate component statistics
    areas = [component["area"] for component in components]

    total_pixels = cleaned.size

    foreground_pixels = np.count_nonzero(filtered_mask)

    foreground_percentage = (
        foreground_pixels / total_pixels
    ) * 100

    if areas:

        mean_area = float(np.mean(areas))
        median_area = float(np.median(areas))
        min_area = int(np.min(areas))
        max_area = int(np.max(areas))
        std_area = float(np.std(areas))

    else:

        mean_area = 0
        median_area = 0
        min_area = 0
        max_area = 0
        std_area = 0

    results = {
        "Image": image_name,
        "Otsu_Threshold": float(threshold_value),
        "Component_Count": len(components),
        "Foreground_Pixels": int(foreground_pixels),
        "Foreground_Percentage": float(foreground_percentage),
        "Mean_Component_Area": mean_area,
        "Median_Component_Area": median_area,
        "Minimum_Component_Area": min_area,
        "Maximum_Component_Area": max_area,
        "Component_Area_Std": std_area
    }

    print("Otsu threshold:", threshold_value)
    print("Detected components:", len(components))
    print(f"Foreground coverage: {foreground_percentage:.2f}%")
    print(f"Mean component area: {mean_area:.2f}")
    print(f"Median component area: {median_area:.2f}")

    return {
        "gray": gray,
        "binary": binary,
        "cleaned": cleaned,
        "filtered_mask": filtered_mask,
        "components": components,
        "results": results,
        "areas": areas
    }


# ==========================================================
# PROCESS BOTH IMAGES
# ==========================================================

source_data = segment_image(source, "Source")
normalised_data = segment_image(normalised, "Normalised")


# ==========================================================
# SAVE COMPONENT STATISTICS
# ==========================================================

summary_path = OUTPUT_DIR / "segmentation_summary.csv"

with open(summary_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(source_data["results"].keys())
    )

    writer.writeheader()
    writer.writerow(source_data["results"])
    writer.writerow(normalised_data["results"])


# Save individual component measurements
component_path = OUTPUT_DIR / "component_details.csv"

with open(component_path, "w", newline="") as file:

    fieldnames = [
        "Image",
        "Component_ID",
        "Area",
        "Width",
        "Height",
        "Centroid_X",
        "Centroid_Y"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for image_name, data in [
        ("Source", source_data),
        ("Normalised", normalised_data)
    ]:

        for component in data["components"]:

            writer.writerow({
                "Image": image_name,
                "Component_ID": component["label"],
                "Area": component["area"],
                "Width": component["width"],
                "Height": component["height"],
                "Centroid_X": component["centroid_x"],
                "Centroid_Y": component["centroid_y"]
            })


# ==========================================================
# PRINT COMPARISON
# ==========================================================

print("\n" + "=" * 65)
print("SEGMENTATION COMPARISON")
print("=" * 65)

for key in source_data["results"]:

    if key == "Image":
        continue

    source_value = source_data["results"][key]
    normalised_value = normalised_data["results"][key]

    print(f"\n{key}")
    print(f"Source: {source_value}")
    print(f"Normalised: {normalised_value}")


# ==========================================================
# GENERATE SEGMENTATION OVERLAYS
# ==========================================================

def create_overlay(image, mask):

    overlay = image.copy()

    colour_mask = np.zeros_like(image)
    colour_mask[mask > 0] = [255, 0, 0]

    result = cv2.addWeighted(
        overlay,
        0.75,
        colour_mask,
        0.25,
        0
    )

    return result


source_overlay = create_overlay(
    source,
    source_data["filtered_mask"]
)

normalised_overlay = create_overlay(
    normalised,
    normalised_data["filtered_mask"]
)


# ==========================================================
# VISUAL COMPARISON
# ==========================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(source)
axes[0, 0].set_title("Original Source")

axes[0, 1].imshow(
    source_data["filtered_mask"],
    cmap="gray"
)
axes[0, 1].set_title("Source Segmentation Mask")

axes[0, 2].imshow(source_overlay)
axes[0, 2].set_title("Source Segmentation Overlay")

axes[1, 0].imshow(normalised)
axes[1, 0].set_title("Normalised Image")

axes[1, 1].imshow(
    normalised_data["filtered_mask"],
    cmap="gray"
)
axes[1, 1].set_title("Normalised Segmentation Mask")

axes[1, 2].imshow(normalised_overlay)
axes[1, 2].set_title("Normalised Segmentation Overlay")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

comparison_path = OUTPUT_DIR / "segmentation_comparison.png"

plt.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# COMPONENT AREA DISTRIBUTION
# ==========================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(
    source_data["areas"],
    bins=30
)
axes[0].set_title("Source Component Area Distribution")
axes[0].set_xlabel("Component Area (pixels)")
axes[0].set_ylabel("Frequency")

axes[1].hist(
    normalised_data["areas"],
    bins=30
)
axes[1].set_title("Normalised Component Area Distribution")
axes[1].set_xlabel("Component Area (pixels)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()

histogram_path = OUTPUT_DIR / "component_area_distribution.png"

plt.savefig(
    histogram_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# COMPLETION
# ==========================================================

print("\n" + "=" * 65)
print("EXPERIMENT 10 COMPLETED")
print("=" * 65)

print("Summary CSV:", summary_path)
print("Component details:", component_path)
print("Segmentation comparison:", comparison_path)
print("Area distribution:", histogram_path)
print("Output directory:", OUTPUT_DIR)
