
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# EXPERIMENT 9: SPATIAL EDGE PRESERVATION
# ==========================================================

print("=" * 65)
print("SPATIAL EDGE PRESERVATION AND THRESHOLD SENSITIVITY")
print("=" * 65)

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
    / "spatial_edge_analysis"
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
    raise ValueError("Images must have matching dimensions.")

source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
normalised_gray = cv2.cvtColor(normalised, cv2.COLOR_RGB2GRAY)

print("\nImages loaded successfully!")
print("Image shape:", source.shape)


# ==========================================================
# EDGE DETECTION THRESHOLDS
# ==========================================================

threshold_pairs = [
    (50, 100),
    (75, 150),
    (100, 200),
    (125, 250)
]

# Allow edges to match within 2 pixels
tolerance = 2

kernel = np.ones((3, 3), dtype=np.uint8)


# ==========================================================
# EDGE DENSITY
# ==========================================================

def edge_density(edge_map):

    return np.count_nonzero(edge_map) / edge_map.size


# ==========================================================
# SPATIAL EDGE AGREEMENT
# ==========================================================

def calculate_edge_agreement(source_edges, normalised_edges):

    source_binary = source_edges > 0
    normalised_binary = normalised_edges > 0

    source_count = np.count_nonzero(source_binary)
    normalised_count = np.count_nonzero(normalised_binary)

    # Dilate both maps to allow small positional differences
    source_dilated = cv2.dilate(
        source_binary.astype(np.uint8),
        kernel,
        iterations=tolerance
    ) > 0

    normalised_dilated = cv2.dilate(
        normalised_binary.astype(np.uint8),
        kernel,
        iterations=tolerance
    ) > 0

    # Normalised edges with nearby source edges
    matched_normalised = np.count_nonzero(
        normalised_binary & source_dilated
    )

    # Source edges with nearby normalised edges
    matched_source = np.count_nonzero(
        source_binary & normalised_dilated
    )

    precision = (
        matched_normalised / normalised_count
        if normalised_count > 0 else 0
    )

    recall = (
        matched_source / source_count
        if source_count > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    # Exact pixel overlap
    intersection = np.count_nonzero(
        source_binary & normalised_binary
    )

    union = np.count_nonzero(
        source_binary | normalised_binary
    )

    iou = intersection / union if union > 0 else 0

    return {
        "Source_Edge_Density": edge_density(source_edges),
        "Normalised_Edge_Density": edge_density(normalised_edges),
        "Edge_Precision_Agreement": precision,
        "Edge_Recall_Agreement": recall,
        "Edge_F1_Agreement": f1,
        "Exact_Edge_IoU": iou
    }


# ==========================================================
# RUN THRESHOLD EXPERIMENTS
# ==========================================================

all_results = []

print("\n" + "=" * 65)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 65)

for lower, upper in threshold_pairs:

    print(f"\nTesting thresholds: {lower}, {upper}")

    source_edges = cv2.Canny(
        source_gray,
        lower,
        upper
    )

    normalised_edges = cv2.Canny(
        normalised_gray,
        lower,
        upper
    )

    metrics = calculate_edge_agreement(
        source_edges,
        normalised_edges
    )

    result = {
        "Lower_Threshold": lower,
        "Upper_Threshold": upper
    }

    result.update(metrics)
    all_results.append(result)

    for metric, value in metrics.items():

        print(f"{metric}: {value:.6f}")

    # Save edge maps for each threshold
    cv2.imwrite(
        str(OUTPUT_DIR / f"source_edges_{lower}_{upper}.png"),
        source_edges
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"normalised_edges_{lower}_{upper}.png"),
        normalised_edges
    )


# ==========================================================
# SAVE RESULTS TO CSV
# ==========================================================

csv_path = OUTPUT_DIR / "spatial_edge_results.csv"

fieldnames = list(all_results[0].keys())

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_results)

print("\nCSV saved:", csv_path)


# ==========================================================
# VISUALISE EDGE OVERLAP
# ==========================================================

# Use 100 and 200 as the main comparison thresholds
source_edges = cv2.Canny(source_gray, 100, 200)
normalised_edges = cv2.Canny(normalised_gray, 100, 200)

source_binary = source_edges > 0
normalised_binary = normalised_edges > 0

# Create a colour overlay:
# Green = source-only edges
# Red = normalised-only edges
# Yellow = exact overlapping edges

overlay = np.zeros(
    (source.shape[0], source.shape[1], 3),
    dtype=np.uint8
)

overlay[source_binary & ~normalised_binary] = [0, 255, 0]
overlay[normalised_binary & ~source_binary] = [255, 0, 0]
overlay[source_binary & normalised_binary] = [255, 255, 0]


# ==========================================================
# PLOT RESULTS
# ==========================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(source_edges, cmap="gray")
axes[0, 0].set_title("Source Edge Map")

axes[0, 1].imshow(normalised_edges, cmap="gray")
axes[0, 1].set_title("Normalised Edge Map")

axes[1, 0].imshow(overlay)
axes[1, 0].set_title(
    "Edge Agreement Overlay\n"
    "Green: Source | Red: Normalised | Yellow: Exact overlap"
)

axes[1, 1].imshow(source_gray, cmap="gray")
axes[1, 1].set_title("Original Source Grayscale")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

comparison_path = OUTPUT_DIR / "spatial_edge_comparison.png"

plt.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# THRESHOLD COMPARISON GRAPH
# ==========================================================

threshold_labels = [
    f"{r['Lower_Threshold']}/{r['Upper_Threshold']}"
    for r in all_results
]

f1_values = [
    r["Edge_F1_Agreement"] * 100
    for r in all_results
]

plt.figure(figsize=(9, 5))

plt.plot(
    threshold_labels,
    f1_values,
    marker="o"
)

plt.xlabel("Canny Threshold Pair")
plt.ylabel("Edge F1 Agreement (%)")
plt.title("Spatial Edge Agreement Across Thresholds")
plt.grid(True)

plt.tight_layout()

graph_path = OUTPUT_DIR / "threshold_sensitivity.png"

plt.savefig(
    graph_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# COMPLETION
# ==========================================================

print("\n" + "=" * 65)
print("EXPERIMENT 9 COMPLETED")
print("=" * 65)

print("Results directory:", OUTPUT_DIR)
print("CSV:", csv_path)
print("Visual comparison:", comparison_path)
print("Threshold graph:", graph_path)
print("Tolerance:", tolerance, "pixels")
