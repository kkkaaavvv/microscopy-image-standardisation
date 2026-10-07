import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

source_path = BASE_DIR / "week1" / "images" / "microscopy_sample.png"
reference_path = BASE_DIR / "week2" / "images" / "reference_blood_smear.jpg"

# IMPORTANT:
# Use the RAW normalized images, NOT the saved visualization figures.
macenko_path = BASE_DIR / "results" / "week2" / "macenko_normalised_raw.png"
vahadane_path = BASE_DIR / "results" / "week2" / "vahadane_normalised_raw.png"

output_path = BASE_DIR / "results" / "week2" / "task11_method_comparison.png"


# ---------------------------------------------------------
# Load images
# ---------------------------------------------------------

source = cv2.imread(str(source_path))
reference = cv2.imread(str(reference_path))
macenko_result = cv2.imread(str(macenko_path))
vahadane_result = cv2.imread(str(vahadane_path))


if source is None:
    raise FileNotFoundError(
        f"Could not load source image: {source_path}"
    )

if reference is None:
    raise FileNotFoundError(
        f"Could not load reference image: {reference_path}"
    )

if macenko_result is None:
    raise FileNotFoundError(
        f"Could not load Macenko raw image: {macenko_path}"
    )

if vahadane_result is None:
    raise FileNotFoundError(
        f"Could not load Vahadane raw image: {vahadane_path}"
    )


print("All four images loaded successfully!")


# ---------------------------------------------------------
# Resize images to common dimensions
# ---------------------------------------------------------

target_size = (800, 800)

source = cv2.resize(
    source,
    target_size,
    interpolation=cv2.INTER_AREA
)

reference = cv2.resize(
    reference,
    target_size,
    interpolation=cv2.INTER_AREA
)

macenko_result = cv2.resize(
    macenko_result,
    target_size,
    interpolation=cv2.INTER_AREA
)

vahadane_result = cv2.resize(
    vahadane_result,
    target_size,
    interpolation=cv2.INTER_AREA
)


# ---------------------------------------------------------
# Calculate statistics
# ---------------------------------------------------------

def calculate_statistics(image):

    # OpenCV loads images as BGR
    b, g, r = cv2.split(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    statistics = {
        "Red Mean": np.mean(r),
        "Green Mean": np.mean(g),
        "Blue Mean": np.mean(b),

        "Red Std": np.std(r),
        "Green Std": np.std(g),
        "Blue Std": np.std(b),

        "Gray Mean": np.mean(gray),
        "Gray Std": np.std(gray),

        "Gray Min": np.min(gray),
        "Gray Max": np.max(gray),
    }

    return statistics


# ---------------------------------------------------------
# Store images
# ---------------------------------------------------------

images = {
    "Original Source": source,
    "Reference": reference,
    "Macenko": macenko_result,
    "Vahadane": vahadane_result
}


# ---------------------------------------------------------
# Calculate statistics for all images
# ---------------------------------------------------------

results = {}

for name, image in images.items():
    results[name] = calculate_statistics(image)


# ---------------------------------------------------------
# Print statistics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TASK 11 — QUANTITATIVE METHOD COMPARISON")
print("=" * 70)

for name, stats in results.items():

    print(f"\n{name}")
    print("-" * 40)

    for key, value in stats.items():
        print(f"{key:15s}: {value:.2f}")


# ---------------------------------------------------------
# Calculate distance from reference
# ---------------------------------------------------------

reference_stats = results["Reference"]

comparison_methods = [
    "Original Source",
    "Macenko",
    "Vahadane"
]


print("\n" + "=" * 70)
print("DISTANCE FROM REFERENCE")
print("=" * 70)


distance_results = {}


for method in comparison_methods:

    mean_distance = np.sqrt(
        (results[method]["Red Mean"]
         - reference_stats["Red Mean"]) ** 2 +

        (results[method]["Green Mean"]
         - reference_stats["Green Mean"]) ** 2 +

        (results[method]["Blue Mean"]
         - reference_stats["Blue Mean"]) ** 2
    )

    std_distance = np.sqrt(
        (results[method]["Red Std"]
         - reference_stats["Red Std"]) ** 2 +

        (results[method]["Green Std"]
         - reference_stats["Green Std"]) ** 2 +

        (results[method]["Blue Std"]
         - reference_stats["Blue Std"]) ** 2
    )

    distance_results[method] = {
        "RGB Mean Distance": mean_distance,
        "RGB Std Distance": std_distance
    }

    print(f"\n{method}")
    print(
        f"RGB mean distance: {mean_distance:.2f}"
    )
    print(
        f"RGB std distance : {std_distance:.2f}"
    )


# ---------------------------------------------------------
# Visual comparison
# ---------------------------------------------------------

fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5)
)


display_images = [
    ("Original", source),
    ("Reference", reference),
    ("Macenko", macenko_result),
    ("Vahadane", vahadane_result)
]


for ax, (title, image) in zip(
    axes,
    display_images
):

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    ax.imshow(rgb_image)
    ax.set_title(title)
    ax.axis("off")


plt.suptitle(
    "Quantitative Comparison of Stain Normalisation Methods",
    fontsize=16
)

plt.tight_layout()


output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# Final output
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TASK 11 COMPLETED")
print("=" * 70)

print("\nComparison figure saved to:")
print(output_path)