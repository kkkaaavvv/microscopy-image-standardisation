
import cv2
import numpy as np
import csv
from pathlib import Path

# ==========================================================
# EXPERIMENT 6: QUANTITATIVE NORMALISATION EVALUATION
# ==========================================================

print("=" * 60)
print("QUANTITATIVE NORMALISATION EVALUATION")
print("=" * 60)

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Image paths
SOURCE_PATH = PROJECT_ROOT / "week1" / "images" / "microscopy_sample.png"

REFERENCE_ROOT = Path(r"D:\Blood Smear Dataset")

reference_matches = list(REFERENCE_ROOT.rglob("patient_05_12.jpg"))

if not reference_matches:
    raise FileNotFoundError(
        "Could not find patient_05_12.jpg inside D:\\Blood Smear Dataset"
    )

REFERENCE_PATH = reference_matches[0]

NORMALISED_PATH = (
    PROJECT_ROOT
    / "results"
    / "week4"
    / "stain_components"
    / "normalised_image.png"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "week4" / "normalisation_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# LOAD IMAGE
# ==========================================================

def load_image(path):

    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


source = load_image(SOURCE_PATH)
reference = load_image(REFERENCE_PATH)
normalised = load_image(NORMALISED_PATH)

print("\nImages loaded successfully!")

print("Source shape:", source.shape)
print("Reference shape:", reference.shape)
print("Normalised shape:", normalised.shape)


# ==========================================================
# IMAGE STATISTICS
# ==========================================================

def calculate_statistics(image):

    image_float = image.astype(np.float32)

    # RGB statistics
    rgb_mean = np.mean(image_float, axis=(0, 1))
    rgb_std = np.std(image_float, axis=(0, 1))

    # Convert RGB to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    lab_mean = np.mean(lab_image, axis=(0, 1))
    lab_std = np.std(lab_image, axis=(0, 1))

    # Optical Density
    safe_image = np.clip(image_float, 1, 255)

    optical_density = -np.log(safe_image / 255.0)

    od_mean = np.mean(optical_density)
    od_std = np.std(optical_density)

    # Grayscale and Laplacian variance
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return {
        "RGB_R_Mean": rgb_mean[0],
        "RGB_G_Mean": rgb_mean[1],
        "RGB_B_Mean": rgb_mean[2],

        "RGB_R_Std": rgb_std[0],
        "RGB_G_Std": rgb_std[1],
        "RGB_B_Std": rgb_std[2],

        "LAB_L_Mean": lab_mean[0],
        "LAB_A_Mean": lab_mean[1],
        "LAB_B_Mean": lab_mean[2],

        "LAB_L_Std": lab_std[0],
        "LAB_A_Std": lab_std[1],
        "LAB_B_Std": lab_std[2],

        "OD_Mean": od_mean,
        "OD_Std": od_std,

        "Laplacian_Variance": sharpness
    }


# ==========================================================
# ANALYSE ALL THREE IMAGES
# ==========================================================

images = {
    "Source": source,
    "Reference": reference,
    "Normalised": normalised
}

results = {}

print("\n" + "=" * 60)
print("CALCULATING IMAGE STATISTICS")
print("=" * 60)

for name, image in images.items():

    print(f"\nAnalysing {name} image...")

    results[name] = calculate_statistics(image)

    print("Analysis completed!")


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)

for name, stats in results.items():

    print(f"\n{name.upper()} IMAGE")

    for metric, value in stats.items():

        print(f"{metric}: {value:.4f}")


# ==========================================================
# SAVE RESULTS TO CSV
# ==========================================================

csv_path = OUTPUT_DIR / "normalisation_evaluation.csv"

fieldnames = ["Image"] + list(results["Source"].keys())

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for name, stats in results.items():

        row = {"Image": name}
        row.update(stats)

        writer.writerow(row)


print("\nCSV saved:", csv_path)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("EXPERIMENT 6 COMPLETED")
print("=" * 60)

print("Source image:", SOURCE_PATH)
print("Reference image:", REFERENCE_PATH)
print("Normalised image:", NORMALISED_PATH)

print("Results saved in:", OUTPUT_DIR)
