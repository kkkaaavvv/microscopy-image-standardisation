
import cv2
import csv
import numpy as np
import torch
import torchstain

from pathlib import Path

# ==========================================================
# EXPERIMENT 12: END-TO-END STANDARDISATION PIPELINE
# ==========================================================

print("=" * 70)
print("END-TO-END MICROSCOPY IMAGE STANDARDISATION PIPELINE")
print("=" * 70)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = Path(r"D:\Blood Smear Dataset")

REFERENCE_PATH = (
    DATASET_DIR / "patient_05" / "patient_05_12.jpg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "week4"
    / "final_pipeline"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = [
    "patient_01",
    "patient_02",
    "patient_03",
    "patient_04",
    "patient_05"
]

IMAGES_PER_PATIENT = 3


# ==========================================================
# IMAGE LOADING AND PREPROCESSING
# ==========================================================

def load_rgb_image(path):

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not load image: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_tensor(image):

    tensor = torch.from_numpy(
        np.ascontiguousarray(image)
    )

    return tensor.permute(2, 0, 1).contiguous().byte()


# ==========================================================
# IMAGE QUALITY STATISTICS
# ==========================================================

def calculate_statistics(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    image_float = image.astype(np.float32)

    # Optical density
    od = -np.log(
        np.clip(image_float, 1, 255) / 255.0
    )

    return {
        "Height": image.shape[0],
        "Width": image.shape[1],

        "RGB_R_Mean": float(np.mean(image[:, :, 0])),
        "RGB_G_Mean": float(np.mean(image[:, :, 1])),
        "RGB_B_Mean": float(np.mean(image[:, :, 2])),

        "RGB_R_Std": float(np.std(image[:, :, 0])),
        "RGB_G_Std": float(np.std(image[:, :, 1])),
        "RGB_B_Std": float(np.std(image[:, :, 2])),

        "LAB_L_Mean": float(np.mean(lab[:, :, 0])),
        "LAB_A_Mean": float(np.mean(lab[:, :, 1])),
        "LAB_B_Mean": float(np.mean(lab[:, :, 2])),

        "OD_Mean": float(np.mean(od)),
        "OD_Std": float(np.std(od)),

        "Laplacian_Variance": float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )
    }


# ==========================================================
# VALIDATE REFERENCE IMAGE
# ==========================================================

print("\nSTEP 1: Loading reference image...")

reference = load_rgb_image(REFERENCE_PATH)

reference_tensor = convert_to_tensor(reference)

print("Reference:", REFERENCE_PATH)
print("Reference shape:", reference.shape)


# ==========================================================
# INITIALISE MACENKO NORMALISER
# ==========================================================

print("\nSTEP 2: Fitting Macenko normaliser...")

normalizer = torchstain.normalizers.MacenkoNormalizer(
    backend="torch"
)

normalizer.fit(I=reference_tensor)

print("Macenko reference fitting successful!")


# ==========================================================
# SELECT TEST IMAGES
# ==========================================================

print("\nSTEP 3: Selecting test images...")

test_images = []

for patient in PATIENTS:

    patient_dir = DATASET_DIR / patient

    if not patient_dir.exists():
        print("WARNING: Missing patient folder:", patient)
        continue

    image_paths = sorted(
        list(patient_dir.glob("*.jpg"))
        + list(patient_dir.glob("*.jpeg"))
        + list(patient_dir.glob("*.png"))
    )

    selected = []

    for path in image_paths:

        if path.resolve() == REFERENCE_PATH.resolve():
            continue

        selected.append(path)

        if len(selected) == IMAGES_PER_PATIENT:
            break

    for path in selected:
        test_images.append((patient, path))

    print(f"{patient}: {len(selected)} images selected")


if not test_images:
    raise RuntimeError("No test images were found.")


# ==========================================================
# PROCESS EACH IMAGE
# ==========================================================

print("\nSTEP 4: Running standardisation pipeline...")

summary_rows = []
successful = 0
failed = 0

for index, (patient, image_path) in enumerate(
    test_images,
    start=1
):

    print("\n" + "-" * 60)
    print(f"Processing image {index}/{len(test_images)}")
    print("Patient:", patient)
    print("Image:", image_path.name)

    try:

        # Load image
        source = load_rgb_image(image_path)

        # Convert to tensor
        source_tensor = convert_to_tensor(source)

        # Calculate original statistics
        source_stats = calculate_statistics(source)

        # Apply Macenko normalisation
        result = normalizer.normalize(
            I=source_tensor,
            stains=False
        )

        normalised_tensor = result[0]

        # Convert output tensor to RGB NumPy image
        normalised = (
            normalised_tensor
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        # Validate dimensions
        if normalised.shape != source.shape:
            raise ValueError(
                "Normalised image dimensions do not match source."
            )

        # Calculate normalised statistics
        normalised_stats = calculate_statistics(normalised)

        # Create patient output directory
        patient_output = OUTPUT_DIR / patient

        patient_output.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            patient_output
            / f"{image_path.stem}_normalised.png"
        )

        # Save standardised image
        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(
                normalised,
                cv2.COLOR_RGB2BGR
            )
        )

        # Store summary
        row = {
            "Patient": patient,
            "Input_Image": image_path.name,
            "Output_Image": str(output_path),
            "Status": "SUCCESS",
        }

        for key, value in source_stats.items():
            row[f"Source_{key}"] = value

        for key, value in normalised_stats.items():
            row[f"Normalised_{key}"] = value

        summary_rows.append(row)

        successful += 1

        print("Source shape:", source.shape)
        print("Output shape:", normalised.shape)
        print("Output saved:", output_path)
        print("Status: SUCCESS")

    except Exception as error:

        failed += 1

        print("Status: FAILED")
        print("Error:", error)

        summary_rows.append({
            "Patient": patient,
            "Input_Image": image_path.name,
            "Output_Image": "",
            "Status": f"FAILED: {error}"
        })


# ==========================================================
# SAVE FINAL VALIDATION REPORT
# ==========================================================

print("\nSTEP 5: Saving validation report...")

csv_path = OUTPUT_DIR / "final_pipeline_validation.csv"

if summary_rows:

    fieldnames = list(
        dict.fromkeys(
            key
            for row in summary_rows
            for key in row.keys()
        )
    )

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(summary_rows)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("FINAL PIPELINE VALIDATION SUMMARY")
print("=" * 70)

print("Total selected images:", len(test_images))
print("Successful:", successful)
print("Failed:", failed)

if test_images:
    print(
        f"Success rate: {successful / len(test_images) * 100:.2f}%"
    )

print("\nReference image:", REFERENCE_PATH)
print("Validation report:", csv_path)
print("Output directory:", OUTPUT_DIR)

print("\n" + "=" * 70)
print("EXPERIMENT 12 COMPLETED")
print("=" * 70)
