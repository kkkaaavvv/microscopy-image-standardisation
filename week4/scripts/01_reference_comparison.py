
import cv2
import csv
import sys
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# PROJECT PATH CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# IMPORT EXISTING MACENKO MODULE
# ==========================================

from week4.modules.stain_normalisation import (
    normalise_stain,
    calculate_rgb_statistics
)


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_DIR = Path(r"D:\Blood Smear Dataset")

SOURCE_PATH = Path(
    "week1/images/microscopy_sample.png"
)

OUTPUT_DIR = Path(
    "results/week4/reference_experiment"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PATIENTS = [
    "patient_01",
    "patient_02",
    "patient_03",
    "patient_04",
    "patient_05"
]


# ==========================================
# STEP 1: LOAD SOURCE IMAGE
# ==========================================

source = cv2.imread(str(SOURCE_PATH))

if source is None:
    raise FileNotFoundError(
        f"Source image not found: {SOURCE_PATH}"
    )

source_rgb = cv2.cvtColor(
    source,
    cv2.COLOR_BGR2RGB
)

print("=" * 60)
print("MACENKO REFERENCE COMPARISON EXPERIMENT")
print("=" * 60)

print("\nSource image:", SOURCE_PATH)
print("Source dimensions:", source_rgb.shape)


# ==========================================
# STEP 2: SELECT INITIAL REFERENCE CANDIDATES
# ==========================================

reference_candidates = []

for patient in PATIENTS:

    patient_folder = DATASET_DIR / patient

    reference_path = (
        patient_folder / f"{patient}_01.jpg"
    )

    if not reference_path.exists():

        print(
            f"\nReference missing for {patient}: "
            f"{reference_path}"
        )

        continue

    reference_candidates.append({
        "patient": patient,
        "path": reference_path
    })


if not reference_candidates:

    raise FileNotFoundError(
        "No reference candidates were found."
    )


print("\nReference candidates found:")

for candidate in reference_candidates:

    print(
        candidate["patient"],
        "->",
        candidate["path"].name
    )


# ==========================================
# STEP 3: RUN MACENKO FOR EACH REFERENCE
# ==========================================

experiment_results = []

for candidate in reference_candidates:

    patient = candidate["patient"]

    reference_path = candidate["path"]

    experiment_dir = OUTPUT_DIR / patient

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 60)

    print("CURRENT EXPERIMENT:", patient)

    print("Reference:", reference_path.name)

    print("=" * 60)

    try:

        normalized_image = normalise_stain(
            source_path=SOURCE_PATH,
            reference_path=reference_path,
            output_dir=experiment_dir
        )

        # Calculate statistics
        statistics = calculate_rgb_statistics(
            normalized_image
        )

        # Load reference for visual comparison
        reference_bgr = cv2.imread(
            str(reference_path)
        )

        reference_rgb = cv2.cvtColor(
            reference_bgr,
            cv2.COLOR_BGR2RGB
        )

        experiment_results.append({

            "patient": patient,

            "reference": reference_path.name,

            "normalized": normalized_image,

            "reference_rgb": reference_rgb,

            "statistics": statistics,

            "status": "Success"

        })

        print(
            f"\n{patient} experiment completed!"
        )

    except Exception as error:

        print(
            f"\n{patient} experiment failed:"
        )

        print(type(error).__name__, error)

        experiment_results.append({

            "patient": patient,

            "reference": reference_path.name,

            "normalized": None,

            "reference_rgb": None,

            "statistics": None,

            "status": "Failed"

        })


# ==========================================
# STEP 4: SAVE STATISTICS
# ==========================================

csv_path = OUTPUT_DIR / "comparison_statistics.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Patient",
        "Reference",
        "Status",
        "Red Mean",
        "Green Mean",
        "Blue Mean",
        "Red Std",
        "Green Std",
        "Blue Std"
    ])

    for result in experiment_results:

        if result["statistics"] is None:

            writer.writerow([
                result["patient"],
                result["reference"],
                result["status"],
                "", "", "", "", "", ""
            ])

            continue

        stats = result["statistics"]

        writer.writerow([
            result["patient"],
            result["reference"],
            result["status"],

            stats["Red"]["mean"],
            stats["Green"]["mean"],
            stats["Blue"]["mean"],

            stats["Red"]["std"],
            stats["Green"]["std"],
            stats["Blue"]["std"]
        ])


# ==========================================
# STEP 5: GENERATE COMPARISON GRID
# ==========================================

successful_results = [
    result for result in experiment_results
    if result["status"] == "Success"
]

if successful_results:

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10)
    )

    axes = axes.flatten()

    # Original source image
    axes[0].imshow(source_rgb)

    axes[0].set_title(
        "Original Source Image"
    )

    axes[0].axis("off")

    # Normalised outputs
    for index, result in enumerate(
        successful_results,
        start=1
    ):

        axes[index].imshow(
            result["normalized"]
        )

        axes[index].set_title(
            f"{result['patient']} Normalised"
        )

        axes[index].axis("off")

    # Hide unused panels
    for index in range(
        len(successful_results) + 1,
        6
    ):

        axes[index].axis("off")

    plt.tight_layout()

    comparison_path = (
        OUTPUT_DIR / "five_reference_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nComparison grid saved:",
        comparison_path
    )


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)

print("REFERENCE COMPARISON COMPLETED")

print("=" * 60)

print(
    "Successful experiments:",
    len(successful_results)
)

print(
    "Failed experiments:",
    len(experiment_results) - len(successful_results)
)

print("\nStatistics saved:", csv_path)

print("\nExperiment directory:", OUTPUT_DIR)
