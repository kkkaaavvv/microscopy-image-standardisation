
import cv2
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from week4.modules.stain_normalisation import (
    normalise_stain
)


# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================

DATASET_DIR = Path(r"D:\Blood Smear Dataset")

SOURCE_PATH = Path(
    "week1/images/microscopy_sample.png"
)

OUTPUT_DIR = Path(
    "results/week4/final_reference_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Three shortlisted references + one outlier
REFERENCES = [
    {
        "name": "patient_05_01",
        "patient": "patient_05",
        "filename": "patient_05_01.jpg",
        "category": "Primary candidate"
    },
    {
        "name": "patient_05_12",
        "patient": "patient_05",
        "filename": "patient_05_12.jpg",
        "category": "Alternative"
    },
    {
        "name": "patient_05_06",
        "patient": "patient_05",
        "filename": "patient_05_06.jpg",
        "category": "Alternative"
    },
    {
        "name": "patient_05_14",
        "patient": "patient_05",
        "filename": "patient_05_14.jpg",
        "category": "Outlier investigation"
    }
]


# ==========================================
# STEP 1: IMAGE STATISTICS
# ==========================================

def calculate_statistics(image):

    rgb = image.astype(np.float32)

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    # Optical Density
    rgb_clipped = np.clip(
        rgb,
        1,
        255
    )

    optical_density = -np.log(
        rgb_clipped / 255.0
    )

    statistics = {}

    channel_names = [
        "Red", "Green", "Blue"
    ]

    # RGB statistics
    for index, channel in enumerate(channel_names):

        statistics[f"{channel}_mean"] = round(
            float(np.mean(rgb[:, :, index])), 3
        )

        statistics[f"{channel}_std"] = round(
            float(np.std(rgb[:, :, index])), 3
        )

    # LAB statistics
    for index, channel in enumerate(["L", "A", "B"]):

        statistics[f"LAB_{channel}_mean"] = round(
            float(np.mean(lab[:, :, index])), 3
        )

        statistics[f"LAB_{channel}_std"] = round(
            float(np.std(lab[:, :, index])), 3
        )

    # Optical Density statistics
    statistics["OD_mean"] = round(
        float(np.mean(optical_density)), 5
    )

    statistics["OD_std"] = round(
        float(np.std(optical_density)), 5
    )

    return statistics


# ==========================================
# STEP 2: LOAD SOURCE IMAGE
# ==========================================

def load_source():

    source = cv2.imread(
        str(SOURCE_PATH)
    )

    if source is None:

        raise FileNotFoundError(
            f"Source image not found: {SOURCE_PATH}"
        )

    source_rgb = cv2.cvtColor(
        source,
        cv2.COLOR_BGR2RGB
    )

    return source_rgb


# ==========================================
# STEP 3: RUN REFERENCE EXPERIMENTS
# ==========================================

def run_experiments(source_rgb):

    results = []

    for candidate in REFERENCES:

        name = candidate["name"]

        reference_path = (
            DATASET_DIR
            / candidate["patient"]
            / candidate["filename"]
        )

        experiment_dir = OUTPUT_DIR / name

        experiment_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print("\n" + "=" * 60)

        print("REFERENCE:", name)

        print("Category:", candidate["category"])

        print("=" * 60)

        if not reference_path.exists():

            print(
                "Reference image missing:",
                reference_path
            )

            continue

        try:

            # Apply Macenko normalisation
            normalized = normalise_stain(
                source_path=SOURCE_PATH,
                reference_path=reference_path,
                output_dir=experiment_dir
            )

            # Load reference image
            reference_bgr = cv2.imread(
                str(reference_path)
            )

            reference_rgb = cv2.cvtColor(
                reference_bgr,
                cv2.COLOR_BGR2RGB
            )

            # Calculate statistics
            normalized_stats = calculate_statistics(
                normalized
            )

            reference_stats = calculate_statistics(
                reference_rgb
            )

            # Save numerical statistics
            results.append({

                "reference": name,

                "category": candidate["category"],

                "status": "Success",

                "normalized_stats": normalized_stats,

                "reference_stats": reference_stats,

                "normalized_image": normalized,

                "reference_image": reference_rgb

            })

            print(
                f"\n{name} completed successfully!"
            )

        except Exception as error:

            print(
                f"\n{name} failed:"
            )

            print(
                type(error).__name__,
                error
            )

            results.append({

                "reference": name,

                "category": candidate["category"],

                "status": "Failed",

                "normalized_stats": None,

                "reference_stats": None,

                "normalized_image": None,

                "reference_image": None

            })

    return results


# ==========================================
# STEP 4: SAVE STATISTICS TO CSV
# ==========================================

def save_statistics(results):

    output_path = (
        OUTPUT_DIR / "final_comparison_statistics.csv"
    )

    rows = []

    for result in results:

        if result["normalized_stats"] is None:
            continue

        row = {
            "reference": result["reference"],
            "category": result["category"],
            "status": result["status"]
        }

        for key, value in result["normalized_stats"].items():

            row[f"normalized_{key}"] = value

        for key, value in result["reference_stats"].items():

            row[f"reference_{key}"] = value

        rows.append(row)

    if not rows:

        print("No successful results to save.")
        return

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()

        writer.writerows(rows)

    print("\nStatistics saved:", output_path)


# ==========================================
# STEP 5: GENERATE COMPARISON GRID
# ==========================================

def generate_comparison(source_rgb, results):

    successful = [
        result for result in results
        if result["status"] == "Success"
    ]

    if not successful:
        print("No successful images to compare.")
        return

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10)
    )

    axes = axes.flatten()

    # Original source
    axes[0].imshow(source_rgb)

    axes[0].set_title(
        "Original Source Image"
    )

    axes[0].axis("off")

    # Normalised outputs
    for index, result in enumerate(
        successful,
        start=1
    ):

        axes[index].imshow(
            result["normalized_image"]
        )

        axes[index].set_title(
            result["reference"]
        )

        axes[index].axis("off")

    # Hide unused panels
    for index in range(
        len(successful) + 1,
        len(axes)
    ):

        axes[index].axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / "final_reference_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nComparison image saved:",
        output_path
    )


# ==========================================
# STEP 6: FINAL SUMMARY
# ==========================================

def print_summary(results):

    successful = [
        result for result in results
        if result["status"] == "Success"
    ]

    failed = len(results) - len(successful)

    print("\n" + "=" * 60)

    print("FINAL REFERENCE COMPARISON SUMMARY")

    print("=" * 60)

    print("Total experiments:", len(results))

    print("Successful:", len(successful))

    print("Failed:", failed)

    for result in results:

        print(
            result["reference"],
            "->",
            result["status"]
        )

    print("\nOutput directory:", OUTPUT_DIR)


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("=" * 60)

    print("FINAL MACENKO REFERENCE COMPARISON")

    print("=" * 60)

    source_rgb = load_source()

    print("\nSource image:", source_rgb.shape)

    results = run_experiments(source_rgb)

    save_statistics(results)

    generate_comparison(
        source_rgb,
        results
    )

    print_summary(results)

    print("\nEXPERIMENT COMPLETED!")
