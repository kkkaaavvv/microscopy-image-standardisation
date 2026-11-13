
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# WEEK 10 - TASK 3
# COMPARATIVE ANALYSIS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "week10" / "images"
RESULT_DIR = PROJECT_ROOT / "results" / "week10"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. IMAGE LOADING
# ==========================================

def load_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to load image: {image_path}"
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ==========================================
# 2. LAB STATISTICS
# ==========================================

def calculate_lab_statistics(image):

    lab_image = cv2.cvtColor(
        image, cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    mean = np.mean(lab_image, axis=(0, 1))
    std = np.std(lab_image, axis=(0, 1))

    return mean, std


# ==========================================
# 3. THREE-PANEL IMAGE COMPARISON
# ==========================================

def generate_image_comparison(
    source, reference, normalised
):

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 6)
    )

    images = [
        source,
        reference,
        normalised
    ]

    titles = [
        "Source Image",
        "Reference Image",
        "Normalised Image"
    ]

    for ax, image, title in zip(
        axes, images, titles
    ):

        ax.imshow(image)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    plt.suptitle(
        "Microscopy Image Colour Normalisation",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "three_panel_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================
# 4. STATISTICAL COMPARISON
# ==========================================

def generate_statistics_comparison(
    source_stats,
    reference_stats,
    normalised_stats
):

    source_mean, source_std = source_stats
    reference_mean, reference_std = reference_stats
    normalised_mean, normalised_std = normalised_stats

    channels = ["L", "A", "B"]
    x = np.arange(len(channels))

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5)
    )

    # Mean comparison
    width = 0.25

    axes[0].bar(
        x - width,
        source_mean,
        width,
        label="Source"
    )

    axes[0].bar(
        x,
        reference_mean,
        width,
        label="Reference"
    )

    axes[0].bar(
        x + width,
        normalised_mean,
        width,
        label="Normalised"
    )

    axes[0].set_title("LAB Mean Comparison")
    axes[0].set_xlabel("LAB Channels")
    axes[0].set_ylabel("Mean Intensity")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channels)
    axes[0].legend()

    # Standard deviation comparison
    axes[1].bar(
        x - width,
        source_std,
        width,
        label="Source"
    )

    axes[1].bar(
        x,
        reference_std,
        width,
        label="Reference"
    )

    axes[1].bar(
        x + width,
        normalised_std,
        width,
        label="Normalised"
    )

    axes[1].set_title("LAB Standard Deviation Comparison")
    axes[1].set_xlabel("LAB Channels")
    axes[1].set_ylabel("Standard Deviation")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channels)
    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "lab_statistics_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================
# 5. PRINT STATISTICAL RESULTS
# ==========================================

def print_comparison(
    source_stats,
    reference_stats,
    normalised_stats
):

    source_mean, source_std = source_stats
    reference_mean, reference_std = reference_stats
    normalised_mean, normalised_std = normalised_stats

    channels = ["L", "A", "B"]

    print("\nLAB STATISTICAL COMPARISON")
    print("=" * 75)

    print(
        f"{'Channel':<10}"
        f"{'Source Mean':<15}"
        f"{'Reference Mean':<18}"
        f"{'Normalised Mean':<18}"
    )

    print("-" * 75)

    for i, channel in enumerate(channels):

        print(
            f"{channel:<10}"
            f"{source_mean[i]:<15.2f}"
            f"{reference_mean[i]:<18.2f}"
            f"{normalised_mean[i]:<18.2f}"
        )

    print("\nSTANDARD DEVIATION COMPARISON")
    print("-" * 75)

    print(
        f"{'Channel':<10}"
        f"{'Source Std':<15}"
        f"{'Reference Std':<18}"
        f"{'Normalised Std':<18}"
    )

    print("-" * 75)

    for i, channel in enumerate(channels):

        print(
            f"{channel:<10}"
            f"{source_std[i]:<15.2f}"
            f"{reference_std[i]:<18.2f}"
            f"{normalised_std[i]:<18.2f}"
        )

    print("\nABSOLUTE DIFFERENCE FROM REFERENCE MEAN")
    print("-" * 50)

    for i, channel in enumerate(channels):

        before = abs(
            source_mean[i] - reference_mean[i]
        )

        after = abs(
            normalised_mean[i] - reference_mean[i]
        )

        print(f"\n{channel} Channel")
        print(f"Before normalisation: {before:.4f}")
        print(f"After normalisation:  {after:.4f}")

    print("\nABSOLUTE DIFFERENCE FROM REFERENCE STD")
    print("-" * 50)

    for i, channel in enumerate(channels):

        before = abs(
            source_std[i] - reference_std[i]
        )

        after = abs(
            normalised_std[i] - reference_std[i]
        )

        print(f"\n{channel} Channel")
        print(f"Before normalisation: {before:.4f}")
        print(f"After normalisation:  {after:.4f}")


# ==========================================
# 6. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("WEEK 10 - COMPARATIVE ANALYSIS")
    print("=" * 60)

    # Load all three images
    source_image = load_image(
        IMAGE_DIR / "source_image.png"
    )

    reference_image = load_image(
        IMAGE_DIR / "reference_image.png"
    )

    normalised_image = load_image(
        RESULT_DIR / "normalised_image.png"
    )

    print("\nAll three images loaded successfully.")

    # Calculate LAB statistics
    source_stats = calculate_lab_statistics(
        source_image
    )

    reference_stats = calculate_lab_statistics(
        reference_image
    )

    normalised_stats = calculate_lab_statistics(
        normalised_image
    )

    # Generate visual comparison
    generate_image_comparison(
        source_image,
        reference_image,
        normalised_image
    )

    # Generate statistical charts
    generate_statistics_comparison(
        source_stats,
        reference_stats,
        normalised_stats
    )

    # Print statistical comparison
    print_comparison(
        source_stats,
        reference_stats,
        normalised_stats
    )

    print("\n" + "=" * 60)
    print("TASK 3 COMPARATIVE ANALYSIS COMPLETED")
    print("=" * 60)

    print(
        "\nThree-panel comparison saved successfully."
    )

    print(
        "LAB statistical comparison saved successfully."
    )

