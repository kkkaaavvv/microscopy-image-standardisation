
import cv2
import numpy as np
from pathlib import Path


# ==========================================
# WEEK 10 - TASK 2
# REINHARD COLOUR NORMALISATION
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
# 2. STATISTICS CALCULATION
# ==========================================

def calculate_statistics(image):

    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))

    return mean, std


# ==========================================
# 3. REINHARD NORMALISATION
# ==========================================

def reinhard_normalisation(source, reference):

    # Convert RGB images to LAB
    source_lab = cv2.cvtColor(
        source, cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    reference_lab = cv2.cvtColor(
        reference, cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    # Calculate source statistics
    source_mean, source_std = calculate_statistics(
        source_lab
    )

    # Calculate reference statistics
    reference_mean, reference_std = calculate_statistics(
        reference_lab
    )

    # Prevent division by zero
    source_std = np.maximum(source_std, 1e-6)

    # Apply Reinhard transformation
    normalised_lab = (
        (source_lab - source_mean)
        * (reference_std / source_std)
        + reference_mean
    )

    # Keep pixel values within valid LAB range
    normalised_lab = np.clip(
        normalised_lab, 0, 255
    ).astype(np.uint8)

    # Convert LAB back to RGB
    normalised_rgb = cv2.cvtColor(
        normalised_lab, cv2.COLOR_LAB2RGB
    )

    return normalised_rgb


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("=" * 50)
    print("REINHARD COLOUR NORMALISATION")
    print("=" * 50)

    # Load images
    source_image = load_image(
        IMAGE_DIR / "source_image.png"
    )

    reference_image = load_image(
        IMAGE_DIR / "reference_image.png"
    )

    print("\nImages loaded successfully.")

    # Perform normalisation
    normalised_image = reinhard_normalisation(
        source_image,
        reference_image
    )

    # Save output
    output_path = RESULT_DIR / "normalised_image.png"

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            normalised_image,
            cv2.COLOR_RGB2BGR
        )
    )

    print("\nNormalisation completed!")
    print("Source dimensions:", source_image.shape)
    print("Reference dimensions:", reference_image.shape)
    print("Output dimensions:", normalised_image.shape)
    print("Saved output:", output_path)

    # Calculate output LAB statistics
    normalised_lab = cv2.cvtColor(
        normalised_image,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    normalised_mean, normalised_std = calculate_statistics(
        normalised_lab
    )

    print("\nNORMALISED IMAGE LAB STATISTICS")
    print("-" * 40)

    for i, channel in enumerate(["L", "A", "B"]):

        print(f"{channel} Channel")
        print(f"Mean: {normalised_mean[i]:.4f}")
        print(f"Standard Deviation: {normalised_std[i]:.4f}")
        print()

    print("=" * 50)
    print("IMPLEMENTATION COMPLETED")
    print("=" * 50)
