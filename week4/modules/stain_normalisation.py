
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchstain

from pathlib import Path


# --------------------------------------------------
# 1. CONVERT NUMPY IMAGE TO PYTORCH TENSOR
# --------------------------------------------------

def convert_to_torch(image):

    """
    Convert an RGB NumPy image (HWC) into
    a PyTorch tensor (CHW).

    Pixel values remain in the 0-255 range.
    """

    tensor = torch.from_numpy(
        np.ascontiguousarray(image)
    )

    # HWC -> CHW
    tensor = tensor.permute(2, 0, 1)

    # Convert to float32
    tensor = tensor.float()

    return tensor


# --------------------------------------------------
# 2. CONVERT NORMALISED OUTPUT TO NUMPY
# --------------------------------------------------

def to_numpy_image(image):

    """
    Convert TorchStain output into an RGB
    NumPy image in HWC format.
    """

    # TorchStain may return multiple outputs
    if isinstance(image, tuple):
        image = image[0]

    # Convert PyTorch tensor to NumPy
    if torch.is_tensor(image):

        image = image.detach().cpu().numpy()

    image = np.asarray(image)

    # CHW -> HWC
    if image.ndim == 3 and image.shape[0] == 3:
        if image.shape[-1] != 3:

            image = np.transpose(
                image,
                (1, 2, 0)
            )

    return image


# --------------------------------------------------
# 3. CALCULATE RGB STATISTICS
# --------------------------------------------------

def calculate_rgb_statistics(image):

    """
    Calculate mean and standard deviation
    for each RGB channel.
    """

    channel_names = [
        "Red",
        "Green",
        "Blue"
    ]

    statistics = {}

    for index, name in enumerate(channel_names):

        channel = image[:, :, index]

        statistics[name] = {
            "mean": round(
                float(np.mean(channel)), 2
            ),

            "std": round(
                float(np.std(channel)), 2
            )
        }

    return statistics


# --------------------------------------------------
# 4. MACENKO STAIN NORMALISATION
# --------------------------------------------------

def normalise_stain(
    source_path,
    reference_path,
    output_dir="results/week4"
):

    """
    Perform reference-based Macenko stain
    normalisation using TorchStain.

    Parameters:
        source_path: Original microscopy image.
        reference_path: Reference microscopy image.
        output_dir: Directory for saving results.

    Returns:
        Normalised RGB NumPy image.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------
    # STEP 1: LOAD SOURCE AND REFERENCE IMAGES
    # ----------------------------------------------

    source = cv2.imread(str(source_path))

    reference = cv2.imread(str(reference_path))

    if source is None:

        raise FileNotFoundError(
            f"Source image not found: {source_path}"
        )

    if reference is None:

        raise FileNotFoundError(
            f"Reference image not found: {reference_path}"
        )

    # ----------------------------------------------
    # STEP 2: CONVERT BGR TO RGB
    # ----------------------------------------------

    source_rgb = cv2.cvtColor(
        source,
        cv2.COLOR_BGR2RGB
    )

    reference_rgb = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2RGB
    )

    source_rgb = np.ascontiguousarray(
        source_rgb.astype(np.uint8)
    )

    reference_rgb = np.ascontiguousarray(
        reference_rgb.astype(np.uint8)
    )

    print("=" * 55)
    print("MACENKO STAIN NORMALISATION")
    print("=" * 55)

    print("\nImage Information")

    print("Source image:", source_rgb.shape)

    print("Reference image:", reference_rgb.shape)

    # ----------------------------------------------
    # STEP 3: CONVERT IMAGES TO PYTORCH TENSORS
    # ----------------------------------------------

    source_tensor = convert_to_torch(
        source_rgb
    )

    reference_tensor = convert_to_torch(
        reference_rgb
    )

    print("\nPyTorch Tensor Information")

    print("Source tensor:", source_tensor.shape)

    print("Reference tensor:", reference_tensor.shape)

    print("Data type:", source_tensor.dtype)

    # ----------------------------------------------
    # STEP 4: INITIALISE MACENKO NORMALISER
    # ----------------------------------------------

    normalizer = torchstain.normalizers.MacenkoNormalizer(
        backend="torch"
    )

    # ----------------------------------------------
    # STEP 5: FIT USING REFERENCE IMAGE
    # ----------------------------------------------

    print("\nEstimating reference stain characteristics...")

    normalizer.fit(
        I=reference_tensor
    )

    print("Reference fitting successful!")

    # ----------------------------------------------
    # STEP 6: NORMALISE SOURCE IMAGE
    # ----------------------------------------------

    print("\nApplying Macenko normalisation...")

    result = normalizer.normalize(
        I=source_tensor,
        stains=True
    )

    # ----------------------------------------------
    # STEP 7: EXTRACT NORMALISED IMAGE
    # ----------------------------------------------

    normalized_rgb = to_numpy_image(result)

    normalized_rgb = np.clip(
        normalized_rgb,
        0,
        255
    ).astype(np.uint8)

    # Validate output dimensions
    if (
        normalized_rgb.ndim != 3
        or normalized_rgb.shape[2] != 3
    ):

        raise ValueError(
            "Invalid normalised image dimensions."
        )

    print("\nNormalisation successful!")

    print(
        "Normalised image shape:",
        normalized_rgb.shape
    )

    # ----------------------------------------------
    # STEP 8: SAVE NORMALISED IMAGE
    # ----------------------------------------------

    output_path = (
        output_dir / "03_macenko_normalised.png"
    )

    normalized_bgr = cv2.cvtColor(
        normalized_rgb,
        cv2.COLOR_RGB2BGR
    )

    success = cv2.imwrite(
        str(output_path),
        normalized_bgr
    )

    if not success:

        raise IOError(
            "Unable to save normalised image."
        )

    # ----------------------------------------------
    # STEP 9: CALCULATE RGB STATISTICS
    # ----------------------------------------------

    source_stats = calculate_rgb_statistics(
        source_rgb
    )

    reference_stats = calculate_rgb_statistics(
        reference_rgb
    )

    normalised_stats = calculate_rgb_statistics(
        normalized_rgb
    )

    print("\n" + "=" * 55)

    print("RGB STATISTICS COMPARISON")

    print("=" * 55)

    for channel in ["Red", "Green", "Blue"]:

        print(f"\n{channel} Channel")

        print(
            "Source Mean:",
            source_stats[channel]["mean"]
        )

        print(
            "Reference Mean:",
            reference_stats[channel]["mean"]
        )

        print(
            "Normalised Mean:",
            normalised_stats[channel]["mean"]
        )

        print(
            "Source Std:",
            source_stats[channel]["std"]
        )

        print(
            "Reference Std:",
            reference_stats[channel]["std"]
        )

        print(
            "Normalised Std:",
            normalised_stats[channel]["std"]
        )

    # ----------------------------------------------
    # STEP 10: GENERATE COMPARISON VISUALISATION
    # ----------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(source_rgb)

    axes[0].set_title(
        "Original Source Image"
    )

    axes[1].imshow(reference_rgb)

    axes[1].set_title(
        "Reference Image"
    )

    axes[2].imshow(normalized_rgb)

    axes[2].set_title(
        "Macenko Normalised Image"
    )

    for ax in axes:

        ax.axis("off")

    plt.tight_layout()

    comparison_path = (
        output_dir / "03_macenko_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ----------------------------------------------
    # STEP 11: SAVE NUMERICAL STATISTICS
    # ----------------------------------------------

    np.savez(
        output_dir / "03_macenko_statistics.npz",

        source=source_stats,

        reference=reference_stats,

        normalised=normalised_stats
    )

    # ----------------------------------------------
    # FINAL OUTPUT
    # ----------------------------------------------

    print("\n" + "=" * 55)

    print("MACENKO NORMALISATION COMPLETED")

    print("=" * 55)

    print("\nNormalised image saved:")
    print(output_path)

    print("\nComparison visualisation saved:")
    print(comparison_path)

    print("\nRGB statistics saved successfully.")

    return normalized_rgb


# --------------------------------------------------
# 5. STANDALONE EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    source_path = (
        "week1/images/microscopy_sample.png"
    )

    reference_path = (
        "week2/images/reference_blood_smear.jpg"
    )

    try:

        result = normalise_stain(
            source_path=source_path,

            reference_path=reference_path
        )

        print(
            "\nMacenko pipeline executed successfully!"
        )

    except Exception as error:

        print("\nNormalisation failed:")

        print(type(error).__name__, ":", error)

        raise
