
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def calculate_optical_density(image, output_dir="results/week4"):

    """
    Convert a microscopy RGB image into Optical Density.

    Returns:
        od_image: Three-channel optical density array.
        od_statistics: Optical density statistics.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate input
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel colour image.")

    if image.dtype != np.uint8:
        raise ValueError("This implementation expects uint8 input.")

    # 2. Convert BGR to RGB
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # 3. Convert pixel values to floating point
    rgb_float = rgb_image.astype(np.float32)

    # 4. Avoid log(0)
    rgb_float = np.clip(rgb_float, 1, 255)

    # 5. Calculate Optical Density
    I0 = 255.0

    od_image = -np.log(rgb_float / I0)

    # 6. Calculate OD statistics
    channel_names = ["Red", "Green", "Blue"]

    od_statistics = {}

    for index, name in enumerate(channel_names):

        channel = od_image[:, :, index]

        od_statistics[name] = {
            "mean": round(float(np.mean(channel)), 4),
            "std": round(float(np.std(channel)), 4),
            "minimum": round(float(np.min(channel)), 4),
            "maximum": round(float(np.max(channel)), 4)
        }

    # 7. Print statistics
    print("=" * 50)
    print("OPTICAL DENSITY ANALYSIS")
    print("=" * 50)

    for channel, values in od_statistics.items():

        print(f"\n{channel} Channel")

        for metric, value in values.items():
            print(f"{metric}: {value}")

    # 8. Visualise OD channels
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    axes[0].imshow(rgb_image)
    axes[0].set_title("Original RGB Image")
    axes[0].axis("off")

    for index, name in enumerate(channel_names):

        axes[index + 1].imshow(
            od_image[:, :, index],
            cmap="gray",
            vmin=0,
            vmax=5.6
        )

        axes[index + 1].set_title(
            f"{name} OD Channel"
        )

        axes[index + 1].axis("off")

    plt.tight_layout()

    output_path = output_dir / "02_optical_density.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # 9. Save raw OD data
    np.save(
        output_dir / "02_optical_density.npy",
        od_image
    )

    print("\nOD visualisation saved:", output_path)
    print("Raw OD data saved successfully.")

    return od_image, od_statistics


if __name__ == "__main__":

    from image_input import load_and_validate_image

    input_path = "week1/images/microscopy_sample.png"

    image, metadata = load_and_validate_image(input_path)

    od_image, statistics = calculate_optical_density(image)

    print("\nOptical Density transformation completed!")
