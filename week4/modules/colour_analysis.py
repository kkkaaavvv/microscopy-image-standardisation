
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyse_colour_spaces(image, output_dir="results/week4"):

    """
    Analyse a microscopy image using RGB, HSV and LAB
    colour spaces.

    Returns:
        Dictionary containing colour representations
        and channel statistics.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convert OpenCV BGR image to RGB
    if image.ndim == 2:
        raise ValueError("Colour analysis requires a colour image.")

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # 2. Convert RGB into HSV and LAB
    hsv_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2HSV
    )

    lab_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2LAB
    )

    colour_spaces = {
        "RGB": rgb_image,
        "HSV": hsv_image,
        "LAB": lab_image
    }

    # 3. Calculate channel statistics
    statistics = {}

    for space_name, colour_image in colour_spaces.items():

        channel_names = {
            "RGB": ["Red", "Green", "Blue"],
            "HSV": ["Hue", "Saturation", "Value"],
            "LAB": ["Lightness", "A", "B"]
        }

        statistics[space_name] = {}

        for index, channel_name in enumerate(
            channel_names[space_name]
        ):

            channel = colour_image[:, :, index]

            statistics[space_name][channel_name] = {
                "mean": round(float(np.mean(channel)), 2),
                "std": round(float(np.std(channel)), 2),
                "min": int(np.min(channel)),
                "max": int(np.max(channel))
            }

    # 4. Print statistics
    print("=" * 55)
    print("MICROSCOPY COLOUR SPACE ANALYSIS")
    print("=" * 55)

    for space, channels in statistics.items():

        print(f"\n{space} COLOUR SPACE")

        for channel, values in channels.items():

            print(f"\n{channel}")
            print(f"Mean: {values['mean']}")
            print(f"Standard deviation: {values['std']}")
            print(f"Minimum: {values['min']}")
            print(f"Maximum: {values['max']}")

    # 5. Visualise colour spaces
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    images = [
        rgb_image,
        hsv_image,
        lab_image
    ]

    titles = [
        "RGB Image",
        "HSV Image",
        "LAB Image"
    ]

    for i, colour_image in enumerate(images):

        # Display each image in RGB format for visualisation
        if i == 0:
            display_image = rgb_image
        elif i == 1:
            display_image = cv2.cvtColor(
                hsv_image,
                cv2.COLOR_HSV2RGB
            )
        else:
            display_image = cv2.cvtColor(
                lab_image,
                cv2.COLOR_LAB2RGB
            )

        axes[0, i].imshow(display_image)
        axes[0, i].set_title(titles[i])
        axes[0, i].axis("off")

        # Plot channel histograms
        for channel_index in range(3):

            channel = colour_image[:, :, channel_index]

            axes[1, i].hist(
                channel.flatten(),
                bins=256,
                range=(0, 256),
                alpha=0.5,
                label=f"Channel {channel_index + 1}"
            )

        axes[1, i].set_title(
            f"{titles[i]} - Histograms"
        )

        axes[1, i].set_xlabel("Pixel Value")
        axes[1, i].set_ylabel("Frequency")
        axes[1, i].legend(fontsize=7)

    plt.tight_layout()

    output_path = output_dir / "01_colour_space_analysis.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("\nVisualisation saved:", output_path)

    return {
        "colour_spaces": colour_spaces,
        "statistics": statistics
    }


if __name__ == "__main__":

    from image_input import load_and_validate_image

    input_path = "week1/images/microscopy_sample.png"

    image, metadata = load_and_validate_image(input_path)

    results = analyse_colour_spaces(image)

    print("\nColour space analysis completed!")
