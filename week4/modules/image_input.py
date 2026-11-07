
import cv2
import numpy as np
from pathlib import Path


def load_and_validate_image(image_path):

    """
    Loads a microscopy image and validates its basic properties.

    Parameters:
        image_path: Path to the input microscopy image.

    Returns:
        image: Validated image in OpenCV BGR format.
        metadata: Dictionary containing image properties.
    """

    image_path = Path(image_path)

    # 1. Check whether the file exists
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # 2. Load the image
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )

    # 3. Check whether loading was successful
    if image is None:
        raise ValueError(
            "Unable to load image. Check the file format."
        )

    # 4. Check image dimensions
    if image.size == 0:
        raise ValueError("Image is empty.")

    if image.ndim not in (2, 3):
        raise ValueError("Unsupported image dimensions.")

    # 5. Identify colour channels
    if image.ndim == 2:
        channels = 1

    else:
        channels = image.shape[2]

        if channels not in (1, 3, 4):
            raise ValueError("Unsupported channel count.")

        if channels == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2BGR
            )
            channels = 3

        elif channels == 1:
            image = image[:, :, 0]
            channels = 1

    # 6. Check data type
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError(
            f"Unsupported data type: {image.dtype}"
        )

    # 7. Extract image properties
    height, width = image.shape[:2]

    metadata = {
        "filename": image_path.name,
        "width": width,
        "height": height,
        "channels": channels,
        "data_type": str(image.dtype),
        "minimum_pixel": int(image.min()),
        "maximum_pixel": int(image.max())
    }

    return image, metadata


# Standalone testing
if __name__ == "__main__":

    input_path = (
        "week1/images/microscopy_sample.png"
    )

    try:
        image, metadata = load_and_validate_image(
            input_path
        )

        print("=" * 45)
        print("IMAGE INPUT AND VALIDATION")
        print("=" * 45)

        for key, value in metadata.items():
            print(f"{key}: {value}")

        print("\nImage validation successful!")

    except (FileNotFoundError, ValueError) as error:
        print(f"Validation failed: {error}")
