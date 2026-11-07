
import cv2
import torch
import torchstain
import numpy as np
import sys

from pathlib import Path


# ==========================================
# PROJECT ROOT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# IMPORT EXISTING MODULE
# ==========================================

from week4.modules.stain_normalisation import (
    convert_to_torch
)


# ==========================================
# CONFIGURATION
# ==========================================

SOURCE_PATH = Path(
    "week1/images/microscopy_sample.png"
)

REFERENCE_PATH = Path(
    r"D:\Blood Smear Dataset\patient_05\patient_05_12.jpg"
)


# ==========================================
# LOAD IMAGES
# ==========================================

def load_rgb(path):

    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(path)

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


source_rgb = load_rgb(SOURCE_PATH)
reference_rgb = load_rgb(REFERENCE_PATH)

source_tensor = convert_to_torch(source_rgb)
reference_tensor = convert_to_torch(reference_rgb)


# ==========================================
# FIT REFERENCE NORMALISER
# ==========================================

print("=" * 60)
print("STAIN VECTOR INVESTIGATION")
print("=" * 60)

normalizer = torchstain.normalizers.MacenkoNormalizer(
    backend="torch"
)

print("\nFitting reference image...")

normalizer.fit(
    I=reference_tensor
)

print("Reference fitting successful!")


# ==========================================
# INSPECT NORMALISER ATTRIBUTES
# ==========================================

print("\n" + "=" * 60)
print("NORMALISER INTERNAL ATTRIBUTES")
print("=" * 60)

print("\nAvailable attributes:")

for attribute in vars(normalizer):

    value = getattr(normalizer, attribute)

    print(f"\n{attribute}")

    if torch.is_tensor(value):

        print("Type:", type(value))
        print("Shape:", value.shape)
        print("Values:\n", value)

    elif isinstance(value, np.ndarray):

        print("Type:", type(value))
        print("Shape:", value.shape)
        print("Values:\n", value)

    else:

        print("Type:", type(value))
        print("Value:", value)


# ==========================================
# INSPECT NORMALIZE OUTPUT
# ==========================================

print("\n" + "=" * 60)
print("NORMALIZE OUTPUT INSPECTION")
print("=" * 60)

result = normalizer.normalize(
    I=source_tensor,
    stains=True
)

# Extract normalisation outputs
normalised_image, hematoxylin_image, eosin_image = result

# Create output directory
output_dir = PROJECT_ROOT / "results" / "week4" / "stain_components"
output_dir.mkdir(parents=True, exist_ok=True)

# Save normalised image
cv2.imwrite(
    str(output_dir / "normalised_image.png"),
    cv2.cvtColor(normalised_image.numpy().astype("uint8"), cv2.COLOR_RGB2BGR)
)

# Save hematoxylin component
cv2.imwrite(
    str(output_dir / "hematoxylin_component.png"),
    cv2.cvtColor(hematoxylin_image.numpy().astype("uint8"), cv2.COLOR_RGB2BGR)
)

# Save eosin component
cv2.imwrite(
    str(output_dir / "eosin_component.png"),
    cv2.cvtColor(eosin_image.numpy().astype("uint8"), cv2.COLOR_RGB2BGR)
)

print("\nAll three stain component images saved successfully!")
print(f"Output directory: {output_dir}")

print("\nOutput type:", type(result))

if isinstance(result, tuple):

    print("Number of returned elements:", len(result))

    for index, item in enumerate(result):

        print(f"\nOutput element {index}")

        print("Type:", type(item))

        if torch.is_tensor(item):

            print("Shape:", item.shape)
            print("Values:", item)

        elif isinstance(item, np.ndarray):

            print("Shape:", item.shape)
            print("Values:", item)

        elif isinstance(item, (tuple, list)):

            print("Number of nested elements:", len(item))

            for sub_index, sub_item in enumerate(item):

                print(
                    f"Nested element {sub_index}:",
                    type(sub_item)
                )

                if torch.is_tensor(sub_item):
                    print("Shape:", sub_item.shape)
                    print("Values:", sub_item)

                else:
                    print("Value:", sub_item)

        else:

            print("Value:", item)

else:

    print("Output shape:", getattr(result, "shape", None))


# ==========================================
# COMPARE SOURCE AND REFERENCE FITTING
# ==========================================

print("\n" + "=" * 60)
print("SOURCE VS REFERENCE STAIN CHARACTERISTICS")
print("=" * 60)

source_normalizer = torchstain.normalizers.MacenkoNormalizer(
    backend="torch"
)

source_normalizer.fit(
    I=source_tensor
)

print("\nSource normaliser attributes:")

for attribute in vars(source_normalizer):

    value = getattr(source_normalizer, attribute)

    if torch.is_tensor(value):

        print(attribute, ":", value.shape)
        print(value)

    else:

        print(attribute, ":", value)


print("\nReference normaliser attributes:")

for attribute in vars(normalizer):

    value = getattr(normalizer, attribute)

    if torch.is_tensor(value):

        print(attribute, ":", value.shape)
        print(value)

    else:

        print(attribute, ":", value)


print("\n" + "=" * 60)
print("STAIN VECTOR INSPECTION COMPLETED")
print("=" * 60)
