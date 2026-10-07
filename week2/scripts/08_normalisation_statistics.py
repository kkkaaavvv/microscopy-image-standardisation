import cv2
import numpy as np
import torch
import torchstain


# --------------------------------------------------
# File paths
# --------------------------------------------------

source_path = "week1/images/microscopy_sample.png"
reference_path = "week2/images/reference_blood_smear.jpg"


# --------------------------------------------------
# Load images
# --------------------------------------------------

source = cv2.imread(source_path)
reference = cv2.imread(reference_path)

if source is None:
    print("Error: Could not load source image.")
    exit()

if reference is None:
    print("Error: Could not load reference image.")
    exit()


# --------------------------------------------------
# Convert BGR → RGB
# --------------------------------------------------

source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
reference_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)


# --------------------------------------------------
# Macenko normalisation
# --------------------------------------------------

source_tensor = torch.from_numpy(source_rgb).permute(2, 0, 1)
reference_tensor = torch.from_numpy(reference_rgb).permute(2, 0, 1)

normalizer = torchstain.normalizers.MacenkoNormalizer(
    backend="torch"
)

normalizer.fit(reference_tensor)

normalized, _, _ = normalizer.normalize(
    I=source_tensor
)

normalized = normalized.cpu().numpy()

normalized = np.clip(
    normalized,
    0,
    255
).astype(np.uint8)


# --------------------------------------------------
# Function to calculate statistics
# --------------------------------------------------

def calculate_statistics(image, name):

    print("\n" + name)
    print("-" * 40)

    channels = ["Red", "Green", "Blue"]

    for i, channel in enumerate(channels):

        values = image[:, :, i]

        print(
            f"{channel}: "
            f"Mean={np.mean(values):.2f}, "
            f"Std={np.std(values):.2f}, "
            f"Min={np.min(values)}, "
            f"Max={np.max(values)}"
        )


# --------------------------------------------------
# Calculate statistics
# --------------------------------------------------

calculate_statistics(
    source_rgb,
    "SOURCE IMAGE"
)

calculate_statistics(
    reference_rgb,
    "REFERENCE IMAGE"
)

calculate_statistics(
    normalized,
    "MACENKO NORMALISED IMAGE"
)