import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchstain


# --------------------------------------------------
# File paths
# --------------------------------------------------

source_path = "week1/images/microscopy_sample.png"

reference_paths = [
    "week2/images/reference_patient_01.jpg",
    "week2/images/reference_patient_02.jpg",
    "week2/images/reference_patient_03.jpg"
]

reference_names = [
    "Patient 01 Reference",
    "Patient 02 Reference",
    "Patient 03 Reference"
]


# --------------------------------------------------
# Load source image
# --------------------------------------------------

source = cv2.imread(source_path)

if source is None:
    print("Error: Could not load source image.")
    exit()

source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

print("Source image loaded successfully!")


# --------------------------------------------------
# Convert source to tensor
# --------------------------------------------------

source_tensor = torch.from_numpy(
    source_rgb
).permute(2, 0, 1)


# --------------------------------------------------
# Store normalized images
# --------------------------------------------------

normalized_images = []


# --------------------------------------------------
# Process each reference
# --------------------------------------------------

for reference_path, reference_name in zip(
    reference_paths,
    reference_names
):

    reference = cv2.imread(reference_path)

    if reference is None:
        print(f"Error: Could not load {reference_name}")
        continue

    reference_rgb = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2RGB
    )

    reference_tensor = torch.from_numpy(
        reference_rgb
    ).permute(2, 0, 1)

    # Create Macenko normalizer
    normalizer = torchstain.normalizers.MacenkoNormalizer(
        backend="torch"
    )

    # Fit to reference
    normalizer.fit(reference_tensor)

    # Normalize source
    normalized, _, _ = normalizer.normalize(
        I=source_tensor
    )

    normalized = normalized.cpu().numpy()

    normalized = np.clip(
        normalized,
        0,
        255
    ).astype(np.uint8)

    normalized_images.append(normalized)

    print(
        f"{reference_name}: "
        "normalisation completed."
    )


# --------------------------------------------------
# Display comparison
# --------------------------------------------------

plt.figure(figsize=(16, 5))


# Source
plt.subplot(1, 4, 1)

plt.imshow(source_rgb)

plt.title("Original Source")

plt.axis("off")


# Normalised images
for i, normalized in enumerate(
    normalized_images
):

    plt.subplot(1, 4, i + 2)

    plt.imshow(normalized)

    plt.title(
        f"Normalised\n{reference_names[i]}"
    )

    plt.axis("off")


plt.tight_layout()


# --------------------------------------------------
# Save result
# --------------------------------------------------

plt.savefig(
    "results/week2/task9_multiple_reference_normalisation.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()