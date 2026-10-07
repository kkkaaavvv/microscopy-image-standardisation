import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

import torch_staintools
from torch_staintools import constants

# Disable torch.compile on Windows
constants.CONFIG.ENABLE_COMPILE = False

from torch_staintools.normalizer import NormalizerBuilder


# --------------------------------------------------
# File paths
# --------------------------------------------------

source_path = "week1/images/microscopy_sample.png"
reference_path = "week2/images/reference_blood_smear.jpg"

raw_output_path = "results/week2/vahadane_normalised_raw.png"
figure_output_path = "results/week2/task10_vahadane_normalisation.png"


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

print("Source image loaded successfully!")
print("Reference image loaded successfully!")


# --------------------------------------------------
# Resize reference for faster Vahadane fitting
# --------------------------------------------------

reference = cv2.resize(
    reference,
    (600, 450),
    interpolation=cv2.INTER_AREA
)

print("Reference resized to:", reference.shape)


# --------------------------------------------------
# Convert BGR → RGB
# --------------------------------------------------

source_rgb = cv2.cvtColor(
    source,
    cv2.COLOR_BGR2RGB
)

reference_rgb = cv2.cvtColor(
    reference,
    cv2.COLOR_BGR2RGB
)


# --------------------------------------------------
# Convert to BCHW float tensors
# --------------------------------------------------

source_tensor = (
    torch.from_numpy(source_rgb)
    .permute(2, 0, 1)
    .float()
    / 255.0
)

reference_tensor = (
    torch.from_numpy(reference_rgb)
    .permute(2, 0, 1)
    .float()
    / 255.0
)

source_tensor = source_tensor.unsqueeze(0)
reference_tensor = reference_tensor.unsqueeze(0)


print("Source tensor shape:", source_tensor.shape)
print("Reference tensor shape:", reference_tensor.shape)


# --------------------------------------------------
# Create Vahadane normalizer
# --------------------------------------------------

normalizer = NormalizerBuilder.build(
    "vahadane",
    sparse_stain_solver="fista",
    concentration_solver="fista"
)


# --------------------------------------------------
# Fit using reference
# --------------------------------------------------

print("Fitting Vahadane normalizer...")

normalizer.fit(reference_tensor)

print("Vahadane normalizer fitted!")


# --------------------------------------------------
# Normalize source
# --------------------------------------------------

print("Normalising source image...")

normalized = normalizer(
    source_tensor
)

print("Vahadane normalisation completed!")


# --------------------------------------------------
# Convert output to RGB uint8
# --------------------------------------------------

normalized = (
    normalized
    .squeeze(0)
    .detach()
    .cpu()
    .numpy()
)

normalized = np.transpose(
    normalized,
    (1, 2, 0)
)

normalized = np.clip(
    normalized * 255.0,
    0,
    255
).astype(np.uint8)

print("Normalized shape:", normalized.shape)


# --------------------------------------------------
# Save raw Vahadane-normalised image
# --------------------------------------------------

cv2.imwrite(
    raw_output_path,
    cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)
)

print(
    "Raw Vahadane image saved to:",
    raw_output_path
)


# --------------------------------------------------
# Display
# --------------------------------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(source_rgb)
plt.title("Source Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(reference_rgb)
plt.title("Reference Blood Smear")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(normalized)
plt.title("Vahadane Normalised")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    figure_output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()