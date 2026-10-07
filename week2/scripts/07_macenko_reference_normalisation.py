import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchstain


# --------------------------------------------------
# File paths
# --------------------------------------------------

source_path = "week1/images/microscopy_sample.png"
reference_path = "week2/images/reference_blood_smear.jpg"

raw_output_path = "results/week2/macenko_normalised_raw.png"
figure_output_path = "results/week2/task7_macenko_reference_normalisation.png"


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
# Convert BGR → RGB
# --------------------------------------------------

source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
reference_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)


# --------------------------------------------------
# Convert images to PyTorch tensors
# --------------------------------------------------

source_tensor = torch.from_numpy(source_rgb).permute(2, 0, 1)
reference_tensor = torch.from_numpy(reference_rgb).permute(2, 0, 1)


# --------------------------------------------------
# Create Macenko normalizer
# --------------------------------------------------

normalizer = torchstain.normalizers.MacenkoNormalizer(
    backend="torch"
)


# --------------------------------------------------
# Fit normalizer using reference image
# --------------------------------------------------

normalizer.fit(reference_tensor)

print("Macenko normalizer fitted to reference image!")


# --------------------------------------------------
# Normalize source image
# --------------------------------------------------

normalized, _, _ = normalizer.normalize(
    I=source_tensor
)

print("Reference-based Macenko normalisation completed!")


# --------------------------------------------------
# Convert normalized image
# --------------------------------------------------

normalized = normalized.cpu().numpy()

print("Normalized shape:", normalized.shape)

normalized = np.clip(
    normalized,
    0,
    255
).astype(np.uint8)


# --------------------------------------------------
# Save raw Macenko-normalised image
# --------------------------------------------------

cv2.imwrite(
    raw_output_path,
    cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)
)

print(
    "Raw Macenko image saved to:",
    raw_output_path
)


# --------------------------------------------------
# Display results
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
plt.title("Macenko Normalised")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    figure_output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()