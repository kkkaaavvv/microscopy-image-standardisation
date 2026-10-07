import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchstain


# Input image
source_path = "week1/images/microscopy_sample.png"

source = cv2.imread(source_path)

if source is None:
    print("Error: Could not load source image.")
else:
    print("Source image loaded successfully!")

    # Convert OpenCV BGR → RGB
    source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    # For this first technical test, use the same image as the reference.
    # A proper reference image will be used in the next experiment.
    target_rgb = source_rgb.copy()

    # Convert images to PyTorch tensors
    source_tensor = torch.from_numpy(source_rgb).permute(2, 0, 1)
    target_tensor = torch.from_numpy(target_rgb).permute(2, 0, 1)

    # Create Macenko normalizer
    normalizer = torchstain.normalizers.MacenkoNormalizer(
        backend="torch"
    )

    # Fit the normalizer using the reference image
    normalizer.fit(target_tensor)

    # Apply Macenko normalisation
    normalized, _, _ = normalizer.normalize(
        I=source_tensor
    )

    print("Macenko normalisation completed!")

    # Convert output tensor to NumPy
    normalized = normalized.cpu().numpy()

    print("Normalized shape:", normalized.shape)

    # Make sure pixel values are valid
    normalized = np.clip(
        normalized,
        0,
        255
    ).astype(np.uint8)

    # Display original and normalised images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(source_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(normalized)
    plt.title("Macenko Normalised")
    plt.axis("off")

    plt.tight_layout()
    plt.show()