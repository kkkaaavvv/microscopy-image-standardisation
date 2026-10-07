import cv2
import numpy as np


# Load image
image_path = "week1/images/microscopy_sample.png"
image = cv2.imread(image_path)


if image is None:
    print("Error: Could not load image.")

else:
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate statistics
    mean_intensity = np.mean(gray)
    median_intensity = np.median(gray)
    std_intensity = np.std(gray)
    min_intensity = np.min(gray)
    max_intensity = np.max(gray)

    print("Image Statistics")
    print("----------------")
    print("Mean intensity:", mean_intensity)
    print("Median intensity:", median_intensity)
    print("Standard deviation:", std_intensity)
    print("Minimum intensity:", min_intensity)
    print("Maximum intensity:", max_intensity)