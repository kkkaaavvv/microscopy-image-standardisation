
import numpy as np


def calculate_statistics(image):
    """
    Calculate colour channel statistics for an image.
    """

    if image is None:
        raise ValueError("Input image is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel RGB image.")

    channel_names = ["Red", "Green", "Blue"]

    statistics = {}

    for index, name in enumerate(channel_names):
        channel = image[:, :, index]

        statistics[name] = {
            "mean": float(np.mean(channel)),
            "std": float(np.std(channel)),
            "min": int(np.min(channel)),
            "max": int(np.max(channel))
        }

    return statistics
