from pathlib import Path
from modules.image_statistics import calculate_statistics
from modules.image_input import load_image
from modules.colour_analysis import convert_colour_spaces
from modules.image_export import save_image
from modules.image_histogram import generate_rgb_histograms

def main():
    print("MICROSCOPY IMAGE STANDARDISATION - WEEK 8")
    print("-" * 45)

    image_path = Path("week8/images/microscopy_sample.png")

    # Load image
    rgb_image = load_image(image_path)
    generate_rgb_histograms(rgb_image)
        # Calculate image statistics
    statistics = calculate_statistics(rgb_image)

    print("\nRGB Image Statistics:")
    for channel, values in statistics.items():
        print(f"\n{channel}:")
        for metric, value in values.items():
            print(f"  {metric}: {value:.2f}")

    # Convert colour spaces
    hsv_image, lab_image = convert_colour_spaces(rgb_image)

    # Save output images
    save_image(rgb_image, "results/week8/rgb_output.png")
    save_image(hsv_image, "results/week8/hsv_output.png")
    save_image(lab_image, "results/week8/lab_output.png")

    print("\nWeek 8 processing completed successfully.")


if __name__ == "__main__":
    main()