# Week 8 – Image I/O and Colour Space Conversion

## Project
Microscopy Image Standardisation for Reliable Automated Blood Cell Analysis.

## Objective
The objective of Week 8 is to implement the initial image standardisation pipeline, focusing on image input/output, colour space conversion and basic image analysis.

## Implemented Components

1. **Image Input:** Loads microscopy images using OpenCV and converts BGR images into RGB format.
2. **Colour Space Conversion:** Converts RGB images into HSV and LAB colour spaces.
3. **Image Statistics:** Calculates the mean, standard deviation, minimum and maximum values for each RGB channel.
4. **RGB Histogram:** Generates a histogram showing the intensity distribution of Red, Green and Blue channels.
5. **Image Output:** Saves RGB, HSV and LAB images in the results directory.

## Technologies Used
- Python
- OpenCV
- NumPy
- Matplotlib

## Project Structure

```text
week8/
├── images/
├── modules/
│   ├── __init__.py
│   ├── image_input.py
│   ├── image_export.py
│   ├── colour_analysis.py
│   ├── image_statistics.py
│   └── image_histogram.py
├── main.py
└── README.md
```

## How to Run

From the project root directory:

```bash
python3 week8/main.py
```

## Output Files

Generated files are saved in `results/week8/`:

- rgb_output.png
- hsv_output.png
- lab_output.png
- rgb_histogram.png

## Current Progress

The initial implementation of image loading, image saving, colour space conversion, RGB statistics and histogram generation has been completed and tested using a microscopy sample image. 
The pipeline was successfully executed using an 800 × 800 RGB microscopy sample image with uint8 data type. RGB, HSV and LAB outputs, channel statistics and the RGB histogram were generated successfully.

Advanced stain normalisation algorithms such as Macenko and Reinhard are not yet implemented.

## Next Steps

- Test the pipeline on additional microscopy images.
- Validate image processing across the project dataset.
- Implement stain normalisation in subsequent development stages.