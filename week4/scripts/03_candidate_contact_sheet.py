
import cv2
import numpy as np

from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_DIR = Path(r"D:\Blood Smear Dataset")

OUTPUT_DIR = Path(
    "results/week4/reference_quality_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CANDIDATES = [
    "patient_05_14.jpg",
    "patient_05_01.jpg",
    "patient_05_12.jpg",
    "patient_05_06.jpg",
    "patient_05_17.jpg"
]


# ==========================================
# GENERATE CONTACT SHEET
# ==========================================

def generate_contact_sheet():

    columns = 3
    cell_width = 400
    cell_height = 330

    rows = 2

    sheet = np.full(
        (rows * cell_height,
         columns * cell_width, 3),
        255,
        dtype=np.uint8
    )

    for index, filename in enumerate(CANDIDATES):

        image_path = (
            DATASET_DIR / "patient_05" / filename
        )

        image = cv2.imread(str(image_path))

        if image is None:

            print("Unable to load:", filename)
            continue

        height, width = image.shape[:2]

        scale = min(
            360 / width,
            250 / height
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        resized = cv2.resize(
            image,
            (new_width, new_height)
        )

        row = index // columns
        column = index % columns

        x = column * cell_width
        y = row * cell_height

        offset_x = (
            x + (cell_width - new_width) // 2
        )

        offset_y = y + 10

        sheet[
            offset_y:offset_y + new_height,
            offset_x:offset_x + new_width
        ] = resized

        cv2.putText(
            sheet,
            filename,
            (x + 10, y + 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

        print("Added:", filename)

    output_path = (
        OUTPUT_DIR / "top_reference_candidates.png"
    )

    cv2.imwrite(
        str(output_path),
        sheet
    )

    print("\nContact sheet saved:", output_path)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    generate_contact_sheet()
