
import cv2
import csv
import numpy as np

from pathlib import Path
from math import ceil


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

PATIENTS = [
    "patient_04",
    "patient_05"
]

SUPPORTED_FORMATS = [
    ".jpg", ".jpeg", ".png",
    ".tif", ".tiff"
]


# ==========================================
# STEP 1: FIND PATIENT IMAGES
# ==========================================

def find_images():

    records = []

    for patient in PATIENTS:

        patient_folder = DATASET_DIR / patient

        if not patient_folder.exists():

            print(
                f"Warning: {patient} folder not found."
            )

            continue

        images = sorted([
            file for file in patient_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_FORMATS
        ])

        for image_path in images:

            records.append({
                "patient": patient,
                "path": image_path
            })

    return records


# ==========================================
# STEP 2: CALCULATE IMAGE STATISTICS
# ==========================================

def analyse_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:

        print("Unable to load:", image_path.name)

        return None

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB
    )

    # RGB statistics
    rgb_mean = np.mean(
        rgb,
        axis=(0, 1)
    )

    rgb_std = np.std(
        rgb,
        axis=(0, 1)
    )

    # LAB statistics
    lab_mean = np.mean(
        lab,
        axis=(0, 1)
    )

    lab_std = np.std(
        lab,
        axis=(0, 1)
    )

    # Brightness and contrast
    lightness = lab[:, :, 0].astype(
        np.float32
    )

    brightness = float(
        np.mean(lightness)
    )

    contrast = float(
        np.std(lightness)
    )

    # Optical Density
    rgb_float = np.clip(
        rgb.astype(np.float32),
        1,
        255
    )

    optical_density = -np.log(
        rgb_float / 255.0
    )

    od_mean = float(
        np.mean(optical_density)
    )

    od_std = float(
        np.std(optical_density)
    )

    # Image sharpness indicator
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    height, width = image.shape[:2]

    return {

        "filename": image_path.name,

        "width": width,
        "height": height,

        "red_mean": round(float(rgb_mean[0]), 2),
        "green_mean": round(float(rgb_mean[1]), 2),
        "blue_mean": round(float(rgb_mean[2]), 2),

        "red_std": round(float(rgb_std[0]), 2),
        "green_std": round(float(rgb_std[1]), 2),
        "blue_std": round(float(rgb_std[2]), 2),

        "lab_l_mean": round(float(lab_mean[0]), 2),
        "lab_a_mean": round(float(lab_mean[1]), 2),
        "lab_b_mean": round(float(lab_mean[2]), 2),

        "lab_l_std": round(float(lab_std[0]), 2),
        "lab_a_std": round(float(lab_std[1]), 2),
        "lab_b_std": round(float(lab_std[2]), 2),

        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),

        "od_mean": round(od_mean, 4),
        "od_std": round(od_std, 4),

        "sharpness": round(sharpness, 2)
    }


# ==========================================
# STEP 3: GENERATE PATIENT CONTACT SHEETS
# ==========================================

def generate_contact_sheet(records, patient):

    patient_records = [
        record for record in records
        if record["patient"] == patient
    ]

    columns = 4

    cell_width = 240
    cell_height = 220

    rows = ceil(
        len(patient_records) / columns
    )

    sheet = np.full(
        (
            rows * cell_height,
            columns * cell_width,
            3
        ),
        255,
        dtype=np.uint8
    )

    for index, record in enumerate(patient_records):

        image = cv2.imread(
            str(record["path"])
        )

        if image is None:
            continue

        height, width = image.shape[:2]

        scale = min(
            215 / width,
            170 / height
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

        offset_y = y + 5

        sheet[
            offset_y:offset_y + new_height,
            offset_x:offset_x + new_width
        ] = resized

        label = record["path"].name

        cv2.putText(
            sheet,
            label,
            (x + 5, y + 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1
        )

    output_path = (
        OUTPUT_DIR / f"{patient}_contact_sheet.png"
    )

    cv2.imwrite(
        str(output_path),
        sheet
    )

    print(
        f"Contact sheet saved: {output_path}"
    )


# ==========================================
# STEP 4: SAVE STATISTICS TO CSV
# ==========================================

def save_statistics(results):

    if not results:
        print("No statistics available.")
        return

    output_path = (
        OUTPUT_DIR / "reference_quality_statistics.csv"
    )

    fieldnames = [
        "patient",
   ] + [
    key for key in results[0].keys()
    if key != "patient"
]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    print(
        "\nStatistics saved:",
        output_path
    )


# ==========================================
# STEP 5: PATIENT SUMMARY
# ==========================================

def print_patient_summary(results):

    print("\n" + "=" * 65)

    print("PATIENT IMAGE QUALITY SUMMARY")

    print("=" * 65)

    for patient in PATIENTS:

        patient_results = [
            result for result in results
            if result["patient"] == patient
        ]

        if not patient_results:
            continue

        print(f"\n{patient.upper()}")

        print("-" * 45)

        print(
            "Number of images:",
            len(patient_results)
        )

        for metric in [
            "brightness",
            "contrast",
            "od_mean",
            "od_std",
            "sharpness"
        ]:

            values = [
                result[metric]
                for result in patient_results
            ]

            print(
                f"{metric}: "
                f"Mean={np.mean(values):.2f}, "
                f"Min={np.min(values):.2f}, "
                f"Max={np.max(values):.2f}"
            )


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("=" * 65)

    print("REFERENCE IMAGE QUALITY ANALYSIS")

    print("=" * 65)

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_DIR}"
        )

    # Find images
    records = find_images()

    print(
        "\nTotal candidate images:",
        len(records)
    )

    results = []

    # Analyse each image
    for index, record in enumerate(
        records,
        start=1
    ):

        print(
            f"Analysing {index}/{len(records)}: "
            f"{record['path'].name}"
        )

        statistics = analyse_image(
            record["path"]
        )

        if statistics is not None:

            statistics["patient"] = (
                record["patient"]
            )

            results.append(statistics)

    # Save numerical results
    save_statistics(results)

    # Generate separate contact sheets
    for patient in PATIENTS:

        generate_contact_sheet(
            records,
            patient
        )

    # Print summaries
    print_patient_summary(results)

    print("\n" + "=" * 65)

    print("REFERENCE QUALITY ANALYSIS COMPLETED")

    print("=" * 65)

    print("Successfully analysed:", len(results))

    print("Output directory:", OUTPUT_DIR)
