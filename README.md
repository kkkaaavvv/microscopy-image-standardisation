# Microscopy Image Standardisation for Reliable Automated Blood Cell Analysis

A modular image-processing and colour standardisation pipeline designed to reduce staining and acquisition-related variation in microscopy images while preserving biologically meaningful cellular structures for reliable automated blood cell analysis.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?logo=numpy)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)

---

## 1. Overview

Microscopy Image Standardisation is a research and implementation project focused on developing a reproducible preprocessing pipeline for peripheral blood smear images, particularly those stained using the **May–Grünwald–Giemsa (MGG)** staining technique.

Automated blood cell analysis systems rely on consistent image characteristics to identify and analyse cellular structures. However, microscopy images acquired from different laboratories frequently exhibit variations caused by:

- Differences in staining protocols, reagent concentrations and staining durations.
- Variations in microscope cameras, sensors and optical configurations.
- Differences in illumination, exposure and white balance.
- Scanner-specific image processing.
- Changes in staining intensity and slide conditions.

These variations introduce **domain shift**, where an automated analysis model trained on images from one acquisition environment may perform differently when presented with images from another.

### Objective

To design, implement and evaluate a standardised image preprocessing pipeline that transforms microscopy images into a more consistent colour representation while preserving the underlying cellular morphology.

The proposed system is intended to operate as an intermediate preprocessing layer between microscopy image storage and downstream automated blood cell analysis systems.

---

## 2. Problem Statement

Automated blood cell analysis models are often trained using microscopy images acquired under specific staining and imaging conditions. When these models encounter images from different microscopes, cameras, magnifications, illumination environments or staining protocols, differences in colour distribution and image appearance can affect their reliability.

The absence of a consistent preprocessing approach makes it difficult to ensure reproducible image characteristics across different acquisition environments.

This project addresses the problem by investigating reference-based colour and stain normalisation techniques, image quality assessment and quantitative evaluation methods to reduce acquisition-related variation while retaining meaningful cellular details.

---

## 3. Project Objectives

1. Analyse microscopy images through pixel-level inspection, colour distributions and image statistics.
2. Investigate RGB, HSV and CIELAB colour representations for microscopy analysis.
3. Explore optical density-based stain analysis and reference-based stain normalisation.
4. Implement and evaluate Macenko, Vahadane and Reinhard normalisation approaches.
5. Investigate image quality characteristics such as blur, intensity distribution and illumination variation.
6. Develop a modular preprocessing architecture with clearly defined processing components.
7. Evaluate normalisation results using channel statistics and quantitative colour-distance measurements.
8. Establish a foundation for structural quality assessment and reliable downstream image analysis.

---

## 4. System Architecture

The proposed architecture consists of five major modules.

| Module | Responsibility |
|---|---|
| Image Loader | Loads microscopy images and validates dimensions, channels and data types. |
| Color Analyzer | Analyses colour spaces, pixel distributions, histograms and image statistics. |
| Reference Template Store | Maintains reference images and associated normalisation parameters. |
| Normalization Engine | Applies the selected reference-based colour or stain normalisation method. |
| Evaluation Module | Evaluates colour consistency and structural preservation. |

### Proposed Processing Workflow

```text
             Input Microscopy Image
                       |
                       v
                Image Loader
                       |
                       v
                Image Validation
                       |
                       v
                 Color Analyzer
                       |
          +------------+------------+
          |                         |
          v                         v
     RGB / HSV / LAB          Image Statistics
          |                         |
          +------------+------------+
                       |
                       v
             Reference Image Selection
                       |
                       v
               Normalization Engine
                       |
             +---------+---------+
             |         |         |
             v         v         v
          Macenko   Reinhard  Vahadane
             |         |         |
             +---------+---------+
                       |
                       v
              Normalised Image
                       |
                       v
                 Evaluation
                       |
              +--------+--------+
              |                 |
              v                 v
          Quality Pass      Quality Failure
              |                 |
              v                 v
       Standardised Output   Manual Review
              |
              v
       Downstream Blood Cell
          Analysis Model
```

**Architecture note:** This represents the proposed system architecture. Not every component or decision branch in the diagram is currently implemented. The implemented functionality is described separately below.

---

## 5. Technical Approach

### 5.1 Image Inspection and Baseline Analysis

The initial development stage focused on understanding the numerical characteristics of microscopy images before applying normalisation.

Activities included:

- Image dimension and channel inspection.
- Pixel-level value inspection.
- Grayscale conversion.
- Histogram generation.
- Mean, median and standard-deviation calculations.
- Minimum and maximum intensity analysis.

The initial practice image measured 800 × 800 pixels, contained three colour channels and used the `uint8` data type.

### 5.2 Colour Space Analysis

Multiple colour representations were investigated to understand the distribution of colour and intensity information.

| Colour space | Application |
|---|---|
| RGB | Standard colour representation and channel analysis. |
| HSV | Inspection of hue, saturation and brightness. |
| CIELAB | Perceptually organised colour analysis and Reinhard normalisation. |
| Optical Density (OD) | Stain-related analysis and Macenko normalisation. |

Optical density is calculated using:

\[
OD=-\log\left(\frac{I}{I_0}\right)
\]

where \(I\) represents pixel intensity and \(I_0\) represents the reference intensity.

### 5.3 Image Quality Assessment

Image quality analysis was introduced to investigate characteristics that could influence subsequent normalisation.

The implemented experimental analyses included:

- **Blur Detection:** Variance of the Laplacian to estimate edge detail.
- **Intensity Analysis:** Grayscale intensity distributions and statistical measurements.
- **Illumination Analysis:** Gaussian smoothing and regional brightness comparisons.

These measurements provide descriptive information about individual images. They are not validated clinical quality thresholds.

### 5.4 Macenko Stain Normalisation

Macenko normalisation was investigated as a reference-based stain normalisation technique.

The method operates in optical-density space and estimates stain contributions to transform a source image towards a selected reference image.

The experimental implementation used `torchstain`.

The investigation included:

- Optical density transformation.
- Reference-based normaliser fitting.
- Stain normalisation.
- RGB statistical comparison.
- Multiple-reference experiments.

### 5.5 Vahadane Stain Normalisation

Vahadane normalisation was explored as an alternative stain separation and reconstruction approach.

The implementation used `torch-staintools`.

The method was benchmarked against Macenko using RGB channel mean and standard-deviation distances.

### 5.6 Reinhard Colour Normalisation

Reinhard colour transfer was implemented as a reference-based colour normalisation workflow.

The implementation converts source and reference images into CIELAB colour space and independently matches the channel means and standard deviations.

The transformation is:

\[
I_{\text{norm},c} =
\left(
\frac{I_{\text{src},c}-\mu_{\text{src},c}}
{\max(\sigma_{\text{src},c},\epsilon)}
\right)
\sigma_{\text{tgt},c}
+\mu_{\text{tgt},c}
\]

Where:

- \(I_{\text{src}}\): Source image.
- \(\mu_{\text{src}}\): Source channel mean.
- \(\sigma_{\text{src}}\): Source channel standard deviation.
- \(\mu_{\text{tgt}}\): Reference channel mean.
- \(\sigma_{\text{tgt}}\): Reference channel standard deviation.
- \(\epsilon\): Small numerical safeguard.

The resulting LAB representation is converted back to RGB and saved as an image.

The implementation includes numerical safeguards, channel clipping, output validation and statistical comparisons.

**Important distinction:** Reinhard performs statistical colour transfer in CIELAB space. It does not perform stain-vector deconvolution or optical-density stain concentration estimation.

---

## 6. Experimental Results

### 6.1 Macenko vs Vahadane

The experiments compared Macenko and Vahadane against a selected MGG reference image.

Euclidean distances between RGB channel statistics were used to evaluate colour distribution alignment.

| Method | RGB Mean Distance ↓ | RGB Std. Deviation Distance ↓ |
|---|---:|---:|
| Original Source | 59.72 | 22.69 |
| Macenko | 14.62 | 0.51 |
| Vahadane | 25.11 | 18.99 |

Macenko produced the closest statistical match to the selected reference in this experiment.

The RGB mean distance decreased by approximately 75.5%, while the standard-deviation distance decreased by approximately 97.8%.

These findings are specific to the tested source-reference pair and do not establish universal superiority across datasets.

### 6.2 Reinhard Normalisation

The Reinhard implementation evaluated colour transfer using two 420 × 420 image regions.

The source region received a controlled simulated darker and more purple stain shift, while the reference region remained unchanged.

| Evaluation Metric | Before | After |
|---|---:|---:|
| LAB Mean Distance | 23.1056 | 0.0773 |
| LAB Standard-Deviation Distance | 3.1180 | 0.0384 |
| RGB Mean Distance | 93.9005 | 0.2627 |
| RGB Standard-Deviation Distance | 11.3087 | 2.9509 |

The results demonstrate substantial alignment of the source image's global colour statistics with the selected reference.

The implementation also generated:

- Source, reference and normalised image comparisons.
- LAB channel distribution histograms.
- RGB and LAB statistical tables.
- Saved normalised image outputs.
- Automated numerical assertions.

**Experimental qualification:** The source and reference regions were extracted from one previously documented microscopy image. The source stain shift was simulated. The experiment is a controlled algorithm demonstration, not validation across independently acquired clinical slides.

---

## 7. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main implementation language. |
| OpenCV | Image loading, colour conversion and image export. |
| NumPy | Numerical operations and image-array processing. |
| Matplotlib | Histograms, visualisations and statistical comparisons. |
| Pillow | Image handling and inspection. |
| PyTorch | Deep learning fundamentals and experimental development. |
| torchstain | Macenko stain normalisation experiments. |
| torch-staintools | Vahadane normalisation experiments. |
| scikit-image | Histogram matching baseline experiments. |
| Jupyter Notebook | Interactive implementation and experimentation. |
| Git and GitHub | Version control and collaboration. |
| Visual Studio Code | Development environment. |

---

## 8. Repository Structure

The repository is organised by development stage, keeping earlier experiments separate from later implementation checkpoints.

```text
microscopy-image-standardisation/
│
├── week1/
│   ├── images/
│   └── scripts/
│
├── week2/
│   ├── scripts/
│   └── results/
│
├── week3/
│   ├── scripts/
│   └── pytorch/
│
├── week4/
│
├── week8/
│   ├── evidence/
│   ├── images/
│   ├── modules/
│   │   ├── image_input.py
│   │   ├── image_export.py
│   │   ├── colour_analysis.py
│   │   ├── image_histogram.py
│   │   ├── image_statistics.py
│   │   └── __init__.py
│   ├── main.py
│   └── README.md
│
├── week10/
│   ├── PS159_Week10_Reinhard.ipynb
│   └── week10_reinhard.py
│
├── results/
│   ├── week3/
│   └── week8/
│
├── .gitignore
└── README.md
```

*The directory overview reflects the documented development organisation. Individual output files and subsequently added folders may differ as development continues.*

---

## 9. Installation and Setup

### Prerequisites

- Python 3.12 (documented development environment).
- Git.
- Visual Studio Code or another Python IDE.
- Jupyter Notebook for interactive experiments.

### Clone the Repository

```bash
git clone https://github.com/kkkaaavvv/microscopy-image-standardisation.git

cd microscopy-image-standardisation
```

### Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Install Core Dependencies

```bash
pip install numpy opencv-python matplotlib pillow
```

For stain-normalisation experiments:

```bash
pip install torch torchvision torchstain
```

Additional experimental dependencies:

```bash
pip install torch-staintools scikit-image
```

Install Jupyter if required:

```bash
pip install notebook
```

*Dependency compatibility may vary across operating systems and package versions. These commands describe the documented technology stack and are not a verified lockfile for every experiment.*

---

## 10. Running the Implementations

### Week 8: Image Processing Pipeline

```bash
python week8/main.py
```

This checkpoint includes image loading, validation, RGB conversion, HSV and LAB conversion, RGB histogram generation, image statistics and output export.

### Week 10: Reinhard Normalisation

```bash
python week10/week10_reinhard.py
```

The Jupyter notebook can also be opened using:

```bash
jupyter notebook week10/PS159_Week10_Reinhard.ipynb
```

Ensure the relevant source and reference image files are available in the expected folder structure before execution.

---

## 11. Development Progress

| Stage | Focus | Outcome |
|---|---|---|
| Week 1 | Image inspection | Pixel inspection, dimensions, grayscale histograms and baseline statistics. |
| Week 2 | Colour and stain analysis | RGB/HSV/LAB analysis, optical density, Macenko and Vahadane experiments. |
| Week 3 | Image quality assessment | Blur detection, intensity analysis, illumination analysis and PyTorch fundamentals. |
| Week 4 | High-level system design | Pipeline flowchart and system block diagram. |
| Week 5 | Project consolidation | Technical documentation, design decisions and portfolio preparation. |
| Week 6 | Detailed system design | Level-0 and Level-1 DFDs and UML sequence diagram. |
| Week 7 | Design review | Consolidated architecture, technical justification, evaluation plan and limitations. |
| Week 8 | Initial implementation | Modular image input/output, colour conversion, histograms and statistics. |
| Week 9 | Portfolio and checkpoint review | Project positioning, implementation review and technical documentation. |
| Week 10 | Colour normalisation implementation | Executable Reinhard workflow, visual comparisons and quantitative evaluation. |

---

## 12. Limitations

The current project is an experimental research and software implementation effort. Several limitations must be considered when interpreting its results.

1. **Limited experimental generalisation:** Earlier Macenko and Vahadane comparisons used a limited source-reference configuration.
2. **Reference sensitivity:** Different reference images can produce visibly different normalisation outputs.
3. **Controlled Reinhard experiment:** The source and reference regions came from the same parent image, and the source stain variation was simulated.
4. **No clinical validation:** The reported colour-distance improvements do not demonstrate improved diagnostic performance.
5. **Structural preservation:** Quantitative colour alignment alone cannot establish that all cellular morphology and local texture are preserved.
6. **Quality-control thresholds:** Proposed SSIM and colour-distance quality gates require further implementation and validation.
7. **Dataset diversity:** Broader validation across independently acquired slides, patients and microscopy environments remains necessary.

The project does not currently claim clinical readiness, diagnostic accuracy improvements or validated cross-laboratory generalisation.

---

## 13. Future Scope

Potential next stages of development include:

- Expand testing using independently acquired MGG-stained blood smear images.
- Develop a unified and configurable normalisation interface.
- Investigate systematic reference image selection and reference profile management.
- Compare Macenko and Reinhard under consistent cross-slide experimental conditions.
- Integrate morphology-oriented quality assessment.
- Investigate SSIM and colour-distance evaluation as part of a quality-control mechanism.
- Validate normalisation robustness across different microscopy acquisition environments.
- Explore integration with downstream blood cell segmentation, detection and classification models.
- Investigate additional image standardisation methods and learned approaches.

---

## 14. Dataset and Attribution

The project references the following publicly available dataset for MGG-stained blood-smear experiments:

**Dataset:** 186 digital images of MGG-stained blood smears from five patients with hereditary spherocytosis.

Delgado-Ortet, M., Molina, A., Alférez, S., Rodellar, J., & Merino, A. (2020). *Dataset A: 186 digital images of MGG-stained blood smears from five patients with hereditary spherocytosis (Version 1).* Mendeley Data.

https://doi.org/10.17632/c37wnbbd3d.1

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).

The dataset was used for earlier reference-based experiments. It was not used to generate the controlled Reinhard source and reference patches.

---

## 15. Research References

1. Reinhard, E., Adhikhmin, M., Gooch, B., & Shirley, P. (2001). Color transfer between images. *IEEE Computer Graphics and Applications, 21*(5), 34–41.

2. Delgado-Ortet, M., et al. (2020). Dataset A: 186 digital images of MGG-stained blood smears from five patients with hereditary spherocytosis. Mendeley Data. https://doi.org/10.17632/c37wnbbd3d.1

3. Macenko, M., et al. (2009). A method for normalizing histology slides for quantitative analysis. *2009 IEEE International Symposium on Biomedical Imaging.*

4. Vahadane, A., et al. (2016). Structure-preserving color normalization and sparse stain separation for histological images. *IEEE Transactions on Medical Imaging, 35*(8).

---

## 16. Repository

**GitHub:** https://github.com/kkkaaavvv/microscopy-image-standardisation

This repository documents the research, experimental investigations, system design and implementation progress of the Microscopy Image Standardisation project.

---

*An ongoing research and engineering project exploring reproducible microscopy image preprocessing and colour standardisation.*
