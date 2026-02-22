# Product Requirements Document (PRD)

## Natural Scene Text Detection and Recognition

| Field            | Value                                      |
| ---------------- | ------------------------------------------ |
| **Project**      | Natural Scene Text Detection & Recognition |
| **Author**       | Samyukta A (23BCA0014)                     |
| **Guide**        | Dr. VijayaRani A                           |
| **Course**       | UBCA398J – Project – I                     |
| **Semester**     | Winter 2025-2026                           |
| **Department**   | Computer Applications, VIT                 |
| **Version**      | 1.0                                        |
| **Last Updated** | 2025                                       |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Objectives](#3-goals--objectives)
4. [Scope](#4-scope)
5. [Target Users & Use Cases](#5-target-users--use-cases)
6. [System Architecture](#6-system-architecture)
7. [Design Decisions & Rationale](#7-design-decisions--rationale)
8. [Module Breakdown](#8-module-breakdown)
9. [Data Flow](#9-data-flow)
10. [Project Structure](#10-project-structure)
11. [Technical Specifications](#11-technical-specifications)
12. [API & Interface Design](#12-api--interface-design)
13. [Preprocessing Pipeline](#13-preprocessing-pipeline)
14. [Dependencies](#14-dependencies)
15. [Testing Strategy](#15-testing-strategy)
16. [Evaluation Metrics](#16-evaluation-metrics)
17. [Datasets](#17-datasets)
18. [Risk Analysis & Mitigation](#18-risk-analysis--mitigation)
19. [Implementation Plan](#19-implementation-plan)
20. [Out of Scope / Future Work](#20-out-of-scope--future-work)
21. [References](#21-references)

---

## 1. Executive Summary

Traditional OCR engines (e.g., Tesseract) are designed for clean, document-style text and perform poorly on natural scene images — street signs, shop boards, billboards, nameplates — where complex backgrounds, variable lighting, distortions, and arbitrary text orientations are the norm.

This project builds a **hybrid two-stage OCR pipeline** that:

1. **Stage 1 — Detection:** Uses the **EAST (Efficient and Accurate Scene Text Detector)** deep learning model to localize text regions in a natural image, producing oriented bounding boxes.
2. **Stage 2 — Recognition:** Crops each detected text region and passes it through **Tesseract OCR** for character-level recognition.

By separating detection from recognition, the system dramatically reduces background interference and improves accuracy on real-world images.

---

## 2. Problem Statement

| Aspect                  | Detail                                                                                                                                        |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **What fails**          | Conventional OCR engines receive the full scene image and misread or miss text due to noise, clutter, and complex backgrounds.                |
| **Why it fails**        | No robust text localization step — the entire image (including irrelevant regions) is fed directly to the recognizer.                         |
| **Impact**              | Low recognition accuracy, missed text regions, high sensitivity to noise, inability to handle multi-oriented or multi-scale text.             |
| **What's needed**       | A two-stage pipeline that first **localizes** text regions with a deep learning detector, then **recognizes** only those cropped regions.     |

---

## 3. Goals & Objectives

### Primary Goal

Develop an efficient and reliable system for detecting and recognizing text from natural scene images.

### Specific Objectives

| #  | Objective                                                                                                   | Priority |
| -- | ----------------------------------------------------------------------------------------------------------- | -------- |
| O1 | Accurately localize text regions in natural images using the EAST text detector.                             | P0       |
| O2 | Recognize characters in detected regions using Tesseract OCR with preprocessing.                            | P0       |
| O3 | Handle real-world challenges: complex backgrounds, varying lighting, noise, multi-oriented text.            | P0       |
| O4 | Provide a simple CLI tool for single-image and batch processing.                                            | P0       |
| O5 | Provide a lightweight web UI (Streamlit) for interactive demo use.                                          | P1       |
| O6 | Evaluate the system on standard benchmarks (ICDAR 2013/2015) and report precision/recall/F1.               | P1       |
| O7 | Keep the codebase small, readable, and modular so it can be extended easily.                                | P0       |

---

## 4. Scope

### In Scope

- Detection and recognition of **printed English text** in natural scene images (JPEG/PNG).
- Image preprocessing: resizing, grayscale conversion, adaptive thresholding, denoising, deskewing.
- EAST-based text detection with Non-Maximum Suppression (NMS).
- Tesseract-based OCR on cropped text regions.
- Confidence-score filtering for both detection and recognition.
- CLI interface for single-image and batch-directory processing.
- Optional Streamlit web interface for demo/evaluation.
- Evaluation scripts computing precision, recall, F1-score, and Character Error Rate (CER).
- Offline image processing (not real-time video).

### Out of Scope

- Handwritten text recognition.
- Multilingual translation of detected text.
- Real-time video stream processing.
- Mobile or embedded deployment.
- Training the EAST model from scratch (we use the pre-trained frozen model).
- Large-scale commercial deployment or cloud hosting.

---

## 5. Target Users & Use Cases

| User                         | Use Case                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| **Researchers / Students**   | Benchmark scene-text pipelines, experiment with preprocessing or swapping detectors.  |
| **Assistive Tech Developers**| Build tools that read street signs/shop names aloud for visually impaired users.      |
| **Navigation App Developers**| Extract text from captured sign images to augment map data.                           |
| **Data Annotators**          | Semi-automatically extract text labels from scene images to speed up annotation.      |

---

## 6. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                              │
│               (natural scene: sign, board, etc.)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   1. PREPROCESSOR                               │
│  • Resize to multiple-of-32 (EAST requirement)                  │
│  • Mean subtraction (R=123.68, G=116.78, B=103.94)             │
│  • Keep original copy for cropping later                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              2. EAST TEXT DETECTOR                               │
│  • Forward pass through frozen_east_text_detection.pb           │
│  • Outputs: score map + geometry map                            │
│  • Decode rotated bounding boxes                                │
│  • Apply Non-Maximum Suppression (NMS)                          │
│  • Filter by confidence threshold (default ≥ 0.5)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│            3. ROI EXTRACTION & PREPROCESSING                    │
│  • For each bounding box:                                       │
│    ├─ Compute rotation-aware crop from original image           │
│    ├─ Add padding (4px) around crop                             │
│    ├─ Convert to grayscale                                      │
│    ├─ Apply adaptive Gaussian thresholding                      │
│    └─ Optional: bilateral filter for denoising                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│             4. TESSERACT OCR RECOGNIZER                         │
│  • PSM 7 (single text line) for each cropped region             │
│  • Whitelist: A-Z a-z 0-9 and common punctuation               │
│  • Return text string + per-word confidence                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│             5. POST-PROCESSOR & OUTPUT                          │
│  • Strip low-confidence results (< 40 %)                        │
│  • Deduplicate overlapping text                                 │
│  • Annotate original image with boxes + recognized text         │
│  • Return JSON list of {bbox, text, confidence}                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Design Decisions & Rationale

### D1. Two-Stage Pipeline (EAST → Tesseract) Instead of End-to-End Model

| Option Considered          | Why Rejected / Chosen                                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **End-to-end (FOTS, etc.)**| Higher accuracy ceiling but requires GPU training, large datasets, and complex code. Overkill for project scope.  |
| **Tesseract only**         | Fails on natural scenes — no text localization.                                                                    |
| ✅ **EAST + Tesseract**    | Best balance of accuracy and simplicity. EAST is fast, well-documented, and has a readily available frozen model. Tesseract is mature, CPU-friendly, and easy to integrate via `pytesseract`. |

### D2. Pre-trained Frozen EAST Model (No Custom Training)

- The EAST model pre-trained on ICDAR datasets is freely available as `frozen_east_text_detection.pb`.
- Avoids the complexity of setting up training pipelines, GPU infrastructure, and large annotated datasets.
- Sufficient for the project's research-level evaluation scope.

### D3. Python as the Implementation Language

- Richest ecosystem for CV/ML: OpenCV, pytesseract, NumPy, imutils.
- Simple, readable, aligns with the "keep code simple" goal.
- Students/researchers can extend easily.

### D4. OpenCV's `dnn` Module for EAST Inference (Not TensorFlow/PyTorch)

- `cv2.dnn.readNet` loads the frozen `.pb` directly — **zero TensorFlow dependency**.
- Keeps the install lightweight (only `opencv-python` needed for detection).
- CPU inference is fast enough since EAST is a single-shot detector.

### D5. Modular File Structure — One Responsibility Per Module

- `detector.py` — only EAST detection logic.
- `recognizer.py` — only Tesseract OCR logic.
- `preprocessor.py` — only image transforms.
- `pipeline.py` — orchestrates the stages.
- Makes unit testing trivial and allows swapping any stage independently.

### D6. CLI-First with Optional Web UI

- The primary interface is a CLI (`main.py`) — no server required, works anywhere.
- Streamlit web UI (`app.py`) is optional and additive — not a core dependency.
- Keeps mandatory dependencies minimal.

### D7. Configuration via a Single `config.py` Constants File

- No YAML/JSON/TOML config files to parse.
- All tunable parameters (thresholds, model path, padding, PSM mode) live in one place.
- Easy to understand, easy to override from CLI args.

### D8. Rotation-Aware Cropping with Padding

- EAST outputs **rotated** bounding boxes (5 geometry values + angle).
- Naïve axis-aligned crops cut off tilted text — so we apply rotation-aware extraction.
- Adding 4px padding around each crop gives Tesseract breathing room, significantly improving recognition.

### D9. Adaptive Thresholding over Global Thresholding

- Natural scenes have uneven illumination.
- Adaptive Gaussian thresholding handles local brightness variations much better than a single global threshold.

### D10. PSM 7 (Single Line) for Tesseract

- Each crop contains a single line/word of text.
- PSM 7 tells Tesseract to expect exactly one text line, improving speed and accuracy over the default full-page mode.

---

## 8. Module Breakdown

### 8.1 `config.py` — Central Configuration

All magic numbers and paths in one file:

```python
# ── Model ──────────────────────────────────────────
EAST_MODEL_PATH   = "models/frozen_east_text_detection.pb"
EAST_INPUT_WIDTH  = 320   # must be multiple of 32
EAST_INPUT_HEIGHT = 320   # must be multiple of 32
EAST_CONF_THRESH  = 0.5
EAST_NMS_THRESH   = 0.4

# ── Preprocessing ──────────────────────────────────
PADDING           = 4     # pixels around each crop
BLUR_KERNEL       = (5, 5)

# ── Tesseract ─────────────────────────────────────
TESS_PSM          = 7     # single text line
TESS_LANG         = "eng"
TESS_CONF_THRESH  = 40    # minimum word confidence (0-100)

# ── Output ────────────────────────────────────────
ANNOTATED_COLOR   = (0, 255, 0)   # green boxes
ANNOTATED_THICK   = 2
```

### 8.2 `preprocessor.py` — Image Transforms

| Function                      | Purpose                                                     |
| ----------------------------- | ----------------------------------------------------------- |
| `resize_for_east(image)`      | Resize image so both dimensions are multiples of 32.        |
| `create_blob(image)`          | Mean-subtracted blob for EAST forward pass.                 |
| `enhance_crop(crop)`          | Grayscale → adaptive threshold → bilateral filter on crop.  |
| `rotate_crop(image, box, angle)` | Extract rotation-aware crop from the original image.     |
| `add_padding(crop, px)`       | Add white-pixel padding around a crop.                      |

### 8.3 `detector.py` — EAST Text Detection

| Function                              | Purpose                                                        |
| ------------------------------------- | -------------------------------------------------------------- |
| `load_east_model(path)`               | Load frozen `.pb` into an `cv2.dnn` net. Called once.          |
| `detect(net, blob, orig_shape)`       | Run forward pass, decode geometry → list of (box, confidence). |
| `decode_predictions(scores, geometry)` | Convert EAST output maps to rotated rects.                    |
| `non_max_suppression(boxes, scores)`  | Apply NMS to remove overlapping detections.                    |

### 8.4 `recognizer.py` — Tesseract OCR

| Function                          | Purpose                                                         |
| --------------------------------- | --------------------------------------------------------------- |
| `recognize(crop, lang, psm)`     | Run `pytesseract.image_to_data` on a preprocessed crop.         |
| `filter_results(ocr_data, min_conf)` | Drop words below confidence threshold, strip whitespace.    |
| `recognize_batch(crops)`         | Process a list of crops, return list of `{text, confidence}`.   |

### 8.5 `pipeline.py` — Orchestrator

| Function                            | Purpose                                                         |
| ----------------------------------- | --------------------------------------------------------------- |
| `process_image(image_path) → dict`  | Full pipeline: load → detect → crop → recognize → return JSON.  |
| `annotate_image(image, results)`    | Draw bounding boxes and text on the original image.             |
| `process_directory(dir_path)`       | Batch process all images in a directory.                        |

### 8.6 `main.py` — CLI Entry Point

```
usage: main.py [-h] --input INPUT [--output OUTPUT] [--east-conf 0.5]
               [--tess-conf 40] [--width 320] [--height 320] [--batch]
```

| Argument       | Description                                     |
| -------------- | ----------------------------------------------- |
| `--input`      | Path to a single image or directory (with `--batch`). |
| `--output`     | Directory to save annotated images and JSON results. |
| `--east-conf`  | EAST confidence threshold (default 0.5).        |
| `--tess-conf`  | Tesseract min word confidence (default 40).     |
| `--width`      | EAST input width, multiple of 32 (default 320). |
| `--height`     | EAST input height, multiple of 32 (default 320).|
| `--batch`      | Treat `--input` as a directory and process all images. |

### 8.7 `app.py` — Streamlit Web UI (Optional)

- File upload widget.
- Sliders for confidence thresholds.
- Display: original image, annotated image side-by-side, table of detected text + confidence.

### 8.8 `evaluate.py` — Evaluation Script

- Compare predicted bounding boxes / text against ground truth.
- Compute: **Precision**, **Recall**, **F1-Score**, **Character Error Rate (CER)**, **Word-level Accuracy**.
- Output a markdown/CSV report.

---

## 9. Data Flow

```
Image File (JPEG/PNG)
    │
    ├──► preprocessor.resize_for_east() ──► resized image
    │       │
    │       └──► preprocessor.create_blob() ──► blob
    │                   │
    │                   └──► detector.detect(net, blob) ──► [(box, conf), ...]
    │                                │
    │   ┌────────────────────────────┘
    │   │   For each (box, angle):
    │   │
    │   ├──► preprocessor.rotate_crop(original, box, angle) ──► raw_crop
    │   │       │
    │   │       └──► preprocessor.add_padding(raw_crop) ──► padded_crop
    │   │               │
    │   │               └──► preprocessor.enhance_crop(padded_crop) ──► clean_crop
    │   │                           │
    │   │                           └──► recognizer.recognize(clean_crop) ──► {text, conf}
    │   │
    │   └──► Collect all results
    │
    └──► pipeline.annotate_image(original, results) ──► annotated image + JSON output
```

---

## 10. Project Structure

```
ml-project/
├── PRD.md                          # This document
├── README.md                       # Quick-start guide
├── requirements.txt                # Python dependencies
├── setup.sh                        # One-command setup script
│
├── models/
│   └── frozen_east_text_detection.pb   # Pre-trained EAST model (~96 MB)
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # All constants & defaults
│   ├── preprocessor.py             # Image transforms
│   ├── detector.py                 # EAST detection
│   ├── recognizer.py               # Tesseract recognition
│   └── pipeline.py                 # Orchestrator
│
├── main.py                         # CLI entry point
├── app.py                          # Streamlit web UI (optional)
├── evaluate.py                     # Benchmark evaluation script
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py
│   ├── test_detector.py
│   ├── test_recognizer.py
│   └── test_pipeline.py
│
├── data/
│   ├── sample_images/              # A few example scene images
│   └── ground_truth/               # GT annotations for evaluation
│
└── output/                         # Default output directory
    ├── annotated/                  # Images with drawn boxes
    └── results/                    # JSON result files
```

---

## 11. Technical Specifications

### 11.1 EAST Text Detector

| Property              | Value                                                |
| --------------------- | ---------------------------------------------------- |
| Model file            | `frozen_east_text_detection.pb` (TensorFlow frozen graph) |
| Architecture          | Fully Convolutional Network (PVANet base)            |
| Input size            | H×W must be multiples of 32 (default 320×320)        |
| Output layers         | `feature_fusion/Conv_7/Sigmoid` (scores), `feature_fusion/concat_3` (geometry) |
| Bounding box format   | Rotated rectangles (4 distances + angle)             |
| NMS algorithm         | `cv2.dnn.NMSBoxesRotated` or custom Python NMS       |
| Inference device      | CPU (via OpenCV DNN); GPU optional if available      |

### 11.2 Tesseract OCR

| Property              | Value                                                |
| --------------------- | ---------------------------------------------------- |
| Version               | Tesseract 4.x+ (LSTM engine)                        |
| Page Segmentation Mode| PSM 7 — treat the image as a single text line        |
| OEM                   | 3 (default — LSTM + legacy combined)                 |
| Language              | `eng`                                                |
| Char whitelist        | `ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?'-&@#` |

### 11.3 Image Preprocessing Parameters

| Step                     | Method                          | Parameters                    |
| ------------------------ | ------------------------------- | ----------------------------- |
| Mean subtraction         | OpenCV `blobFromImage`          | R=123.68, G=116.78, B=103.94 |
| Grayscale conversion     | `cv2.cvtColor` BGR2GRAY        | —                             |
| Adaptive thresholding    | `cv2.adaptiveThreshold`        | Gaussian, blockSize=11, C=2   |
| Denoising (optional)     | `cv2.bilateralFilter`          | d=9, sigmaColor=75, sigmaSpace=75 |
| Padding                  | `cv2.copyMakeBorder`           | 4px, white (255)              |

---

## 12. API & Interface Design

### 12.1 Core Python API

```python
from src.pipeline import process_image, annotate_image

# Process a single image
results = process_image("data/sample_images/street_sign.jpg")
# Returns:
# {
#     "image_path": "data/sample_images/street_sign.jpg",
#     "detections": [
#         {
#             "bbox": [x1, y1, x2, y2, x3, y3, x4, y4],
#             "text": "STOP",
#             "confidence": 92.3
#         },
#         ...
#     ],
#     "processing_time_ms": 245.7
# }

# Get annotated image (numpy array)
import cv2
image = cv2.imread("data/sample_images/street_sign.jpg")
annotated = annotate_image(image, results["detections"])
cv2.imwrite("output/annotated/street_sign.jpg", annotated)
```

### 12.2 CLI Interface

```bash
# Single image
python main.py --input data/sample_images/sign.jpg --output output/

# Batch processing
python main.py --input data/sample_images/ --output output/ --batch

# Custom thresholds
python main.py --input photo.jpg --east-conf 0.6 --tess-conf 50 --width 640 --height 640
```

### 12.3 Output JSON Format

```json
{
  "image_path": "data/sample_images/sign.jpg",
  "image_size": [480, 640],
  "detections": [
    {
      "id": 1,
      "bbox": [102, 55, 310, 55, 310, 98, 102, 98],
      "text": "MAIN STREET",
      "confidence": 87.5
    }
  ],
  "total_detections": 1,
  "processing_time_ms": 312.4
}
```

---

## 13. Preprocessing Pipeline

The preprocessing is intentionally kept **simple and deterministic** — no learned components.

### Why These Steps (and Not More)

| Step                     | Why Included                                                 | Why Not Something Fancier                            |
| ------------------------ | ------------------------------------------------------------ | ---------------------------------------------------- |
| Resize to 320×320        | EAST requires multiples of 32; 320 is a good speed/accuracy trade-off. | 640×640 is more accurate but 4× slower on CPU.     |
| Mean subtraction         | Required by the EAST model (trained with ImageNet means).    | —                                                    |
| Grayscale                | Tesseract works best on single-channel images.               | Color adds no value for character recognition.       |
| Adaptive threshold       | Handles uneven lighting in natural scenes.                   | Global Otsu fails when one half of the crop is shadowed. |
| Bilateral filter         | Smooths noise while preserving edges (text edges).           | Gaussian blur would smudge thin strokes.             |
| Padding                  | Gives Tesseract context around text edges.                   | Without it, characters touching the border get clipped. |

### Steps NOT Included (Simplicity Decision)

- **Super-resolution / upscaling:** Adds complexity and latency. EAST already handles multi-scale.
- **Morphological operations (erosion/dilation):** Risk destroying thin characters. Only useful for specific edge cases.
- **Color-space tricks (HSV/LAB):** Marginal gains, harder to tune, scene-dependent.
- **Perspective correction (full homography):** Rotation-aware crop handles most cases. Full warp is fragile.

---

## 14. Dependencies

### `requirements.txt`

```
opencv-python>=4.5.0
opencv-contrib-python>=4.5.0
numpy>=1.21.0
pytesseract>=0.3.10
imutils>=0.5.4
Pillow>=9.0.0
streamlit>=1.20.0       # optional — only for web UI
```

### System Dependencies

| Dependency          | Install Command (Ubuntu/Debian)            |
| ------------------- | ------------------------------------------ |
| Tesseract OCR 4.x+  | `sudo apt install tesseract-ocr`           |
| English language data| `sudo apt install tesseract-ocr-eng`       |
| Python 3.8+         | `sudo apt install python3 python3-pip`     |

### `setup.sh`

```bash
#!/bin/bash
set -e

# Install system deps
sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-eng

# Install Python deps
pip install -r requirements.txt

# Download EAST model if not present
EAST_MODEL="models/frozen_east_text_detection.pb"
if [ ! -f "$EAST_MODEL" ]; then
    mkdir -p models
    echo "Downloading EAST model..."
    wget -q -O "$EAST_MODEL" \
      "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
    echo "Model downloaded to $EAST_MODEL"
fi

echo "Setup complete!"
```

---

## 15. Testing Strategy

### 15.1 Unit Tests

| Test File                  | What It Tests                                                    |
| -------------------------- | ---------------------------------------------------------------- |
| `test_preprocessor.py`     | resize_for_east returns correct dimensions; enhance_crop doesn't crash on various sizes; padding is correct. |
| `test_detector.py`         | Model loads successfully; detect() returns list of tuples; NMS reduces box count; empty image returns []. |
| `test_recognizer.py`       | Clean text image ("HELLO") is recognized correctly; confidence filtering works; empty crop returns "". |
| `test_pipeline.py`         | End-to-end: known test image produces expected text; JSON output has correct schema; batch mode works. |

### 15.2 Integration Test

- Process 5 curated sample images with known ground truth.
- Assert that at least 80% of expected text strings are detected and recognized.

### 15.3 Manual / Visual Test

- Run the Streamlit app, upload images, visually inspect bounding boxes and recognized text.
- Useful for catching edge cases that automated tests miss.

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_detector.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## 16. Evaluation Metrics

### 16.1 Detection Metrics

| Metric         | Definition                                                       | Target     |
| -------------- | ---------------------------------------------------------------- | ---------- |
| **Precision**  | (True positive detections) / (All predicted detections)          | ≥ 0.75     |
| **Recall**     | (True positive detections) / (All ground truth text regions)     | ≥ 0.70     |
| **F1-Score**   | Harmonic mean of precision and recall                            | ≥ 0.72     |
| **IoU threshold** | A detection is "correct" if IoU with GT box ≥ 0.5            | 0.5        |

### 16.2 Recognition Metrics

| Metric                     | Definition                                               | Target     |
| -------------------------- | -------------------------------------------------------- | ---------- |
| **Word-level Accuracy**    | % of detected words that exactly match ground truth      | ≥ 0.60     |
| **Character Error Rate (CER)** | Edit distance / GT length, averaged over all detections | ≤ 0.25     |
| **Case-insensitive Accuracy** | Same as word-level but ignoring case                  | ≥ 0.70     |

### 16.3 Performance Metrics

| Metric                     | Target                            |
| -------------------------- | --------------------------------- |
| **Latency per image (320×320, CPU)** | ≤ 500 ms              |
| **Peak RAM usage**         | ≤ 500 MB                         |
| **Model file size (EAST)** | ~96 MB (fixed)                    |

---

## 17. Datasets

### 17.1 For Development & Demo

| Dataset              | Size          | Description                                 |
| -------------------- | ------------- | ------------------------------------------- |
| Custom sample set    | 20-30 images  | Manually collected street signs, shop boards from the web. Stored in `data/sample_images/`. |

### 17.2 For Evaluation (Optional / If Time Permits)

| Dataset              | Size           | Description                                       | Link                             |
| -------------------- | -------------- | ------------------------------------------------- | -------------------------------- |
| ICDAR 2013           | 462 images     | Focused scene text, horizontal.                   | rrc.cvc.uab.es                   |
| ICDAR 2015           | 1500 images    | Incidental scene text, multi-oriented.            | rrc.cvc.uab.es                   |
| SVT (Street View Text)| 647 images   | Google Street View images with word-level annotations. | —                            |

---

## 18. Risk Analysis & Mitigation

| #  | Risk                                                  | Likelihood | Impact | Mitigation                                                       |
| -- | ----------------------------------------------------- | ---------- | ------ | ---------------------------------------------------------------- |
| R1 | EAST model download link becomes unavailable.         | Low        | High   | Bundle model in repo (Git LFS) or provide mirror link.           |
| R2 | Tesseract fails on heavily distorted/curved text.     | Medium     | Medium | Document known limitation. Future work: swap Tesseract for a CRNN recognizer. |
| R3 | EAST misses small text at 320×320 resolution.         | Medium     | Medium | Allow user to increase `--width`/`--height` (e.g., 640×640) via CLI. |
| R4 | Preprocessing over-thresholds, destroying text.       | Low        | Medium | Keep raw crop alongside enhanced crop; fall back to raw if OCR returns empty. |
| R5 | Tesseract not installed on user's system.             | Medium     | High   | `setup.sh` installs it automatically. Clear error message in code if missing. |
| R6 | OpenCV version incompatibilities.                     | Low        | Medium | Pin minimum version in `requirements.txt`. Test on 4.5+.        |

---

## 19. Implementation Plan

### Phase 1 — Core Pipeline (Week 1-2)

| Task                                        | Deliverable                          |
| ------------------------------------------- | ------------------------------------ |
| Set up project structure, `config.py`       | Skeleton repo                        |
| Implement `preprocessor.py`                 | All image transform functions        |
| Implement `detector.py` (EAST loading + inference + NMS) | Text detection working   |
| Implement `recognizer.py` (Tesseract wrapper) | OCR on cropped images working      |
| Implement `pipeline.py` (orchestrator)      | End-to-end single-image processing   |
| Implement `main.py` (CLI)                   | CLI tool fully functional            |

### Phase 2 — Polish & Testing (Week 3)

| Task                                        | Deliverable                          |
| ------------------------------------------- | ------------------------------------ |
| Write unit tests for all modules            | `tests/` directory complete          |
| Collect sample images, write ground truth   | `data/` directory populated          |
| Implement `evaluate.py`                     | Evaluation script with metrics       |
| Tune thresholds on sample data              | Updated `config.py` defaults         |

### Phase 3 — Web UI & Documentation (Week 4)

| Task                                        | Deliverable                          |
| ------------------------------------------- | ------------------------------------ |
| Build Streamlit app (`app.py`)              | Working web demo                     |
| Write `README.md` with setup & usage guide  | Complete documentation               |
| Run evaluation on ICDAR dataset (if available)| Metrics report                     |
| Final code review and cleanup               | Release-ready codebase               |

---

## 20. Out of Scope / Future Work

These are explicitly **not** part of the current project but are natural extensions:

| Feature                          | Why Deferred                                                    |
| -------------------------------- | --------------------------------------------------------------- |
| **Video / real-time processing** | Requires frame-by-frame optimization, tracking, and GPU setup.  |
| **Handwritten text**             | Fundamentally different problem; EAST + Tesseract won't handle it. |
| **Multilingual support**         | Needs language-specific Tesseract data and potentially different detectors. |
| **Curved text (TextSnake/PSENet)** | More complex detection models; would replace EAST entirely.   |
| **CRNN-based recognizer**        | Could replace Tesseract for better scene-text accuracy. Requires training. |
| **Mobile deployment (TFLite/ONNX)** | Model conversion and optimization needed. Different project.  |
| **Cloud API / REST service**     | Out of scope for research-level evaluation.                     |
| **Text-to-Speech integration**   | Application-layer feature; not core to detection/recognition.   |

---

## 21. References

1. D. Cao, X. Sun, H. Yu, C. Guo — "Scene Text Detection in Natural Images: A Review" (Symmetry, 2020)
2. Y. Liu, Z. Chen, J. Zhang — "From Detection to Understanding: A Systematic Survey of Deep Learning for Scene Text Processing" (Applied Sciences, 2025)
3. F. Naiemi, V. Ghods, H. Khalesi — "Scene Text Detection and Recognition: A Survey" (Multimedia Tools & Applications, 2022)
4. S. Fang, X. Liu, C. Xu — "Scene Text Detection and Recognition: The Deep Learning Era" (IJCV, 2018)
5. X. Chen, L. Jin, Y. Zhu, C. Luo, T. Wang — "Text Detection and Recognition in the Wild: A Review" (Pattern Recognition, 2020)
6. U. Pal, P. P. Roy, J. Lladós — "A Comprehensive Review on Text Detection and Recognition in Scene Images" (Pattern Recognition, 2024)
7. E. Eli, R. Mandal, A. Basu — "A Comprehensive Review of Non-Latin Natural Scene Text Recognition" (EAAI, 2025)
8. A. Kumar, R. Singh — "A Survey on Text Recognition from Natural Scene Images" (IJERT, 2021)
9. W. Wang et al. — "Shape Robust Text Detection with Progressive Scale Expansion Network (PSENet)" (CVPR, 2019)
10. R. Smith — "An Overview of the Tesseract OCR Engine" (ICDAR, 2007)
11. X. Zhou et al. — "EAST: An Efficient and Accurate Scene Text Detector" (CVPR, 2017)

---

## Appendix A: Quick-Start Commands

```bash
# 1. Clone and setup
git clone <repo-url> && cd ml-project
chmod +x setup.sh && ./setup.sh

# 2. Run on a single image
python main.py --input data/sample_images/sign.jpg --output output/

# 3. Run batch processing
python main.py --input data/sample_images/ --output output/ --batch

# 4. Launch web UI
streamlit run app.py

# 5. Run tests
python -m pytest tests/ -v

# 6. Evaluate on benchmark
python evaluate.py --predictions output/results/ --ground-truth data/ground_truth/
```

---

## Appendix B: Key Simplicity Principles

1. **No custom model training.** We use pre-trained EAST as-is.
2. **No deep learning framework dependency.** OpenCV's `dnn` module is enough.
3. **No complex config system.** One Python file with constants.
4. **No microservices.** Single-process Python scripts.
5. **No database.** JSON files for input/output.
6. **Each module does one thing.** Swap any stage without touching others.
7. **Fail gracefully.** If a crop produces no text, skip it — don't crash.
8. **Log, don't silently fail.** Use Python's `logging` module at INFO level by default.

---

*End of PRD — version 1.0*