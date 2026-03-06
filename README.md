# 🔤 Natural Scene Text Detection & Recognition

A hybrid OCR pipeline that combines **EAST** (Efficient and Accurate Scene Text Detector) with **Tesseract OCR** to detect and recognize text in natural scene images — street signs, shop boards, billboards, nameplates, and more.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![Tesseract](https://img.shields.io/badge/Tesseract-4.x%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [CLI — Single Image](#cli--single-image)
  - [CLI — Batch Processing](#cli--batch-processing)
  - [Web UI (Streamlit)](#web-ui-streamlit)
  - [Python API](#python-api)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Design Decisions](#design-decisions)
- [References](#references)

---

## Overview

Traditional OCR engines like Tesseract are designed for clean, document-style text and perform poorly on natural scene images where text appears against complex backgrounds with variable lighting, distortions, and arbitrary orientations.

This project solves the problem with a **two-stage pipeline**:

1. **Stage 1 — Detection:** The EAST deep learning model localizes text regions in the image, producing oriented bounding boxes around candidate text areas.
2. **Stage 2 — Recognition:** Each detected region is cropped, preprocessed (grayscale, adaptive threshold, denoising), and passed to Tesseract OCR for character recognition.

By separating detection from recognition, the system dramatically reduces background interference and improves accuracy on real-world images.

---

## How It Works

```
Input Image (street sign, shop board, etc.)
        │
        ▼
┌─────────────────────┐
│  1. Preprocessing   │  Resize to 320×320 (multiple of 32), mean subtraction
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  2. EAST Detector   │  Score map + geometry map → rotated bounding boxes → NMS
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  3. Crop & Enhance  │  Rotation-aware crop → padding → grayscale → adaptive threshold
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  4. Tesseract OCR   │  PSM 7 (single text line) → word text + confidence
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  5. Output          │  Annotated image + JSON results with bboxes, text, confidence
└─────────────────────┘
```

---

## Project Structure

```
ml-project/
├── PRD.md                              # Product Requirements Document
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── setup.sh                            # One-command setup script
│
├── models/
│   └── frozen_east_text_detection.pb   # Pre-trained EAST model (~96 MB)
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # All constants & tunable parameters
│   ├── preprocessor.py                 # Image transforms (resize, crop, enhance)
│   ├── detector.py                     # EAST text detection + NMS
│   ├── recognizer.py                   # Tesseract OCR integration
│   └── pipeline.py                     # Orchestrator tying everything together
│
├── main.py                             # CLI entry point
├── app.py                              # Streamlit web UI (optional)
├── evaluate.py                         # Benchmark evaluation script
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py
│   ├── test_detector.py
│   ├── test_recognizer.py
│   └── test_pipeline.py
│
├── data/
│   ├── sample_images/                  # Example scene images
│   └── ground_truth/                   # GT annotations for evaluation
│
└── output/
    ├── annotated/                      # Images with drawn bounding boxes
    └── results/                        # JSON result files
```

---

## Prerequisites

| Requirement       | Version   | Purpose                              |
| ----------------- | --------- | ------------------------------------ |
| **Python**        | 3.8+      | Runtime                              |
| **Tesseract OCR** | 4.x+      | Character recognition engine         |
| **pip**           | Latest    | Python package installer             |

> **Note:** No GPU is required. The EAST model runs on CPU via OpenCV's `dnn` module — no TensorFlow or PyTorch needed.

---

## Quick Start

### Option A: Automated Setup (Recommended)

```bash
# Clone the repository
git clone <repo-url>
cd ml-project

# Run the setup script (installs everything + downloads the EAST model)
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Check your Python version
2. Install Tesseract OCR (if not already installed)
3. Create a Python virtual environment
4. Install all Python dependencies
5. Download the pre-trained EAST model (~96 MB)
6. Create output directories

After setup, activate the virtual environment:

```bash
source venv/bin/activate
```

### Option B: Manual Setup

```bash
sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-eng

pip install -r requirements.txt

mkdir -p models
wget -O models/frozen_east_text_detection.pb \
  "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"

mkdir -p output/annotated output/results data/sample_images data/ground_truth
```

### Verify Installation

```bash
python -c "
import cv2, pytesseract, numpy, imutils
print(f'OpenCV:     {cv2.__version__}')
print(f'NumPy:      {numpy.__version__}')
print(f'Tesseract:  {pytesseract.get_tesseract_version()}')
print('All dependencies OK!')
"
```

---

## Usage

### CLI — Single Image

Process a single natural scene image:

```bash
python main.py --input data/sample_images/sign.jpg --output output/
```

This will:
- Detect text regions using EAST
- Recognize text in each region using Tesseract
- Save an annotated image to `output/annotated/`
- Save JSON results to `output/results/`
- Print recognized text to the terminal

**Example output:**

```
============================================================
Image: /path/to/sign.jpg
Size:  640x480
Time:  312.4 ms
============================================================
  Found 2 text region(s):

  [1] "MAIN STREET"
      OCR confidence:       87.5%
      Detection confidence: 94.2%
      Source:               enhanced

  [2] "ONE WAY"
      OCR confidence:       91.0%
      Detection confidence: 88.7%
      Source:               enhanced
```

### CLI — Batch Processing

Process all images in a directory:

```bash
python main.py --input data/sample_images/ --output output/ --batch
```

Process images recursively (including subdirectories):

```bash
python main.py --input data/sample_images/ --output output/ --batch --recursive
```

### CLI — Custom Thresholds

Tune detection and recognition sensitivity:

```bash
# Lower EAST confidence to detect more text (may include false positives)
python main.py --input photo.jpg --east-conf 0.3

# Higher Tesseract confidence to only keep high-quality recognition
python main.py --input photo.jpg --tess-conf 70

# Larger EAST input for better detection of small text (slower)
python main.py --input photo.jpg --width 640 --height 640

# Print results only — don't save files
python main.py --input photo.jpg --no-save

# Verbose debug logging
python main.py --input photo.jpg --log-level DEBUG
```

### Full CLI Reference

```
usage: main.py [-h] --input INPUT [--output OUTPUT] [--batch] [--recursive]
               [--east-conf EAST_CONF] [--east-nms EAST_NMS]
               [--tess-conf TESS_CONF] [--width WIDTH] [--height HEIGHT]
               [--no-save] [--log-level {DEBUG,INFO,WARNING,ERROR}]

Arguments:
  --input, -i       Path to image file or directory (with --batch)
  --output, -o      Output directory (default: output/)
  --batch, -b       Process all images in --input directory
  --recursive, -r   With --batch, also process subdirectories
  --east-conf       EAST detection confidence, 0.0–1.0 (default: 0.5)
  --east-nms        EAST NMS IoU threshold, 0.0–1.0 (default: 0.4)
  --tess-conf       Tesseract min word confidence, 0–100 (default: 40)
  --width           EAST input width, multiple of 32 (default: 320)
  --height          EAST input height, multiple of 32 (default: 320)
  --no-save         Print results to stdout only, don't save files
  --log-level       Logging verbosity (default: INFO)
```

### Web UI (Streamlit)

Launch the interactive web interface:

```bash
streamlit run app.py
```

This opens a browser-based UI where you can:
- Upload images via drag-and-drop
- Adjust all thresholds with sliders in real-time
- View original and annotated images side-by-side
- Browse detected text in a structured table
- Download annotated images and JSON results

### Python API

Use the pipeline programmatically in your own code:

```python
from src.pipeline import SceneTextPipeline, annotate_image
import cv2

# Initialize pipeline (loads EAST model once)
pipeline = SceneTextPipeline(
    east_width=320,
    east_height=320,
    east_conf=0.5,
    tess_conf=40,
)

# Process a single image
result = pipeline.process_image("data/sample_images/sign.jpg")

# Access results
print(f"Found {result['total_detections']} text regions")
for det in result["detections"]:
    if det["text"]:
        print(f"  '{det['text']}' (confidence: {det['confidence']:.1f}%)")

# Generate an annotated image
image = cv2.imread("data/sample_images/sign.jpg")
annotated = annotate_image(image, result["detections"])
cv2.imwrite("output/annotated_result.jpg", annotated)

# Batch process a directory
results, errors = pipeline.process_directory("data/sample_images/")
print(f"Processed {len(results)} images, {len(errors)} failed")
```

**One-shot convenience function** (creates a new pipeline each time — use `SceneTextPipeline` for batch work):

```python
from src.pipeline import process_image

result = process_image("photo.jpg", east_conf=0.6, tess_conf=50)
```

### Output JSON Format

Each processed image produces a JSON file with this structure:

```json
{
  "image_path": "/absolute/path/to/image.jpg",
  "image_size": [480, 640],
  "detections": [
    {
      "id": 1,
      "bbox": [102, 55, 310, 55, 310, 98, 102, 98],
      "text": "MAIN STREET",
      "confidence": 87.5,
      "detection_confidence": 94.2,
      "source": "enhanced"
    }
  ],
  "total_detections": 1,
  "processing_time_ms": 312.4
}
```

| Field                    | Description                                                      |
| ------------------------ | ---------------------------------------------------------------- |
| `bbox`                   | Flattened 8-point polygon: [x1,y1, x2,y2, x3,y3, x4,y4]        |
| `text`                   | Recognized text string                                           |
| `confidence`             | Average word-level OCR confidence (0–100)                        |
| `detection_confidence`   | EAST detection confidence for this region (0–100)                |
| `source`                 | Which crop produced the result: `"enhanced"`, `"raw"`, or `"none"` |

---

## Configuration

All tunable parameters live in a single file: **`src/config.py`**

You can edit this file directly, or override values via CLI arguments.

### Key Parameters

| Parameter                 | Default | Description                                                    |
| ------------------------- | ------- | -------------------------------------------------------------- |
| `EAST_INPUT_WIDTH`        | 320     | EAST input width (must be multiple of 32)                      |
| `EAST_INPUT_HEIGHT`       | 320     | EAST input height (must be multiple of 32)                     |
| `EAST_CONF_THRESHOLD`     | 0.5     | Min confidence for EAST to keep a detection (0.0–1.0)          |
| `EAST_NMS_THRESHOLD`      | 0.4     | NMS IoU threshold — higher = more aggressive merging           |
| `CROP_PADDING`            | 4       | Pixels of padding around each crop before OCR                  |
| `TESSERACT_PSM`           | 7       | Page Segmentation Mode (7 = single text line)                  |
| `TESSERACT_CONF_THRESHOLD`| 40      | Min word confidence from Tesseract (0–100)                     |
| `TESSERACT_LANG`          | `"eng"` | Tesseract language code                                        |

### Tuning Tips

| Problem                        | Solution                                                   |
| ------------------------------ | ---------------------------------------------------------- |
| Missing small text             | Increase `EAST_INPUT_WIDTH` and `HEIGHT` to 640            |
| Too many false positives       | Increase `EAST_CONF_THRESHOLD` to 0.6 or 0.7              |
| Poor OCR on detected regions   | Decrease `TESSERACT_CONF_THRESHOLD` to 20                  |
| Characters clipped at edges    | Increase `CROP_PADDING` to 6 or 8                          |
| Overlapping duplicate boxes    | Decrease `EAST_NMS_THRESHOLD` to 0.3                       |
| Slow processing on CPU         | Decrease `EAST_INPUT_WIDTH/HEIGHT` to 160 or 192           |

---

## Evaluation

Evaluate the pipeline against ground truth annotations:

```bash
# Using existing prediction JSON files
python evaluate.py --predictions output/results/ --ground-truth data/ground_truth/

# Or run the pipeline on images first, then evaluate
python evaluate.py --images data/sample_images/ --ground-truth data/ground_truth/

# Custom IoU threshold
python evaluate.py --predictions output/results/ --ground-truth data/ground_truth/ --iou 0.6

# Save structured results as JSON
python evaluate.py --predictions output/results/ --ground-truth data/ground_truth/ --save-json
```

### Ground Truth Format

Place one JSON file per image in the `data/ground_truth/` directory. The filename stem must match the image filename stem.

```json
{
  "image": "sign.jpg",
  "annotations": [
    {
      "bbox": [102, 55, 310, 55, 310, 98, 102, 98],
      "text": "MAIN STREET"
    },
    {
      "bbox": [50, 200, 180, 200, 180, 240, 50, 240],
      "text": "ONE WAY"
    }
  ]
}
```

### Metrics Computed

**Detection:**
- **Precision** — What fraction of predicted boxes are correct
- **Recall** — What fraction of ground truth boxes were found
- **F1-Score** — Harmonic mean of precision and recall

**Recognition:**
- **Word Accuracy (exact)** — Fraction of matched words that are identical
- **Word Accuracy (case-insensitive)** — Same, ignoring case
- **Character Error Rate (CER)** — Edit distance normalized by GT length

---

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test module
python -m pytest tests/test_detector.py -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run with verbose output
python -m pytest tests/ -v -s
```

---

## Troubleshooting

### "EAST model not found"

The pre-trained model file is not at `models/frozen_east_text_detection.pb`.

```bash
# Download it manually
mkdir -p models
wget -O models/frozen_east_text_detection.pb \
  "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
```

### "Tesseract is not installed or not found on PATH"

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-eng

# Fedora
sudo dnf install tesseract tesseract-langpack-eng

# macOS
brew install tesseract

# Verify
tesseract --version
```

### "EAST input dimensions must be multiples of 32"

The `--width` and `--height` arguments must be multiples of 32. Valid examples: 160, 192, 224, 256, 288, 320, 384, 448, 512, 576, 640.

### No text detected in an image

- **Lower the detection threshold:** `--east-conf 0.3`
- **Increase resolution:** `--width 640 --height 640`
- **Check image quality:** Very blurry, tiny, or heavily distorted text may not be detected
- **Enable debug logging:** `--log-level DEBUG` to see what the pipeline is doing

### Text detected but recognition is poor

- **Lower Tesseract confidence:** `--tess-conf 20` to see all recognized text including low-confidence results
- **Check the crops:** The pipeline uses adaptive thresholding which can sometimes invert or destroy text in unusual lighting. It automatically falls back to the raw grayscale crop if the enhanced version fails
- **Increase padding:** Edit `CROP_PADDING` in `src/config.py` to 6 or 8

### Import errors

Make sure you're running from the project root directory:

```bash
cd ml-project
python main.py --input photo.jpg
```

---

## Design Decisions

This project prioritizes **simplicity and readability** over bleeding-edge accuracy. Here are the key design choices and why:

| Decision | Rationale |
| --- | --- |
| **EAST + Tesseract** (not end-to-end) | Best balance of accuracy and simplicity. No GPU training needed. |
| **OpenCV DNN** (not TensorFlow/PyTorch) | Zero framework dependency — just `opencv-python`. |
| **Pre-trained model** (no custom training) | Avoids GPU infra, large datasets, and training complexity. |
| **Python constants file** (not YAML/JSON config) | One file, zero parsing code, easy to understand. |
| **CLI-first** with optional web UI | Works anywhere without a server. Streamlit is additive, not required. |
| **One module per responsibility** | Swap any stage independently. Unit test each in isolation. |
| **Adaptive threshold** (not global) | Handles uneven lighting that's common in natural scenes. |
| **Rotation-aware crop with padding** | EAST outputs rotated boxes; naïve crops miss tilted text. |
| **Fallback from enhanced to raw crop** | If thresholding destroys text, the pipeline retries with the grayscale version. |
| **PSM 7** for Tesseract | Each crop is a single text line — telling Tesseract this improves speed and accuracy. |

For the full rationale, see [PRD.md](PRD.md).

---

## References

1. X. Zhou et al. — "EAST: An Efficient and Accurate Scene Text Detector" (CVPR, 2017)
2. R. Smith — "An Overview of the Tesseract OCR Engine" (ICDAR, 2007)
3. D. Cao, X. Sun, H. Yu, C. Guo — "Scene Text Detection in Natural Images: A Review" (Symmetry, 2020)
4. S. Fang, X. Liu, C. Xu — "Scene Text Detection and Recognition: The Deep Learning Era" (IJCV, 2018)
5. W. Wang et al. — "Shape Robust Text Detection with PSENet" (CVPR, 2019)

---

## License

This project is developed as part of the UBCA398J coursework (Winter 2025-2026) at VIT, School of Computer Science Engineering and Information Systems.

---

**Author:** Samyukta A (23BCA0014)
**Guide:** Dr. VijayaRani A
**Department:** Computer Applications