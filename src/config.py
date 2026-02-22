"""
Central configuration for the Scene Text Detection & Recognition pipeline.

All tunable parameters live here. Override them via CLI arguments or by
editing this file directly.
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EAST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "frozen_east_text_detection.pb")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# ──────────────────────────────────────────────────────────────────────────────
# EAST Text Detector
# ──────────────────────────────────────────────────────────────────────────────
# Input dimensions MUST be multiples of 32. 320 is a good speed/accuracy
# trade-off on CPU. Use 640 for better small-text detection if latency allows.
EAST_INPUT_WIDTH = 320
EAST_INPUT_HEIGHT = 320

# Minimum confidence for a detected text region to be kept (0.0 – 1.0).
EAST_CONF_THRESHOLD = 0.5

# Non-Maximum Suppression threshold — higher means more aggressive merging
# of overlapping boxes (0.0 – 1.0).
EAST_NMS_THRESHOLD = 0.4

# EAST output layer names (do not change unless using a different model).
EAST_OUTPUT_LAYERS = (
    "feature_fusion/Conv_7/Sigmoid",  # score map
    "feature_fusion/concat_3",  # geometry map
)

# ImageNet mean values used during EAST training (BGR order).
EAST_MEAN = (123.68, 116.78, 103.94)

# ──────────────────────────────────────────────────────────────────────────────
# Image Preprocessing
# ──────────────────────────────────────────────────────────────────────────────
# Padding (in pixels) added around each cropped text region before OCR.
# Gives Tesseract breathing room so characters at the border aren't clipped.
CROP_PADDING = 4

# Bilateral filter parameters for denoising cropped text regions.
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# Adaptive threshold parameters for binarizing cropped text regions.
ADAPTIVE_THRESH_BLOCK_SIZE = 11
ADAPTIVE_THRESH_C = 2

# Supported image file extensions (lowercase).
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

# ──────────────────────────────────────────────────────────────────────────────
# Tesseract OCR
# ──────────────────────────────────────────────────────────────────────────────
# Page Segmentation Mode:
#   6 = Assume a single uniform block of text
#   7 = Treat the image as a single text line
#   8 = Treat the image as a single word
TESSERACT_PSM = 7

# OCR Engine Mode:
#   0 = Legacy only
#   1 = LSTM only
#   2 = Legacy + LSTM
#   3 = Default (based on what's available)
TESSERACT_OEM = 3

# Language for Tesseract (must be installed on the system).
TESSERACT_LANG = "eng"

# Character whitelist — only these characters will be returned by Tesseract.
TESSERACT_CHAR_WHITELIST = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?'-&@#"
)

# Minimum word-level confidence (0–100) from Tesseract. Words below this
# threshold are discarded.
TESSERACT_CONF_THRESHOLD = 40

# ──────────────────────────────────────────────────────────────────────────────
# Annotation / Visualization
# ──────────────────────────────────────────────────────────────────────────────
# Bounding box color (BGR) and thickness for annotated output images.
ANNOTATION_BOX_COLOR = (0, 255, 0)  # green
ANNOTATION_BOX_THICKNESS = 2

# Text label settings drawn above each bounding box.
ANNOTATION_FONT_SCALE = 0.6
ANNOTATION_FONT_THICKNESS = 2
ANNOTATION_TEXT_COLOR = (0, 0, 255)  # red
ANNOTATION_TEXT_BG_COLOR = (0, 255, 0)  # green background behind text

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
