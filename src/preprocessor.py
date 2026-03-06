"""
Image preprocessing utilities for the Scene Text Detection & Recognition pipeline.

This module handles all image transformations:
  - Resizing images for EAST model input
  - Creating DNN blobs for forward pass
  - Rotation-aware cropping of detected text regions
  - Enhancing cropped text regions for better OCR accuracy
"""

import logging

import cv2
import numpy as np

from src.config import (
    ADAPTIVE_THRESH_BLOCK_SIZE,
    ADAPTIVE_THRESH_C,
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    CROP_BOX_SCALE,
    CROP_PADDING,
    EAST_INPUT_HEIGHT,
    EAST_INPUT_WIDTH,
    EAST_MEAN,
    MORPH_ITERATIONS,
    MORPH_KERNEL_SIZE,
)

logger = logging.getLogger(__name__)


def resize_for_east(image, width=EAST_INPUT_WIDTH, height=EAST_INPUT_HEIGHT):
    """
    Resize an image so both dimensions are multiples of 32 (EAST requirement).

    Parameters
    ----------
    image : np.ndarray
        Input BGR image (H, W, 3).
    width : int
        Target width (must be a multiple of 32).
    height : int
        Target height (must be a multiple of 32).

    Returns
    -------
    resized : np.ndarray
        Resized image.
    ratio_w : float
        Horizontal scale factor (original_width / new_width).
    ratio_h : float
        Vertical scale factor (original_height / new_height).
    """
    if width % 32 != 0 or height % 32 != 0:
        raise ValueError(
            f"EAST input dimensions must be multiples of 32, got {width}x{height}"
        )

    orig_h, orig_w = image.shape[:2]
    ratio_w = orig_w / float(width)
    ratio_h = orig_h / float(height)

    resized = cv2.resize(image, (width, height))
    logger.debug(
        "Resized image from %dx%d to %dx%d (ratios: %.2f, %.2f)",
        orig_w,
        orig_h,
        width,
        height,
        ratio_w,
        ratio_h,
    )
    return resized, ratio_w, ratio_h


def create_blob(image):
    """
    Create a mean-subtracted blob suitable for EAST forward pass.

    Parameters
    ----------
    image : np.ndarray
        Resized BGR image (H, W, 3) — should already be resized via
        ``resize_for_east``.

    Returns
    -------
    blob : np.ndarray
        4D blob (1, 3, H, W) with mean subtraction applied.
    """
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0,
        size=(image.shape[1], image.shape[0]),
        mean=EAST_MEAN,
        swapRB=True,
        crop=False,
    )
    return blob


def rotate_crop(image, box, angle):
    """
    Extract a rotation-aware crop from the original image using a rotated
    bounding box.

    For boxes with negligible rotation (|angle| < 1°), a simple axis-aligned
    crop is used for speed and robustness. For rotated boxes, we compute the
    affine transform to deskew the region.

    Parameters
    ----------
    image : np.ndarray
        Original BGR image (full resolution).
    box : np.ndarray
        Four corner points of the rotated bounding box, shape (4, 2).
        Points are expected in order: top-left, top-right, bottom-right,
        bottom-left.
    angle : float
        Rotation angle of the bounding box in degrees.

    Returns
    -------
    crop : np.ndarray
        Cropped (and optionally deskewed) text region, or None if the crop
        is invalid (zero area, out of bounds, etc.).
    """
    h, w = image.shape[:2]

    if abs(angle) < 1.0:
        # Fast path: axis-aligned crop
        xs = box[:, 0]
        ys = box[:, 1]
        x_min = max(0, int(np.min(xs)))
        x_max = min(w, int(np.max(xs)))
        y_min = max(0, int(np.min(ys)))
        y_max = min(h, int(np.max(ys)))

        if x_max <= x_min or y_max <= y_min:
            logger.warning(
                "Invalid axis-aligned crop bounds: x(%d-%d), y(%d-%d)",
                x_min,
                x_max,
                y_min,
                y_max,
            )
            return None

        crop = image[y_min:y_max, x_min:x_max]
    else:
        # Rotation-aware crop using the minimum area rect
        rect = cv2.minAreaRect(box.astype(np.float32))
        center, (rect_w, rect_h), rect_angle = rect

        # Ensure width > height (landscape orientation for text)
        if rect_w < rect_h:
            rect_w, rect_h = rect_h, rect_w
            rect_angle += 90.0

        # Get rotation matrix to deskew
        rotation_matrix = cv2.getRotationMatrix2D(center, rect_angle, 1.0)

        # Rotate the entire image
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Crop the deskewed region
        crop_w = int(rect_w)
        crop_h = int(rect_h)
        cx, cy = int(center[0]), int(center[1])

        x_min = max(0, cx - crop_w // 2)
        x_max = min(w, cx + crop_w // 2)
        y_min = max(0, cy - crop_h // 2)
        y_max = min(h, cy + crop_h // 2)

        if x_max <= x_min or y_max <= y_min:
            logger.warning(
                "Invalid rotated crop bounds: x(%d-%d), y(%d-%d)",
                x_min,
                x_max,
                y_min,
                y_max,
            )
            return None

        crop = rotated[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        logger.warning("Crop resulted in empty image")
        return None

    return crop


def add_padding(crop, padding=CROP_PADDING, color=255):
    """
    Add a constant border (padding) around a cropped text region.

    This gives Tesseract breathing room so characters touching the edge
    of the crop are not clipped during recognition.

    Parameters
    ----------
    crop : np.ndarray
        Cropped text region image.
    padding : int
        Number of pixels to add on each side.
    color : int or tuple
        Border color. Use 255 (white) for grayscale, (255, 255, 255) for BGR.

    Returns
    -------
    padded : np.ndarray
        Padded image.
    """
    if crop is None or crop.size == 0:
        return crop

    padded = cv2.copyMakeBorder(
        crop,
        top=padding,
        bottom=padding,
        left=padding,
        right=padding,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded


def enhance_crop(crop):
    """
    Enhance a cropped text region for better OCR accuracy.

    Pipeline:
      1. Convert to grayscale (if BGR).
      2. Apply bilateral filter to reduce noise while preserving edges.
      3. Apply adaptive Gaussian thresholding to binarize the text.

    Parameters
    ----------
    crop : np.ndarray
        Cropped text region (BGR or grayscale).

    Returns
    -------
    enhanced : np.ndarray
        Binarized single-channel image ready for Tesseract.
    """
    if crop is None or crop.size == 0:
        logger.warning("enhance_crop received empty or None image, returning as-is")
        return crop

    # Convert to grayscale if needed
    if len(crop.shape) == 3 and crop.shape[2] == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    elif len(crop.shape) == 2:
        gray = crop.copy()
    else:
        logger.warning(
            "Unexpected image shape %s, attempting grayscale conversion", crop.shape
        )
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Bilateral filter: smooths noise while keeping text edges sharp
    denoised = cv2.bilateralFilter(
        gray,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )

    # Boost local contrast via CLAHE before binarization
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    contrasted = clahe.apply(denoised)

    # Adaptive Gaussian thresholding handles uneven illumination
    binary = cv2.adaptiveThreshold(
        contrasted,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=ADAPTIVE_THRESH_BLOCK_SIZE,
        C=ADAPTIVE_THRESH_C,
    )

    # Morphological closing reconnects broken strokes/thin gaps
    kernel_size = max(1, MORPH_KERNEL_SIZE)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    iterations = max(1, MORPH_ITERATIONS)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return closed


def scale_box(box, scale, image_shape):
    """
    Uniformly scale the box around its centroid while clamping to image bounds.
    """
    if scale <= 1.0:
        return box

    center = np.mean(box, axis=0)
    scaled = (box - center) * scale + center

    h, w = image_shape[:2]
    scaled[:, 0] = np.clip(scaled[:, 0], 0, w - 1)
    scaled[:, 1] = np.clip(scaled[:, 1], 0, h - 1)
    return scaled


def preprocess_crop(image, box, angle, padding=CROP_PADDING):
    """
    Full preprocessing for a single detected text region: crop → pad → enhance.

    This is a convenience function that chains ``rotate_crop``,
    ``add_padding``, and ``enhance_crop`` together.

    If cropping fails (invalid region), returns (None, None) so the caller
    can skip this detection gracefully.

    Parameters
    ----------
    image : np.ndarray
        Original full-resolution BGR image.
    box : np.ndarray
        Four corner points of the rotated bounding box, shape (4, 2).
    angle : float
        Rotation angle of the bounding box in degrees.
    padding : int
        Pixels of padding to add around the crop.

    Returns
    -------
    raw_crop : np.ndarray or None
        The raw (unenhanced) padded crop, useful as a fallback if the
        enhanced version produces poor OCR results.
    enhanced_crop : np.ndarray or None
        The fully preprocessed (grayscale, denoised, binarized) crop
        ready for Tesseract.
    """
    scaled_box = scale_box(box, CROP_BOX_SCALE, image.shape)

    crop = rotate_crop(image, scaled_box, angle)
    if crop is None:
        return None, None

    padded = add_padding(
        crop, padding=padding, color=255 if len(crop.shape) == 2 else (255, 255, 255)
    )
    if padded is None or padded.size == 0:
        return None, None

    enhanced = enhance_crop(padded)

    # Keep the raw padded crop in grayscale for potential fallback
    if len(padded.shape) == 3:
        raw_gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    else:
        raw_gray = padded

    return raw_gray, enhanced


def load_image(image_path):
    """
    Load an image from disk with validation.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    image : np.ndarray
        BGR image.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If OpenCV cannot decode the file.
    """
    import os

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(
            f"Could not decode image (corrupt or unsupported format): {image_path}"
        )

    logger.info("Loaded image %s — shape: %s", image_path, image.shape)
    return image
