"""
EAST (Efficient and Accurate Scene Text Detector) integration.

This module handles:
  - Loading the pre-trained frozen EAST model via OpenCV's DNN module
  - Running forward passes to get score and geometry maps
  - Decoding rotated bounding boxes from EAST output
  - Applying Non-Maximum Suppression to filter overlapping detections

No TensorFlow/PyTorch dependency — everything runs through cv2.dnn.
"""

import logging
import math
import os

import cv2
import numpy as np

from src.config import (
    EAST_CONF_THRESHOLD,
    EAST_MODEL_PATH,
    EAST_NMS_THRESHOLD,
    EAST_OUTPUT_LAYERS,
)

logger = logging.getLogger(__name__)


def load_east_model(model_path=EAST_MODEL_PATH):
    """
    Load the frozen EAST text detection model into an OpenCV DNN network.

    This should be called once at startup and the returned net object reused
    for all subsequent detections.

    Parameters
    ----------
    model_path : str
        Path to the ``frozen_east_text_detection.pb`` file.

    Returns
    -------
    net : cv2.dnn.Net
        Loaded network ready for forward passes.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the given path.
    RuntimeError
        If OpenCV fails to load the model (corrupt file, unsupported format).
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"EAST model not found at '{model_path}'. "
            f"Run setup.sh to download it, or set EAST_MODEL_PATH in config.py."
        )

    logger.info("Loading EAST model from %s ...", model_path)
    try:
        net = cv2.dnn.readNet(model_path)
    except cv2.error as e:
        raise RuntimeError(f"Failed to load EAST model: {e}") from e

    logger.info("EAST model loaded successfully.")
    return net


def detect(
    net,
    blob,
    orig_h,
    orig_w,
    new_h,
    new_w,
    conf_threshold=EAST_CONF_THRESHOLD,
    nms_threshold=EAST_NMS_THRESHOLD,
):
    """
    Run EAST text detection on a preprocessed image blob.

    Parameters
    ----------
    net : cv2.dnn.Net
        The loaded EAST network (from ``load_east_model``).
    blob : np.ndarray
        4D blob created by ``preprocessor.create_blob`` (1, 3, H, W).
    orig_h : int
        Original image height (before resizing).
    orig_w : int
        Original image width (before resizing).
    new_h : int
        Height of the resized image fed to EAST.
    new_w : int
        Width of the resized image fed to EAST.
    conf_threshold : float
        Minimum confidence score for a detection to be kept (0.0–1.0).
    nms_threshold : float
        IoU threshold for Non-Maximum Suppression (0.0–1.0).

    Returns
    -------
    boxes : list of np.ndarray
        List of rotated bounding boxes. Each box is an array of shape (4, 2)
        containing the four corner points in the original image coordinate
        space: [top-left, top-right, bottom-right, bottom-left].
    confidences : list of float
        Confidence score for each corresponding box.
    angles : list of float
        Rotation angle in degrees for each corresponding box.
    """
    # Forward pass
    net.setInput(blob)
    scores, geometry = net.forward(list(EAST_OUTPUT_LAYERS))

    # Decode raw predictions into boxes and confidences
    raw_boxes, raw_confs, raw_angles = decode_predictions(
        scores, geometry, conf_threshold
    )

    if len(raw_boxes) == 0:
        logger.info("No text regions detected above confidence %.2f", conf_threshold)
        return [], [], []

    logger.debug(
        "Decoded %d candidate boxes (before NMS) from EAST output", len(raw_boxes)
    )

    # Apply Non-Maximum Suppression
    kept_boxes, kept_confs, kept_angles = non_max_suppression(
        raw_boxes, raw_confs, raw_angles, nms_threshold
    )

    logger.info(
        "EAST detected %d text regions after NMS (from %d candidates)",
        len(kept_boxes),
        len(raw_boxes),
    )

    # Scale boxes back to original image dimensions
    ratio_w = orig_w / float(new_w)
    ratio_h = orig_h / float(new_h)

    scaled_boxes = []
    for box in kept_boxes:
        scaled = box.astype(np.float64)
        scaled[:, 0] *= ratio_w
        scaled[:, 1] *= ratio_h
        scaled_boxes.append(scaled.astype(np.int32))

    return scaled_boxes, kept_confs, kept_angles


def decode_predictions(scores, geometry, conf_threshold=EAST_CONF_THRESHOLD):
    """
    Decode the EAST model's raw output maps into rotated bounding boxes.

    The EAST model outputs two tensors:
      - **scores**: shape (1, 1, H/4, W/4) — per-pixel text confidence.
      - **geometry**: shape (1, 5, H/4, W/4) — per-pixel bounding box
        parameters: [d_top, d_right, d_bottom, d_left, angle].

    For each pixel whose score exceeds ``conf_threshold``, we reconstruct
    the rotated bounding box in the resized image coordinate space.

    Parameters
    ----------
    scores : np.ndarray
        Score map from EAST, shape (1, 1, H/4, W/4).
    geometry : np.ndarray
        Geometry map from EAST, shape (1, 5, H/4, W/4).
    conf_threshold : float
        Minimum confidence to keep a detection.

    Returns
    -------
    boxes : list of np.ndarray
        Each box is shape (4, 2) with corner points in resized-image coords.
    confidences : list of float
        Confidence for each box.
    angles : list of float
        Rotation angle in degrees for each box.
    """
    num_rows, num_cols = scores.shape[2:4]

    boxes = []
    confidences = []
    angles = []

    for y in range(num_rows):
        # Extract row data
        score_row = scores[0, 0, y]
        d_top = geometry[0, 0, y]
        d_right = geometry[0, 1, y]
        d_bottom = geometry[0, 2, y]
        d_left = geometry[0, 3, y]
        angle_row = geometry[0, 4, y]

        for x in range(num_cols):
            if score_row[x] < conf_threshold:
                continue

            # Compute the offset — each cell in the output corresponds
            # to a 4×4 block in the input image
            offset_x = x * 4.0
            offset_y = y * 4.0

            angle = angle_row[x]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Box height and width from the four distances
            h = d_top[x] + d_bottom[x]
            w = d_right[x] + d_left[x]

            if h <= 0 or w <= 0:
                continue

            # Compute the center of the bounding box
            # The offset point is roughly the center-bottom of the text line
            end_x = offset_x + (cos_a * d_right[x]) + (sin_a * d_bottom[x])
            end_y = offset_y - (sin_a * d_right[x]) + (cos_a * d_bottom[x])

            start_x = end_x - w
            start_y = end_y - h

            # Compute the four rotated corner points
            cx = (start_x + end_x) / 2.0
            cy = (start_y + end_y) / 2.0

            box = _rotated_rect_corners(cx, cy, w, h, -angle)

            boxes.append(box)
            confidences.append(float(score_row[x]))
            angles.append(float(math.degrees(-angle)))

    return boxes, confidences, angles


def _rotated_rect_corners(cx, cy, w, h, angle_rad):
    """
    Compute the four corner points of a rotated rectangle.

    Parameters
    ----------
    cx, cy : float
        Center of the rectangle.
    w, h : float
        Width and height of the rectangle.
    angle_rad : float
        Rotation angle in radians.

    Returns
    -------
    corners : np.ndarray
        Shape (4, 2) — [top-left, top-right, bottom-right, bottom-left].
    """
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Half extents
    hw = w / 2.0
    hh = h / 2.0

    # Corner offsets relative to center (before rotation)
    # Order: top-left, top-right, bottom-right, bottom-left
    dx = [-hw, hw, hw, -hw]
    dy = [-hh, -hh, hh, hh]

    corners = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        corners[i, 0] = cx + dx[i] * cos_a - dy[i] * sin_a
        corners[i, 1] = cy + dx[i] * sin_a + dy[i] * cos_a

    return corners


def non_max_suppression(boxes, confidences, angles, nms_threshold=EAST_NMS_THRESHOLD):
    """
    Apply Non-Maximum Suppression to remove overlapping bounding boxes.

    Uses a simple IoU-based greedy NMS on the axis-aligned bounding rectangles
    enclosing each rotated box. This is sufficient for most scene text
    scenarios and avoids the complexity of rotated-rect NMS.

    Parameters
    ----------
    boxes : list of np.ndarray
        Rotated bounding boxes, each shape (4, 2).
    confidences : list of float
        Confidence scores corresponding to each box.
    angles : list of float
        Rotation angles in degrees corresponding to each box.
    nms_threshold : float
        IoU threshold. Boxes with IoU above this with a higher-scoring box
        are suppressed.

    Returns
    -------
    kept_boxes : list of np.ndarray
        Surviving boxes after NMS.
    kept_confs : list of float
        Corresponding confidences.
    kept_angles : list of float
        Corresponding angles.
    """
    if len(boxes) == 0:
        return [], [], []

    # Convert rotated boxes to axis-aligned bounding rects for NMS
    aa_rects = []
    for box in boxes:
        x_min = float(np.min(box[:, 0]))
        y_min = float(np.min(box[:, 1]))
        w = float(np.max(box[:, 0]) - x_min)
        h = float(np.max(box[:, 1]) - y_min)
        aa_rects.append([x_min, y_min, w, h])

    # Use OpenCV's built-in NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes=aa_rects,
        scores=confidences,
        score_threshold=0.0,  # already filtered by conf_threshold
        nms_threshold=nms_threshold,
    )

    # OpenCV returns indices in different formats depending on version
    if indices is None or len(indices) == 0:
        return [], [], []

    # Flatten — older OpenCV returns shape (N, 1), newer returns (N,)
    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()
    else:
        indices = [i for i in indices]

    kept_boxes = [boxes[i] for i in indices]
    kept_confs = [confidences[i] for i in indices]
    kept_angles = [angles[i] for i in indices]

    return kept_boxes, kept_confs, kept_angles


def detect_from_image(
    net,
    image,
    east_width,
    east_height,
    conf_threshold=EAST_CONF_THRESHOLD,
    nms_threshold=EAST_NMS_THRESHOLD,
):
    """
    High-level convenience function: run full detection on a raw BGR image.

    This combines resizing, blob creation, and detection into one call.
    Useful when you don't need fine-grained control over each step.

    Parameters
    ----------
    net : cv2.dnn.Net
        Loaded EAST model.
    image : np.ndarray
        Original BGR image.
    east_width : int
        Target width for EAST (multiple of 32).
    east_height : int
        Target height for EAST (multiple of 32).
    conf_threshold : float
        Detection confidence threshold.
    nms_threshold : float
        NMS IoU threshold.

    Returns
    -------
    boxes : list of np.ndarray
        Detected text bounding boxes in original image coordinates.
    confidences : list of float
        Detection confidence scores.
    angles : list of float
        Box rotation angles in degrees.
    """
    from src.preprocessor import create_blob, resize_for_east

    orig_h, orig_w = image.shape[:2]

    resized, _, _ = resize_for_east(image, width=east_width, height=east_height)
    blob = create_blob(resized)

    return detect(
        net,
        blob,
        orig_h=orig_h,
        orig_w=orig_w,
        new_h=east_height,
        new_w=east_width,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )
