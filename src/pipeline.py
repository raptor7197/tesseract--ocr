"""
Pipeline orchestrator for the Scene Text Detection & Recognition system.

This module ties together the detector, preprocessor, and recognizer into
a single cohesive pipeline. It is the main entry point for processing
images programmatically.

Usage:
    from src.pipeline import SceneTextPipeline

    pipeline = SceneTextPipeline()
    results = pipeline.process_image("path/to/image.jpg")
    annotated = pipeline.annotate_image(image, results["detections"])
"""

import json
import logging
import os
import time

import cv2
import numpy as np

from src.config import (
    ANNOTATED_DIR,
    ANNOTATION_BOX_COLOR,
    ANNOTATION_BOX_THICKNESS,
    ANNOTATION_FONT_SCALE,
    ANNOTATION_FONT_THICKNESS,
    ANNOTATION_TEXT_BG_COLOR,
    ANNOTATION_TEXT_COLOR,
    EAST_CONF_THRESHOLD,
    EAST_INPUT_HEIGHT,
    EAST_INPUT_WIDTH,
    EAST_MODEL_PATH,
    EAST_NMS_THRESHOLD,
    RESULTS_DIR,
    SUPPORTED_EXTENSIONS,
    TESSERACT_CONF_THRESHOLD,
)
from src.detector import detect_from_image, load_east_model
from src.preprocessor import load_image, preprocess_crop
from src.recognizer import recognize_with_fallback

logger = logging.getLogger(__name__)


class SceneTextPipeline:
    """
    End-to-end scene text detection and recognition pipeline.

    Loads the EAST model once on initialization, then exposes methods
    for processing single images, batch directories, and annotating
    results onto images.

    Parameters
    ----------
    east_model_path : str
        Path to the frozen EAST model file (.pb).
    east_width : int
        Input width for EAST (must be multiple of 32).
    east_height : int
        Input height for EAST (must be multiple of 32).
    east_conf : float
        Minimum detection confidence for EAST (0.0–1.0).
    east_nms : float
        NMS IoU threshold for EAST (0.0–1.0).
    tess_conf : float
        Minimum word-level confidence for Tesseract (0–100).
    """

    def __init__(
        self,
        east_model_path=EAST_MODEL_PATH,
        east_width=EAST_INPUT_WIDTH,
        east_height=EAST_INPUT_HEIGHT,
        east_conf=EAST_CONF_THRESHOLD,
        east_nms=EAST_NMS_THRESHOLD,
        tess_conf=TESSERACT_CONF_THRESHOLD,
    ):
        self.east_width = east_width
        self.east_height = east_height
        self.east_conf = east_conf
        self.east_nms = east_nms
        self.tess_conf = tess_conf

        logger.info("Initializing SceneTextPipeline...")
        self.net = load_east_model(east_model_path)
        logger.info("Pipeline ready.")

    def process_image(self, image_path):
        """
        Run the full detection + recognition pipeline on a single image.

        Steps:
          1. Load the image from disk.
          2. Run EAST text detection to find text bounding boxes.
          3. For each detected region, crop, preprocess, and run OCR.
          4. Collect results into a structured dictionary.

        Parameters
        ----------
        image_path : str
            Path to the input image file.

        Returns
        -------
        result : dict
            Dictionary containing:
              - ``"image_path"`` (str): Absolute path to the input image.
              - ``"image_size"`` (list[int]): [height, width] of the original image.
              - ``"detections"`` (list[dict]): List of detection results, each with:
                  - ``"id"`` (int): 1-based detection index.
                  - ``"bbox"`` (list[int]): Flattened 8-point bounding box
                    [x1,y1, x2,y2, x3,y3, x4,y4].
                  - ``"text"`` (str): Recognized text string.
                  - ``"confidence"`` (float): OCR confidence (0–100).
                  - ``"source"`` (str): Which crop variant produced the result
                    (``"enhanced"``, ``"raw"``, or ``"none"``).
              - ``"total_detections"`` (int): Number of detections with non-empty text.
              - ``"processing_time_ms"`` (float): Total wall-clock time in milliseconds.
        """
        start_time = time.time()

        # Step 1: Load image
        image = load_image(image_path)
        orig_h, orig_w = image.shape[:2]

        # Step 2: Detect text regions
        boxes, confidences, angles = detect_from_image(
            net=self.net,
            image=image,
            east_width=self.east_width,
            east_height=self.east_height,
            conf_threshold=self.east_conf,
            nms_threshold=self.east_nms,
        )

        # Step 3: Crop, preprocess, and recognize each region
        detections = []
        for i, (box, det_conf, angle) in enumerate(zip(boxes, confidences, angles)):
            raw_crop, enhanced_crop = preprocess_crop(image, box, angle)

            if raw_crop is None and enhanced_crop is None:
                logger.debug("Detection %d: crop failed, skipping.", i + 1)
                continue

            ocr_result, source = recognize_with_fallback(
                enhanced_crop=enhanced_crop,
                raw_crop=raw_crop,
                min_confidence=self.tess_conf,
            )

            # Flatten the box from (4, 2) to a flat list of 8 ints
            bbox_flat = box.flatten().tolist()

            detection = {
                "id": i + 1,
                "bbox": bbox_flat,
                "text": ocr_result["text"],
                "confidence": ocr_result["confidence"],
                "detection_confidence": round(det_conf * 100, 2),
                "source": source,
            }
            detections.append(detection)

            if ocr_result["text"]:
                logger.info(
                    "Detection %d: '%s' (conf=%.1f, source=%s)",
                    i + 1,
                    ocr_result["text"],
                    ocr_result["confidence"],
                    source,
                )
            else:
                logger.debug("Detection %d: no text recognized.", i + 1)

        elapsed_ms = (time.time() - start_time) * 1000.0

        # Filter to only detections that produced text for the count
        text_detections = [d for d in detections if d["text"]]

        result = {
            "image_path": os.path.abspath(image_path),
            "image_size": [orig_h, orig_w],
            "detections": detections,
            "total_detections": len(text_detections),
            "processing_time_ms": round(elapsed_ms, 2),
        }

        logger.info(
            "Processed %s: %d text regions found in %.1f ms",
            image_path,
            len(text_detections),
            elapsed_ms,
        )

        return result

    def process_directory(self, dir_path, recursive=False):
        """
        Batch-process all supported images in a directory.

        Parameters
        ----------
        dir_path : str
            Path to the directory containing images.
        recursive : bool
            If True, also process images in subdirectories.

        Returns
        -------
        results : list[dict]
            List of result dicts (one per image), same format as
            ``process_image`` output.
        errors : list[dict]
            List of ``{"image_path": str, "error": str}`` for images that
            failed to process.
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        image_paths = self._find_images(dir_path, recursive=recursive)

        if not image_paths:
            logger.warning("No supported images found in %s", dir_path)
            return [], []

        logger.info("Batch processing %d images from %s", len(image_paths), dir_path)

        results = []
        errors = []

        for i, img_path in enumerate(image_paths, 1):
            logger.info("[%d/%d] Processing %s ...", i, len(image_paths), img_path)
            try:
                result = self.process_image(img_path)
                results.append(result)
            except Exception as e:
                logger.error("Failed to process %s: %s", img_path, e)
                errors.append({"image_path": img_path, "error": str(e)})

        logger.info(
            "Batch complete: %d succeeded, %d failed", len(results), len(errors)
        )

        return results, errors

    def _find_images(self, dir_path, recursive=False):
        """
        Find all images with supported extensions in a directory.

        Parameters
        ----------
        dir_path : str
            Root directory to search.
        recursive : bool
            Whether to search subdirectories.

        Returns
        -------
        paths : list[str]
            Sorted list of absolute image file paths.
        """
        paths = []

        if recursive:
            for root, _dirs, files in os.walk(dir_path):
                for fname in files:
                    if fname.lower().endswith(SUPPORTED_EXTENSIONS):
                        paths.append(os.path.join(root, fname))
        else:
            for fname in os.listdir(dir_path):
                full = os.path.join(dir_path, fname)
                if os.path.isfile(full) and fname.lower().endswith(
                    SUPPORTED_EXTENSIONS
                ):
                    paths.append(full)

        paths.sort()
        return paths


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def annotate_image(image, detections):
    """
    Draw bounding boxes and recognized text labels on an image.

    Parameters
    ----------
    image : np.ndarray
        Original BGR image (will be copied, not modified in place).
    detections : list[dict]
        List of detection dicts from ``SceneTextPipeline.process_image``.
        Each must contain ``"bbox"`` (list of 8 ints) and ``"text"`` (str).

    Returns
    -------
    annotated : np.ndarray
        Copy of the image with boxes and text drawn on it.
    """
    annotated = image.copy()

    for det in detections:
        text = det.get("text", "")
        if not text:
            continue

        bbox = det["bbox"]
        confidence = det.get("confidence", 0.0)

        # Reshape flat bbox [x1,y1,x2,y2,...] into (4, 2) array of points
        pts = np.array(bbox, dtype=np.int32).reshape(4, 2)

        # Draw the rotated bounding box
        cv2.polylines(
            annotated,
            [pts],
            isClosed=True,
            color=ANNOTATION_BOX_COLOR,
            thickness=ANNOTATION_BOX_THICKNESS,
        )

        # Prepare label text
        label = f"{text} ({confidence:.0f}%)"

        # Position the label just above the top-left corner of the box
        label_x = int(np.min(pts[:, 0]))
        label_y = int(np.min(pts[:, 1])) - 8

        # Ensure label doesn't go above the image
        if label_y < 15:
            label_y = int(np.max(pts[:, 1])) + 20

        # Draw a small background rectangle behind the text for readability
        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            ANNOTATION_FONT_SCALE,
            ANNOTATION_FONT_THICKNESS,
        )
        cv2.rectangle(
            annotated,
            (label_x, label_y - th - 4),
            (label_x + tw + 4, label_y + 4),
            ANNOTATION_TEXT_BG_COLOR,
            cv2.FILLED,
        )

        # Draw the label text
        cv2.putText(
            annotated,
            label,
            (label_x + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            ANNOTATION_FONT_SCALE,
            ANNOTATION_TEXT_COLOR,
            ANNOTATION_FONT_THICKNESS,
        )

    return annotated


def save_results(result, output_dir=None):
    """
    Save pipeline results to disk: annotated image + JSON file.

    Parameters
    ----------
    result : dict
        Output from ``SceneTextPipeline.process_image``.
    output_dir : str or None
        Base output directory. If None, uses the default from config.
        Creates ``annotated/`` and ``results/`` subdirectories.

    Returns
    -------
    saved : dict
        Paths to the saved files:
          - ``"annotated_image"`` (str or None): Path to annotated image.
          - ``"json_file"`` (str): Path to JSON results file.
    """
    annotated_dir = (
        os.path.join(output_dir, "annotated") if output_dir else ANNOTATED_DIR
    )
    results_dir = os.path.join(output_dir, "results") if output_dir else RESULTS_DIR

    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Derive filenames from the input image name
    image_path = result["image_path"]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save JSON results
    json_path = os.path.join(results_dir, f"{base_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON results to %s", json_path)

    # Save annotated image
    annotated_path = None
    if result["detections"]:
        try:
            image = cv2.imread(image_path)
            if image is not None:
                annotated = annotate_image(image, result["detections"])
                annotated_path = os.path.join(
                    annotated_dir, f"{base_name}_annotated.jpg"
                )
                cv2.imwrite(annotated_path, annotated)
                logger.info("Saved annotated image to %s", annotated_path)
            else:
                logger.warning("Could not reload image for annotation: %s", image_path)
        except Exception as e:
            logger.error("Failed to save annotated image: %s", e)

    return {"annotated_image": annotated_path, "json_file": json_path}


def save_batch_results(results, errors=None, output_dir=None):
    """
    Save results from a batch processing run.

    Parameters
    ----------
    results : list[dict]
        List of result dicts from ``SceneTextPipeline.process_directory``.
    errors : list[dict] or None
        List of error dicts, if any.
    output_dir : str or None
        Base output directory.

    Returns
    -------
    summary : dict
        Summary of saved files including a batch summary JSON.
    """
    results_dir = os.path.join(output_dir, "results") if output_dir else RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    saved_files = []
    for result in results:
        saved = save_results(result, output_dir=output_dir)
        saved_files.append(saved)

    # Write a batch summary
    summary = {
        "total_images": len(results) + (len(errors) if errors else 0),
        "successful": len(results),
        "failed": len(errors) if errors else 0,
        "total_text_detections": sum(r["total_detections"] for r in results),
        "avg_processing_time_ms": (
            round(sum(r["processing_time_ms"] for r in results) / len(results), 2)
            if results
            else 0.0
        ),
        "errors": errors or [],
    }

    summary_path = os.path.join(results_dir, "_batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved batch summary to %s", summary_path)

    return summary


# ---------------------------------------------------------------------------
# Convenience function for quick single-image processing
# ---------------------------------------------------------------------------


def process_image(image_path, **kwargs):
    """
    One-shot convenience function: create a pipeline, process one image, return results.

    This creates a new ``SceneTextPipeline`` instance each time, so it's
    best suited for scripts that process a single image. For batch work,
    instantiate the pipeline once and reuse it.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    **kwargs
        Keyword arguments forwarded to ``SceneTextPipeline.__init__``
        (e.g. ``east_conf``, ``tess_conf``, ``east_width``).

    Returns
    -------
    result : dict
        Same as ``SceneTextPipeline.process_image`` output.
    """
    pipeline = SceneTextPipeline(**kwargs)
    return pipeline.process_image(image_path)
