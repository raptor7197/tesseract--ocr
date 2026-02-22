"""
Evaluation script for the Natural Scene Text Detection & Recognition pipeline.

Computes detection and recognition metrics by comparing pipeline predictions
against ground truth annotations.

Detection metrics:
  - Precision, Recall, F1-Score (IoU-based matching)

Recognition metrics:
  - Word-level Accuracy (exact match)
  - Case-insensitive Accuracy
  - Character Error Rate (CER) via edit distance

Usage:
    # Evaluate predictions against ground truth
    python evaluate.py --predictions output/results/ --ground-truth data/ground_truth/

    # Run pipeline on images first, then evaluate
    python evaluate.py --images data/sample_images/ --ground-truth data/ground_truth/ --output output/

    # Custom IoU threshold
    python evaluate.py --predictions output/results/ --ground-truth data/ground_truth/ --iou 0.6

Ground truth format (JSON per image):
    {
        "image": "sign.jpg",
        "annotations": [
            {
                "bbox": [x1, y1, x2, y2, x3, y3, x4, y4],
                "text": "STOP"
            },
            ...
        ]
    }
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np

from src.config import LOG_FORMAT, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edit distance (Levenshtein) — pure Python, no external dependency
# ---------------------------------------------------------------------------


def levenshtein_distance(s1, s2):
    """
    Compute the Levenshtein (edit) distance between two strings.

    Uses the standard dynamic programming approach with O(min(m, n)) space.

    Parameters
    ----------
    s1 : str
        First string.
    s2 : str
        Second string.

    Returns
    -------
    distance : int
        Minimum number of single-character edits (insert, delete, substitute)
        to transform s1 into s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insert_cost = prev_row[j + 1] + 1
            delete_cost = curr_row[j] + 1
            substitute_cost = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insert_cost, delete_cost, substitute_cost))
        prev_row = curr_row

    return prev_row[-1]


def character_error_rate(prediction, ground_truth):
    """
    Compute the Character Error Rate (CER) between a prediction and ground truth.

    CER = edit_distance(prediction, ground_truth) / len(ground_truth)

    Parameters
    ----------
    prediction : str
        Predicted text string.
    ground_truth : str
        Ground truth text string.

    Returns
    -------
    cer : float
        Character error rate. 0.0 means perfect match, values > 1.0 are
        possible when the prediction is much longer than the ground truth.
        Returns 0.0 if both strings are empty, 1.0 if only ground truth is empty.
    """
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0

    dist = levenshtein_distance(prediction, ground_truth)
    return dist / len(ground_truth)


# ---------------------------------------------------------------------------
# IoU computation for bounding box matching
# ---------------------------------------------------------------------------


def polygon_to_axis_aligned(bbox_flat):
    """
    Convert a flat 8-point polygon bbox to an axis-aligned bounding rectangle.

    Parameters
    ----------
    bbox_flat : list[int or float]
        Flattened 8-point bbox: [x1, y1, x2, y2, x3, y3, x4, y4].

    Returns
    -------
    rect : tuple
        (x_min, y_min, x_max, y_max) axis-aligned bounding rectangle.
    """
    xs = [bbox_flat[i] for i in range(0, 8, 2)]
    ys = [bbox_flat[i] for i in range(1, 8, 2)]
    return (min(xs), min(ys), max(xs), max(ys))


def compute_iou(rect_a, rect_b):
    """
    Compute Intersection over Union (IoU) between two axis-aligned rectangles.

    Parameters
    ----------
    rect_a : tuple
        (x_min, y_min, x_max, y_max) for rectangle A.
    rect_b : tuple
        (x_min, y_min, x_max, y_max) for rectangle B.

    Returns
    -------
    iou : float
        Intersection over Union, in [0.0, 1.0].
    """
    x_left = max(rect_a[0], rect_b[0])
    y_top = max(rect_a[1], rect_b[1])
    x_right = min(rect_a[2], rect_b[2])
    y_bottom = min(rect_a[3], rect_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (rect_a[2] - rect_a[0]) * (rect_a[3] - rect_a[1])
    area_b = (rect_b[2] - rect_b[0]) * (rect_b[3] - rect_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ---------------------------------------------------------------------------
# Matching predictions to ground truth
# ---------------------------------------------------------------------------


def match_detections(pred_detections, gt_annotations, iou_threshold=0.5):
    """
    Match predicted detections to ground truth annotations using greedy
    IoU-based matching.

    Each ground truth annotation is matched to at most one prediction
    (the one with the highest IoU above the threshold). Predictions are
    processed in order of decreasing detection confidence.

    Parameters
    ----------
    pred_detections : list[dict]
        Predicted detections, each with ``"bbox"`` (list of 8 ints) and
        ``"text"`` (str). Optionally ``"confidence"`` and
        ``"detection_confidence"``.
    gt_annotations : list[dict]
        Ground truth annotations, each with ``"bbox"`` (list of 8 ints)
        and ``"text"`` (str).
    iou_threshold : float
        Minimum IoU for a prediction to be matched to a ground truth box.

    Returns
    -------
    matches : list[dict]
        List of matched pairs, each containing:
          - ``"pred"`` (dict): The prediction dict.
          - ``"gt"`` (dict): The matched ground truth dict.
          - ``"iou"`` (float): IoU between the two boxes.
    unmatched_preds : list[dict]
        Predictions that could not be matched (false positives).
    unmatched_gts : list[dict]
        Ground truth annotations that were not matched (false negatives).
    """
    if not pred_detections or not gt_annotations:
        return [], list(pred_detections), list(gt_annotations)

    # Sort predictions by detection confidence (descending) for greedy matching
    sorted_preds = sorted(
        pred_detections,
        key=lambda d: d.get("detection_confidence", d.get("confidence", 0)),
        reverse=True,
    )

    # Convert all bboxes to axis-aligned rects for IoU computation
    pred_rects = [polygon_to_axis_aligned(p["bbox"]) for p in sorted_preds]
    gt_rects = [polygon_to_axis_aligned(g["bbox"]) for g in gt_annotations]

    matched_gt_indices = set()
    matches = []
    unmatched_preds = []

    for pred_idx, pred in enumerate(sorted_preds):
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_rect in enumerate(gt_rects):
            if gt_idx in matched_gt_indices:
                continue

            iou = compute_iou(pred_rects[pred_idx], gt_rect)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt_indices.add(best_gt_idx)
            matches.append(
                {
                    "pred": pred,
                    "gt": gt_annotations[best_gt_idx],
                    "iou": best_iou,
                }
            )
        else:
            unmatched_preds.append(pred)

    unmatched_gts = [
        gt for i, gt in enumerate(gt_annotations) if i not in matched_gt_indices
    ]

    return matches, unmatched_preds, unmatched_gts


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_detection_metrics(all_matches, all_unmatched_preds, all_unmatched_gts):
    """
    Compute aggregate detection metrics across all images.

    Parameters
    ----------
    all_matches : list[dict]
        All matched (pred, gt) pairs across every image.
    all_unmatched_preds : list[dict]
        All false positive predictions across every image.
    all_unmatched_gts : list[dict]
        All missed ground truth annotations across every image.

    Returns
    -------
    metrics : dict
        Dictionary with keys: ``precision``, ``recall``, ``f1``,
        ``true_positives``, ``false_positives``, ``false_negatives``.
    """
    tp = len(all_matches)
    fp = len(all_unmatched_preds)
    fn = len(all_unmatched_gts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def compute_recognition_metrics(all_matches):
    """
    Compute aggregate recognition metrics across all matched detections.

    Only matched detections (true positives from the detection stage) are
    evaluated for recognition quality.

    Parameters
    ----------
    all_matches : list[dict]
        Matched pairs, each with ``"pred"`` and ``"gt"`` dicts containing
        ``"text"`` fields.

    Returns
    -------
    metrics : dict
        Dictionary with keys:
          - ``word_accuracy``: Fraction of exact text matches.
          - ``word_accuracy_ci``: Fraction of case-insensitive matches.
          - ``avg_cer``: Average Character Error Rate.
          - ``median_cer``: Median Character Error Rate.
          - ``total_matched``: Number of matched detections evaluated.
          - ``per_match_details``: List of per-match breakdowns.
    """
    if not all_matches:
        return {
            "word_accuracy": 0.0,
            "word_accuracy_ci": 0.0,
            "avg_cer": 1.0,
            "median_cer": 1.0,
            "total_matched": 0,
            "per_match_details": [],
        }

    exact_matches = 0
    ci_matches = 0
    cer_values = []
    details = []

    for match in all_matches:
        pred_text = match["pred"].get("text", "").strip()
        gt_text = match["gt"].get("text", "").strip()

        # Exact match
        is_exact = pred_text == gt_text
        if is_exact:
            exact_matches += 1

        # Case-insensitive match
        is_ci = pred_text.lower() == gt_text.lower()
        if is_ci:
            ci_matches += 1

        # Character Error Rate
        cer = character_error_rate(pred_text, gt_text)
        cer_values.append(cer)

        details.append(
            {
                "ground_truth": gt_text,
                "prediction": pred_text,
                "exact_match": is_exact,
                "case_insensitive_match": is_ci,
                "cer": round(cer, 4),
                "iou": round(match["iou"], 4),
            }
        )

    n = len(all_matches)
    avg_cer = sum(cer_values) / n
    sorted_cer = sorted(cer_values)
    median_cer = (
        sorted_cer[n // 2]
        if n % 2 == 1
        else (sorted_cer[n // 2 - 1] + sorted_cer[n // 2]) / 2.0
    )

    return {
        "word_accuracy": round(exact_matches / n, 4),
        "word_accuracy_ci": round(ci_matches / n, 4),
        "avg_cer": round(avg_cer, 4),
        "median_cer": round(median_cer, 4),
        "total_matched": n,
        "per_match_details": details,
    }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def load_ground_truth(gt_dir):
    """
    Load all ground truth JSON files from a directory.

    Each JSON file should follow the format:
        {
            "image": "filename.jpg",
            "annotations": [
                {"bbox": [x1,y1,...,x4,y4], "text": "WORD"},
                ...
            ]
        }

    Parameters
    ----------
    gt_dir : str
        Path to the ground truth directory.

    Returns
    -------
    gt_map : dict
        Mapping from image base name (without extension) to list of annotations.
    """
    gt_map = {}

    if not os.path.isdir(gt_dir):
        raise NotADirectoryError(f"Ground truth directory not found: {gt_dir}")

    for fname in sorted(os.listdir(gt_dir)):
        if not fname.lower().endswith(".json"):
            continue

        fpath = os.path.join(gt_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load ground truth file %s: %s", fpath, e)
            continue

        # Determine the key: use the "image" field if present, else the filename stem
        if "image" in data:
            key = os.path.splitext(os.path.basename(data["image"]))[0]
        else:
            key = os.path.splitext(fname)[0]

        annotations = data.get("annotations", [])

        # Validate annotations
        valid = []
        for ann in annotations:
            bbox = ann.get("bbox")
            text = ann.get("text", "")
            if bbox and len(bbox) == 8:
                valid.append({"bbox": bbox, "text": text})
            else:
                logger.warning(
                    "Skipping annotation with invalid bbox in %s: %s", fpath, bbox
                )

        gt_map[key] = valid
        logger.debug("Loaded %d annotations for '%s'", len(valid), key)

    logger.info("Loaded ground truth for %d images from %s", len(gt_map), gt_dir)
    return gt_map


def load_predictions(pred_dir):
    """
    Load all prediction JSON files from a directory.

    These are the output files produced by the pipeline (via ``save_results``).
    Each file is expected to have the standard pipeline output format with
    ``"image_path"`` and ``"detections"`` fields.

    Parameters
    ----------
    pred_dir : str
        Path to the predictions directory.

    Returns
    -------
    pred_map : dict
        Mapping from image base name (without extension) to list of detections.
    """
    pred_map = {}

    if not os.path.isdir(pred_dir):
        raise NotADirectoryError(f"Predictions directory not found: {pred_dir}")

    for fname in sorted(os.listdir(pred_dir)):
        if not fname.lower().endswith(".json"):
            continue
        # Skip batch summary files
        if fname.startswith("_"):
            continue

        fpath = os.path.join(pred_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load prediction file %s: %s", fpath, e)
            continue

        # Determine key from image_path or filename
        if "image_path" in data:
            key = os.path.splitext(os.path.basename(data["image_path"]))[0]
        else:
            key = os.path.splitext(fname)[0]

        detections = data.get("detections", [])
        pred_map[key] = detections
        logger.debug("Loaded %d detections for '%s'", len(detections), key)

    logger.info("Loaded predictions for %d images from %s", len(pred_map), pred_dir)
    return pred_map


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    detection_metrics, recognition_metrics, per_image_results, iou_threshold
):
    """
    Generate a formatted evaluation report string.

    Parameters
    ----------
    detection_metrics : dict
        Output of ``compute_detection_metrics``.
    recognition_metrics : dict
        Output of ``compute_recognition_metrics``.
    per_image_results : list[dict]
        Per-image breakdown of results.
    iou_threshold : float
        The IoU threshold used for matching.

    Returns
    -------
    report : str
        Formatted multi-line report string suitable for printing or saving.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  SCENE TEXT DETECTION & RECOGNITION — EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Detection metrics
    lines.append("DETECTION METRICS (IoU threshold = {:.2f})".format(iou_threshold))
    lines.append("-" * 50)
    lines.append(
        "  Precision:        {:.4f}  ({}/{})".format(
            detection_metrics["precision"],
            detection_metrics["true_positives"],
            detection_metrics["true_positives"] + detection_metrics["false_positives"],
        )
    )
    lines.append(
        "  Recall:           {:.4f}  ({}/{})".format(
            detection_metrics["recall"],
            detection_metrics["true_positives"],
            detection_metrics["true_positives"] + detection_metrics["false_negatives"],
        )
    )
    lines.append("  F1-Score:         {:.4f}".format(detection_metrics["f1"]))
    lines.append("  True Positives:   {}".format(detection_metrics["true_positives"]))
    lines.append("  False Positives:  {}".format(detection_metrics["false_positives"]))
    lines.append("  False Negatives:  {}".format(detection_metrics["false_negatives"]))
    lines.append("")

    # Recognition metrics
    lines.append("RECOGNITION METRICS (on matched detections)")
    lines.append("-" * 50)
    lines.append(
        "  Matched Pairs:           {}".format(recognition_metrics["total_matched"])
    )
    lines.append(
        "  Word Accuracy (exact):   {:.4f}".format(recognition_metrics["word_accuracy"])
    )
    lines.append(
        "  Word Accuracy (CI):      {:.4f}".format(
            recognition_metrics["word_accuracy_ci"]
        )
    )
    lines.append(
        "  Avg CER:                 {:.4f}".format(recognition_metrics["avg_cer"])
    )
    lines.append(
        "  Median CER:              {:.4f}".format(recognition_metrics["median_cer"])
    )
    lines.append("")

    # Per-image summary
    lines.append("PER-IMAGE BREAKDOWN")
    lines.append("-" * 50)
    lines.append(
        "{:<30s}  {:>4s}  {:>4s}  {:>4s}  {:>4s}  {:>6s}".format(
            "Image", "TP", "FP", "FN", "GT", "AvgCER"
        )
    )
    lines.append("-" * 70)

    for img_result in per_image_results:
        name = img_result["image"]
        tp = img_result["true_positives"]
        fp = img_result["false_positives"]
        fn = img_result["false_negatives"]
        gt = img_result["ground_truth_count"]
        avg_cer = img_result.get("avg_cer", "N/A")
        cer_str = "{:.4f}".format(avg_cer) if isinstance(avg_cer, float) else avg_cer

        # Truncate long names
        if len(name) > 28:
            name = name[:25] + "..."

        lines.append(
            "{:<30s}  {:>4d}  {:>4d}  {:>4d}  {:>4d}  {:>6s}".format(
                name, tp, fp, fn, gt, cer_str
            )
        )

    lines.append("")

    # Per-match recognition details (up to 50 entries)
    details = recognition_metrics.get("per_match_details", [])
    if details:
        lines.append("RECOGNITION DETAILS (up to 50 matches)")
        lines.append("-" * 70)
        lines.append(
            "{:<20s}  {:<20s}  {:>5s}  {:>5s}  {:>6s}".format(
                "Ground Truth", "Prediction", "Exact", "CI", "CER"
            )
        )
        lines.append("-" * 70)

        for detail in details[:50]:
            gt_text = detail["ground_truth"]
            pred_text = detail["prediction"]
            exact = "Yes" if detail["exact_match"] else "No"
            ci = "Yes" if detail["case_insensitive_match"] else "No"
            cer = "{:.4f}".format(detail["cer"])

            # Truncate long strings
            if len(gt_text) > 18:
                gt_text = gt_text[:15] + "..."
            if len(pred_text) > 18:
                pred_text = pred_text[:15] + "..."

            lines.append(
                "{:<20s}  {:<20s}  {:>5s}  {:>5s}  {:>6s}".format(
                    gt_text, pred_text, exact, ci, cer
                )
            )

        if len(details) > 50:
            lines.append(
                "  ... and {} more matches (truncated)".format(len(details) - 50)
            )

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------


def evaluate(pred_map, gt_map, iou_threshold=0.5):
    """
    Run the full evaluation: match detections, compute metrics, generate report.

    Parameters
    ----------
    pred_map : dict
        Mapping from image name to list of prediction dicts.
    gt_map : dict
        Mapping from image name to list of ground truth annotation dicts.
    iou_threshold : float
        IoU threshold for detection matching.

    Returns
    -------
    report : str
        Formatted evaluation report.
    results : dict
        Structured results dictionary containing all metrics and per-image details.
    """
    all_matches = []
    all_unmatched_preds = []
    all_unmatched_gts = []
    per_image_results = []

    # Get the union of all image keys
    all_keys = sorted(set(list(gt_map.keys()) + list(pred_map.keys())))

    for key in all_keys:
        gt_anns = gt_map.get(key, [])
        pred_dets = pred_map.get(key, [])

        # Only consider detections that actually have text for recognition eval,
        # but use all detections (including empty) for detection eval
        matches, unmatched_preds, unmatched_gts = match_detections(
            pred_dets, gt_anns, iou_threshold=iou_threshold
        )

        all_matches.extend(matches)
        all_unmatched_preds.extend(unmatched_preds)
        all_unmatched_gts.extend(unmatched_gts)

        # Per-image CER
        img_cer_values = []
        for m in matches:
            pred_text = m["pred"].get("text", "").strip()
            gt_text = m["gt"].get("text", "").strip()
            if gt_text:
                img_cer_values.append(character_error_rate(pred_text, gt_text))

        per_image_results.append(
            {
                "image": key,
                "true_positives": len(matches),
                "false_positives": len(unmatched_preds),
                "false_negatives": len(unmatched_gts),
                "ground_truth_count": len(gt_anns),
                "prediction_count": len(pred_dets),
                "avg_cer": (
                    round(sum(img_cer_values) / len(img_cer_values), 4)
                    if img_cer_values
                    else "N/A"
                ),
            }
        )

        if not gt_anns and not pred_dets:
            logger.debug("Image '%s': no GT and no predictions (skipped)", key)
        elif not gt_anns:
            logger.info(
                "Image '%s': %d predictions but no ground truth", key, len(pred_dets)
            )
        elif not pred_dets:
            logger.info(
                "Image '%s': %d GT annotations but no predictions", key, len(gt_anns)
            )
        else:
            logger.info(
                "Image '%s': %d matches, %d FP, %d FN",
                key,
                len(matches),
                len(unmatched_preds),
                len(unmatched_gts),
            )

    # Compute aggregate metrics
    detection_metrics = compute_detection_metrics(
        all_matches, all_unmatched_preds, all_unmatched_gts
    )
    recognition_metrics = compute_recognition_metrics(all_matches)

    # Generate report
    report = generate_report(
        detection_metrics, recognition_metrics, per_image_results, iou_threshold
    )

    # Structured results
    results = {
        "iou_threshold": iou_threshold,
        "detection": detection_metrics,
        "recognition": {
            k: v for k, v in recognition_metrics.items() if k != "per_match_details"
        },
        "per_image": per_image_results,
        "recognition_details": recognition_metrics.get("per_match_details", []),
    }

    return report, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Evaluate scene text detection & recognition against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate.py --predictions output/results/ --ground-truth data/ground_truth/\n"
            "  python evaluate.py --images data/sample_images/ --ground-truth data/ground_truth/\n"
            "  python evaluate.py --predictions output/results/ --ground-truth data/ground_truth/ --iou 0.6\n"
        ),
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--predictions",
        "-p",
        help="Directory containing prediction JSON files (from pipeline output).",
    )
    input_group.add_argument(
        "--images",
        help="Directory of images to run the pipeline on before evaluating. "
        "Results are saved to --output before evaluation.",
    )

    parser.add_argument(
        "--ground-truth",
        "-g",
        required=True,
        help="Directory containing ground truth JSON annotation files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Output directory for saving evaluation report and results (default: output/).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to ground truth (default: 0.5).",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=False,
        help="Also save structured results as a JSON file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    args = parser.parse_args(argv)

    # Validate IoU threshold
    if not 0.0 < args.iou <= 1.0:
        parser.error(f"--iou must be between 0.0 (exclusive) and 1.0, got {args.iou}")

    return args


def main(argv=None):
    """Main entry point for the evaluation script."""
    args = parse_args(argv)

    # Set log level
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    # Load ground truth
    logger.info("Loading ground truth from: %s", args.ground_truth)
    gt_map = load_ground_truth(args.ground_truth)

    if not gt_map:
        logger.error("No ground truth annotations found. Cannot evaluate.")
        print("Error: No ground truth annotations found in", args.ground_truth)
        sys.exit(1)

    # Get predictions — either from existing files or by running the pipeline
    if args.predictions:
        logger.info("Loading predictions from: %s", args.predictions)
        pred_map = load_predictions(args.predictions)
    else:
        # Run the pipeline on images first
        logger.info("Running pipeline on images in: %s", args.images)

        if not os.path.isdir(args.images):
            print(f"Error: Images directory not found: {args.images}")
            sys.exit(1)

        from src.pipeline import SceneTextPipeline, save_batch_results

        pipeline = SceneTextPipeline()
        results, errors = pipeline.process_directory(args.images)

        if errors:
            logger.warning("%d images failed to process", len(errors))

        # Save results so they can be reused
        save_batch_results(results, errors=errors, output_dir=args.output)

        # Build pred_map from results
        pred_map = {}
        for result in results:
            key = os.path.splitext(os.path.basename(result["image_path"]))[0]
            pred_map[key] = result.get("detections", [])

        logger.info("Pipeline produced predictions for %d images", len(pred_map))

    if not pred_map:
        logger.error("No predictions found. Cannot evaluate.")
        print("Error: No prediction files found.")
        sys.exit(1)

    # Check overlap between prediction and GT keys
    pred_keys = set(pred_map.keys())
    gt_keys = set(gt_map.keys())
    common_keys = pred_keys & gt_keys

    if not common_keys:
        logger.error(
            "No matching images between predictions and ground truth. "
            "Prediction keys: %s, GT keys: %s",
            sorted(pred_keys)[:5],
            sorted(gt_keys)[:5],
        )
        print("Error: No matching image names between predictions and ground truth.")
        print(
            f"  Prediction images: {sorted(pred_keys)[:5]}{'...' if len(pred_keys) > 5 else ''}"
        )
        print(
            f"  Ground truth images: {sorted(gt_keys)[:5]}{'...' if len(gt_keys) > 5 else ''}"
        )
        sys.exit(1)

    logger.info(
        "Found %d matching images (%d predictions, %d ground truths)",
        len(common_keys),
        len(pred_keys),
        len(gt_keys),
    )

    # Run evaluation
    report, results = evaluate(pred_map, gt_map, iou_threshold=args.iou)

    # Print report
    print(report)

    # Save report
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Saved evaluation report to %s", report_path)

    # Optionally save structured JSON results
    if args.save_json:
        json_path = os.path.join(args.output, "evaluation_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Saved JSON results to %s", json_path)
        print(f"\nJSON results saved to: {json_path}")

    print(f"\nReport saved to: {report_path}")

    # Return non-zero exit code if F1 is below 0.5 (useful for CI)
    f1 = results["detection"]["f1"]
    if f1 < 0.5:
        logger.warning("F1-score %.4f is below 0.5 threshold", f1)

    return results


if __name__ == "__main__":
    main()
