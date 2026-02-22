"""
Tesseract OCR integration for text recognition on cropped scene-text regions.

This module handles:
  - Running Tesseract OCR on preprocessed crop images
  - Extracting word-level text and confidence scores
  - Filtering low-confidence results
  - Batch recognition across multiple crops

Requires ``tesseract-ocr`` to be installed on the system and the
``pytesseract`` Python wrapper.
"""

import logging
import shutil

import cv2
import numpy as np
import pytesseract

from src.config import (
    TESSERACT_CHAR_WHITELIST,
    TESSERACT_CONF_THRESHOLD,
    TESSERACT_LANG,
    TESSERACT_OEM,
    TESSERACT_PSM,
)

logger = logging.getLogger(__name__)


def _check_tesseract_installed():
    """
    Verify that the ``tesseract`` binary is available on the system PATH.

    Raises
    ------
    EnvironmentError
        If Tesseract is not found.
    """
    if shutil.which("tesseract") is None:
        raise EnvironmentError(
            "Tesseract OCR is not installed or not found on PATH. "
            "Install it with: sudo apt install tesseract-ocr tesseract-ocr-eng"
        )


def _build_tesseract_config(
    psm=TESSERACT_PSM, oem=TESSERACT_OEM, whitelist=TESSERACT_CHAR_WHITELIST
):
    """
    Build the Tesseract CLI config string from parameters.

    Parameters
    ----------
    psm : int
        Page Segmentation Mode (e.g. 7 = single text line).
    oem : int
        OCR Engine Mode (e.g. 3 = default LSTM + legacy).
    whitelist : str or None
        If provided, restrict output to only these characters.

    Returns
    -------
    config : str
        Tesseract config string suitable for ``pytesseract`` calls.
    """
    parts = [f"--psm {psm}", f"--oem {oem}"]
    if whitelist:
        parts.append(f"-c tessedit_char_whitelist={whitelist}")
    return " ".join(parts)


def recognize(
    crop,
    lang=TESSERACT_LANG,
    psm=TESSERACT_PSM,
    oem=TESSERACT_OEM,
    whitelist=TESSERACT_CHAR_WHITELIST,
):
    """
    Run Tesseract OCR on a single preprocessed crop image.

    Uses ``pytesseract.image_to_data`` to get word-level text and
    confidence scores, then aggregates them into a single result.

    Parameters
    ----------
    crop : np.ndarray
        Preprocessed (grayscale/binary) text region image. Should already
        be enhanced via ``preprocessor.enhance_crop``.
    lang : str
        Tesseract language code (e.g. ``"eng"``).
    psm : int
        Page Segmentation Mode.
    oem : int
        OCR Engine Mode.
    whitelist : str or None
        Character whitelist to restrict Tesseract output.

    Returns
    -------
    result : dict
        Dictionary with keys:
          - ``"text"`` (str): Recognized text, stripped of excess whitespace.
          - ``"confidence"`` (float): Average word-level confidence (0–100),
            or 0.0 if no words were recognized.
          - ``"words"`` (list[dict]): Per-word breakdown, each containing
            ``"word"`` (str) and ``"confidence"`` (float).
    """
    empty_result = {"text": "", "confidence": 0.0, "words": []}

    if crop is None or crop.size == 0:
        logger.warning(
            "recognize() received empty or None crop, returning empty result"
        )
        return empty_result

    # Ensure the crop is a valid numpy array that pytesseract can handle
    if not isinstance(crop, np.ndarray):
        logger.warning("recognize() received non-ndarray input of type %s", type(crop))
        return empty_result

    config = _build_tesseract_config(psm=psm, oem=oem, whitelist=whitelist)

    try:
        data = pytesseract.image_to_data(
            crop,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractError as e:
        logger.error("Tesseract failed on crop: %s", e)
        return empty_result
    except Exception as e:
        logger.error("Unexpected error during OCR: %s", e)
        return empty_result

    return _parse_tesseract_data(data)


def _parse_tesseract_data(data):
    """
    Parse the raw dictionary output from ``pytesseract.image_to_data``
    into a structured result.

    Parameters
    ----------
    data : dict
        Raw output from pytesseract with keys: ``text``, ``conf``, etc.

    Returns
    -------
    result : dict
        Parsed result with ``text``, ``confidence``, and ``words`` fields.
    """
    words = []
    texts = data.get("text", [])
    confs = data.get("conf", [])

    for i, word_text in enumerate(texts):
        # pytesseract returns conf as int or string; handle both
        try:
            conf = float(confs[i])
        except (ValueError, TypeError, IndexError):
            conf = -1.0

        # Skip empty/whitespace-only entries and invalid confidences
        cleaned = word_text.strip() if isinstance(word_text, str) else ""
        if not cleaned or conf < 0:
            continue

        words.append({"word": cleaned, "confidence": conf})

    if not words:
        return {"text": "", "confidence": 0.0, "words": []}

    full_text = " ".join(w["word"] for w in words)
    avg_conf = sum(w["confidence"] for w in words) / len(words)

    return {
        "text": full_text,
        "confidence": round(avg_conf, 2),
        "words": words,
    }


def filter_results(result, min_confidence=TESSERACT_CONF_THRESHOLD):
    """
    Filter OCR results by removing words below a minimum confidence threshold.

    If all words are removed, the result text and confidence are reset to
    empty/zero.

    Parameters
    ----------
    result : dict
        OCR result from ``recognize()``, containing ``text``, ``confidence``,
        and ``words`` fields.
    min_confidence : float
        Minimum word-level confidence (0–100). Words below this are dropped.

    Returns
    -------
    filtered : dict
        New result dict with low-confidence words removed and text/confidence
        recomputed.
    """
    if not result or not result.get("words"):
        return {"text": "", "confidence": 0.0, "words": []}

    kept = [w for w in result["words"] if w["confidence"] >= min_confidence]

    if not kept:
        return {"text": "", "confidence": 0.0, "words": []}

    full_text = " ".join(w["word"] for w in kept)
    avg_conf = sum(w["confidence"] for w in kept) / len(kept)

    return {
        "text": full_text,
        "confidence": round(avg_conf, 2),
        "words": kept,
    }


def recognize_and_filter(crop, min_confidence=TESSERACT_CONF_THRESHOLD, **kwargs):
    """
    Convenience function: run OCR on a crop and immediately filter by confidence.

    Combines ``recognize()`` and ``filter_results()`` into a single call.

    Parameters
    ----------
    crop : np.ndarray
        Preprocessed text region image.
    min_confidence : float
        Minimum word-level confidence (0–100).
    **kwargs
        Additional keyword arguments passed to ``recognize()`` (e.g.
        ``lang``, ``psm``, ``oem``, ``whitelist``).

    Returns
    -------
    result : dict
        Filtered OCR result.
    """
    raw = recognize(crop, **kwargs)
    return filter_results(raw, min_confidence=min_confidence)


def recognize_batch(crops, min_confidence=TESSERACT_CONF_THRESHOLD, **kwargs):
    """
    Run OCR on a list of cropped text regions and return filtered results.

    Skips ``None`` entries in the crops list gracefully.

    Parameters
    ----------
    crops : list of np.ndarray
        List of preprocessed crop images.
    min_confidence : float
        Minimum word-level confidence (0–100).
    **kwargs
        Additional keyword arguments passed to ``recognize()``.

    Returns
    -------
    results : list of dict
        One result dict per crop, in the same order. Crops that produced
        no text (or were None) will have ``{"text": "", "confidence": 0.0,
        "words": []}``.
    """
    results = []

    for i, crop in enumerate(crops):
        if crop is None:
            logger.debug("Skipping crop %d (None)", i)
            results.append({"text": "", "confidence": 0.0, "words": []})
            continue

        result = recognize_and_filter(crop, min_confidence=min_confidence, **kwargs)
        logger.debug(
            "Crop %d: text='%s' confidence=%.1f",
            i,
            result["text"],
            result["confidence"],
        )
        results.append(result)

    logger.info(
        "Batch OCR complete: %d crops processed, %d produced text",
        len(crops),
        sum(1 for r in results if r["text"]),
    )

    return results


def recognize_with_fallback(
    enhanced_crop, raw_crop, min_confidence=TESSERACT_CONF_THRESHOLD, **kwargs
):
    """
    Attempt OCR on the enhanced (binarized) crop first; if the result is
    empty or below threshold, fall back to the raw (grayscale) crop.

    This handles cases where adaptive thresholding destroys thin characters
    or inverts contrast in unusual lighting conditions.

    Parameters
    ----------
    enhanced_crop : np.ndarray or None
        The binarized/preprocessed crop (preferred).
    raw_crop : np.ndarray or None
        The raw grayscale crop (fallback).
    min_confidence : float
        Minimum word-level confidence (0–100).
    **kwargs
        Additional keyword arguments passed to ``recognize()``.

    Returns
    -------
    result : dict
        The best OCR result from either the enhanced or raw crop.
    source : str
        Either ``"enhanced"`` or ``"raw"``, indicating which crop produced
        the returned result.
    """
    empty = {"text": "", "confidence": 0.0, "words": []}

    # Try enhanced crop first
    if enhanced_crop is not None and enhanced_crop.size > 0:
        enhanced_result = recognize_and_filter(
            enhanced_crop, min_confidence=min_confidence, **kwargs
        )
        if enhanced_result["text"]:
            return enhanced_result, "enhanced"

    # Fall back to raw crop
    if raw_crop is not None and raw_crop.size > 0:
        logger.debug("Enhanced crop produced no text, falling back to raw crop")
        raw_result = recognize_and_filter(
            raw_crop, min_confidence=min_confidence, **kwargs
        )
        if raw_result["text"]:
            return raw_result, "raw"

    logger.debug("Neither enhanced nor raw crop produced text")
    return empty, "none"
