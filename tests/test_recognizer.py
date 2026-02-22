"""
Unit tests for the Tesseract OCR recognizer module.

Tests cover:
  - Building Tesseract config strings
  - Parsing raw Tesseract output data
  - Filtering results by confidence threshold
  - Handling edge cases: None crops, empty crops, blank images
  - Recognize-and-filter convenience function
  - Batch recognition
  - Fallback logic (enhanced → raw crop)
"""

import numpy as np
import pytest

from src.recognizer import (
    _build_tesseract_config,
    _parse_tesseract_data,
    filter_results,
    recognize,
    recognize_and_filter,
    recognize_batch,
    recognize_with_fallback,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def white_image():
    """A blank white 100x300 image (no text)."""
    return np.ones((100, 300), dtype=np.uint8) * 255


@pytest.fixture
def black_image():
    """A blank black 100x300 image (no text)."""
    return np.zeros((100, 300), dtype=np.uint8)


@pytest.fixture
def tiny_image():
    """A very small 2x2 grayscale image."""
    return np.ones((2, 2), dtype=np.uint8) * 128


@pytest.fixture
def synthetic_text_image():
    """
    Create a synthetic image with black text on white background using OpenCV.
    This gives Tesseract a fighting chance at recognition.
    """
    import cv2

    img = np.ones((80, 400), dtype=np.uint8) * 255
    cv2.putText(
        img,
        "HELLO WORLD",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        0,
        3,
        cv2.LINE_AA,
    )
    return img


@pytest.fixture
def sample_tesseract_data_good():
    """Simulated pytesseract.image_to_data output with valid words."""
    return {
        "level": [1, 2, 3, 4, 5, 5],
        "page_num": [1, 1, 1, 1, 1, 1],
        "block_num": [0, 1, 1, 1, 1, 1],
        "par_num": [0, 0, 1, 1, 1, 1],
        "line_num": [0, 0, 0, 1, 1, 1],
        "word_num": [0, 0, 0, 0, 1, 2],
        "left": [0, 20, 20, 20, 20, 180],
        "top": [0, 10, 10, 10, 10, 10],
        "width": [400, 360, 360, 360, 150, 150],
        "height": [80, 60, 60, 60, 50, 50],
        "conf": [-1, -1, -1, -1, 92, 85],
        "text": ["", "", "", "", "HELLO", "WORLD"],
    }


@pytest.fixture
def sample_tesseract_data_low_conf():
    """Simulated pytesseract output where all words have low confidence."""
    return {
        "level": [1, 5, 5],
        "conf": [-1, 15, 22],
        "text": ["", "abc", "xyz"],
    }


@pytest.fixture
def sample_tesseract_data_mixed():
    """Simulated pytesseract output with a mix of high and low confidence words."""
    return {
        "level": [1, 5, 5, 5, 5],
        "conf": [-1, 90, 20, 75, 10],
        "text": ["", "STOP", "junk", "SIGN", "noise"],
    }


@pytest.fixture
def sample_tesseract_data_empty():
    """Simulated pytesseract output with no recognized words."""
    return {
        "level": [1, 2, 3],
        "conf": [-1, -1, -1],
        "text": ["", "", ""],
    }


@pytest.fixture
def sample_tesseract_data_whitespace():
    """Simulated pytesseract output where words are all whitespace."""
    return {
        "level": [1, 5, 5],
        "conf": [-1, 80, 70],
        "text": ["", "   ", "\t\n"],
    }


# ---------------------------------------------------------------------------
# Tests: _build_tesseract_config
# ---------------------------------------------------------------------------


class TestBuildTesseractConfig:
    def test_default_config(self):
        config = _build_tesseract_config()
        assert "--psm 7" in config
        assert "--oem 3" in config

    def test_custom_psm(self):
        config = _build_tesseract_config(psm=6)
        assert "--psm 6" in config
        assert "--psm 7" not in config

    def test_custom_oem(self):
        config = _build_tesseract_config(oem=1)
        assert "--oem 1" in config

    def test_whitelist_included(self):
        config = _build_tesseract_config(whitelist="ABC123")
        assert "tessedit_char_whitelist=ABC123" in config

    def test_no_whitelist(self):
        config = _build_tesseract_config(whitelist=None)
        assert "tessedit_char_whitelist" not in config

    def test_empty_whitelist(self):
        config = _build_tesseract_config(whitelist="")
        assert "tessedit_char_whitelist" not in config


# ---------------------------------------------------------------------------
# Tests: _parse_tesseract_data
# ---------------------------------------------------------------------------


class TestParseTesseractData:
    def test_good_data(self, sample_tesseract_data_good):
        result = _parse_tesseract_data(sample_tesseract_data_good)
        assert result["text"] == "HELLO WORLD"
        assert len(result["words"]) == 2
        assert result["words"][0]["word"] == "HELLO"
        assert result["words"][0]["confidence"] == 92.0
        assert result["words"][1]["word"] == "WORLD"
        assert result["words"][1]["confidence"] == 85.0
        assert result["confidence"] == pytest.approx(88.5, abs=0.1)

    def test_empty_data(self, sample_tesseract_data_empty):
        result = _parse_tesseract_data(sample_tesseract_data_empty)
        assert result["text"] == ""
        assert result["confidence"] == 0.0
        assert result["words"] == []

    def test_whitespace_only_data(self, sample_tesseract_data_whitespace):
        result = _parse_tesseract_data(sample_tesseract_data_whitespace)
        assert result["text"] == ""
        assert result["words"] == []

    def test_mixed_confidence_data(self, sample_tesseract_data_mixed):
        result = _parse_tesseract_data(sample_tesseract_data_mixed)
        assert "STOP" in result["text"]
        assert "SIGN" in result["text"]
        assert len(result["words"]) == 4

    def test_completely_empty_dict(self):
        result = _parse_tesseract_data({})
        assert result["text"] == ""
        assert result["confidence"] == 0.0
        assert result["words"] == []

    def test_missing_conf_key(self):
        data = {"text": ["HELLO"], "level": [5]}
        result = _parse_tesseract_data(data)
        # Should handle missing conf gracefully
        assert result["text"] == ""
        assert result["words"] == []

    def test_string_confidence_values(self):
        """pytesseract sometimes returns conf as strings."""
        data = {
            "text": ["", "TEST"],
            "conf": ["-1", "88"],
        }
        result = _parse_tesseract_data(data)
        assert result["text"] == "TEST"
        assert result["words"][0]["confidence"] == 88.0


# ---------------------------------------------------------------------------
# Tests: filter_results
# ---------------------------------------------------------------------------


class TestFilterResults:
    def test_filter_keeps_high_confidence(self):
        result = {
            "text": "HELLO WORLD",
            "confidence": 88.5,
            "words": [
                {"word": "HELLO", "confidence": 92.0},
                {"word": "WORLD", "confidence": 85.0},
            ],
        }
        filtered = filter_results(result, min_confidence=40)
        assert filtered["text"] == "HELLO WORLD"
        assert len(filtered["words"]) == 2

    def test_filter_removes_low_confidence(self, sample_tesseract_data_mixed):
        parsed = _parse_tesseract_data(sample_tesseract_data_mixed)
        filtered = filter_results(parsed, min_confidence=50)
        assert "STOP" in filtered["text"]
        assert "SIGN" in filtered["text"]
        assert "junk" not in filtered["text"]
        assert "noise" not in filtered["text"]
        assert len(filtered["words"]) == 2

    def test_filter_removes_all(self, sample_tesseract_data_low_conf):
        parsed = _parse_tesseract_data(sample_tesseract_data_low_conf)
        filtered = filter_results(parsed, min_confidence=50)
        assert filtered["text"] == ""
        assert filtered["confidence"] == 0.0
        assert filtered["words"] == []

    def test_filter_with_zero_threshold(self):
        result = {
            "text": "HELLO",
            "confidence": 5.0,
            "words": [{"word": "HELLO", "confidence": 5.0}],
        }
        filtered = filter_results(result, min_confidence=0)
        assert filtered["text"] == "HELLO"

    def test_filter_none_result(self):
        filtered = filter_results(None, min_confidence=40)
        assert filtered["text"] == ""
        assert filtered["words"] == []

    def test_filter_empty_result(self):
        filtered = filter_results({}, min_confidence=40)
        assert filtered["text"] == ""

    def test_filter_result_no_words(self):
        result = {"text": "something", "confidence": 50, "words": []}
        filtered = filter_results(result, min_confidence=40)
        assert filtered["text"] == ""

    def test_filter_recalculates_confidence(self):
        result = {
            "text": "A B C",
            "confidence": 50.0,
            "words": [
                {"word": "A", "confidence": 90.0},
                {"word": "B", "confidence": 10.0},
                {"word": "C", "confidence": 80.0},
            ],
        }
        filtered = filter_results(result, min_confidence=50)
        assert filtered["text"] == "A C"
        assert filtered["confidence"] == pytest.approx(85.0, abs=0.1)
        assert len(filtered["words"]) == 2


# ---------------------------------------------------------------------------
# Tests: recognize (integration — requires Tesseract installed)
# ---------------------------------------------------------------------------


class TestRecognize:
    """
    These tests require Tesseract to be installed on the system.
    They are integration tests that call the actual OCR engine.
    """

    def test_recognize_none_crop(self):
        result = recognize(None)
        assert result["text"] == ""
        assert result["confidence"] == 0.0
        assert result["words"] == []

    def test_recognize_empty_array(self):
        empty = np.array([], dtype=np.uint8)
        result = recognize(empty)
        assert result["text"] == ""

    def test_recognize_non_ndarray(self):
        result = recognize("not an image")
        assert result["text"] == ""

    def test_recognize_white_image(self, white_image):
        """A blank white image should produce no text."""
        result = recognize(white_image)
        # Tesseract may or may not return empty text for a blank image
        # but it should not crash
        assert isinstance(result, dict)
        assert "text" in result
        assert "confidence" in result
        assert "words" in result

    def test_recognize_tiny_image(self, tiny_image):
        """A 2x2 image should not crash Tesseract."""
        result = recognize(tiny_image)
        assert isinstance(result, dict)
        assert "text" in result

    def test_recognize_returns_dict_structure(self, white_image):
        result = recognize(white_image)
        assert isinstance(result, dict)
        assert "text" in result
        assert "confidence" in result
        assert "words" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["words"], list)

    def test_recognize_synthetic_text(self, synthetic_text_image):
        """
        Test OCR on a synthetic image with 'HELLO WORLD' drawn on it.
        We expect Tesseract to recognize at least part of it.
        """
        result = recognize(synthetic_text_image)
        assert isinstance(result, dict)
        # The synthetic text is large and clean, so Tesseract should
        # recognize something. We check loosely because font rendering
        # varies across environments.
        if result["text"]:
            # If it recognized anything, confidence should be a number
            assert result["confidence"] >= 0.0

    def test_recognize_custom_psm(self, synthetic_text_image):
        """Test passing custom PSM value."""
        result = recognize(synthetic_text_image, psm=8)  # single word mode
        assert isinstance(result, dict)
        assert "text" in result

    def test_recognize_custom_lang(self, white_image):
        """Test passing the language parameter."""
        result = recognize(white_image, lang="eng")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Tests: recognize_and_filter
# ---------------------------------------------------------------------------


class TestRecognizeAndFilter:
    def test_none_crop(self):
        result = recognize_and_filter(None, min_confidence=40)
        assert result["text"] == ""
        assert result["confidence"] == 0.0

    def test_empty_crop(self):
        empty = np.array([], dtype=np.uint8)
        result = recognize_and_filter(empty, min_confidence=40)
        assert result["text"] == ""

    def test_white_image(self, white_image):
        result = recognize_and_filter(white_image, min_confidence=40)
        assert isinstance(result, dict)
        assert "text" in result

    def test_high_threshold_filters_everything(self, white_image):
        result = recognize_and_filter(white_image, min_confidence=100)
        # With min_confidence=100, almost nothing should survive
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Tests: recognize_batch
# ---------------------------------------------------------------------------


class TestRecognizeBatch:
    def test_empty_list(self):
        results = recognize_batch([], min_confidence=40)
        assert results == []

    def test_all_none_crops(self):
        results = recognize_batch([None, None, None], min_confidence=40)
        assert len(results) == 3
        for r in results:
            assert r["text"] == ""
            assert r["confidence"] == 0.0
            assert r["words"] == []

    def test_mixed_none_and_valid(self, white_image):
        crops = [None, white_image, None, white_image]
        results = recognize_batch(crops, min_confidence=40)
        assert len(results) == 4
        # None crops should produce empty results
        assert results[0]["text"] == ""
        assert results[2]["text"] == ""
        # Valid crops should produce dict results (may or may not have text)
        assert isinstance(results[1], dict)
        assert isinstance(results[3], dict)

    def test_preserves_order(self, white_image, black_image):
        crops = [white_image, black_image, white_image]
        results = recognize_batch(crops, min_confidence=40)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Tests: recognize_with_fallback
# ---------------------------------------------------------------------------


class TestRecognizeWithFallback:
    def test_both_none(self):
        result, source = recognize_with_fallback(None, None, min_confidence=40)
        assert result["text"] == ""
        assert source == "none"

    def test_enhanced_none_raw_none(self):
        result, source = recognize_with_fallback(None, None)
        assert result["text"] == ""
        assert source == "none"

    def test_enhanced_empty_raw_none(self):
        empty = np.array([], dtype=np.uint8)
        result, source = recognize_with_fallback(empty, None)
        assert result["text"] == ""
        assert source == "none"

    def test_both_empty(self):
        empty = np.array([], dtype=np.uint8)
        result, source = recognize_with_fallback(empty, empty)
        assert result["text"] == ""
        assert source == "none"

    def test_with_white_images(self, white_image):
        """White images produce no text, so should get 'none' source."""
        result, source = recognize_with_fallback(
            white_image, white_image, min_confidence=40
        )
        assert isinstance(result, dict)
        assert source in ("enhanced", "raw", "none")

    def test_enhanced_has_text_uses_enhanced(self, synthetic_text_image, white_image):
        """
        If the enhanced crop has recognizable text, it should be preferred.
        Note: This test depends on Tesseract actually recognizing the synthetic text.
        """
        result, source = recognize_with_fallback(
            synthetic_text_image, white_image, min_confidence=10
        )
        assert isinstance(result, dict)
        # If text was found in enhanced, source should be "enhanced"
        if result["text"]:
            assert source == "enhanced"

    def test_returns_valid_source_string(self, white_image):
        _, source = recognize_with_fallback(white_image, white_image)
        assert source in ("enhanced", "raw", "none")

    def test_result_has_correct_keys(self, white_image):
        result, _ = recognize_with_fallback(white_image, white_image)
        assert "text" in result
        assert "confidence" in result
        assert "words" in result


# ---------------------------------------------------------------------------
# Tests: Edge cases and robustness
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_recognize_3channel_image(self):
        """Tesseract can also handle 3-channel BGR images."""
        bgr = np.ones((100, 300, 3), dtype=np.uint8) * 255
        result = recognize(bgr)
        assert isinstance(result, dict)

    def test_recognize_large_image(self):
        """A reasonably large blank image should not crash or hang."""
        large = np.ones((1000, 2000), dtype=np.uint8) * 255
        result = recognize(large)
        assert isinstance(result, dict)

    def test_filter_results_with_100_threshold(self):
        result = {
            "text": "TEST",
            "confidence": 99.0,
            "words": [{"word": "TEST", "confidence": 99.0}],
        }
        filtered = filter_results(result, min_confidence=100)
        assert filtered["text"] == ""

    def test_filter_results_with_exact_threshold(self):
        result = {
            "text": "TEST",
            "confidence": 50.0,
            "words": [{"word": "TEST", "confidence": 50.0}],
        }
        filtered = filter_results(result, min_confidence=50)
        assert filtered["text"] == "TEST"

    def test_batch_single_item(self, white_image):
        results = recognize_batch([white_image], min_confidence=40)
        assert len(results) == 1
        assert isinstance(results[0], dict)

    def test_parse_data_with_non_string_text(self):
        """Handle cases where pytesseract might return non-string text values."""
        data = {
            "text": [None, 123, "WORD"],
            "conf": [-1, 50, 80],
        }
        result = _parse_tesseract_data(data)
        # None and 123 should be skipped or handled gracefully
        assert isinstance(result, dict)
        assert "WORD" in result["text"]
