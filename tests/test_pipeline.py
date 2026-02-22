"""
Integration tests for the full Scene Text Detection & Recognition pipeline.

These tests verify the end-to-end flow from image loading through detection
and recognition, as well as the annotation and saving utilities.

Some tests require the EAST model to be present and are skipped if it's not
available (marked with @pytest.mark.skipif).
"""

import json
import os
import shutil
import tempfile

import cv2
import numpy as np
import pytest

from src.config import EAST_MODEL_PATH, SUPPORTED_EXTENSIONS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EAST_MODEL_AVAILABLE = os.path.isfile(EAST_MODEL_PATH)

skip_no_model = pytest.mark.skipif(
    not EAST_MODEL_AVAILABLE,
    reason=f"EAST model not found at {EAST_MODEL_PATH}. Run setup.sh to download it.",
)


def _tesseract_available():
    """Check whether the tesseract binary is on PATH."""
    return shutil.which("tesseract") is not None


skip_no_tesseract = pytest.mark.skipif(
    not _tesseract_available(),
    reason="Tesseract OCR is not installed or not on PATH.",
)


def make_text_image(text="HELLO", width=300, height=80, font_scale=2.0, thickness=3):
    """
    Create a synthetic BGR image with white background and black text.

    This produces a clean image that EAST *might* detect (no guarantee on
    synthetic images) and that Tesseract can recognise easily.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(0, (width - tw) // 2)
    y = max(th, (height + th) // 2)

    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness)
    return img


def make_scene_image(width=640, height=480):
    """
    Create a colourful synthetic scene image with a text-like region.

    Not expected to produce real detections — used for smoke-testing that
    the pipeline does not crash on arbitrary input.
    """
    img = np.random.randint(60, 200, (height, width, 3), dtype=np.uint8)

    # Draw a white rectangle with black text to simulate a sign
    cv2.rectangle(img, (100, 150), (400, 220), (255, 255, 255), cv2.FILLED)
    cv2.putText(
        img,
        "TEST SIGN",
        (120, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2,
    )
    return img


def save_image(image, directory, filename="test_image.jpg"):
    """Save an image to a directory and return the full path."""
    path = os.path.join(directory, filename)
    cv2.imwrite(path, image)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="scene_text_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_image_path(tmp_dir):
    """Save a synthetic scene image to a temp dir and return the path."""
    img = make_scene_image()
    return save_image(img, tmp_dir)


@pytest.fixture
def tmp_text_image_path(tmp_dir):
    """Save a clean text image to a temp dir and return the path."""
    img = make_text_image("STOP")
    return save_image(img, tmp_dir, "stop_sign.jpg")


@pytest.fixture
def tmp_batch_dir(tmp_dir):
    """Create a temp directory with several images for batch processing."""
    for i, word in enumerate(["OPEN", "EXIT", "CAFE"]):
        img = make_text_image(word, width=250, height=70)
        save_image(img, tmp_dir, f"img_{i}.jpg")

    # Also add a non-image file that should be ignored
    with open(os.path.join(tmp_dir, "notes.txt"), "w") as f:
        f.write("not an image")

    return tmp_dir


# ---------------------------------------------------------------------------
# Tests: annotate_image
# ---------------------------------------------------------------------------


class TestAnnotateImage:
    """Tests for pipeline.annotate_image (no model required)."""

    def test_returns_copy_not_mutating_original(self):
        from src.pipeline import annotate_image

        image = np.zeros((200, 400, 3), dtype=np.uint8)
        original_copy = image.copy()

        detections = [
            {
                "bbox": [10, 10, 200, 10, 200, 50, 10, 50],
                "text": "HELLO",
                "confidence": 90.0,
            }
        ]

        annotated = annotate_image(image, detections)

        # Original must be unmodified
        np.testing.assert_array_equal(image, original_copy)
        # Annotated must differ (has drawings on it)
        assert not np.array_equal(image, annotated)

    def test_empty_detections_returns_identical_copy(self):
        from src.pipeline import annotate_image

        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        annotated = annotate_image(image, [])

        np.testing.assert_array_equal(image, annotated)
        assert image is not annotated  # must be a copy

    def test_skips_detections_without_text(self):
        from src.pipeline import annotate_image

        image = np.ones((100, 300, 3), dtype=np.uint8) * 200
        original_copy = image.copy()

        detections = [
            {"bbox": [5, 5, 100, 5, 100, 40, 5, 40], "text": "", "confidence": 0.0},
            {"bbox": [5, 50, 100, 50, 100, 90, 5, 90], "text": "", "confidence": 0.0},
        ]

        annotated = annotate_image(image, detections)
        # No text means nothing drawn — should look identical to original
        np.testing.assert_array_equal(annotated, original_copy)

    def test_multiple_detections_all_drawn(self):
        from src.pipeline import annotate_image

        image = np.ones((300, 400, 3), dtype=np.uint8) * 255

        detections = [
            {
                "bbox": [10, 10, 150, 10, 150, 50, 10, 50],
                "text": "AAA",
                "confidence": 80.0,
            },
            {
                "bbox": [10, 100, 150, 100, 150, 140, 10, 140],
                "text": "BBB",
                "confidence": 70.0,
            },
            {
                "bbox": [10, 200, 150, 200, 150, 240, 10, 240],
                "text": "CCC",
                "confidence": 60.0,
            },
        ]

        annotated = annotate_image(image, detections)
        # The annotated image should differ from the blank white original
        diff = cv2.absdiff(image, annotated)
        assert diff.sum() > 0

    def test_handles_bbox_near_top_edge(self):
        """Label placement should not crash when the box is near the top."""
        from src.pipeline import annotate_image

        image = np.ones((100, 300, 3), dtype=np.uint8) * 200

        detections = [
            {
                "bbox": [10, 2, 150, 2, 150, 20, 10, 20],
                "text": "TOP",
                "confidence": 95.0,
            },
        ]

        # Should not raise
        annotated = annotate_image(image, detections)
        assert annotated.shape == image.shape


# ---------------------------------------------------------------------------
# Tests: save_results
# ---------------------------------------------------------------------------


class TestSaveResults:
    """Tests for pipeline.save_results (no model required)."""

    def test_saves_json_file(self, tmp_dir):
        from src.pipeline import save_results

        # Create a fake result dict (no real image needed for JSON-only)
        fake_image = make_text_image("X")
        img_path = save_image(fake_image, tmp_dir, "fake.jpg")

        result = {
            "image_path": img_path,
            "image_size": [80, 300],
            "detections": [],
            "total_detections": 0,
            "processing_time_ms": 100.0,
        }

        saved = save_results(result, output_dir=tmp_dir)

        assert saved["json_file"] is not None
        assert os.path.isfile(saved["json_file"])

        with open(saved["json_file"], "r") as f:
            data = json.load(f)

        assert data["processing_time_ms"] == 100.0
        assert data["detections"] == []

    def test_saves_annotated_image_when_detections_exist(self, tmp_dir):
        from src.pipeline import save_results

        fake_image = make_text_image("HI")
        img_path = save_image(fake_image, tmp_dir, "hi.jpg")

        result = {
            "image_path": img_path,
            "image_size": [80, 300],
            "detections": [
                {
                    "id": 1,
                    "bbox": [10, 10, 100, 10, 100, 50, 10, 50],
                    "text": "HI",
                    "confidence": 85.0,
                    "detection_confidence": 90.0,
                    "source": "enhanced",
                }
            ],
            "total_detections": 1,
            "processing_time_ms": 50.0,
        }

        saved = save_results(result, output_dir=tmp_dir)

        assert saved["annotated_image"] is not None
        assert os.path.isfile(saved["annotated_image"])

        # Verify the annotated image can be loaded
        ann_img = cv2.imread(saved["annotated_image"])
        assert ann_img is not None
        assert ann_img.shape[0] > 0


# ---------------------------------------------------------------------------
# Tests: SceneTextPipeline (require model + tesseract)
# ---------------------------------------------------------------------------


@skip_no_model
@skip_no_tesseract
class TestSceneTextPipeline:
    """Integration tests that run the full pipeline end-to-end."""

    def test_pipeline_initialises(self):
        from src.pipeline import SceneTextPipeline

        pipeline = SceneTextPipeline()
        assert pipeline.net is not None

    def test_process_image_returns_correct_schema(self, tmp_image_path):
        from src.pipeline import SceneTextPipeline

        pipeline = SceneTextPipeline()
        result = pipeline.process_image(tmp_image_path)

        # Verify top-level keys
        assert "image_path" in result
        assert "image_size" in result
        assert "detections" in result
        assert "total_detections" in result
        assert "processing_time_ms" in result

        # Types
        assert isinstance(result["image_path"], str)
        assert isinstance(result["image_size"], list)
        assert len(result["image_size"]) == 2
        assert isinstance(result["detections"], list)
        assert isinstance(result["total_detections"], int)
        assert isinstance(result["processing_time_ms"], float)

        # Processing time must be positive
        assert result["processing_time_ms"] > 0

    def test_process_image_detection_schema(self, tmp_image_path):
        from src.pipeline import SceneTextPipeline

        pipeline = SceneTextPipeline(east_conf=0.1)  # low threshold to get detections
        result = pipeline.process_image(tmp_image_path)

        for det in result["detections"]:
            assert "id" in det
            assert "bbox" in det
            assert "text" in det
            assert "confidence" in det
            assert "source" in det

            # bbox must be a list of 8 numbers
            assert isinstance(det["bbox"], list)
            assert len(det["bbox"]) == 8

            # source must be one of the expected values
            assert det["source"] in ("enhanced", "raw", "none")

    def test_process_image_nonexistent_file_raises(self):
        from src.pipeline import SceneTextPipeline

        pipeline = SceneTextPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.process_image("/nonexistent/path/image.jpg")

    def test_process_image_invalid_file_raises(self, tmp_dir):
        from src.pipeline import SceneTextPipeline

        # Create a non-image file
        bad_path = os.path.join(tmp_dir, "bad.jpg")
        with open(bad_path, "w") as f:
            f.write("this is not an image")

        pipeline = SceneTextPipeline()
        with pytest.raises(ValueError, match="Could not decode"):
            pipeline.process_image(bad_path)

    def test_process_directory_basic(self, tmp_batch_dir):
        from src.pipeline import SceneTextPipeline

        pipeline = SceneTextPipeline()
        results, errors = pipeline.process_directory(tmp_batch_dir)

        # Should have processed exactly 3 images (notes.txt is ignored)
        assert len(results) == 3
        assert len(errors) == 0

        for result in results:
            assert "image_path" in result
            assert "detections" in result

    def test_process_directory_nonexistent_raises(self):
        from src.pipeline import SceneTextPipeline

        pipeline = SceneTextPipeline()
        with pytest.raises(NotADirectoryError):
            pipeline.process_directory("/nonexistent/dir/")

    def test_process_directory_empty(self, tmp_dir):
        from src.pipeline import SceneTextPipeline

        pipeline = SceneTextPipeline()
        results, errors = pipeline.process_directory(tmp_dir)

        assert results == []
        assert errors == []

    def test_end_to_end_clean_text_image(self, tmp_text_image_path):
        """
        A clean synthetic text image should at minimum not crash the pipeline.

        We don't assert specific OCR output because EAST may or may not detect
        synthetic text reliably, but the pipeline must handle it gracefully.
        """
        from src.pipeline import SceneTextPipeline

        pipeline = SceneTextPipeline(east_conf=0.1, tess_conf=10)
        result = pipeline.process_image(tmp_text_image_path)

        assert result is not None
        assert result["processing_time_ms"] > 0
        assert isinstance(result["detections"], list)

    def test_save_and_reload_results(self, tmp_image_path, tmp_dir):
        from src.pipeline import SceneTextPipeline, save_results

        pipeline = SceneTextPipeline()
        result = pipeline.process_image(tmp_image_path)

        out_dir = os.path.join(tmp_dir, "output")
        saved = save_results(result, output_dir=out_dir)

        assert os.path.isfile(saved["json_file"])

        with open(saved["json_file"], "r") as f:
            reloaded = json.load(f)

        # Round-trip check
        assert reloaded["total_detections"] == result["total_detections"]
        assert len(reloaded["detections"]) == len(result["detections"])


# ---------------------------------------------------------------------------
# Tests: save_batch_results
# ---------------------------------------------------------------------------


class TestSaveBatchResults:
    """Tests for pipeline.save_batch_results (no model required)."""

    def test_saves_summary_json(self, tmp_dir):
        from src.pipeline import save_batch_results

        results = [
            {
                "image_path": "/fake/img1.jpg",
                "image_size": [100, 200],
                "detections": [],
                "total_detections": 0,
                "processing_time_ms": 100.0,
            },
            {
                "image_path": "/fake/img2.jpg",
                "image_size": [100, 200],
                "detections": [
                    {
                        "id": 1,
                        "bbox": [0, 0, 10, 0, 10, 10, 0, 10],
                        "text": "A",
                        "confidence": 90.0,
                        "detection_confidence": 95.0,
                        "source": "enhanced",
                    }
                ],
                "total_detections": 1,
                "processing_time_ms": 200.0,
            },
        ]

        errors = [{"image_path": "/fake/bad.jpg", "error": "corrupt file"}]

        summary = save_batch_results(results, errors=errors, output_dir=tmp_dir)

        assert summary["total_images"] == 3
        assert summary["successful"] == 2
        assert summary["failed"] == 1
        assert summary["total_text_detections"] == 1
        assert summary["avg_processing_time_ms"] == 150.0

        # Check the summary file was written
        summary_path = os.path.join(tmp_dir, "results", "_batch_summary.json")
        assert os.path.isfile(summary_path)


# ---------------------------------------------------------------------------
# Tests: process_image convenience function
# ---------------------------------------------------------------------------


@skip_no_model
@skip_no_tesseract
class TestProcessImageConvenience:
    """Tests for the top-level convenience function pipeline.process_image."""

    def test_convenience_function_works(self, tmp_image_path):
        from src.pipeline import process_image

        result = process_image(tmp_image_path)

        assert "image_path" in result
        assert "detections" in result
        assert "processing_time_ms" in result

    def test_convenience_function_accepts_kwargs(self, tmp_image_path):
        from src.pipeline import process_image

        result = process_image(
            tmp_image_path,
            east_conf=0.9,
            tess_conf=80,
            east_width=320,
            east_height=320,
        )

        assert result is not None


# ---------------------------------------------------------------------------
# Tests: _find_images helper (indirectly via process_directory)
# ---------------------------------------------------------------------------


@skip_no_model
@skip_no_tesseract
class TestFindImages:
    """Test the internal image-finding logic."""

    def test_only_supported_extensions(self, tmp_dir):
        from src.pipeline import SceneTextPipeline

        # Create files with various extensions
        for ext in [".jpg", ".png", ".bmp", ".txt", ".pdf", ".py"]:
            fname = f"file{ext}"
            if ext in SUPPORTED_EXTENSIONS:
                img = np.ones((32, 32, 3), dtype=np.uint8) * 200
                cv2.imwrite(os.path.join(tmp_dir, fname), img)
            else:
                with open(os.path.join(tmp_dir, fname), "w") as f:
                    f.write("not an image")

        pipeline = SceneTextPipeline()
        results, errors = pipeline.process_directory(tmp_dir)

        # Should have processed only the image files
        total = len(results) + len(errors)
        # .jpg, .png, .bmp are supported
        assert total == 3

    def test_recursive_finds_subdirectory_images(self, tmp_dir):
        from src.pipeline import SceneTextPipeline

        # Top-level image
        img = np.ones((32, 32, 3), dtype=np.uint8) * 200
        cv2.imwrite(os.path.join(tmp_dir, "top.jpg"), img)

        # Subdirectory image
        sub = os.path.join(tmp_dir, "subdir")
        os.makedirs(sub)
        cv2.imwrite(os.path.join(sub, "nested.jpg"), img)

        pipeline = SceneTextPipeline()

        # Non-recursive should only find top-level
        results_flat, _ = pipeline.process_directory(tmp_dir, recursive=False)
        assert len(results_flat) == 1

        # Recursive should find both
        results_rec, _ = pipeline.process_directory(tmp_dir, recursive=True)
        assert len(results_rec) == 2
