"""
Unit tests for the preprocessor module.

Tests cover:
  - resize_for_east: correct output dimensions and scale ratios
  - create_blob: correct blob shape and dtype
  - add_padding: correct border addition
  - enhance_crop: grayscale conversion, binarization, and edge cases
  - rotate_crop: axis-aligned and rotated extraction
  - preprocess_crop: full preprocessing chain
  - load_image: file validation and error handling
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from src.preprocessor import (
    add_padding,
    create_blob,
    enhance_crop,
    load_image,
    preprocess_crop,
    resize_for_east,
    rotate_crop,
)

# ---------------------------------------------------------------------------
# Fixtures — reusable test images
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_bgr_image():
    """Create a simple 480x640 BGR image with some variation."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a white rectangle in the center to simulate text area
    cv2.rectangle(img, (200, 150), (440, 330), (255, 255, 255), -1)
    # Add some "text-like" dark marks inside the white area
    cv2.putText(img, "HELLO", (220, 270), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return img


@pytest.fixture
def small_bgr_image():
    """Create a small 100x200 BGR image."""
    return np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)


@pytest.fixture
def grayscale_image():
    """Create a 100x200 grayscale image."""
    return np.random.randint(0, 256, (100, 200), dtype=np.uint8)


@pytest.fixture
def text_crop_bgr():
    """Create a small BGR crop that looks like white text on dark background."""
    crop = np.zeros((40, 120, 3), dtype=np.uint8)
    cv2.putText(
        crop, "TEST", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    return crop


@pytest.fixture
def text_crop_gray():
    """Create a small grayscale crop with text."""
    crop = np.zeros((40, 120), dtype=np.uint8)
    cv2.putText(crop, "TEST", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    return crop


@pytest.fixture
def axis_aligned_box():
    """An axis-aligned bounding box as 4 corner points (4, 2)."""
    return np.array(
        [
            [100, 100],  # top-left
            [300, 100],  # top-right
            [300, 200],  # bottom-right
            [100, 200],  # bottom-left
        ],
        dtype=np.int32,
    )


@pytest.fixture
def rotated_box():
    """A slightly rotated bounding box as 4 corner points (4, 2)."""
    # ~15 degree rotation around center (200, 150)
    cx, cy = 200, 150
    w, h = 200, 80
    angle_rad = np.radians(15)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    hw, hh = w / 2, h / 2

    corners = []
    for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
        rx = cx + dx * cos_a - dy * sin_a
        ry = cy + dx * sin_a + dy * cos_a
        corners.append([int(rx), int(ry)])

    return np.array(corners, dtype=np.int32)


# ---------------------------------------------------------------------------
# Tests for resize_for_east
# ---------------------------------------------------------------------------


class TestResizeForEast:
    def test_default_dimensions(self, sample_bgr_image):
        resized, rw, rh = resize_for_east(sample_bgr_image)
        assert resized.shape[1] == 320
        assert resized.shape[0] == 320
        assert resized.shape[2] == 3

    def test_custom_dimensions(self, sample_bgr_image):
        resized, rw, rh = resize_for_east(sample_bgr_image, width=640, height=640)
        assert resized.shape[1] == 640
        assert resized.shape[0] == 640

    def test_scale_ratios(self, sample_bgr_image):
        resized, rw, rh = resize_for_east(sample_bgr_image, width=320, height=320)
        assert rw == pytest.approx(640 / 320.0)
        assert rh == pytest.approx(480 / 320.0)

    def test_invalid_width_not_multiple_of_32(self, sample_bgr_image):
        with pytest.raises(ValueError, match="multiples of 32"):
            resize_for_east(sample_bgr_image, width=300, height=320)

    def test_invalid_height_not_multiple_of_32(self, sample_bgr_image):
        with pytest.raises(ValueError, match="multiples of 32"):
            resize_for_east(sample_bgr_image, width=320, height=300)

    def test_output_dtype_preserved(self, sample_bgr_image):
        resized, _, _ = resize_for_east(sample_bgr_image)
        assert resized.dtype == np.uint8

    def test_small_image(self, small_bgr_image):
        resized, rw, rh = resize_for_east(small_bgr_image, width=320, height=320)
        assert resized.shape == (320, 320, 3)
        assert rw == pytest.approx(200 / 320.0)
        assert rh == pytest.approx(100 / 320.0)

    def test_already_correct_size(self):
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        resized, rw, rh = resize_for_east(img, width=320, height=320)
        assert resized.shape == (320, 320, 3)
        assert rw == pytest.approx(1.0)
        assert rh == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests for create_blob
# ---------------------------------------------------------------------------


class TestCreateBlob:
    def test_blob_shape(self, sample_bgr_image):
        resized, _, _ = resize_for_east(sample_bgr_image)
        blob = create_blob(resized)
        # blob should be (1, 3, H, W)
        assert blob.ndim == 4
        assert blob.shape[0] == 1
        assert blob.shape[1] == 3
        assert blob.shape[2] == 320
        assert blob.shape[3] == 320

    def test_blob_dtype(self, sample_bgr_image):
        resized, _, _ = resize_for_east(sample_bgr_image)
        blob = create_blob(resized)
        assert blob.dtype == np.float32

    def test_blob_from_small_image(self, small_bgr_image):
        resized, _, _ = resize_for_east(small_bgr_image, width=160, height=160)
        blob = create_blob(resized)
        assert blob.shape == (1, 3, 160, 160)


# ---------------------------------------------------------------------------
# Tests for add_padding
# ---------------------------------------------------------------------------


class TestAddPadding:
    def test_default_padding_bgr(self, text_crop_bgr):
        padded = add_padding(text_crop_bgr, padding=4, color=(255, 255, 255))
        expected_h = text_crop_bgr.shape[0] + 2 * 4
        expected_w = text_crop_bgr.shape[1] + 2 * 4
        assert padded.shape[0] == expected_h
        assert padded.shape[1] == expected_w
        assert padded.shape[2] == 3

    def test_default_padding_gray(self, text_crop_gray):
        padded = add_padding(text_crop_gray, padding=4, color=255)
        expected_h = text_crop_gray.shape[0] + 2 * 4
        expected_w = text_crop_gray.shape[1] + 2 * 4
        assert padded.shape[0] == expected_h
        assert padded.shape[1] == expected_w

    def test_custom_padding(self, text_crop_gray):
        padded = add_padding(text_crop_gray, padding=10, color=255)
        assert padded.shape[0] == text_crop_gray.shape[0] + 20
        assert padded.shape[1] == text_crop_gray.shape[1] + 20

    def test_zero_padding(self, text_crop_gray):
        padded = add_padding(text_crop_gray, padding=0, color=255)
        assert padded.shape == text_crop_gray.shape

    def test_border_color_white_gray(self, text_crop_gray):
        padded = add_padding(text_crop_gray, padding=5, color=255)
        # Top border row should be all white
        assert np.all(padded[0, :] == 255)
        # Bottom border row
        assert np.all(padded[-1, :] == 255)
        # Left border column
        assert np.all(padded[:, 0] == 255)
        # Right border column
        assert np.all(padded[:, -1] == 255)

    def test_none_input_returns_none(self):
        result = add_padding(None, padding=4, color=255)
        assert result is None

    def test_empty_input_returns_empty(self):
        empty = np.array([], dtype=np.uint8)
        result = add_padding(empty, padding=4, color=255)
        assert result is not None
        assert result.size == 0

    def test_content_preserved(self, text_crop_gray):
        padding = 4
        padded = add_padding(text_crop_gray, padding=padding, color=255)
        inner = padded[padding:-padding, padding:-padding]
        np.testing.assert_array_equal(inner, text_crop_gray)


# ---------------------------------------------------------------------------
# Tests for enhance_crop
# ---------------------------------------------------------------------------


class TestEnhanceCrop:
    def test_bgr_input_produces_single_channel(self, text_crop_bgr):
        enhanced = enhance_crop(text_crop_bgr)
        assert enhanced is not None
        assert enhanced.ndim == 2, "Enhanced crop should be single-channel (grayscale)"

    def test_gray_input_produces_single_channel(self, text_crop_gray):
        enhanced = enhance_crop(text_crop_gray)
        assert enhanced is not None
        assert enhanced.ndim == 2

    def test_output_is_binary(self, text_crop_bgr):
        enhanced = enhance_crop(text_crop_bgr)
        unique_values = np.unique(enhanced)
        # Adaptive threshold should produce only 0 and 255
        assert all(v in (0, 255) for v in unique_values)

    def test_output_dtype(self, text_crop_bgr):
        enhanced = enhance_crop(text_crop_bgr)
        assert enhanced.dtype == np.uint8

    def test_output_dimensions_preserved(self, text_crop_bgr):
        enhanced = enhance_crop(text_crop_bgr)
        assert enhanced.shape[0] == text_crop_bgr.shape[0]
        assert enhanced.shape[1] == text_crop_bgr.shape[1]

    def test_none_input(self):
        result = enhance_crop(None)
        assert result is None

    def test_empty_input(self):
        empty = np.array([], dtype=np.uint8)
        result = enhance_crop(empty)
        assert result is not None
        assert result.size == 0

    def test_minimum_size_image(self):
        """Adaptive threshold needs at least blockSize pixels; test a very small crop."""
        tiny = np.full((15, 15, 3), 128, dtype=np.uint8)
        enhanced = enhance_crop(tiny)
        assert enhanced is not None
        assert enhanced.shape == (15, 15)

    def test_all_white_image(self):
        white = np.full((50, 100, 3), 255, dtype=np.uint8)
        enhanced = enhance_crop(white)
        assert enhanced is not None
        assert enhanced.ndim == 2

    def test_all_black_image(self):
        black = np.zeros((50, 100, 3), dtype=np.uint8)
        enhanced = enhance_crop(black)
        assert enhanced is not None
        assert enhanced.ndim == 2


# ---------------------------------------------------------------------------
# Tests for rotate_crop
# ---------------------------------------------------------------------------


class TestRotateCrop:
    def test_axis_aligned_crop(self, sample_bgr_image, axis_aligned_box):
        crop = rotate_crop(sample_bgr_image, axis_aligned_box, angle=0.0)
        assert crop is not None
        assert crop.shape[0] > 0 and crop.shape[1] > 0
        # Should roughly match the box dimensions (100 height, 200 width)
        assert crop.shape[0] == 100
        assert crop.shape[1] == 200

    def test_small_angle_uses_fast_path(self, sample_bgr_image, axis_aligned_box):
        """Angles < 1 degree should use the fast axis-aligned path."""
        crop = rotate_crop(sample_bgr_image, axis_aligned_box, angle=0.5)
        assert crop is not None
        # Fast path should give exact dimensions
        assert crop.shape[0] == 100
        assert crop.shape[1] == 200

    def test_rotated_crop_produces_output(self, sample_bgr_image, rotated_box):
        crop = rotate_crop(sample_bgr_image, rotated_box, angle=15.0)
        assert crop is not None
        assert crop.size > 0

    def test_out_of_bounds_box_is_clamped(self, sample_bgr_image):
        """A box partially outside the image should be clamped, not crash."""
        box = np.array(
            [
                [-50, -50],
                [100, -50],
                [100, 50],
                [-50, 50],
            ],
            dtype=np.int32,
        )
        crop = rotate_crop(sample_bgr_image, box, angle=0.0)
        # Should return a valid crop (clamped to image bounds) or None
        if crop is not None:
            assert crop.size > 0

    def test_zero_area_box_returns_none(self, sample_bgr_image):
        """A degenerate box with zero area should return None."""
        box = np.array(
            [
                [100, 100],
                [100, 100],
                [100, 100],
                [100, 100],
            ],
            dtype=np.int32,
        )
        crop = rotate_crop(sample_bgr_image, box, angle=0.0)
        assert crop is None

    def test_crop_has_three_channels(self, sample_bgr_image, axis_aligned_box):
        crop = rotate_crop(sample_bgr_image, axis_aligned_box, angle=0.0)
        assert crop is not None
        assert crop.shape[2] == 3


# ---------------------------------------------------------------------------
# Tests for preprocess_crop
# ---------------------------------------------------------------------------


class TestPreprocessCrop:
    def test_returns_two_crops(self, sample_bgr_image, axis_aligned_box):
        raw, enhanced = preprocess_crop(sample_bgr_image, axis_aligned_box, angle=0.0)
        assert raw is not None
        assert enhanced is not None

    def test_raw_is_grayscale(self, sample_bgr_image, axis_aligned_box):
        raw, _ = preprocess_crop(sample_bgr_image, axis_aligned_box, angle=0.0)
        assert raw.ndim == 2

    def test_enhanced_is_binary(self, sample_bgr_image, axis_aligned_box):
        _, enhanced = preprocess_crop(sample_bgr_image, axis_aligned_box, angle=0.0)
        unique = np.unique(enhanced)
        assert all(v in (0, 255) for v in unique)

    def test_includes_padding(self, sample_bgr_image, axis_aligned_box):
        padding = 4
        raw, enhanced = preprocess_crop(
            sample_bgr_image, axis_aligned_box, angle=0.0, padding=padding
        )
        # The box is 200x100 so with 4px padding each side:
        # width = 200 + 8 = 208, height = 100 + 8 = 108
        assert raw.shape[0] == 100 + 2 * padding
        assert raw.shape[1] == 200 + 2 * padding
        assert enhanced.shape[0] == 100 + 2 * padding
        assert enhanced.shape[1] == 200 + 2 * padding

    def test_invalid_box_returns_none_pair(self, sample_bgr_image):
        degenerate_box = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
        raw, enhanced = preprocess_crop(sample_bgr_image, degenerate_box, angle=0.0)
        assert raw is None
        assert enhanced is None

    def test_with_rotation(self, sample_bgr_image, rotated_box):
        raw, enhanced = preprocess_crop(sample_bgr_image, rotated_box, angle=15.0)
        assert raw is not None
        assert enhanced is not None
        assert raw.ndim == 2
        assert enhanced.ndim == 2


# ---------------------------------------------------------------------------
# Tests for load_image
# ---------------------------------------------------------------------------


class TestLoadImage:
    def test_load_valid_image(self, sample_bgr_image):
        """Write a temp image, load it, verify shape."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, sample_bgr_image)
            tmp_path = tmp.name

        try:
            loaded = load_image(tmp_path)
            assert loaded is not None
            assert loaded.shape[2] == 3
            # JPEG compression may slightly alter dimensions but shape should match
            assert loaded.shape[0] == sample_bgr_image.shape[0]
            assert loaded.shape[1] == sample_bgr_image.shape[1]
        finally:
            os.unlink(tmp_path)

    def test_load_png(self, sample_bgr_image):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, sample_bgr_image)
            tmp_path = tmp.name

        try:
            loaded = load_image(tmp_path)
            assert loaded is not None
            np.testing.assert_array_equal(loaded, sample_bgr_image)
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError, match="Image not found"):
            load_image("/nonexistent/path/image.jpg")

    def test_load_corrupt_file(self):
        """A file that exists but is not a valid image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, mode="w") as tmp:
            tmp.write("this is not an image")
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="Could not decode"):
                load_image(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_load_directory_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_image(tmpdir)
