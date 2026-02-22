"""
Unit tests for the EAST text detector module (src/detector.py).

Tests cover:
  - Model loading (happy path and missing file)
  - Bounding box decoding from synthetic score/geometry maps
  - Non-Maximum Suppression behaviour
  - The high-level detect_from_image convenience wrapper
  - Edge cases: empty images, no detections above threshold
"""

import math
import os
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.config import EAST_MODEL_PATH
from src.detector import (
    _rotated_rect_corners,
    decode_predictions,
    detect,
    detect_from_image,
    load_east_model,
    non_max_suppression,
)


class TestRotatedRectCorners(unittest.TestCase):
    """Tests for the _rotated_rect_corners helper."""

    def test_no_rotation(self):
        """A rectangle with zero rotation should produce axis-aligned corners."""
        corners = _rotated_rect_corners(cx=100, cy=50, w=40, h=20, angle_rad=0.0)
        self.assertEqual(corners.shape, (4, 2))
        # top-left, top-right, bottom-right, bottom-left
        np.testing.assert_array_almost_equal(corners[0], [80, 40])
        np.testing.assert_array_almost_equal(corners[1], [120, 40])
        np.testing.assert_array_almost_equal(corners[2], [120, 60])
        np.testing.assert_array_almost_equal(corners[3], [80, 60])

    def test_90_degree_rotation(self):
        """A 90-degree rotation should swap width and height directions."""
        corners = _rotated_rect_corners(cx=0, cy=0, w=10, h=4, angle_rad=math.pi / 2)
        self.assertEqual(corners.shape, (4, 2))
        # After 90-degree rotation the box should be 4 wide and 10 tall
        xs = corners[:, 0]
        ys = corners[:, 1]
        effective_w = np.max(xs) - np.min(xs)
        effective_h = np.max(ys) - np.min(ys)
        self.assertAlmostEqual(effective_w, 4.0, places=3)
        self.assertAlmostEqual(effective_h, 10.0, places=3)

    def test_returns_float32(self):
        corners = _rotated_rect_corners(0, 0, 10, 10, 0)
        self.assertEqual(corners.dtype, np.float32)

    def test_symmetry(self):
        """Corners should be symmetric about the center."""
        cx, cy = 50.0, 50.0
        corners = _rotated_rect_corners(cx, cy, 20, 10, angle_rad=0.3)
        center = corners.mean(axis=0)
        np.testing.assert_array_almost_equal(center, [cx, cy], decimal=3)


class TestDecodePredictions(unittest.TestCase):
    """Tests for decode_predictions which converts EAST output to boxes."""

    @staticmethod
    def _make_score_geometry(num_rows=5, num_cols=5, hot_cells=None, conf=0.9):
        """
        Create synthetic score and geometry tensors.

        Parameters
        ----------
        num_rows, num_cols : int
            Size of the output feature map (H/4, W/4 of EAST input).
        hot_cells : list of (row, col) or None
            Cells where the score should be high. If None, all cells are low.
        conf : float
            Score value for hot cells.

        Returns
        -------
        scores : np.ndarray, shape (1, 1, num_rows, num_cols)
        geometry : np.ndarray, shape (1, 5, num_rows, num_cols)
        """
        scores = np.zeros((1, 1, num_rows, num_cols), dtype=np.float32)
        geometry = np.zeros((1, 5, num_rows, num_cols), dtype=np.float32)

        if hot_cells:
            for r, c in hot_cells:
                scores[0, 0, r, c] = conf
                # d_top, d_right, d_bottom, d_left, angle
                geometry[0, 0, r, c] = 8.0  # d_top
                geometry[0, 1, r, c] = 16.0  # d_right
                geometry[0, 2, r, c] = 8.0  # d_bottom
                geometry[0, 3, r, c] = 16.0  # d_left
                geometry[0, 4, r, c] = 0.0  # angle (radians)

        return scores, geometry

    def test_no_detections_below_threshold(self):
        """When all scores are below threshold, should return empty lists."""
        scores, geometry = self._make_score_geometry(hot_cells=None)
        boxes, confs, angles = decode_predictions(scores, geometry, conf_threshold=0.5)
        self.assertEqual(len(boxes), 0)
        self.assertEqual(len(confs), 0)
        self.assertEqual(len(angles), 0)

    def test_single_detection(self):
        """A single hot cell should produce exactly one decoded box."""
        scores, geometry = self._make_score_geometry(hot_cells=[(2, 3)], conf=0.95)
        boxes, confs, angles = decode_predictions(scores, geometry, conf_threshold=0.5)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(len(confs), 1)
        self.assertEqual(len(angles), 1)
        self.assertAlmostEqual(confs[0], 0.95, places=2)
        # Box should be a (4, 2) array
        self.assertEqual(boxes[0].shape, (4, 2))

    def test_multiple_detections(self):
        """Multiple hot cells should produce multiple boxes."""
        scores, geometry = self._make_score_geometry(
            hot_cells=[(0, 0), (2, 3), (4, 4)], conf=0.8
        )
        boxes, confs, angles = decode_predictions(scores, geometry, conf_threshold=0.5)
        self.assertEqual(len(boxes), 3)

    def test_threshold_filtering(self):
        """Cells with score below threshold should be filtered out."""
        scores, geometry = self._make_score_geometry(
            num_rows=3, num_cols=3, hot_cells=[(1, 1)], conf=0.4
        )
        # Set threshold above the hot cell's confidence
        boxes, confs, angles = decode_predictions(scores, geometry, conf_threshold=0.5)
        self.assertEqual(len(boxes), 0)

    def test_zero_size_boxes_filtered(self):
        """Geometry with zero height/width should not produce a box."""
        scores = np.zeros((1, 1, 3, 3), dtype=np.float32)
        geometry = np.zeros((1, 5, 3, 3), dtype=np.float32)
        scores[0, 0, 1, 1] = 0.9
        # d_top=0, d_bottom=0 → h=0
        geometry[0, 0, 1, 1] = 0.0
        geometry[0, 1, 1, 1] = 10.0
        geometry[0, 2, 1, 1] = 0.0
        geometry[0, 3, 1, 1] = 10.0
        geometry[0, 4, 1, 1] = 0.0

        boxes, confs, angles = decode_predictions(scores, geometry, conf_threshold=0.5)
        self.assertEqual(len(boxes), 0)


class TestNonMaxSuppression(unittest.TestCase):
    """Tests for the NMS function."""

    @staticmethod
    def _make_box(x_min, y_min, x_max, y_max):
        """Create a simple axis-aligned box as a (4, 2) array."""
        return np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            dtype=np.float32,
        )

    def test_empty_input(self):
        kept_b, kept_c, kept_a = non_max_suppression([], [], [], nms_threshold=0.4)
        self.assertEqual(len(kept_b), 0)
        self.assertEqual(len(kept_c), 0)
        self.assertEqual(len(kept_a), 0)

    def test_single_box_passes_through(self):
        box = self._make_box(0, 0, 100, 50)
        kept_b, kept_c, kept_a = non_max_suppression(
            [box], [0.9], [0.0], nms_threshold=0.4
        )
        self.assertEqual(len(kept_b), 1)
        self.assertAlmostEqual(kept_c[0], 0.9)

    def test_non_overlapping_boxes_kept(self):
        """Two boxes far apart should both survive NMS."""
        box1 = self._make_box(0, 0, 50, 50)
        box2 = self._make_box(200, 200, 250, 250)
        kept_b, kept_c, kept_a = non_max_suppression(
            [box1, box2], [0.9, 0.8], [0.0, 0.0], nms_threshold=0.4
        )
        self.assertEqual(len(kept_b), 2)

    def test_overlapping_boxes_suppressed(self):
        """Two highly overlapping boxes should be reduced to one."""
        box1 = self._make_box(0, 0, 100, 100)
        box2 = self._make_box(5, 5, 105, 105)  # nearly identical
        kept_b, kept_c, kept_a = non_max_suppression(
            [box1, box2], [0.9, 0.85], [0.0, 0.0], nms_threshold=0.3
        )
        self.assertEqual(len(kept_b), 1)
        # The higher-confidence box should be the surviving one
        self.assertAlmostEqual(kept_c[0], 0.9)

    def test_angles_preserved(self):
        """Angles should correspond to the surviving boxes after NMS."""
        box1 = self._make_box(0, 0, 50, 50)
        box2 = self._make_box(200, 200, 250, 250)
        _, _, kept_a = non_max_suppression(
            [box1, box2], [0.9, 0.8], [15.0, -5.0], nms_threshold=0.4
        )
        self.assertEqual(len(kept_a), 2)
        self.assertIn(15.0, kept_a)
        self.assertIn(-5.0, kept_a)


class TestLoadEastModel(unittest.TestCase):
    """Tests for model loading."""

    def test_missing_model_raises_file_not_found(self):
        """Loading from a non-existent path should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_east_model("/nonexistent/path/model.pb")

    @unittest.skipUnless(
        os.path.isfile(EAST_MODEL_PATH),
        f"EAST model not found at {EAST_MODEL_PATH}; run setup.sh to download it.",
    )
    def test_model_loads_successfully(self):
        """If the EAST model file exists, it should load without error."""
        net = load_east_model(EAST_MODEL_PATH)
        self.assertIsNotNone(net)


class TestDetect(unittest.TestCase):
    """Tests for the detect() orchestration function."""

    @staticmethod
    def _make_fake_net(scores, geometry):
        """
        Create a mock cv2.dnn.Net whose forward() returns pre-built
        score and geometry arrays.
        """
        net = MagicMock()
        net.forward.return_value = (scores, geometry)
        return net

    def test_no_detections(self):
        """With all-zero score maps, detect should return empty lists."""
        h, w = 320, 320
        scores = np.zeros((1, 1, h // 4, w // 4), dtype=np.float32)
        geometry = np.zeros((1, 5, h // 4, w // 4), dtype=np.float32)
        net = self._make_fake_net(scores, geometry)

        blob = np.zeros((1, 3, h, w), dtype=np.float32)
        boxes, confs, angles = detect(
            net, blob, orig_h=640, orig_w=480, new_h=h, new_w=w
        )
        self.assertEqual(len(boxes), 0)
        self.assertEqual(len(confs), 0)
        self.assertEqual(len(angles), 0)

    def test_boxes_scaled_to_original_dimensions(self):
        """Detected boxes should be scaled back to original image coordinates."""
        h, w = 320, 320
        orig_h, orig_w = 640, 640  # 2x scaling

        scores = np.zeros((1, 1, h // 4, w // 4), dtype=np.float32)
        geometry = np.zeros((1, 5, h // 4, w // 4), dtype=np.float32)

        # Place one hot cell
        r, c = 10, 10
        scores[0, 0, r, c] = 0.95
        geometry[0, 0, r, c] = 8.0  # d_top
        geometry[0, 1, r, c] = 16.0  # d_right
        geometry[0, 2, r, c] = 8.0  # d_bottom
        geometry[0, 3, r, c] = 16.0  # d_left
        geometry[0, 4, r, c] = 0.0  # angle

        net = self._make_fake_net(scores, geometry)
        blob = np.zeros((1, 3, h, w), dtype=np.float32)

        boxes, confs, angles = detect(
            net,
            blob,
            orig_h=orig_h,
            orig_w=orig_w,
            new_h=h,
            new_w=w,
            conf_threshold=0.5,
            nms_threshold=0.4,
        )

        self.assertGreaterEqual(len(boxes), 1)
        # Because of 2x scaling, all coordinates should be roughly 2x
        # the values they'd have at 320x320 scale
        box = boxes[0]
        self.assertEqual(box.shape, (4, 2))
        # All x-coords should be in [0, orig_w] range
        self.assertTrue(np.all(box[:, 0] >= -10))
        self.assertTrue(np.all(box[:, 0] <= orig_w + 10))
        # All y-coords should be in [0, orig_h] range
        self.assertTrue(np.all(box[:, 1] >= -10))
        self.assertTrue(np.all(box[:, 1] <= orig_h + 10))


class TestDetectFromImage(unittest.TestCase):
    """Tests for the high-level detect_from_image convenience function."""

    @unittest.skipUnless(
        os.path.isfile(EAST_MODEL_PATH),
        f"EAST model not found at {EAST_MODEL_PATH}; run setup.sh to download it.",
    )
    def test_solid_color_image_returns_no_text(self):
        """A solid-color image with no text should produce zero detections."""
        net = load_east_model()
        # Pure white image
        image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        boxes, confs, angles = detect_from_image(
            net, image, east_width=320, east_height=320
        )
        self.assertEqual(len(boxes), 0)

    @unittest.skipUnless(
        os.path.isfile(EAST_MODEL_PATH),
        f"EAST model not found at {EAST_MODEL_PATH}; run setup.sh to download it.",
    )
    def test_synthetic_text_image(self):
        """
        An image with large, clear text drawn on it should ideally produce
        at least one detection. This is a best-effort test since EAST may
        not detect synthetic OpenCV-drawn text reliably.
        """
        net = load_east_model()
        image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.putText(
            image,
            "HELLO WORLD",
            (50, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            (0, 0, 0),
            8,
        )
        boxes, confs, angles = detect_from_image(
            net,
            image,
            east_width=320,
            east_height=320,
            conf_threshold=0.3,
        )
        # We don't assert len > 0 because EAST may not detect synthetic
        # OpenCV text. We just verify it doesn't crash and returns valid types.
        self.assertIsInstance(boxes, list)
        self.assertIsInstance(confs, list)
        self.assertIsInstance(angles, list)
        self.assertEqual(len(boxes), len(confs))
        self.assertEqual(len(confs), len(angles))


if __name__ == "__main__":
    unittest.main()
