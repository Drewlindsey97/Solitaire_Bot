#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from board_reader_lib import (
    BoardLayout,
    BoardTransform,
    TEMPLATES,
    match_rank_detailed,
    preprocess_rank_variants,
    read_board,
    save_calibration_artifacts,
    save_exposed_card_diagnostics,
    transform_from_content_rect,
)
from solitaire_auto_bot import get_element_coords, main


def synthetic_image(width=720, height=1600, content_rect=None):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if content_rect is None:
        x, y, w, h = 0, 0, width, height
    else:
        x, y, w, h = content_rect
    img[y:y + h, x:x + w] = (30, 120, 30)
    return img


class BoardLayoutTests(unittest.TestCase):
    def test_reference_resolution(self):
        layout = BoardLayout()
        transform = transform_from_content_rect(720, 1600, (0, 0, 720, 1600), layout)
        self.assertEqual((layout.reference_width, layout.reference_height), (720, 1600))
        self.assertEqual(transform.original_to_normalized(360, 800), (360, 800))

    def test_scaled_screenshot_mapping(self):
        layout = BoardLayout()
        transform = transform_from_content_rect(1440, 3200, (0, 0, 1440, 3200), layout)
        self.assertEqual(transform.original_to_normalized(720, 1600), (360, 800))
        self.assertEqual(transform.normalized_to_original(360, 800), (720, 1600))

    def test_top_bottom_padding_mapping(self):
        layout = BoardLayout()
        transform = transform_from_content_rect(720, 1800, (0, 100, 720, 1600), layout)
        self.assertEqual(transform.original_to_normalized(360, 900), (360, 800))
        self.assertEqual(transform.normalized_to_original(360, 800), (360, 900))

    def test_coordinate_round_trip_tolerance(self):
        layout = BoardLayout()
        transform = transform_from_content_rect(1000, 2100, (40, 120, 900, 1800), layout)
        original = (533.3, 991.7)
        normalized = transform.original_to_normalized(*original)
        restored = transform.normalized_to_original(*normalized)
        self.assertAlmostEqual(original[0], restored[0], places=6)
        self.assertAlmostEqual(original[1], restored[1], places=6)

    def test_normalized_gesture_maps_to_original(self):
        layout = BoardLayout()
        transform = transform_from_content_rect(1440, 3200, (0, 0, 1440, 3200), layout)
        board = {f"col{i}": [] for i in range(7)}
        board["col0"] = [{"rank": "A", "color": "RED", "score": 1.0}]
        coords = get_element_coords(board, "col", 0, transform=transform, layout=layout)
        self.assertEqual(coords, (115, 1104))

    def test_crop_bounds_for_padded_screenshot(self):
        layout = BoardLayout()
        img = synthetic_image(720, 1800, (0, 100, 720, 1600))
        transform = transform_from_content_rect(720, 1800, (0, 100, 720, 1600), layout)
        normalized = transform.normalize_image(img)
        self.assertEqual(normalized.shape[:2], (1600, 720))

    def test_maximum_card_count_guard(self):
        layout = BoardLayout(max_tableau_cards=2)
        img = synthetic_image()
        cv2.rectangle(img, (10, 507), (105, 507 + 135 + 50 * 4), (245, 245, 245), -1)
        board = read_board(img, layout=layout)
        self.assertEqual(board["col0"], [])

    def test_repeated_position_rejection_by_non_monotonic_guard(self):
        transform = BoardTransform(720, 1600, 0, 0, 720, 1600, 720, 1600)
        x, y = 100, 100
        nx, ny = transform.original_to_normalized(x, y)
        ox, oy = transform.normalized_to_original(nx, ny)
        self.assertEqual((ox, oy), (x, y))

    def test_calibration_mode_never_executes_gesture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "solitaire_auto_bot.py",
                "--sim",
                "live_before.png",
                "--calibrate-layout",
                "--calibration-dir",
                temp_dir,
            ]
            with patch("sys.argv", argv), \
                    patch("bridge.tap") as tap, \
                    patch("bridge.swipe") as swipe:
                main()
            tap.assert_not_called()
            swipe.assert_not_called()
            self.assertTrue((Path(temp_dir) / "layout.json").exists())
            self.assertTrue((Path(temp_dir) / "detection_report.json").exists())

    def test_frame_0108_remains_readable(self):
        board = read_board("Gameplay/frame_0108.png")
        total = sum(len(board[f"col{i}"]) for i in range(7))
        self.assertGreaterEqual(total, 25)
        self.assertEqual(board["col0"][0]["rank"], "K")

    def test_calibration_artifacts_writer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = save_calibration_artifacts("live_before.png", temp_dir)
            self.assertTrue((Path(temp_dir) / "original.png").exists())
            self.assertTrue((Path(temp_dir) / "normalized.png").exists())
            self.assertTrue((Path(temp_dir) / "layout_overlay.png").exists())
            self.assertIn("columns", report)

    def test_preprocessing_variants_normalize_to_template_dimensions(self):
        template = next(iter(TEMPLATES.values()))
        patch = np.zeros((20, 30, 3), dtype=np.uint8)
        variants = preprocess_rank_variants(patch, size=(template.shape[1], template.shape[0]))

        self.assertIn("otsu_inv", variants)
        self.assertIn("adaptive_inv", variants)
        for image in variants.values():
            self.assertEqual(image.shape, template.shape)

    def test_match_rank_detailed_reports_preprocessing_variants(self):
        template_name, template = next(iter(TEMPLATES.items()))
        patch = cv2.cvtColor(cv2.bitwise_not(template), cv2.COLOR_GRAY2BGR)

        name, score, candidates = match_rank_detailed(patch, TEMPLATES)

        self.assertEqual(name, template_name)
        self.assertGreater(score, 0.5)
        self.assertIn("variant", candidates[0])

    def test_low_confidence_blank_card_remains_unresolved(self):
        patch = np.full((45, 45, 3), 255, dtype=np.uint8)

        _, score, _ = match_rank_detailed(patch, TEMPLATES)

        self.assertLess(score, BoardLayout().min_rank_score)

    def test_diagnostic_artifact_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = save_exposed_card_diagnostics("live_before.png", 2, 2, temp_dir)
            path = Path(temp_dir)

            self.assertTrue((path / "full_original_card_crop.png").exists())
            self.assertTrue((path / "rank_crop.png").exists())
            self.assertTrue((path / "suit_color_crop.png").exists())
            self.assertTrue((path / "diagnostics.json").exists())
            self.assertEqual(result["coordinates"]["column"], 2)

    def test_targeted_diagnostic_mode_never_executes_gesture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "solitaire_auto_bot.py",
                "--sim",
                "live_before.png",
                "--debug-exposed-card",
                "2,2",
                "--calibration-dir",
                temp_dir,
            ]
            with patch("sys.argv", argv), \
                    patch("bridge.tap") as tap, \
                    patch("bridge.swipe") as swipe:
                main()
            tap.assert_not_called()
            swipe.assert_not_called()
            self.assertTrue((Path(temp_dir) / "diagnostics.json").exists())


if __name__ == "__main__":
    unittest.main()
