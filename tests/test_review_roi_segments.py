import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.review_roi_segments import (
    HTML,
    OCR_REVIEW_HTML,
    SegmentReviewApp,
    parse_edit_distance_tolerance,
    parse_similarity_ratio,
)


class ReviewRoiSegmentsTests(unittest.TestCase):
    def test_state_keeps_empty_runs_between_subtitle_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            rows = [
                {"image": "frame_000.jpg", "has_subtitle": True, "segment_marker": "seg-a"},
                {"image": "frame_030.jpg", "has_subtitle": False},
                {"image": "frame_060.jpg", "has_subtitle": False},
                {"image": "frame_090.jpg", "has_subtitle": True, "segment_marker": "seg-b"},
            ]
            (root / "annotations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            state = SegmentReviewApp(root).state()

        self.assertEqual(
            [group["segment_id"] for group in state["groups"]],
            ["seg-a", "__no_subtitle_1__", "seg-b"],
        )
        self.assertEqual([group["count"] for group in state["groups"]], [1, 2, 1])
        self.assertTrue(state["groups"][1]["no_subtitle"])

    def test_ocr_merge_keeps_next_candidate_after_merge(self):
        self.assertNotIn("candidateIndex = Math.max(0, candidateIndex - 1);", OCR_REVIEW_HTML)
        self.assertIn("Keeping the same index selects the next refreshed candidate", OCR_REVIEW_HTML)

    def test_ocr_review_keyboard_shortcuts_ignore_focused_controls(self):
        for page in (HTML, OCR_REVIEW_HTML):
            self.assertIn("function isTextEntryFocus()", page)
            self.assertIn("function suppressFocusedButtonActivation(event)", page)
            self.assertIn("if (isTextEntryFocus()) return;", page)
            self.assertIn("if (suppressFocusedButtonActivation(event)) return;", page)

    def test_navigation_shortcuts_use_cancelable_hold_controller_not_native_repeat_queue(
        self,
    ):
        for page in (HTML, OCR_REVIEW_HTML):
            self.assertIn("function createNavigationHoldController(shortcuts)", page)
            self.assertIn("if (event.repeat) return true;", page)
            self.assertIn("window.setTimeout(runStep", page)
            self.assertIn("window.clearTimeout(activeHold.timerId);", page)
            self.assertIn(
                'document.addEventListener("keyup", navigationHold.releaseKey);',
                page,
            )
            self.assertIn(
                'window.addEventListener("blur", navigationHold.releaseAll);',
                page,
            )
            self.assertIn('document.addEventListener("visibilitychange"', page)
            self.assertNotIn("const heldNavigationKeys = new Set();", page)
            self.assertNotIn("queuedNavigationAction", page)
            self.assertNotIn("window.requestAnimationFrame", page)
            self.assertNotIn("cancelQueuedNavigation", page)
            self.assertNotIn("setInterval", page)

    def test_ocr_candidates_accept_custom_tolerance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            rows = [
                {
                    "image": "frame_000.jpg",
                    "has_subtitle": True,
                    "ocr_text": "字幕识别结果AAAA",
                },
                {
                    "image": "frame_030.jpg",
                    "has_subtitle": True,
                    "ocr_text": "字幕识别结果BBBB",
                },
            ]
            (root / "annotations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            app = SegmentReviewApp(root)

            default_state = app.state()
            loose_state = app.state(similarity_ratio=0.6, edit_distance_tolerance=4)

        self.assertEqual(default_state["similar_candidates"], [])
        self.assertEqual(len(loose_state["similar_candidates"]), 1)
        self.assertEqual(loose_state["candidate_thresholds"]["similarity_ratio"], 0.6)
        self.assertEqual(loose_state["candidate_thresholds"]["edit_distance_tolerance"], 4)

    def test_ocr_tolerance_query_values_are_clamped(self):
        self.assertEqual(parse_similarity_ratio("1.7"), 1.0)
        self.assertEqual(parse_similarity_ratio("-0.2"), 0.0)
        self.assertEqual(parse_edit_distance_tolerance("100"), 30)
        self.assertEqual(parse_edit_distance_tolerance("-3"), 0)


if __name__ == "__main__":
    unittest.main()
