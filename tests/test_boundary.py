"""Tests for the adopted terminator trace and its disk accounting."""

import unittest

import numpy as np

from image_lab.validation import Invalid
from moondesk import analysis, boundary, model


def scene(height=120, width=200):
    luma = np.full((height, width), 20.0)
    for y in range(10, height - 10):
        x = 60 + int(round(0.15 * (y - 10)))
        luma[y, x + 1:] -= 5.0
    for y in range(50, 58):
        luma[y, 73:] -= 24.0
    return luma, np.ones((height, width), dtype=bool)


SETTINGS = {"inspect": [0, 0, 200, 120], "sun_side": "left", "arcsec_per_pixel": 2,
            "threshold": 10, "boundary": "trace"}


class TraceTest(unittest.TestCase):
    def test_prefers_long_weak_chain_over_short_strong_spur(self):
        luma, disk = scene()
        trace = boundary.trace(disk, luma, SETTINGS)
        points = dict(trace["points"])
        self.assertEqual(len(points), len(set(points)))
        for y in range(50, 58):
            self.assertLess(points[y], 70.0, "short strong spur must not be adopted")
        self.assertEqual(len(trace["segments"]), 1)
        self.assertEqual(trace["segments"][0]["rows"], 100)
        self.assertAlmostEqual(trace["segments"][0]["mean_jump"], 5.0, places=1)
        self.assertEqual(len(trace["alternatives"]), 1)
        self.assertEqual(trace["alternatives"][0]["mean_jump"], 24.0)

    def test_rows_without_observation_stay_blank(self):
        luma, disk = scene()
        trace = boundary.trace(disk, luma, SETTINGS)
        blank = [row["y"] for row in trace["rows"] if row["status"] == "留空"]
        self.assertEqual(len(blank), 20)
        self.assertEqual(len(trace["points"]), 100)
        for row in trace["rows"]:
            self.assertIn(row["status"], ("采用", "留空"))

    def test_at_most_one_point_per_row(self):
        luma, disk = scene()
        trace = boundary.trace(disk, luma, SETTINGS)
        rows = [y for y, _ in trace["points"]]
        self.assertEqual(rows, sorted(set(rows)))

    def test_counts_add_up_to_disk(self):
        luma, disk = scene()
        trace = boundary.trace(disk, luma, SETTINGS)
        totals, _ = boundary.counts(disk, SETTINGS, trace["points"])
        self.assertEqual(totals["light"] + totals["dark"] + totals["pending"], totals["disk"])
        self.assertEqual(totals["boundary_rows"], 100)
        self.assertEqual(totals["blank_rows"], 20)
        self.assertGreater(totals["pending"], 0)

    def test_sun_side_swaps_light_and_dark(self):
        luma, disk = scene()
        trace = boundary.trace(disk, luma, SETTINGS)
        left = boundary.counts(disk, SETTINGS, trace["points"])[0]
        right = boundary.counts(disk, dict(SETTINGS, sun_side="right"), trace["points"])[0]
        self.assertLess(left["light"], left["dark"])
        self.assertEqual(right["light"], left["dark"])
        self.assertEqual(right["dark"], left["light"])

    def test_pending_only_where_no_point(self):
        luma, disk = scene()
        trace = boundary.trace(disk, luma, SETTINGS)
        labels = boundary.labels(disk, SETTINGS, trace["points"])
        blank = {row["y"] for row in trace["rows"] if row["status"] == "留空"}
        pending_rows = {int(y) for y in np.flatnonzero((labels == boundary.PENDING).any(axis=1))}
        self.assertEqual(pending_rows, blank)


class NoiseTest(unittest.TestCase):
    def test_noisy_step_edge_is_followed_without_spikes(self):
        height, width = 300, 300
        rng = np.random.default_rng(3)
        luma = np.empty((height, width))
        for y in range(height):
            x = 40 + int(round(20 * y / (height - 1)))
            luma[y] = 80.0
            luma[y, x:] = 30.0
        luma += rng.normal(0, 4, (height, width))
        disk = np.ones((height, width), dtype=bool)
        settings = dict(SETTINGS, inspect=[0, 0, width, height])
        trace = boundary.trace(disk, luma, settings)
        self.assertEqual(len(trace["segments"]), 1)
        self.assertEqual(trace["segments"][0]["rows"], height)
        self.assertLessEqual(trace["segments"][0]["max_step"], 2.0)
        true_x = [40 + int(round(20 * y / (height - 1))) + 0.5 for y in range(height)]
        for (y, x), expected in zip(trace["points"], true_x):
            self.assertLessEqual(abs(x - expected), 3.0)

    def test_flat_noise_yields_no_invented_boundary(self):
        height, width = 120, 160
        rng = np.random.default_rng(7)
        luma = 50 + rng.normal(0, 6, (height, width))
        disk = np.ones((height, width), dtype=bool)
        settings = dict(SETTINGS, inspect=[0, 0, width, height])
        trace = boundary.trace(disk, luma, settings)
        self.assertEqual(trace["points"], [])
        self.assertIn("留空", trace["summary"])


class GapTest(unittest.TestCase):
    def test_real_gap_is_left_blank_and_not_crossed(self):
        height, width = 100, 160
        luma = np.full((height, width), 20.0)
        for y in range(10, 90):
            luma[y, 70:] -= 4.0
        for y in range(40, 46):
            luma[y] = 20.0
        disk = np.ones((height, width), dtype=bool)
        settings = dict(SETTINGS, inspect=[0, 0, width, height])
        trace = boundary.trace(disk, luma, settings)
        blank = {row["y"] for row in trace["rows"] if row["status"] == "留空"}
        self.assertTrue(set(range(40, 46)) <= blank)
        adopted = {y for y, _ in trace["points"]}
        self.assertIn(39, adopted)
        self.assertIn(46, adopted)
        self.assertFalse(adopted & set(range(40, 46)))
        self.assertEqual([(segment["y0"], segment["y1"]) for segment in trace["segments"]],
                         [(10, 39), (46, 89)])


class OverlayTest(unittest.TestCase):
    def test_overlay_fill_matches_table_counts(self):
        from io import BytesIO

        from PIL import Image

        from moondesk import output

        luma, disk = scene()
        points = boundary.trace(disk, luma, SETTINGS)["points"]
        fills = {boundary.LIGHT: (250, 199, 122, 80), boundary.DARK: (72, 92, 168, 96),
                 boundary.PENDING: (226, 84, 170, 120)}
        labels = boundary.labels(disk, SETTINGS, points)
        rgb = np.stack([luma, luma, luma], axis=-1).clip(0, 255).astype(np.uint8)
        marks = np.zeros(disk.shape + (4,), dtype=np.uint8)
        for code, rgba in fills.items():
            marks[(labels == code) & disk] = rgba
        composed = Image.alpha_composite(Image.fromarray(rgb).convert("RGBA"),
                                         Image.fromarray(marks)).convert("RGB")
        base = rgb.astype(int)
        composed_values = np.array(composed).astype(int)
        totals, _ = boundary.counts(disk, SETTINGS, points)
        for code, rgba in fills.items():
            alpha = rgba[3] / 255
            target = np.array(rgba[:3]) * alpha + base * (1 - alpha)
            painted = ((np.abs(composed_values - target).max(axis=2) <= 1) & disk).sum()
            self.assertEqual(int(painted), totals[{boundary.LIGHT: "light",
                                                   boundary.DARK: "dark",
                                                   boundary.PENDING: "pending"}[code]])
        self.assertGreater(totals["pending"], 0)
        trace = boundary.trace(disk, luma, SETTINGS)
        text = output.csv_text({"mode": "illumination", "rows": [totals], "trace": trace,
                                "settings": dict(SETTINGS, light="a")})
        self.assertIn("boundary_summary", text)
        self.assertIn("segment", text)


class ModelTest(unittest.TestCase):
    def test_boundary_defaults_to_trace(self):
        panel = {"width": 200, "height": 120, "sources": {"a": {}},
                 "settings": model.defaults(200, 120)}
        chosen = model.settings({}, panel)
        self.assertEqual(chosen["boundary"], "trace")

    def test_unknown_boundary_rejected(self):
        panel = {"width": 200, "height": 120, "sources": {"a": {}},
                 "settings": model.defaults(200, 120)}
        with self.assertRaises(Invalid):
            model.settings({"boundary": "blend"}, panel)


class IlluminationTest(unittest.TestCase):
    def test_threshold_mode_still_available(self):
        luma, disk = scene()
        bright = luma > 18
        traced, trace, _ = analysis.illumination(disk, luma, bright, SETTINGS)
        legacy = dict(SETTINGS, boundary="threshold")
        threshold, none_trace, _ = analysis.illumination(disk, luma, bright, legacy)
        self.assertIsNotNone(trace)
        self.assertIsNone(none_trace)
        self.assertEqual(threshold[0]["pending_px"], 0)
        self.assertNotEqual(threshold[0]["light_px"], traced[0]["light_px"])


if __name__ == "__main__":
    unittest.main()
