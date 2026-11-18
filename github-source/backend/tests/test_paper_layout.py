"""Offline checks for matching translated paragraphs to PDF page geometry."""

import unittest

from backend.app.services.paper_reader import match_page_blocks


class PaperLayoutTests(unittest.TestCase):
    def test_matches_blocks_without_covering_figure_or_vertical_stamp(self):
        blocks = [
            (100, 40, 260, 60, "Layer Normalization\n", 0, 0),
            (100, 90, 420, 190, "Training state-of-the-art networks\nreduces training time.\n", 1, 0),
            (100, 210, 430, 390, "", 2, 1),
            (12, 100, 35, 410, "arXiv:1607.06450v1\n", 3, 0),
        ]
        paragraphs = [
            {"id": "title", "text": "Layer Normalization"},
            {"id": "body", "text": "Training state-of-the-art networks\nreduces training time."},
            {"id": "stamp", "text": "arXiv:1607.06450v1"},
            {"id": "unmatched", "text": "No matching text in this PDF page"},
        ]
        regions = match_page_blocks(paragraphs, blocks)
        self.assertEqual([item["paragraph_id"] for item in regions], ["title", "body"])
        self.assertEqual(regions[1]["bbox"], [100.0, 90.0, 420.0, 190.0])

    def test_normalizes_superscript_markup(self):
        blocks = [(10, 10, 60, 25, "x2 baseline\n", 0, 0)]
        regions = match_page_blocks([{"id": "formula", "text": "x<sup>2</sup> baseline"}], blocks)
        self.assertEqual(regions[0]["match_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
