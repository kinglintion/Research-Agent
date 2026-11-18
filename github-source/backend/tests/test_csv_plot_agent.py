"""Exercise model-code retries and the real local PNG renderer without a network call."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.services import csv_plot_agent as plot


class PlotAgentTests(unittest.TestCase):
    def test_code_must_read_uploaded_rows(self):
        with self.assertRaisesRegex(ValueError, "rows"):
            plot._check_code("fig, ax = plt.subplots(); ax.bar([1], [2])")
        with self.assertRaisesRegex(ValueError, "Import"):
            plot._check_code("import os\nfig, ax = plt.subplots(); ax.bar([x['value'] for x in rows], [1])")

    def test_failed_code_is_repaired_and_png_is_verified(self):
        outputs = ["value = float(rows[0]['value']) / 0\nfig, ax = plt.subplots()",
                   "fig, ax = plt.subplots()\nax.bar([row['time'] for row in rows], [float(row['value']) for row in rows])"]
        def completion(*args, **kwargs):
            code = outputs.pop(0)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=code))], model="test-model")
        client = SimpleNamespace(with_options=lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=completion))))
        attempts = []
        with patch.object(plot, "get_llm_client", return_value=client):
            result = plot.generate_csv_plot("sample.csv", "time,value\nt1,2\nt2,3\n", "画柱状图",
                                            on_attempt=attempts.append)
        self.assertEqual([item["status"] for item in attempts], ["failed", "success"])
        self.assertEqual(result["plot_model"], "test-model")
        self.assertGreater(len(result["generated_png"]), 1000)
        self.assertIn("rows", result["generated_code"])


if __name__ == "__main__":
    unittest.main()
