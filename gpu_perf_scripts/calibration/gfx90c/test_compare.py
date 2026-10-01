"""Regression coverage for the calibration gate's exit status."""
import contextlib
import io
import unittest
from unittest.mock import patch

import compare


class ComparisonGateTests(unittest.TestCase):
    def gate(self, sim):
        with patch.object(compare, "load_pairs", side_effect=[{"a": 100, "b": 100}, sim]), \
                patch.object(compare, "load_production_targets", return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            return compare.main()

    def test_all_required_benchmarks_pass(self):
        self.assertEqual(self.gate({"a": 100, "b": 105}), 0)

    def test_outside_threshold_fails(self):
        self.assertEqual(self.gate({"a": 100, "b": 120}), 1)

    def test_missing_benchmark_fails(self):
        self.assertEqual(self.gate({"a": 100}), 1)

    def test_invalid_measurements_fail(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertEqual(self.gate({"a": 100, "b": value}), 1)


if __name__ == "__main__":
    unittest.main()
