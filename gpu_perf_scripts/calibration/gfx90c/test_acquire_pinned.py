import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acquire_pinned as acquire


def sample(clock=1600, policy="high", temperature=50_000):
    return acquire.TraceSample(
        monotonic_seconds=1.0,
        epoch_ns=1,
        policy=policy,
        selected_clock_mhz=clock,
        temperature_millidegrees=temperature,
        raw_sclk=f"2: {clock}Mhz *",
    )


class StatisticsTests(unittest.TestCase):
    def test_summary(self):
        result = acquire.summarize(
            [1.0, 2.0, 3.0, 4.0, 100.0],
            bootstrap_resamples=500,
            bootstrap_seed=7,
        )
        self.assertEqual(result["median_us"], 3.0)
        self.assertEqual(result["mad_us"], 1.0)
        self.assertEqual(result["minimum_us"], 1.0)
        self.assertEqual(result["maximum_us"], 100.0)
        low, high = result["bootstrap95_median_us"]
        self.assertLessEqual(low, 3.0)
        self.assertGreaterEqual(high, 3.0)

    def test_bootstrap_is_deterministic(self):
        first = acquire.bootstrap_median_interval(
            [1.0, 2.0, 3.0], resamples=100, seed=11
        )
        second = acquire.bootstrap_median_interval(
            [1.0, 2.0, 3.0], resamples=100, seed=11
        )
        self.assertEqual(first, second)


class ParsingTests(unittest.TestCase):
    def test_parse_selected_clock(self):
        text = "0: 200Mhz  1: 700Mhz\n2: 1600Mhz *\n"
        self.assertEqual(acquire.parse_selected_clock_mhz(text), 1600)
        self.assertIsNone(acquire.parse_selected_clock_mhz("0: 200Mhz"))

    def test_parse_metric_requires_one_positive_finite_value(self):
        self.assertEqual(acquire.parse_metric("matrixmult 42.25\n", "matrixmult"), 42.25)
        with self.assertRaises(ValueError):
            acquire.parse_metric("matrixmult 1\nmatrixmult 2\n", "matrixmult")
        with self.assertRaises(ValueError):
            acquire.parse_metric("matrixmult nan\n", "matrixmult")

    def test_reserved_passthrough_arguments_are_rejected(self):
        self.assertEqual(
            acquire.validate_passthrough_args(["--", "--matrix-size", "64"]),
            ["--matrix-size", "64"],
        )
        with self.assertRaises(ValueError):
            acquire.validate_passthrough_args(["--iters=2"])

    def test_kmeans_fixture_selection_tracks_last_geometry_option(self):
        fixtures, sources = acquire.kmeans_fixture_files(
            ["--points", "1024", "--points", "8192", "--features", "16"]
        )
        self.assertEqual(
            [path.name for path in fixtures],
            ["kmeans_8192_features.f32", "kmeans_8192_membership.i32"],
        )
        self.assertEqual(
            [path.name for path in sources],
            ["generate_fixtures.go", "kmeans.go"],
        )

    def test_kmeans_fixture_selection_rejects_unmatched_geometry(self):
        with self.assertRaisesRegex(ValueError, "reference acquisition"):
            acquire.kmeans_fixture_files(["--points", "3072"])
        with self.assertRaisesRegex(ValueError, "reference acquisition"):
            acquire.kmeans_fixture_files(["--features", "32"])
        with self.assertRaisesRegex(ValueError, "separate value"):
            acquire.kmeans_fixture_files(["--points=8192"])


class AcceptanceTests(unittest.TestCase):
    def assess(self, trace, **overrides):
        arguments = {
            "returncode": 0,
            "trace": trace,
            "verification_mode": "marker",
            "verification_pattern": r"verification Passed!",
            "stdout": "matrixmult 42.0\n",
            "stderr": "matrixmult verification Passed!\n",
            "maximum_temperature_millidegrees": 75_000,
            "minimum_active_samples": 3,
        }
        arguments.update(overrides)
        return acquire.assess_batch(**arguments)

    def test_accepts_idle_edges_around_contiguous_active_interval(self):
        result = self.assess(
            [sample(200), sample(), sample(), sample(), sample(0)]
        )
        self.assertTrue(result.accepted, result.reasons)
        self.assertEqual(result.active_samples, 3)

    def test_rejects_clock_drop_inside_active_interval(self):
        result = self.assess(
            [sample(), sample(700), sample(), sample()]
        )
        self.assertFalse(result.accepted)
        self.assertTrue(any("left 1600" in reason for reason in result.reasons))

    def test_rejects_policy_thermal_rc_and_verification_failures(self):
        result = self.assess(
            [sample(policy="auto", temperature=76_000)] * 3,
            returncode=3,
            stderr="mismatch\n",
        )
        self.assertFalse(result.accepted)
        joined = " ".join(result.reasons)
        self.assertIn("rc=3", joined)
        self.assertIn("not high", joined)
        self.assertIn("temperature", joined)
        self.assertIn("marker", joined)


class CommandTests(unittest.TestCase):
    def test_reference_defaults_are_strict(self):
        args = acquire.parse_args(["matrixmult"])
        self.assertEqual(args.batches, 9)
        self.assertEqual(args.warmup, 20)
        self.assertEqual(args.cooldown_seconds, 30.0)
        self.assertEqual(args.max_start_temp_c, 50.0)
        self.assertEqual(args.max_temp_c, 58.0)
        self.assertEqual(args.min_active_samples, 10)

    def test_command_enforces_serial_batch_options_and_passes_geometry(self):
        command = acquire.build_container_command(
            podman="podman",
            image="image",
            container_name="batch-1",
            binary=acquire.HERE / "build/isca10_bench",
            benchmark="matrixmult",
            warmup=20,
            iterations=1000,
            benchmark_args=["--matrix-size", "64"],
        )
        self.assertIn("--rm", command)
        self.assertEqual(command[command.index("--only") + 1], "matrixmult")
        self.assertEqual(command[command.index("--warmup") + 1], "20")
        self.assertEqual(command[command.index("--iters") + 1], "1000")
        self.assertEqual(command[-2:], ["--matrix-size", "64"])

    def test_verification_auto_mode_rejects_unverified_benchmark(self):
        args = argparse.Namespace(
            verification_regex=None,
            rc_guarded_verification=False,
            benchmark="relu",
        )
        with self.assertRaises(ValueError):
            acquire.resolve_verification(args)

    def test_kmeans_auto_mode_requires_explicit_success_marker(self):
        args = argparse.Namespace(
            verification_regex=None,
            rc_guarded_verification=False,
            benchmark="kmeans",
        )
        self.assertEqual(
            acquire.resolve_verification(args),
            ("marker", r"kmeans .*verification Passed!"),
        )


if __name__ == "__main__":
    unittest.main()
