import argparse
import contextlib
import io
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acquire_pinned as acquire


def sample(clock=1600, policy="high", temperature=50_000, monotonic=1.0):
    return acquire.TraceSample(
        monotonic_seconds=monotonic,
        epoch_ns=1,
        policy=policy,
        selected_clock_mhz=clock,
        temperature_millidegrees=temperature,
        raw_sclk=f"2: {clock}Mhz *",
    )


def timed_marker(start_seconds, end_seconds):
    return (
        f"{acquire.TIMED_WINDOW_MARKER} "
        f"start_monotonic_ns={round(start_seconds * 1_000_000_000)} "
        f"end_monotonic_ns={round(end_seconds * 1_000_000_000)}"
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

    def test_parse_timed_window_requires_one_exact_ordered_marker(self):
        marker = timed_marker(1.25, 1.5)
        self.assertEqual(
            acquire.parse_timed_window(f"noise\n{marker}\nmore noise"),
            acquire.TimedWindow(
                marker=marker,
                start_monotonic_ns=1_250_000_000,
                end_monotonic_ns=1_500_000_000,
            ),
        )
        invalid_outputs = [
            "no marker",
            f"{marker}\n{marker}",
            f"prefix {marker}",
            (
                f"{acquire.TIMED_WINDOW_MARKER} "
                "start_monotonic_ns=2 end_monotonic_ns=1"
            ),
            (
                f"{acquire.TIMED_WINDOW_MARKER} "
                "start_monotonic_ns=-1 end_monotonic_ns=2"
            ),
        ]
        for output in invalid_outputs:
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    acquire.parse_timed_window(output)

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
        window = overrides.pop("window", None)
        if window is None and trace:
            window = (
                trace[0].monotonic_seconds,
                trace[-1].monotonic_seconds,
            )
        marker = "" if window is None else timed_marker(*window)
        arguments = {
            "returncode": 0,
            "trace": trace,
            "verification_mode": "marker",
            "verification_pattern": r"verification Passed!",
            "stdout": "matrixmult 42.0\n",
            "stderr": f"matrixmult verification Passed!\n{marker}\n",
            "maximum_temperature_millidegrees": 75_000,
            "minimum_active_samples": 3,
        }
        arguments.update(overrides)
        return acquire.assess_batch(**arguments)

    def test_accepts_idle_edges_around_contiguous_active_interval(self):
        result = self.assess(
            [
                sample(200, monotonic=0.0),
                sample(monotonic=1.0),
                sample(monotonic=2.0),
                sample(monotonic=3.0),
                sample(0, monotonic=4.0),
            ],
            window=(1.0, 3.0),
        )
        self.assertTrue(result.accepted, result.reasons)
        self.assertEqual(result.active_samples, 3)
        self.assertEqual(result.timed_window_samples, 3)

    def test_rejects_clock_drop_inside_active_interval(self):
        result = self.assess(
            [
                sample(monotonic=0.0),
                sample(700, monotonic=1.0),
                sample(monotonic=2.0),
                sample(monotonic=3.0),
            ]
        )
        self.assertFalse(result.accepted)
        self.assertTrue(any("timed window" in reason for reason in result.reasons))

    def test_rejects_missing_multiple_malformed_and_out_of_trace_markers(self):
        trace = [
            sample(monotonic=1.0),
            sample(monotonic=2.0),
            sample(monotonic=3.0),
        ]
        cases = {
            "missing": "matrixmult verification Passed!\n",
            "multiple": (
                "matrixmult verification Passed!\n"
                f"{timed_marker(1.0, 3.0)}\n{timed_marker(1.0, 3.0)}\n"
            ),
            "malformed": (
                "matrixmult verification Passed!\n"
                f"{acquire.TIMED_WINDOW_MARKER} start=1 end=3\n"
            ),
            "outside": (
                "matrixmult verification Passed!\n"
                f"{timed_marker(0.0, 4.0)}\n"
            ),
        }
        for name, stderr in cases.items():
            with self.subTest(name=name):
                result = self.assess(trace, stderr=stderr)
                self.assertFalse(result.accepted)
                self.assertTrue(
                    any("timed-window" in reason for reason in result.reasons),
                    result.reasons,
                )

    def test_rejects_policy_thermal_rc_and_verification_failures(self):
        result = self.assess(
            [
                sample(policy="auto", temperature=76_000, monotonic=index)
                for index in range(3)
            ],
            returncode=3,
            stderr="mismatch\n",
        )
        self.assertFalse(result.accepted)
        joined = " ".join(result.reasons)
        self.assertIn("rc=3", joined)
        self.assertIn("not high", joined)
        self.assertIn("temperature", joined)
        self.assertIn("marker", joined)

    def test_rejects_forbidden_process_with_exact_evidence(self):
        match = acquire.ForbiddenProcessMatch(
            pattern="verilator", pid=4321, command="/opt/bin/verilator --build"
        )
        result = self.assess(
            [sample(monotonic=index) for index in range(3)],
            forbidden_process_match=match,
        )
        self.assertFalse(result.accepted)
        self.assertIn("pid=4321", " ".join(result.reasons))
        self.assertIn("/opt/bin/verilator --build", " ".join(result.reasons))


class ProcessGuardTests(unittest.TestCase):
    def test_disabled_guard_does_not_scan_processes(self):
        def unexpected_snapshot():
            self.fail("disabled process guard must not scan /proc")

        guard = acquire.ForbiddenProcessGuard(
            [], snapshotter=unexpected_snapshot, collector_pid=100
        )
        self.assertIsNone(guard.check())

    def test_excludes_collector_and_ancestor_but_matches_external_process(self):
        processes = [
            acquire.ProcessSnapshot(100, 50, "collector --forbid-process-regex busy"),
            acquire.ProcessSnapshot(50, 1, "shell busy"),
            acquire.ProcessSnapshot(1, 0, "init"),
            acquire.ProcessSnapshot(200, 1, "worker busy --repeat"),
        ]
        guard = acquire.ForbiddenProcessGuard(
            ["busy"], snapshotter=lambda: processes, collector_pid=100
        )
        self.assertEqual(guard.excluded_pids, {1, 50, 100})
        self.assertEqual(
            guard.check(),
            acquire.ForbiddenProcessMatch(
                pattern="busy", pid=200, command="worker busy --repeat"
            ),
        )

        processes.pop()
        self.assertIsNone(guard.check())

    def test_require_clear_raises_structured_match(self):
        processes = [
            acquire.ProcessSnapshot(100, 1, "collector"),
            acquire.ProcessSnapshot(1, 0, "init"),
            acquire.ProcessSnapshot(777, 1, "Vgfx9_compute_unit_tb test"),
        ]
        guard = acquire.ForbiddenProcessGuard(
            [r"Vgfx9_compute_unit_tb|verilator_bin"],
            snapshotter=lambda: processes,
            collector_pid=100,
        )
        with self.assertRaises(acquire.ForbiddenProcessError) as caught:
            guard.require_clear()
        self.assertEqual(caught.exception.match.pid, 777)
        self.assertIn("command='Vgfx9_compute_unit_tb test'", str(caught.exception))

    def test_readiness_checks_guard_before_sysfs(self):
        match = acquire.ForbiddenProcessMatch("busy", 888, "busy worker")
        processes = [
            acquire.ProcessSnapshot(100, 1, "collector"),
            acquire.ProcessSnapshot(1, 0, "init"),
            acquire.ProcessSnapshot(888, 1, "busy worker"),
        ]
        guard = acquire.ForbiddenProcessGuard(
            ["busy"], snapshotter=lambda: processes, collector_pid=100
        )
        with self.assertRaises(acquire.ForbiddenProcessError) as caught:
            acquire.wait_until_ready(
                policy_file=Path("/does/not/exist/policy"),
                temperature_file=Path("/does/not/exist/temp"),
                start_temperature_millidegrees=50_000,
                deadline=1.0,
                next_eligible_time=0.0,
                process_guard=guard,
            )
        self.assertEqual(caught.exception.match, match)


class TelemetryMonitorTests(unittest.TestCase):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def monotonic(self):
            return self.now

    class FakeEvent:
        def __init__(self, clock):
            self.clock = clock
            self.waits = []
            self.stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, timeout):
            self.waits.append(timeout)
            self.clock.now += timeout
            return self.stopped

    def test_sampling_uses_requested_cadence(self):
        clock = self.FakeClock()
        stop_event = self.FakeEvent(clock)
        samples = []
        complete = threading.Event()

        def collect(trace_sample):
            samples.append(trace_sample)
            if len(samples) == 3:
                stop_event.set()
                complete.set()

        monitor = acquire.TelemetryMonitor(
            sample_reader=sample,
            sample_sink=collect,
            should_continue=lambda: True,
            interval_seconds=0.005,
            maximum_temperature_millidegrees=58_000,
            abort_callback=lambda: self.fail("healthy samples must not abort"),
            monotonic=clock.monotonic,
            stop_event=stop_event,
        )
        monitor.start()
        self.assertTrue(complete.wait(timeout=1))
        monitor.stop_and_join()

        self.assertEqual(len(samples), 3)
        self.assertEqual(stop_event.waits, [0.0, 0.005, 0.005])
        self.assertFalse(monitor.is_alive)

    def test_policy_violation_aborts_immediately(self):
        aborted = threading.Event()
        monitor = acquire.TelemetryMonitor(
            sample_reader=lambda: sample(policy="auto"),
            sample_sink=lambda _: None,
            should_continue=lambda: True,
            interval_seconds=60.0,
            maximum_temperature_millidegrees=58_000,
            abort_callback=aborted.set,
        )
        monitor.start()
        self.assertTrue(aborted.wait(timeout=1))
        monitor.stop_and_join()

        self.assertEqual(monitor.sampling_error, "performance policy left high")
        self.assertFalse(monitor.thermal_abort)

    def test_thermal_limit_aborts_immediately(self):
        aborted = threading.Event()
        monitor = acquire.TelemetryMonitor(
            sample_reader=lambda: sample(temperature=58_000),
            sample_sink=lambda _: None,
            should_continue=lambda: True,
            interval_seconds=60.0,
            maximum_temperature_millidegrees=58_000,
            abort_callback=aborted.set,
        )
        monitor.start()
        self.assertTrue(aborted.wait(timeout=1))
        monitor.stop_and_join()

        self.assertTrue(monitor.thermal_abort)
        self.assertIsNone(monitor.sampling_error)

    def test_stop_wakes_sampler_and_joins_thread(self):
        sampled = threading.Event()
        monitor = acquire.TelemetryMonitor(
            sample_reader=sample,
            sample_sink=lambda _: sampled.set(),
            should_continue=lambda: True,
            interval_seconds=60.0,
            maximum_temperature_millidegrees=58_000,
            abort_callback=lambda: None,
        )
        monitor.start()
        self.assertTrue(sampled.wait(timeout=1))
        monitor.stop_and_join()

        self.assertFalse(monitor.is_alive)


class CommandTests(unittest.TestCase):
    def test_reference_defaults_are_strict(self):
        args = acquire.parse_args(["matrixmult"])
        self.assertEqual(args.batches, 9)
        self.assertEqual(args.warmup, 20)
        self.assertEqual(args.cooldown_seconds, 30.0)
        self.assertEqual(args.max_start_temp_c, 50.0)
        self.assertEqual(args.max_temp_c, 58.0)
        self.assertEqual(args.min_active_samples, 10)
        self.assertEqual(args.forbid_process_regex, [])

    def test_process_guard_regex_is_repeatable_and_validated(self):
        args = acquire.parse_args(
            [
                "--forbid-process-regex",
                "verilator_bin",
                "--forbid-process-regex",
                "pytest.*miaow_gcn4",
                "matrixmult",
            ]
        )
        self.assertEqual(
            args.forbid_process_regex,
            ["verilator_bin", "pytest.*miaow_gcn4"],
        )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                acquire.parse_args(
                    ["--forbid-process-regex", "[invalid", "matrixmult"]
                )

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
