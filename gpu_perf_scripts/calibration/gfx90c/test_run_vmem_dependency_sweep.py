import contextlib
import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_vmem_dependency_sweep as sweep


class PlanTests(unittest.TestCase):
    def test_guard_matches_workload_tokens_but_not_inspection_arguments(self):
        guard = re.compile(sweep.FORBIDDEN_PROCESS_REGEX)
        self.assertIsNotNone(guard.search("/tmp/build/Vgfx9_compute_unit_tb --run"))
        self.assertIsNotNone(guard.search("python3 -m pytest tests"))
        self.assertIsNone(
            guard.search(
                "rg -n 'Vgfx9_compute_unit_tb|verilator_bin|pytest|miaow_gcn4' /tmp"
            )
        )
        self.assertIsNone(guard.search("rg -n result /tmp/pytest-artifact"))

    def test_points_keep_matched_zero_and_body_adjacent(self):
        self.assertEqual(
            [(p.mode, p.workgroups, p.repeats) for p in sweep.sweep_points()],
            [
                ("independent2", 16, 0),
                ("independent2", 16, 64),
                ("independent2", 28, 0),
                ("independent2", 28, 64),
                ("independent8", 16, 0),
                ("independent8", 16, 64),
                ("independent8", 28, 0),
                ("independent8", 28, 64),
            ],
        )

    def test_point_filters_keep_a_focused_pair(self):
        self.assertEqual(
            sweep.sweep_points(("independent8",), (28,)),
            (
                sweep.SweepPoint("independent8", 28, 0),
                sweep.SweepPoint("independent8", 28, 64),
            ),
        )

    def test_command_has_strict_controls_and_exact_geometry(self):
        point = sweep.SweepPoint("independent8", 28, 64)
        command = sweep.build_acquire_command(
            point,
            Path("/tmp/explicit-point"),
            batches=9,
            python="python-test",
            acquire_script=Path("collector.py"),
        )
        expected_pairs = {
            "--batches": "9",
            "--warmup": "20",
            "--iters": "6500",
            "--cooldown-seconds": "30",
            "--max-start-temp-c": "50",
            "--max-temp-c": "58",
            "--sample-interval-seconds": "0.005",
            "--min-active-samples": "10",
            "--forbid-process-regex": sweep.FORBIDDEN_PROCESS_REGEX,
            "--output-dir": "/tmp/explicit-point",
            "--vmem-mode": "independent8",
            "--vmem-repeats": "64",
            "--vmem-workgroups": "28",
        }
        self.assertEqual(command[:2], ["python-test", "collector.py"])
        for option, value in expected_pairs.items():
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertIn("vmemloadshape", command)

    def test_zero_and_body_iterations_target_the_timed_window(self):
        for repeats, expected in ((0, "10000"), (64, "6500")):
            command = sweep.build_acquire_command(
                sweep.SweepPoint("independent2", 16, repeats),
                Path("/tmp/explicit-point"),
                batches=1,
            )
            self.assertEqual(command[command.index("--iters") + 1], expected)

    def test_default_is_one_batch_dry_run_and_production_is_nine(self):
        pilot = sweep.parse_args([])
        production = sweep.parse_args(["--execute", "--production"])
        self.assertFalse(pilot.execute)
        self.assertFalse(pilot.production)
        self.assertTrue(production.execute)
        self.assertTrue(production.production)
        focused = sweep.parse_args(
            ["--mode", "independent8", "--workgroups", "28"]
        )
        self.assertEqual(focused.mode, ["independent8"])
        self.assertEqual(focused.workgroups, [28])


class ExecutionTests(unittest.TestCase):
    def test_dry_run_does_not_create_output_or_invoke_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "not-created"

            def unexpected_runner(*args, **kwargs):
                self.fail("dry run invoked subprocess")

            with contextlib.redirect_stdout(io.StringIO()) as output:
                rc = sweep.run_sweep(
                    output_root=root,
                    batches=1,
                    execute=False,
                    runner=unexpected_runner,
                    sleeper=lambda _: self.fail("dry run slept"),
                )
            self.assertEqual(rc, 0)
            self.assertFalse(root.exists())
            self.assertIn("dry run", output.getvalue())
            self.assertEqual(output.getvalue().count("acquire_pinned.py"), 8)

    def test_execute_stops_at_first_failure_and_records_artifact_paths(self):
        calls = []

        def fake_runner(command, *, check):
            self.assertFalse(check)
            calls.append(command)
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir()
            (output_dir / "manifest.json").write_text("{}\n")
            return subprocess.CompletedProcess(
                command, 7 if len(calls) == 2 else 0
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "new-run"
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                rc = sweep.run_sweep(
                    output_root=root,
                    batches=1,
                    execute=True,
                    runner=fake_runner,
                    sleeper=lambda _: None,
                    guard_waiter=lambda: None,
                )
            self.assertEqual(rc, 7)
            self.assertEqual(len(calls), 2)
            manifest = json.loads((root / "sweep_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failed_point"], "independent2-g16-r64")
            self.assertEqual(len(manifest["completed"]), 2)
            for record in manifest["completed"]:
                self.assertTrue(Path(record["collector_manifest"]).is_file())

    def test_execute_cools_down_between_successful_points(self):
        sleeps = []

        def fake_runner(command, *, check):
            self.assertFalse(check)
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir()
            (output_dir / "manifest.json").write_text("{}\n")
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as temporary:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = sweep.run_sweep(
                    output_root=Path(temporary) / "new-run",
                    batches=1,
                    execute=True,
                    runner=fake_runner,
                    sleeper=sleeps.append,
                    guard_waiter=lambda: None,
                )

        self.assertEqual(rc, 0)
        self.assertEqual(
            sleeps,
            [sweep.INTER_POINT_COOLDOWN_SECONDS] * 7,
        )

    def test_execute_requires_stable_guard_before_every_point(self):
        waits = []

        def fake_runner(command, *, check):
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir()
            (output_dir / "manifest.json").write_text("{}\n")
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as temporary:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = sweep.run_sweep(
                    output_root=Path(temporary) / "new-run",
                    batches=1,
                    execute=True,
                    runner=fake_runner,
                    sleeper=lambda _: None,
                    guard_waiter=lambda: waits.append("clear"),
                )

        self.assertEqual(rc, 0)
        self.assertEqual(waits, ["clear"] * 8)

    def test_stable_guard_resets_after_a_match(self):
        class FakeGuard:
            def __init__(self):
                self.matches = iter((None, object(), None, None, None))

            def check(self):
                return next(self.matches)

        times = iter((0.0, 0.0, 1.0, 2.0, 3.0, 4.0))
        sleeps = []
        sweep.wait_for_stable_guard_clear(
            FakeGuard(),
            stable_seconds=2.0,
            timeout_seconds=10.0,
            poll_seconds=1.0,
            monotonic=lambda: next(times),
            sleeper=sleeps.append,
        )
        self.assertEqual(sleeps, [1.0, 1.0, 1.0, 1.0])

    def test_guard_interrupt_is_recorded(self):
        def interrupt():
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "new-run"
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                rc = sweep.run_sweep(
                    output_root=root,
                    batches=1,
                    execute=True,
                    guard_waiter=interrupt,
                )
            manifest = json.loads((root / "sweep_manifest.json").read_text())

        self.assertEqual(rc, 130)
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual(manifest["failed_point"], "independent2-g16-r0")

    def test_execute_requires_a_new_run_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                sweep.run_sweep(
                    output_root=Path(temporary),
                    batches=1,
                    execute=True,
                )


if __name__ == "__main__":
    unittest.main()
