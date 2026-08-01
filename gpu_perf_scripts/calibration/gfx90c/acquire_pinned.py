#!/usr/bin/env python3
"""Acquire serialized, pinned-clock gfx90c hardware timing batches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_IMAGE = "docker.io/rocm/dev-ubuntu-24.04:7.1.1"
REQUIRED_POLICY = "high"
REQUIRED_CLOCK_MHZ = 1600
TIMED_WINDOW_MARKER = "MGPUSIM_TIMED_WINDOW_V1"
TIMED_WINDOW_PATTERN = re.compile(
    rf"^{TIMED_WINDOW_MARKER} "
    r"start_monotonic_ns=(\d+) end_monotonic_ns=(\d+)$"
)

BENCHMARK_HSACO = {
    "vectoradd": "amd/benchmarks/amdappsdk/vectoradd/kernels_gfx90c.hsaco",
    "relu": "amd/benchmarks/dnn/layer_benchmarks/relu/kernels_gfx90c.hsaco",
    "matrixmult": (
        "amd/benchmarks/amdappsdk/matrixmultiplication/"
        "kernels_gfx90c.hsaco"
    ),
    "matrixtranspose": (
        "amd/benchmarks/amdappsdk/matrixtranspose/kernels_gfx90c.hsaco"
    ),
    "bitonicsort": "amd/benchmarks/amdappsdk/bitonicsort/kernels_gfx90c.hsaco",
    "aes": "amd/benchmarks/heteromark/aes/kernels_gfx90c.hsaco",
    "fir": "amd/benchmarks/heteromark/fir/kernels_gfx90c.hsaco",
    "fp32fma": (
        "amd/benchmarks/microbench/fp32throughput/kernels_gfx90c.hsaco"
    ),
    "vmemloadshape": (
        "amd/benchmarks/microbench/vmemloadshape/kernels_gfx90c.hsaco"
    ),
    "cache_latency": (
        "amd/benchmarks/microbench/cachelatency/kernels_gfx90c.hsaco"
    ),
    "storestride": (
        "amd/benchmarks/microbench/storestride/kernels_gfx90c.hsaco"
    ),
    "scratchspill": (
        "amd/benchmarks/microbench/scratchspill/kernels_gfx90c.hsaco"
    ),
    "kmeans": "amd/benchmarks/heteromark/kmeans/kernels_gfx90c.hsaco",
    "pagerank": "amd/benchmarks/heteromark/pagerank/kernels_gfx90c.hsaco",
    "nw": "amd/benchmarks/rodinia/nw/kernels_gfx90c.hsaco",
}

# Marker verification requires an explicit success message. RC-guarded
# verification means the harness checks every result after timing and exits 3
# on a mismatch, but currently emits no success marker.
AUTO_VERIFICATION = {
    "matrixmult": ("marker", r"matrixmult .*verification Passed!"),
    "fp32fma": ("marker", r"fp32fma verification Passed!"),
    "vmemloadshape": ("marker", r"vmemloadshape verification Passed!"),
    "kmeans": ("marker", r"kmeans .*verification Passed!"),
    "cache_latency": ("rc-guarded", None),
    "storestride": ("rc-guarded", None),
    "scratchspill": ("rc-guarded", None),
}

RESERVED_BENCHMARK_ARGS = ("--only", "--warmup", "--iters")


@dataclass(frozen=True)
class TraceSample:
    monotonic_seconds: float
    epoch_ns: int
    policy: str
    selected_clock_mhz: int | None
    temperature_millidegrees: int
    raw_sclk: str


@dataclass(frozen=True)
class BatchAssessment:
    accepted: bool
    reasons: tuple[str, ...]
    active_samples: int
    active_interval_samples: int
    maximum_temperature_millidegrees: int | None
    timed_window_marker: str | None
    timed_window_start_monotonic_ns: int | None
    timed_window_end_monotonic_ns: int | None
    timed_window_samples: int


@dataclass(frozen=True)
class TimedWindow:
    marker: str
    start_monotonic_ns: int
    end_monotonic_ns: int


@dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    parent_pid: int
    command: str


@dataclass(frozen=True)
class ForbiddenProcessMatch:
    pattern: str
    pid: int
    command: str


class ForbiddenProcessError(RuntimeError):
    def __init__(self, match: ForbiddenProcessMatch):
        self.match = match
        super().__init__(format_forbidden_process_match(match))


def format_forbidden_process_match(match: ForbiddenProcessMatch) -> str:
    return (
        f"forbidden process matched {match.pattern!r}: "
        f"pid={match.pid} command={match.command!r}"
    )


def read_process_snapshot(proc_root: Path = Path("/proc")) -> list[ProcessSnapshot]:
    """Read live command lines without retaining unrelated process contents."""
    snapshots: list[ProcessSnapshot] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return snapshots
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            after_name = stat[stat.rfind(")") + 1 :].split()
            parent_pid = int(after_name[1])
            argv = [
                part.decode(errors="replace")
                for part in (entry / "cmdline").read_bytes().split(b"\0")
                if part
            ]
            if argv:
                command = shlex.join(argv)
            else:
                command = f"[{(entry / 'comm').read_text().strip()}]"
        except (OSError, ValueError, IndexError):
            # Processes can disappear between directory enumeration and reads.
            continue
        snapshots.append(
            ProcessSnapshot(
                pid=int(entry.name), parent_pid=parent_pid, command=command
            )
        )
    return snapshots


def process_lineage_pids(
    snapshots: Sequence[ProcessSnapshot], start_pid: int
) -> set[int]:
    parents = {process.pid: process.parent_pid for process in snapshots}
    lineage: set[int] = set()
    pid = start_pid
    while pid > 0 and pid not in lineage:
        lineage.add(pid)
        pid = parents.get(pid, 0)
    return lineage


class ForbiddenProcessGuard:
    def __init__(
        self,
        patterns: Sequence[str],
        *,
        snapshotter: Callable[[], Sequence[ProcessSnapshot]] = read_process_snapshot,
        collector_pid: int | None = None,
    ) -> None:
        self.patterns = tuple(patterns)
        self._compiled = tuple(re.compile(pattern) for pattern in self.patterns)
        self._snapshotter = snapshotter
        initial = list(snapshotter()) if self._compiled else []
        self.excluded_pids = process_lineage_pids(
            initial, os.getpid() if collector_pid is None else collector_pid
        )

    def check(self) -> ForbiddenProcessMatch | None:
        if not self._compiled:
            return None
        for process in self._snapshotter():
            if process.pid in self.excluded_pids:
                continue
            for pattern, compiled in zip(self.patterns, self._compiled):
                if compiled.search(process.command):
                    return ForbiddenProcessMatch(
                        pattern=pattern,
                        pid=process.pid,
                        command=process.command,
                    )
        return None

    def require_clear(self) -> None:
        if match := self.check():
            raise ForbiddenProcessError(match)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_selected_clock_mhz(text: str) -> int | None:
    """Return the starred DPM clock, accepting the kernel's multiline form."""
    matches = re.findall(r"\b\d+:\s*(\d+)Mhz\s*(\*)?", text, re.IGNORECASE)
    selected = [int(clock) for clock, star in matches if star]
    if len(selected) != 1:
        return None
    return selected[0]


def parse_metric(output: str, metric_name: str) -> float:
    pattern = re.compile(
        rf"^{re.escape(metric_name)}\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
    )
    values = [float(match.group(1)) for line in output.splitlines()
              if (match := pattern.match(line.strip()))]
    if len(values) != 1:
        raise ValueError(
            f"expected exactly one {metric_name!r} metric, found {len(values)}"
        )
    if not math.isfinite(values[0]) or values[0] <= 0:
        raise ValueError(f"invalid {metric_name!r} metric: {values[0]!r}")
    return values[0]


def parse_timed_window(output: str) -> TimedWindow:
    marker_lines = [
        line.strip()
        for line in output.splitlines()
        if TIMED_WINDOW_MARKER in line
    ]
    if not marker_lines:
        raise ValueError("timed-window marker was not found")
    if len(marker_lines) != 1:
        raise ValueError(
            "expected exactly one timed-window marker, found "
            f"{len(marker_lines)}"
        )
    marker = marker_lines[0]
    match = TIMED_WINDOW_PATTERN.fullmatch(marker)
    if match is None:
        raise ValueError(f"malformed timed-window marker: {marker!r}")
    start_ns, end_ns = (int(value) for value in match.groups())
    if end_ns <= start_ns:
        raise ValueError(
            "timed-window marker is not an ordered positive interval: "
            f"start={start_ns}, end={end_ns}"
        )
    return TimedWindow(
        marker=marker,
        start_monotonic_ns=start_ns,
        end_monotonic_ns=end_ns,
    )


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0 <= probability <= 1:
        raise ValueError("percentile probability must be in [0, 1]")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1 - fraction)
        + sorted_values[upper] * fraction
    )


def bootstrap_median_interval(
    values: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sequence")
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    rng = random.Random(seed)
    count = len(values)
    medians = [
        statistics.median(values[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    medians.sort()
    return percentile(medians, 0.025), percentile(medians, 0.975)


def summarize(
    values: Sequence[float],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, float | int | list[float]]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    mean = statistics.fmean(values)
    if mean == 0:
        raise ValueError("coefficient of variation is undefined for zero mean")
    low, high = bootstrap_median_interval(
        values, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    return {
        "count": len(values),
        "median_us": median,
        "mad_us": mad,
        "mean_us": mean,
        "population_cv_percent": statistics.pstdev(values) / abs(mean) * 100,
        "minimum_us": min(values),
        "maximum_us": max(values),
        "bootstrap95_median_us": [low, high],
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
    }


def assess_batch(
    *,
    returncode: int,
    trace: Sequence[TraceSample],
    verification_mode: str,
    verification_pattern: str | None,
    stdout: str,
    stderr: str,
    maximum_temperature_millidegrees: int,
    minimum_active_samples: int,
    sampling_error: str | None = None,
    thermal_abort: bool = False,
    forbidden_process_match: ForbiddenProcessMatch | None = None,
) -> BatchAssessment:
    reasons: list[str] = []
    if returncode != 0:
        reasons.append(f"container exited with rc={returncode}")
    if sampling_error:
        reasons.append(f"sampling failed: {sampling_error}")
    if thermal_abort:
        reasons.append("thermal limit reached during the container run")
    if forbidden_process_match:
        reasons.append(format_forbidden_process_match(forbidden_process_match))
    if not trace:
        reasons.append("no policy/clock/temperature samples were captured")

    if any(sample.policy != REQUIRED_POLICY for sample in trace):
        reasons.append("performance policy was not high for every sample")

    observed_maximum = max(
        (sample.temperature_millidegrees for sample in trace), default=None
    )
    if (
        observed_maximum is not None
        and observed_maximum >= maximum_temperature_millidegrees
    ):
        reasons.append(
            "temperature reached "
            f"{observed_maximum / 1000:.1f} C (limit "
            f"{maximum_temperature_millidegrees / 1000:.1f} C)"
        )

    combined_output = stdout + "\n" + stderr
    timed_window: TimedWindow | None = None
    try:
        timed_window = parse_timed_window(combined_output)
    except ValueError as error:
        reasons.append(str(error))

    timed_trace: Sequence[TraceSample] = ()
    if timed_window is not None and trace:
        monotonic_ns = [
            round(sample.monotonic_seconds * 1_000_000_000)
            for sample in trace
        ]
        if any(
            current < previous
            for previous, current in zip(monotonic_ns, monotonic_ns[1:])
        ):
            reasons.append("telemetry trace monotonic timestamps are not ordered")
        elif (
            timed_window.start_monotonic_ns < monotonic_ns[0]
            or timed_window.end_monotonic_ns > monotonic_ns[-1]
        ):
            reasons.append(
                "timed-window marker lies outside the telemetry trace: "
                f"window=[{timed_window.start_monotonic_ns}, "
                f"{timed_window.end_monotonic_ns}], "
                f"trace=[{monotonic_ns[0]}, {monotonic_ns[-1]}]"
            )
        else:
            timed_trace = [
                sample
                for sample, timestamp_ns in zip(trace, monotonic_ns)
                if timed_window.start_monotonic_ns
                <= timestamp_ns
                <= timed_window.end_monotonic_ns
            ]

    active_indices = [
        index
        for index, sample in enumerate(timed_trace)
        if sample.selected_clock_mhz == REQUIRED_CLOCK_MHZ
    ]
    active_count = len(active_indices)
    active_interval_count = len(timed_trace)
    if active_count < minimum_active_samples:
        reasons.append(
            f"captured only {active_count} active {REQUIRED_CLOCK_MHZ} MHz "
            f"samples (minimum {minimum_active_samples})"
        )
    if timed_trace:
        unstable = [
            sample.selected_clock_mhz
            for sample in timed_trace
            if sample.selected_clock_mhz != REQUIRED_CLOCK_MHZ
        ]
        if unstable:
            reasons.append(
                "selected clock left 1600 MHz inside the timed window"
            )

    if verification_mode == "marker":
        if not verification_pattern or not re.search(
            verification_pattern, combined_output
        ):
            reasons.append("verification success marker was not found")
    elif verification_mode != "rc-guarded":
        reasons.append(f"unknown verification mode {verification_mode!r}")

    return BatchAssessment(
        accepted=not reasons,
        reasons=tuple(reasons),
        active_samples=active_count,
        active_interval_samples=active_interval_count,
        maximum_temperature_millidegrees=observed_maximum,
        timed_window_marker=(
            None if timed_window is None else timed_window.marker
        ),
        timed_window_start_monotonic_ns=(
            None if timed_window is None else timed_window.start_monotonic_ns
        ),
        timed_window_end_monotonic_ns=(
            None if timed_window is None else timed_window.end_monotonic_ns
        ),
        timed_window_samples=len(timed_trace),
    )


def validate_passthrough_args(arguments: Sequence[str]) -> list[str]:
    args = list(arguments)
    if args and args[0] == "--":
        args = args[1:]
    for argument in args:
        if any(
            argument == reserved or argument.startswith(reserved + "=")
            for reserved in RESERVED_BENCHMARK_ARGS
        ):
            raise ValueError(
                f"benchmark argument {argument!r} overrides an enforced option"
            )
    return args


def passthrough_option(
    arguments: Sequence[str], option: str, default: str
) -> str:
    """Return the last value for a harness option, matching its argv parser."""
    value = default
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value")
            value = arguments[index + 1]
            index += 2
            continue
        if argument.startswith(option + "="):
            raise ValueError(
                f"{option} must use a separate value, matching the harness parser"
            )
        index += 1
    return value


def kmeans_fixture_files(arguments: Sequence[str]) -> tuple[list[Path], list[Path]]:
    """Resolve matched K-means fixtures and the sources that generate them."""
    try:
        points = int(passthrough_option(arguments, "--points", "4096"))
        features = int(passthrough_option(arguments, "--features", "16"))
        clusters = int(passthrough_option(arguments, "--clusters", "5"))
    except ValueError as error:
        raise ValueError(f"invalid K-means geometry: {error}") from error
    if points not in (1024, 2048, 4096, 8192) or features != 16 or clusters != 5:
        raise ValueError(
            "K-means reference acquisition requires --points "
            "{1024,2048,4096,8192} --features 16 --clusters 5"
        )

    fixture_prefix = HERE / "build" / f"kmeans_{points}_"
    fixtures = [
        Path(str(fixture_prefix) + "features.f32"),
        Path(str(fixture_prefix) + "membership.i32"),
    ]
    sources = [
        HERE / "generate_fixtures.go",
        ROOT / "amd/benchmarks/heteromark/kmeans/kmeans.go",
    ]
    return fixtures, sources


def build_container_command(
    *,
    podman: str,
    image: str,
    container_name: str,
    binary: Path,
    benchmark: str,
    warmup: int,
    iterations: int,
    benchmark_args: Sequence[str],
) -> list[str]:
    return [
        podman,
        "run",
        "--rm",
        "--name",
        container_name,
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--group-add=video",
        "--group-add=render",
        "--security-opt=label=disable",
        "-v",
        f"{ROOT}:{ROOT}:z",
        "-w",
        str(HERE),
        image,
        "env",
        f"MGPUSIM_ROOT={ROOT}",
        str(binary),
        "--only",
        benchmark,
        "--warmup",
        str(warmup),
        "--iters",
        str(iterations),
        *benchmark_args,
    ]


def discover_device_sysfs(explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    candidates = sorted(
        path.parent
        for path in Path("/sys/class/drm").glob(
            "card*/device/power_dpm_force_performance_level"
        )
        if (path.parent / "pp_dpm_sclk").is_file()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "could not uniquely identify the GPU sysfs directory; use "
            "--device-sysfs"
        )
    return candidates[0].resolve()


def discover_temperature_file(device_sysfs: Path, explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    candidates = sorted(device_sysfs.glob("hwmon/hwmon*/temp1_input"))
    if len(candidates) != 1:
        raise RuntimeError(
            "could not uniquely identify the GPU edge-temperature file; use "
            "--temperature-file"
        )
    return candidates[0].resolve()


def read_trace_sample(
    policy_file: Path, clock_file: Path, temperature_file: Path
) -> TraceSample:
    policy = policy_file.read_text().strip()
    raw_sclk = clock_file.read_text().strip()
    temperature = int(temperature_file.read_text().strip())
    return TraceSample(
        monotonic_seconds=time.monotonic(),
        epoch_ns=time.time_ns(),
        policy=policy,
        selected_clock_mhz=parse_selected_clock_mhz(raw_sclk),
        temperature_millidegrees=temperature,
        raw_sclk=" ".join(raw_sclk.split()),
    )


class TelemetryMonitor:
    """Sample GPU state on cadence independently of slower control checks."""

    def __init__(
        self,
        *,
        sample_reader: Callable[[], TraceSample],
        sample_sink: Callable[[TraceSample], None],
        should_continue: Callable[[], bool],
        interval_seconds: float,
        maximum_temperature_millidegrees: int,
        abort_callback: Callable[[], None],
        monotonic: Callable[[], float] = time.monotonic,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._sample_reader = sample_reader
        self._sample_sink = sample_sink
        self._should_continue = should_continue
        self._interval_seconds = interval_seconds
        self._maximum_temperature_millidegrees = (
            maximum_temperature_millidegrees
        )
        self._abort_callback = abort_callback
        self._monotonic = monotonic
        self._stop_event = stop_event or threading.Event()
        self._thread: threading.Thread | None = None
        self.sampling_error: str | None = None
        self.thermal_abort = False

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry monitor can only be started once")
        self._thread = threading.Thread(
            target=self._run,
            name="gfx90c-telemetry",
            daemon=False,
        )
        self._thread.start()

    def stop_and_join(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def _abort(self) -> None:
        try:
            self._abort_callback()
        except Exception as error:
            # This runs on a worker thread, so retain every abort failure in
            # the batch assessment instead of losing it to a thread traceback.
            suffix = f"container abort failed: {error}"
            self.sampling_error = (
                suffix
                if self.sampling_error is None
                else f"{self.sampling_error}; {suffix}"
            )

    def _run(self) -> None:
        next_sample_time = self._monotonic()
        while not self._stop_event.is_set() and self._should_continue():
            delay = max(0.0, next_sample_time - self._monotonic())
            if self._stop_event.wait(delay):
                return
            if not self._should_continue():
                return
            try:
                sample = self._sample_reader()
                self._sample_sink(sample)
            except Exception as error:
                # Sampling and trace-writing failures must reject the batch.
                self.sampling_error = f"{type(error).__name__}: {error}"
                self._abort()
                return
            if sample.policy != REQUIRED_POLICY:
                self.sampling_error = "performance policy left high"
                self._abort()
                return
            if (
                sample.temperature_millidegrees
                >= self._maximum_temperature_millidegrees
            ):
                self.thermal_abort = True
                self._abort()
                return
            next_sample_time += self._interval_seconds
            now = self._monotonic()
            if next_sample_time < now:
                next_sample_time = now


def stop_container(podman: str, container_name: str, process: subprocess.Popen) -> None:
    subprocess.run(
        [podman, "stop", "--time", "1", container_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def wait_until_ready(
    *,
    policy_file: Path,
    temperature_file: Path,
    start_temperature_millidegrees: int,
    deadline: float,
    next_eligible_time: float,
    process_guard: ForbiddenProcessGuard | None = None,
) -> None:
    while True:
        if process_guard:
            process_guard.require_clear()
        policy = policy_file.read_text().strip()
        if policy != REQUIRED_POLICY:
            raise RuntimeError(
                f"GPU policy is {policy!r}, expected {REQUIRED_POLICY!r}"
            )
        now = time.monotonic()
        temperature = int(temperature_file.read_text().strip())
        if now >= next_eligible_time and temperature <= start_temperature_millidegrees:
            return
        if now >= deadline:
            raise RuntimeError(
                "timed out waiting for cooldown/start temperature; last "
                f"temperature was {temperature / 1000:.1f} C"
            )
        time.sleep(min(1.0, max(0.05, next_eligible_time - now)))


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return result.stdout.strip()


def make_output_directory(benchmark: str, explicit: Path | None) -> Path:
    if explicit:
        explicit.mkdir(parents=True, exist_ok=False)
        return explicit.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(
        tempfile.mkdtemp(prefix=f"gfx90c-pinned-{benchmark}-{timestamp}-", dir="/tmp")
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--cooldown-seconds", type=float, default=30.0)
    parser.add_argument("--max-start-temp-c", type=float, default=50.0)
    parser.add_argument("--max-temp-c", type=float, default=58.0)
    parser.add_argument("--ready-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.02)
    parser.add_argument("--min-active-samples", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--metric-name")
    parser.add_argument("--verification-regex")
    parser.add_argument(
        "--forbid-process-regex",
        action="append",
        default=[],
        metavar="REGEX",
        help=(
            "reject the point if a non-ancestor process command line matches; "
            "repeat for multiple patterns"
        ),
    )
    parser.add_argument(
        "--rc-guarded-verification",
        action="store_true",
        help="assert that this harness path exits nonzero on every mismatch",
    )
    parser.add_argument("--image", default=os.environ.get("ROCM_IMAGE", DEFAULT_IMAGE))
    parser.add_argument("--podman", default="podman")
    parser.add_argument("--binary", type=Path, default=HERE / "build/isca10_bench")
    parser.add_argument("--device-sysfs", type=Path)
    parser.add_argument("--temperature-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("benchmark", choices=sorted(BENCHMARK_HSACO))
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.batches < 1 or args.warmup < 0 or args.iters < 1:
        parser.error("batches/iters must be positive and warmup must be nonnegative")
    if args.cooldown_seconds < 30:
        parser.error("cooldown must be at least 30 seconds")
    if not 0 < args.max_start_temp_c < args.max_temp_c:
        parser.error("start temperature must be positive and below max temperature")
    if args.ready_timeout_seconds <= 0 or args.sample_interval_seconds <= 0:
        parser.error("timeouts and sample intervals must be positive")
    if args.min_active_samples < 1 or args.bootstrap_resamples < 1:
        parser.error("active samples and bootstrap resamples must be positive")
    try:
        args.benchmark_args = validate_passthrough_args(args.benchmark_args)
        for pattern in args.forbid_process_regex:
            re.compile(pattern)
    except ValueError as error:
        parser.error(str(error))
    except re.error as error:
        parser.error(f"invalid --forbid-process-regex: {error}")
    return args


def resolve_verification(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.verification_regex and args.rc_guarded_verification:
        raise ValueError("choose either verification regex or rc-guarded verification")
    if args.verification_regex:
        re.compile(args.verification_regex)
        return "marker", args.verification_regex
    if args.rc_guarded_verification:
        return "rc-guarded", None
    if args.benchmark not in AUTO_VERIFICATION:
        raise ValueError(
            f"{args.benchmark!r} has no known output verifier; provide "
            "--verification-regex after adding a marker, or explicitly assert "
            "--rc-guarded-verification only if the harness checks all output"
        )
    return AUTO_VERIFICATION[args.benchmark]


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    args = parse_args(argv)
    try:
        verification_mode, verification_pattern = resolve_verification(args)
        process_guard = ForbiddenProcessGuard(args.forbid_process_regex)
        device_sysfs = discover_device_sysfs(args.device_sysfs)
        temperature_file = discover_temperature_file(
            device_sysfs, args.temperature_file
        )
    except (OSError, RuntimeError, ValueError, re.error) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    policy_file = device_sysfs / "power_dpm_force_performance_level"
    clock_file = device_sysfs / "pp_dpm_sclk"
    binary = args.binary.resolve()
    source = HERE / "isca10_bench.cpp"
    manifest_path = HERE / "hsaco_manifest.txt"
    hsaco = ROOT / BENCHMARK_HSACO[args.benchmark]
    fixture_files: list[Path] = []
    fixture_sources: list[Path] = []
    if args.benchmark == "kmeans":
        try:
            fixture_files, fixture_sources = kmeans_fixture_files(
                args.benchmark_args
            )
        except ValueError as error:
            print(f"configuration error: {error}", file=sys.stderr)
            return 2
    required_files = [
        binary,
        source,
        manifest_path,
        hsaco,
        *fixture_sources,
        *fixture_files,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        print("missing required files: " + ", ".join(missing), file=sys.stderr)
        return 2
    if binary.stat().st_mtime_ns < max(source.stat().st_mtime_ns, hsaco.stat().st_mtime_ns):
        print("exact-HSACO harness binary is older than its source/code object", file=sys.stderr)
        return 2
    if fixture_files:
        newest_generator = max(path.stat().st_mtime_ns for path in fixture_sources)
        stale_fixtures = [
            str(path)
            for path in fixture_files
            if path.stat().st_mtime_ns < newest_generator
        ]
        if stale_fixtures:
            print(
                "K-means fixtures are older than their generator source: "
                + ", ".join(stale_fixtures),
                file=sys.stderr,
            )
            return 2

    output_dir = make_output_directory(args.benchmark, args.output_dir)
    metric_name = args.metric_name or args.benchmark
    start_limit = round(args.max_start_temp_c * 1000)
    thermal_limit = round(args.max_temp_c * 1000)
    run_identifier = f"{int(time.time())}-{os.getpid()}"
    command_template = build_container_command(
        podman=args.podman,
        image=args.image,
        container_name="<per-batch>",
        binary=binary,
        benchmark=args.benchmark,
        warmup=args.warmup,
        iterations=args.iters,
        benchmark_args=args.benchmark_args,
    )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_porcelain": git_output("status", "--porcelain"),
        "benchmark": args.benchmark,
        "benchmark_args": args.benchmark_args,
        "metric_name": metric_name,
        "batches": args.batches,
        "warmup": args.warmup,
        "iterations": args.iters,
        "cooldown_seconds": args.cooldown_seconds,
        "maximum_start_temperature_c": args.max_start_temp_c,
        "maximum_temperature_c": args.max_temp_c,
        "sample_interval_seconds": args.sample_interval_seconds,
        "minimum_active_samples": args.min_active_samples,
        "required_policy": REQUIRED_POLICY,
        "required_active_clock_mhz": REQUIRED_CLOCK_MHZ,
        "timed_window_marker": TIMED_WINDOW_MARKER,
        "verification_mode": verification_mode,
        "verification_pattern": verification_pattern,
        "forbidden_process_patterns": list(process_guard.patterns),
        "image": args.image,
        "device_sysfs": str(device_sysfs),
        "temperature_file": str(temperature_file),
        "command_template": command_template,
        "hashes": {
            str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
            sha256_file(path)
            for path in required_files
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    result_rows: list[dict[str, object]] = []
    accepted_values: list[float] = []
    last_end = time.monotonic() - args.cooldown_seconds
    overall_status = "complete"
    fatal_reason: str | None = None
    forbidden_process_matches: list[dict[str, object]] = []

    for batch in range(1, args.batches + 1):
        prefix = output_dir / f"batch_{batch:03d}"
        metadata_path = prefix.with_suffix(".json")
        try:
            wait_until_ready(
                policy_file=policy_file,
                temperature_file=temperature_file,
                start_temperature_millidegrees=start_limit,
                deadline=time.monotonic() + args.ready_timeout_seconds,
                next_eligible_time=last_end + args.cooldown_seconds,
                process_guard=process_guard,
            )
            pre_sample = read_trace_sample(
                policy_file, clock_file, temperature_file
            )
            if pre_sample.policy != REQUIRED_POLICY:
                raise RuntimeError("performance policy changed before launch")
            if pre_sample.temperature_millidegrees > start_limit:
                raise RuntimeError(
                    "temperature rose above the start threshold before launch"
                )
            process_guard.require_clear()
        except KeyboardInterrupt:
            overall_status = "interrupted"
            fatal_reason = f"operator interrupted during batch {batch} readiness"
            metadata_path.write_text(
                json.dumps(
                    {
                        "batch": batch,
                        "accepted": False,
                        "phase": "readiness",
                        "status": "interrupted",
                        "reasons": [fatal_reason],
                        "forbidden_process_patterns": list(process_guard.patterns),
                        "forbidden_process_match": None,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            break
        except (OSError, RuntimeError, ValueError) as error:
            print(f"batch {batch}: readiness failed: {error}", file=sys.stderr)
            overall_status = "failed"
            fatal_reason = f"batch {batch} readiness failed: {error}"
            forbidden_match = (
                error.match if isinstance(error, ForbiddenProcessError) else None
            )
            if forbidden_match:
                forbidden_process_matches.append(asdict(forbidden_match))
            metadata_path.write_text(
                json.dumps(
                    {
                        "batch": batch,
                        "accepted": False,
                        "phase": "readiness",
                        "status": "failed",
                        "reasons": [fatal_reason],
                        "forbidden_process_patterns": list(process_guard.patterns),
                        "forbidden_process_match": (
                            None
                            if forbidden_match is None
                            else asdict(forbidden_match)
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            break

        stdout_path = prefix.with_suffix(".stdout.txt")
        stderr_path = prefix.with_suffix(".stderr.txt")
        trace_path = prefix.with_suffix(".trace.tsv")
        container_name = f"gfx90c-pinned-{run_identifier}-{batch:03d}"
        command = build_container_command(
            podman=args.podman,
            image=args.image,
            container_name=container_name,
            binary=binary,
            benchmark=args.benchmark,
            warmup=args.warmup,
            iterations=args.iters,
            benchmark_args=args.benchmark_args,
        )

        trace: list[TraceSample] = []
        sampling_error: str | None = None
        thermal_abort = False
        forbidden_process_match: ForbiddenProcessMatch | None = None
        operator_interrupted = False
        started_utc = datetime.now(timezone.utc).isoformat()
        with (
            stdout_path.open("wb") as stdout_file,
            stderr_path.open("wb") as stderr_file,
            trace_path.open("w", newline="") as trace_file,
        ):
            trace_writer = csv.writer(trace_file, delimiter="\t")
            trace_writer.writerow(
                [
                    "epoch_ns",
                    "monotonic_seconds",
                    "policy",
                    "selected_clock_mhz",
                    "temperature_millidegrees",
                    "raw_sclk",
                ]
            )
            trace_file.flush()
            process: subprocess.Popen | None = None
            telemetry_monitor: TelemetryMonitor | None = None
            try:
                process = subprocess.Popen(
                    command, stdout=stdout_file, stderr=stderr_file
                )
                if process is not None:
                    stop_lock = threading.Lock()
                    container_stopped = False

                    def stop_container_once() -> None:
                        nonlocal container_stopped
                        with stop_lock:
                            if not container_stopped and process.poll() is None:
                                container_stopped = True
                                stop_container(
                                    args.podman, container_name, process
                                )

                    def record_sample(sample: TraceSample) -> None:
                        trace.append(sample)
                        trace_writer.writerow(
                            [
                                sample.epoch_ns,
                                f"{sample.monotonic_seconds:.9f}",
                                sample.policy,
                                (
                                    ""
                                    if sample.selected_clock_mhz is None
                                    else sample.selected_clock_mhz
                                ),
                                sample.temperature_millidegrees,
                                sample.raw_sclk,
                            ]
                        )
                        trace_file.flush()

                    telemetry_monitor = TelemetryMonitor(
                        sample_reader=lambda: read_trace_sample(
                            policy_file, clock_file, temperature_file
                        ),
                        sample_sink=record_sample,
                        should_continue=lambda: process.poll() is None,
                        interval_seconds=args.sample_interval_seconds,
                        maximum_temperature_millidegrees=thermal_limit,
                        abort_callback=stop_container_once,
                    )
                    telemetry_monitor.start()
                    try:
                        if process_guard.patterns:
                            while process.poll() is None:
                                if match := process_guard.check():
                                    forbidden_process_match = match
                                    stop_container_once()
                                    break
                        else:
                            # Avoid a CPU-intensive poll loop when the optional
                            # guard is disabled; telemetry remains independent.
                            process.wait()
                    finally:
                        telemetry_monitor.stop_and_join()
                    sampling_error = telemetry_monitor.sampling_error
                    thermal_abort = telemetry_monitor.thermal_abort
                    returncode = process.wait()
                    try:
                        # Bound the recorded trace after the harness marker even
                        # when the process exits between periodic samples.
                        record_sample(
                            read_trace_sample(
                                policy_file, clock_file, temperature_file
                            )
                        )
                    except (OSError, ValueError) as error:
                        sampling_error = sampling_error or (
                            f"terminal in-run sample: {error}"
                        )
                    # A final scan closes the interval after the last in-run
                    # scan, including short runs that finish during that scan.
                    if match := process_guard.check():
                        forbidden_process_match = (
                            forbidden_process_match or match
                        )
            except OSError as error:
                returncode = 127
                sampling_error = f"container launch failed: {error}"
            except KeyboardInterrupt:
                operator_interrupted = True
                sampling_error = "operator interrupted the container run"
                if telemetry_monitor is not None:
                    telemetry_monitor.stop_and_join()
                if process is not None and process.poll() is None:
                    stop_container(args.podman, container_name, process)
                returncode = (
                    130
                    if process is None or process.returncode is None
                    else process.returncode
                )
        last_end = time.monotonic()
        try:
            post_sample = read_trace_sample(
                policy_file, clock_file, temperature_file
            )
        except (OSError, ValueError) as error:
            post_sample = None
            sampling_error = sampling_error or f"post-run sample: {error}"

        stdout = stdout_path.read_text(errors="replace")
        stderr = stderr_path.read_text(errors="replace")
        metric: float | None = None
        metric_error: str | None = None
        try:
            metric = parse_metric(stdout, metric_name)
        except ValueError as error:
            metric_error = str(error)

        assessment = assess_batch(
            returncode=returncode,
            trace=trace,
            verification_mode=verification_mode,
            verification_pattern=verification_pattern,
            stdout=stdout,
            stderr=stderr,
            maximum_temperature_millidegrees=thermal_limit,
            minimum_active_samples=args.min_active_samples,
            sampling_error=sampling_error,
            thermal_abort=thermal_abort,
            forbidden_process_match=forbidden_process_match,
        )
        reasons = list(assessment.reasons)
        if metric_error:
            reasons.append(metric_error)
        accepted = not reasons
        if accepted and metric is not None:
            accepted_values.append(metric)

        metadata = {
            "batch": batch,
            "container_name": container_name,
            "command": command,
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": returncode,
            "metric_us": metric,
            "accepted": accepted,
            "reasons": reasons,
            "operator_interrupted": operator_interrupted,
            "forbidden_process_patterns": list(process_guard.patterns),
            "forbidden_process_match": (
                None
                if forbidden_process_match is None
                else asdict(forbidden_process_match)
            ),
            "assessment": asdict(assessment),
            "timed_window": (
                None
                if assessment.timed_window_marker is None
                else {
                    "marker": assessment.timed_window_marker,
                    "start_monotonic_ns": (
                        assessment.timed_window_start_monotonic_ns
                    ),
                    "end_monotonic_ns": (
                        assessment.timed_window_end_monotonic_ns
                    ),
                    "duration_ns": (
                        assessment.timed_window_end_monotonic_ns
                        - assessment.timed_window_start_monotonic_ns
                    ),
                    "trace_samples": assessment.timed_window_samples,
                }
            ),
            "pre_sample": asdict(pre_sample),
            "post_sample": None if post_sample is None else asdict(post_sample),
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
            "trace_sha256": sha256_file(trace_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        result_rows.append(
            {
                "batch": batch,
                "accepted": accepted,
                "metric_us": "" if metric is None else metric,
                "returncode": returncode,
                "active_samples": assessment.active_samples,
                "active_interval_samples": assessment.active_interval_samples,
                "maximum_temperature_millidegrees": (
                    "" if assessment.maximum_temperature_millidegrees is None
                    else assessment.maximum_temperature_millidegrees
                ),
                "reasons": "; ".join(reasons),
            }
        )
        print(
            f"batch {batch}/{args.batches}: "
            f"{'accepted' if accepted else 'REJECTED'} metric={metric!r} "
            f"active_samples={assessment.active_samples} "
            f"max_temp_mC={assessment.maximum_temperature_millidegrees}"
        )
        if forbidden_process_match:
            forbidden_process_matches.append(asdict(forbidden_process_match))
        if not accepted:
            overall_status = "interrupted" if operator_interrupted else "failed"
            fatal_reason = f"batch {batch} rejected: {'; '.join(reasons)}"
            break

    with (output_dir / "results.tsv").open("w", newline="") as file:
        fieldnames = [
            "batch",
            "accepted",
            "metric_us",
            "returncode",
            "active_samples",
            "active_interval_samples",
            "maximum_temperature_millidegrees",
            "reasons",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(result_rows)

    summary: dict[str, object] = {
        "status": overall_status,
        "requested_batches": args.batches,
        "accepted_batches": len(accepted_values),
        "artifact_directory": str(output_dir),
        "failure_reason": fatal_reason,
        "forbidden_process_patterns": list(process_guard.patterns),
        "forbidden_process_matches": forbidden_process_matches,
    }
    if len(accepted_values) == args.batches:
        summary["statistics"] = summarize(
            accepted_values,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if overall_status == "complete" else 1


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "status": "interrupted",
                    "failure_reason": "operator interrupted outside a batch",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        exit_code = 130
    raise SystemExit(exit_code)
