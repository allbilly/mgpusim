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
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_IMAGE = "docker.io/rocm/dev-ubuntu-24.04:7.1.1"
REQUIRED_POLICY = "high"
REQUIRED_CLOCK_MHZ = 1600

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
) -> BatchAssessment:
    reasons: list[str] = []
    if returncode != 0:
        reasons.append(f"container exited with rc={returncode}")
    if sampling_error:
        reasons.append(f"sampling failed: {sampling_error}")
    if thermal_abort:
        reasons.append("thermal limit reached during the container run")
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

    active_indices = [
        index
        for index, sample in enumerate(trace)
        if sample.selected_clock_mhz == REQUIRED_CLOCK_MHZ
    ]
    active_count = len(active_indices)
    active_interval_count = 0
    if active_count < minimum_active_samples:
        reasons.append(
            f"captured only {active_count} active {REQUIRED_CLOCK_MHZ} MHz "
            f"samples (minimum {minimum_active_samples})"
        )
    if active_indices:
        first, last = active_indices[0], active_indices[-1]
        active_interval = trace[first : last + 1]
        active_interval_count = len(active_interval)
        unstable = [
            sample.selected_clock_mhz
            for sample in active_interval
            if sample.selected_clock_mhz != REQUIRED_CLOCK_MHZ
        ]
        if unstable:
            reasons.append(
                "selected clock left 1600 MHz inside the observed active interval"
            )

    combined_output = stdout + "\n" + stderr
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
) -> None:
    while True:
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
    except ValueError as error:
        parser.error(str(error))
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
    required_files = [binary, source, manifest_path, hsaco]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        print("missing required files: " + ", ".join(missing), file=sys.stderr)
        return 2
    if binary.stat().st_mtime_ns < max(source.stat().st_mtime_ns, hsaco.stat().st_mtime_ns):
        print("exact-HSACO harness binary is older than its source/code object", file=sys.stderr)
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
        "verification_mode": verification_mode,
        "verification_pattern": verification_pattern,
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

    for batch in range(1, args.batches + 1):
        try:
            wait_until_ready(
                policy_file=policy_file,
                temperature_file=temperature_file,
                start_temperature_millidegrees=start_limit,
                deadline=time.monotonic() + args.ready_timeout_seconds,
                next_eligible_time=last_end + args.cooldown_seconds,
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
        except (OSError, RuntimeError, ValueError) as error:
            print(f"batch {batch}: readiness failed: {error}", file=sys.stderr)
            overall_status = "failed"
            fatal_reason = f"batch {batch} readiness failed: {error}"
            break

        prefix = output_dir / f"batch_{batch:03d}"
        stdout_path = prefix.with_suffix(".stdout.txt")
        stderr_path = prefix.with_suffix(".stderr.txt")
        trace_path = prefix.with_suffix(".trace.tsv")
        metadata_path = prefix.with_suffix(".json")
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
            try:
                process = subprocess.Popen(
                    command, stdout=stdout_file, stderr=stderr_file
                )
            except OSError as error:
                process = None
                returncode = 127
                sampling_error = f"container launch failed: {error}"
            if process is not None:
                try:
                    while process.poll() is None:
                        try:
                            sample = read_trace_sample(
                                policy_file, clock_file, temperature_file
                            )
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
                            if sample.policy != REQUIRED_POLICY:
                                sampling_error = "performance policy left high"
                                stop_container(
                                    args.podman, container_name, process
                                )
                                break
                            if sample.temperature_millidegrees >= thermal_limit:
                                thermal_abort = True
                                stop_container(
                                    args.podman, container_name, process
                                )
                                break
                        except (OSError, ValueError) as error:
                            sampling_error = str(error)
                            stop_container(args.podman, container_name, process)
                            break
                        time.sleep(args.sample_interval_seconds)
                except KeyboardInterrupt:
                    stop_container(args.podman, container_name, process)
                    raise
                returncode = process.wait()
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
            "assessment": asdict(assessment),
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
        if not accepted:
            overall_status = "failed"
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
    raise SystemExit(main())
