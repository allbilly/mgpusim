#!/usr/bin/env python3
"""Run matched RX 570 hardware or simulator size sweeps.

The hardware mode uses the exact gfx803 HSACO harness. The simulator mode
uses the same benchmark packages, launch shapes, and size parameter. Results
are CSV so compare_size_sweeps.py can evaluate holdout points and slopes.
"""

import argparse
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

SPECS = {
    "vectoradd": ([4096, 16384, 65536, 262144], lambda n: n),
    "relu": ([4096, 16384, 65536, 131072, 262144], lambda n: n),
    "matrixmult": ([64, 96, 128, 160], lambda n: n**3),
    "matrixtranspose": (
        [128, 256, 320, 384, 448, 512, 576, 640], lambda n: n**2
    ),
    "bitonicsort": (
        [1024, 2048, 4096, 8192],
        lambda n: n * math.log2(n) * (math.log2(n) + 1) / 2,
    ),
    "aes": ([1024, 2048, 4096, 8192], lambda n: n),
    "fir": ([2048, 4096, 8192, 16384], lambda n: n),
    "nw": ([64, 128, 192, 256], lambda n: n**2),
}

MULTIPLES = {
    "vectoradd": 64,
    "relu": 64,
    "matrixmult": 32,
    "matrixtranspose": 64,
    "bitonicsort": 64,
    "aes": 16,
    "fir": 256,
    "nw": 64,
}


def parse_benchmarks(value):
    if not value:
        return list(SPECS)
    names = []
    for item in value:
        names.extend(part for part in item.split(",") if part)
    unknown = sorted(set(names) - set(SPECS))
    if unknown:
        raise SystemExit(f"unknown benchmark(s): {', '.join(unknown)}")
    return list(dict.fromkeys(names))


def validate_size(benchmark, size):
    multiple = MULTIPLES[benchmark]
    if size <= 0 or size % multiple:
        raise ValueError(
            f"{benchmark} size must be a positive multiple of {multiple}: {size}"
        )
    if benchmark == "bitonicsort" and size & (size - 1):
        raise ValueError(f"bitonicsort size must be a power of two: {size}")


def extract_time(output, benchmark):
    pattern = re.compile(rf"^{re.escape(benchmark)}\s+([0-9]+(?:\.[0-9]+)?)$", re.M)
    match = pattern.search(output)
    if not match:
        raise RuntimeError(f"no timing row for {benchmark} in output:\n{output[-2000:]}")
    return float(match.group(1))


def run_checked(command, env, timeout):
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )
    return result.stdout


def run_sim_point(benchmark, size, timeout):
    with tempfile.TemporaryDirectory(prefix=f"rx570-{benchmark}-{size}-") as out:
        env = os.environ.copy()
        env.update({
            "ONLY": benchmark,
            "SWEEP_SIZE": str(size),
            "SIM_JOBS": "1",
        })
        output = run_checked(
            ["bash", str(HERE / "run_sim.sh"), out], env, timeout
        )
    return extract_time(output, benchmark)


def run_sim(points, jobs, timeout):
    rows = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run_sim_point, benchmark, size, timeout): (benchmark, size)
            for benchmark, size in points
        }
        for future in as_completed(futures):
            benchmark, size = futures[future]
            try:
                time_us = future.result()
                status = "ok"
                print(
                    f"sim {benchmark} size={size}: {time_us:.3f} us",
                    file=sys.stderr,
                )
            except Exception as error:  # Preserve all other sweep evidence.
                time_us = None
                status = "failed"
                print(
                    f"sim {benchmark} size={size}: FAILED\n{error}",
                    file=sys.stderr,
                )
            rows.append((benchmark, size, time_us, status))
    return rows


def run_hardware(points, warmup, warmup_ms, iters, trials, timeout):
    if not os.access("/dev/kfd", os.R_OK | os.W_OK):
        raise SystemExit(
            "hardware mode requires read/write access to /dev/kfd; "
            "add this user to the render group and start a new login session"
        )
    rows = []
    invocation = 0
    for benchmark, size in points:
        samples = []
        failed = False
        for trial in range(1, trials + 1):
            env = os.environ.copy()
            if invocation == 0:
                env.pop("RX570_REUSE_BUILD", None)
            else:
                env["RX570_REUSE_BUILD"] = "1"
            invocation += 1
            try:
                output = run_checked(
                    [
                        "bash", str(HERE / "build_and_run.sh"),
                        "--only", benchmark,
                        "--size", str(size),
                        "--warmup", str(warmup),
                        "--warmup-us", str(warmup_ms * 1000),
                        "--iters", str(iters),
                    ],
                    env,
                    timeout,
                )
                sample_us = extract_time(output, benchmark)
                samples.append(sample_us)
                print(
                    f"hardware {benchmark} size={size} trial={trial}: "
                    f"{sample_us:.3f} us",
                    file=sys.stderr,
                )
            except Exception as error:  # Keep other hardware points usable.
                failed = True
                print(
                    f"hardware {benchmark} size={size} trial={trial}: "
                    f"FAILED\n{error}",
                    file=sys.stderr,
                )
        time_us = statistics.median(samples) if samples else None
        status = "failed" if failed else "ok"
        if time_us is not None:
            print(
                f"hardware {benchmark} size={size} median: {time_us:.3f} us",
                file=sys.stderr,
            )
        rows.append((benchmark, size, time_us, status))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sim", "hardware"), required=True)
    parser.add_argument(
        "--benchmark", "-b", action="append",
        help="benchmark name or comma-separated names; default: all",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        help="override sizes; requires exactly one benchmark",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--warmup-ms", type=int, default=50,
        help="minimum GPU-active warmup duration per hardware process",
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--trials", type=int, default=3,
        help="independent hardware processes per point; output is the median",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path("-"))
    args = parser.parse_args()

    benchmarks = parse_benchmarks(args.benchmark)
    if args.sizes and len(benchmarks) != 1:
        parser.error("--sizes requires exactly one --benchmark")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
    if args.warmup_ms < 0:
        parser.error("--warmup-ms must not be negative")
    if args.iters < 1:
        parser.error("--iters must be positive")
    if args.trials < 1:
        parser.error("--trials must be positive")

    points = []
    for benchmark in benchmarks:
        sizes = args.sizes or SPECS[benchmark][0]
        for size in sizes:
            try:
                validate_size(benchmark, size)
            except ValueError as error:
                parser.error(str(error))
            points.append((benchmark, size))

    if args.mode == "sim":
        measured = run_sim(points, args.jobs, args.timeout)
    else:
        if args.jobs != 1:
            print("hardware mode is serialized; ignoring --jobs", file=sys.stderr)
        measured = run_hardware(
            points, args.warmup, args.warmup_ms, args.iters, args.trials,
            args.timeout
        )

    order = {(benchmark, size): i for i, (benchmark, size) in enumerate(points)}
    measured.sort(key=lambda row: order[(row[0], row[1])])
    output = sys.stdout if args.output == Path("-") else args.output.open("w", newline="")
    try:
        writer = csv.DictWriter(
            output,
            fieldnames=("benchmark", "size", "work", "time_us", "status"),
        )
        writer.writeheader()
        for benchmark, size, time_us, status in measured:
            work = SPECS[benchmark][1](size)
            writer.writerow({
                "benchmark": benchmark,
                "size": size,
                "work": f"{work:.6g}",
                "time_us": "" if time_us is None else f"{time_us:.3f}",
                "status": status,
            })
    finally:
        if output is not sys.stdout:
            output.close()

    if any(status != "ok" for _, _, _, status in measured):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
