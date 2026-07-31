#!/usr/bin/env python3
"""Compare matched RX 570 hardware and simulator size-sweep CSV files."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load(path):
    rows = {}
    failed = []
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (row["benchmark"], int(row["size"]))
            if key in rows or key in failed:
                raise SystemExit(
                    f"duplicate row in {path}: {key[0]} size={key[1]}"
                )
            if row.get("status", "ok") != "ok" or not row.get("time_us"):
                failed.append(key)
                continue
            rows[key] = (float(row["work"]), float(row["time_us"]))
    return rows, failed


def summarize(keys):
    grouped = defaultdict(int)
    for benchmark, _ in keys:
        grouped[benchmark] += 1
    return ", ".join(f"{name}({count})" for name, count in sorted(grouped.items()))


def slope(points):
    x_mean = sum(x for x, _ in points) / len(points)
    y_mean = sum(y for _, y in points) / len(points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hardware", type=Path)
    parser.add_argument("sim", type=Path)
    args = parser.parse_args()

    hardware, hardware_failed = load(args.hardware)
    simulator, simulator_failed = load(args.sim)
    shared = sorted(set(hardware) & set(simulator))
    if not shared:
        raise SystemExit("no shared benchmark/size rows")

    hardware_only = set(hardware) - set(simulator)
    simulator_only = set(simulator) - set(hardware)
    if hardware_only:
        print(f"NOTE: hardware-only points: {summarize(hardware_only)}")
    if simulator_only:
        print(f"NOTE: simulator points awaiting hardware: {summarize(simulator_only)}")

    for label, failed in (
        ("hardware", hardware_failed), ("sim", simulator_failed)
    ):
        for benchmark, size in failed:
            print(f"WARNING: {label} point failed: {benchmark} size={size}")

    by_benchmark = defaultdict(list)
    errors = []
    print(f"{'Benchmark':<18} {'Size':>9} {'HW us':>10} {'Sim us':>10} {'Err%':>8}")
    for benchmark, size in shared:
        hw_work, hw_us = hardware[(benchmark, size)]
        sim_work, sim_us = simulator[(benchmark, size)]
        if hw_work != sim_work:
            raise SystemExit(f"work mismatch for {benchmark} size={size}")
        error = abs(sim_us - hw_us) / hw_us * 100
        errors.append(error)
        by_benchmark[benchmark].append((hw_work, hw_us, sim_us))
        print(f"{benchmark:<18} {size:>9} {hw_us:>10.3f} {sim_us:>10.3f} {error:>8.1f}")

    print()
    print(f"{'Benchmark':<18} {'HW slope':>12} {'Sim slope':>12} {'Sim/HW':>9}")
    for benchmark, rows in sorted(by_benchmark.items()):
        if len(rows) < 2:
            continue
        hw_slope = slope([(work, hw_us) for work, hw_us, _ in rows])
        sim_slope = slope([(work, sim_us) for work, _, sim_us in rows])
        ratio = sim_slope / hw_slope if hw_slope else float("nan")
        print(f"{benchmark:<18} {hw_slope:>12.6g} {sim_slope:>12.6g} {ratio:>9.3f}")

    print()
    print(f"points={len(errors)} MARE={sum(errors)/len(errors):.1f}% max={max(errors):.1f}%")


if __name__ == "__main__":
    main()
