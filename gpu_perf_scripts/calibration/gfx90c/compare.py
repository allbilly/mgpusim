#!/usr/bin/env python3
"""Print sim vs HW table for gfx90c ISCA-10 calibration."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The checked-in matrix-multiplication hardware result predates the corrected
# global-row indexing and regenerated HSACO. Keep it visible for provenance,
# but do not include it in scored calibration aggregates.
UNSCORED = {
    "matrixmult": "stale HW target (predates corrected kernel)",
}


def load_pairs(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = float(parts[1])
    return out


def main():
    hw_path = HERE / "hw_ground_truth.txt"
    sim_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "sim_ground_truth.txt"
    hw = load_pairs(hw_path)
    sim = load_pairs(sim_path)
    names = list(hw.keys())
    print(
        f"{'Benchmark':<18} {'Sim(µs)':>10} {'HW(µs)':>10} "
        f"{'HW/Sim':>8} {'err%':>8}  Status"
    )
    ratios = []
    absolute_errors = []
    for k in names:
        if k not in sim:
            print(f"{k:<18} {'—':>10} {hw[k]:10.1f} {'—':>8} {'—':>8}")
            continue
        r = hw[k] / sim[k]
        err = (r - 1.0) * 100
        if k in UNSCORED:
            status = f"not scored: {UNSCORED[k]}"
        else:
            ratios.append(r)
            absolute_errors.append(abs(err))
            status = "pass" if abs(err) < 10.0 else "FAIL"
        print(
            f"{k:<18} {sim[k]:10.1f} {hw[k]:10.1f} "
            f"{r:8.2f} {err:7.1f}%  {status}"
        )
    if ratios:
        geo = 1.0
        for r in ratios:
            geo *= r
        geo **= 1 / len(ratios)
        print(f"\nscored geometric mean HW/Sim = {geo:.2f}x  (n={len(ratios)})")
        print(
            "scored mean absolute relative error = "
            f"{sum(absolute_errors) / len(absolute_errors):.1f}%"
        )


if __name__ == "__main__":
    main()
