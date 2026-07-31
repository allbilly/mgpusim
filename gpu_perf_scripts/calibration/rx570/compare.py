#!/usr/bin/env python3
"""Print sim vs HW table for RX 570 (gfx803) ISCA-10 calibration.

Mirrors the gfx90c compare.py but is honest about the unpinned-clock caveat:
without root, the RX 570 host cannot pin power_dpm_force_performance_level=high,
so absolute hipEvent microsecond numbers are DIAGNOSTIC, not reference. The
HW/Sim µs ratio is still printed (it is informative when the clock happens to
boost), but the headline accuracy signal is the cycle-counter ratio column,
which is invariant to clock speed.

Input file format (whitespace-separated, one benchmark per line):
    HW file:  <name> <hw_us> [<hw_cycles>]
              or <name> <cold_us> <steady_us> <cold_minus_steady_us>
    Sim file: <name> <sim_us> [<sim_cycles>]

Lines starting with '#' are comments. Missing values are written as '-'.
"""
import argparse
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_pairs(path, is_sim):
    """Load simulator or hardware measurements.

    Hardware rows with three values are cold µs, steady µs, and their
    difference. Two-value hardware rows retain the older µs/cycle format.
    """
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        name = parts[0]
        vals = []
        for value in parts[1:]:
            parsed = float(value) if value != "-" else None
            vals.append(parsed if parsed is None or math.isfinite(parsed) else None)
        if is_sim:
            # Sim file: (sim_us[, sim_cyc])
            sim_us = vals[0] if len(vals) > 0 else None
            sim_cyc = vals[1] if len(vals) > 1 else None
            out[name] = {
                "sim_us": sim_us,
                "sim_cyc": sim_cyc,
            }
        else:
            out[name] = {
                "cold_us": vals[0] if len(vals) > 0 else None,
                "steady_us": vals[1] if len(vals) >= 3 else None,
                "hw_cyc": vals[1] if len(vals) == 2 else None,
            }
    return out


def fmt(v, w, prec=1):
    if v is None:
        return f"{'-':>{w}}"
    return f"{v:>{w}.{prec}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sim_file",
        nargs="?",
        type=Path,
        default=HERE / "sim_ground_truth.txt",
    )
    parser.add_argument(
        "--hw-mode",
        choices=("steady", "cold"),
        default="steady",
        help=(
            "compare with steady GPU timing (default) or cold hipEvent timing "
            "that also contains first-use runtime costs"
        ),
    )
    args = parser.parse_args()

    hw_path = HERE / "hw_ground_truth.txt"
    sim_path = args.sim_file
    hw = load_pairs(hw_path, is_sim=False)
    sim = load_pairs(sim_path, is_sim=True)
    names = list(hw.keys())

    hw_label = "HWsteady" if args.hw_mode == "steady" else "HWcold"
    print(f"{'Benchmark':<16} {hw_label:>9} {'Simµs':>9} {'HW/Simµs':>9} {'Err%':>7} "
          f"{'HWcyc':>9} {'Simcyc':>9} {'cyc ratio':>10}")
    us_pairs = []
    cyc_pairs = []
    for k in names:
        hw_us = hw[k][f"{args.hw_mode}_us"]
        hw_cyc = hw[k]["hw_cyc"]
        sim_row = sim.get(k, {})
        sim_us = sim_row.get("sim_us")
        sim_cyc = sim_row.get("sim_cyc")
        us_ratio = (hw_us / sim_us) if (hw_us and sim_us) else None
        us_error = (
            abs(sim_us - hw_us) / hw_us * 100
            if hw_us is not None and sim_us is not None
            else None
        )
        cyc_ratio = (hw_cyc / sim_cyc) if (hw_cyc and sim_cyc) else None
        if us_ratio is not None:
            us_pairs.append((hw_us, sim_us))
        if cyc_ratio is not None:
            cyc_pairs.append((hw_cyc, sim_cyc))
        print(f"{k:<16} {fmt(hw_us,9)} {fmt(sim_us,9)} "
              f"{fmt(us_ratio,9,2)} {fmt(us_error,7,1)} "
              f"{fmt(hw_cyc,9,0)} {fmt(sim_cyc,9,0)} "
              f"{fmt(cyc_ratio,10,2)}")

    print()
    if us_pairs:
        geo_us = 1.0
        for hw_us, sim_us in us_pairs:
            geo_us *= hw_us / sim_us
        geo_us **= 1 / len(us_pairs)
        mare = (
            sum(abs(sim_us - hw_us) / hw_us for hw_us, sim_us in us_pairs)
            / len(us_pairs)
            * 100
        )
        print(f"geometric mean HW/Sim (µs, DIAGNOSTIC) = {geo_us:.2f}x  "
              f"(n={len(us_pairs)}, MARE={mare:.1f}%)")
        non_streaming = [
            (
                hw[name][f"{args.hw_mode}_us"],
                sim[name]["sim_us"],
            )
            for name in names
            if name not in {"vectoradd", "relu"}
            and name in sim
            and hw[name][f"{args.hw_mode}_us"] is not None
            and sim[name]["sim_us"] is not None
        ]
        if non_streaming:
            non_streaming_mare = (
                sum(
                    abs(sim_us - hw_us) / hw_us
                    for hw_us, sim_us in non_streaming
                )
                / len(non_streaming)
                * 100
            )
            print(
                "MARE excluding vectoradd/ReLU "
                f"= {non_streaming_mare:.1f}% (n={len(non_streaming)})"
            )
    if cyc_pairs:
        geo_cyc = 1.0
        for hw_cyc, sim_cyc in cyc_pairs:
            geo_cyc *= hw_cyc / sim_cyc
        geo_cyc **= 1 / len(cyc_pairs)
        mare_c = (
            sum(abs(sim_cyc - hw_cyc) / hw_cyc for hw_cyc, sim_cyc in cyc_pairs)
            / len(cyc_pairs)
            * 100
        )
        print(f"geometric mean HW/Sim (cycles, reference) = {geo_cyc:.2f}x  "
              f"(n={len(cyc_pairs)}, MARE={mare_c:.1f}%)")
    if not cyc_pairs:
        print(
            "NOTE: no cycle-counter data present. µs ratios above are "
            "DIAGNOSTIC (unpinned clock); do not treat as reference."
        )


if __name__ == "__main__":
    main()
