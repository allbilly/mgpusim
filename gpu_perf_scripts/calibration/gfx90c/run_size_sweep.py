#!/usr/bin/env python3
"""Verify matrix and k-means geometric sweeps using existing sample binaries."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True,
                        help="Directory containing matrixmult.bin and kmeans.bin from run_sim.sh")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for logs, metrics, and results.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binaries = {name: (args.bin_dir / f"{name}.bin").resolve()
                for name in ("matrixmult", "kmeans")}
    manifest = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=HERE, text=True),
        "binary_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in binaries.items()},
        "target_sha256": hashlib.sha256(
            (HERE / "hw_production_targets.json").read_bytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    targets = json.loads((HERE / "hw_production_targets.json").read_text())["targets"]
    cases = [("matrixmult", n, ["-x", str(n), "-y", str(n), "-z", str(n)])
             for n in (32, 64, 128)]
    cases += [("kmeans", n, ["-points", str(n), "-features", "16",
                            "-clusters", "5", "-max-iter", "1"])
              for n in (1024, 2048, 4096, 8192)]
    results = []
    failed = False
    for benchmark, size, parameters in cases:
        name = f"{benchmark}_{size}"
        metric = (args.output / name).resolve()
        command = [str(binaries[benchmark]),
                   "-timing", "-arch", "gcn5", "-gpu", "gfx90c",
                   "-disable-rtm", "-verify", "-metric-file-name", str(metric)] + parameters
        print(f"Running {name}", flush=True)
        with (args.output / f"{name}.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record = {"benchmark": benchmark, "size": size, "command": command,
                  "returncode": completed.returncode}
        try:
            if completed.returncode:
                raise ValueError(f"simulation exited {completed.returncode}")
            with sqlite3.connect(f"file:{metric}.sqlite3?mode=ro", uri=True) as db:
                rows = db.execute("SELECT Value FROM mgpusim_metrics WHERE "
                                  "What='kernel_time' AND Location='Driver'").fetchall()
            if len(rows) != 1:
                raise ValueError(f"expected one kernel_time metric, found {len(rows)}")
            sim = float(rows[0][0]) * 1e6
            if not math.isfinite(sim) or sim <= 0:
                raise ValueError("invalid kernel time")
            record["sim_us"] = sim
            key = f"matrixmult_n{size}" if benchmark == "matrixmult" else None
            if key in targets:
                hw = float(targets[key]["median"])
                error = (hw / sim - 1) * 100
                record.update(hw_us=hw, error_pct=error, gate="pass" if abs(error) < 10 else "FAIL")
                failed |= abs(error) >= 10
            else:
                record["gate"] = "unscored: no matched production target"
        except (ValueError, sqlite3.Error) as error:
            record.update(gate="FAIL", error=str(error))
            failed = True
        results.append(record)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
