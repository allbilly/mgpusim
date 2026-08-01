#!/usr/bin/env python3
"""Plan or execute the guarded gfx90c VMEM dependency-window sweep."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import acquire_pinned as acquire


HERE = Path(__file__).resolve().parent
ACQUIRE = HERE / "acquire_pinned.py"
FORBIDDEN_PROCESS_REGEX = (
    r"Vgfx9_compute_unit_tb|verilator_bin|pytest|miaow_gcn4"
)
INTER_POINT_COOLDOWN_SECONDS = 30.0
STABLE_GUARD_CLEAR_SECONDS = 30.0
GUARD_READY_TIMEOUT_SECONDS = 900.0
GUARD_POLL_SECONDS = 5.0


@dataclass(frozen=True)
class SweepPoint:
    mode: str
    workgroups: int
    repeats: int

    @property
    def name(self) -> str:
        return f"{self.mode}-g{self.workgroups}-r{self.repeats}"


def sweep_points(
    modes: Sequence[str] = ("independent2", "independent8"),
    workgroup_counts: Sequence[int] = (16, 28),
) -> tuple[SweepPoint, ...]:
    """Return matched zero/body points, with each pair kept adjacent."""
    return tuple(
        SweepPoint(mode, workgroups, repeats)
        for mode in modes
        for workgroups in workgroup_counts
        for repeats in (0, 64)
    )


def build_acquire_command(
    point: SweepPoint,
    output_dir: Path,
    *,
    batches: int,
    python: str = sys.executable,
    acquire_script: Path = ACQUIRE,
) -> list[str]:
    """Build one collector invocation without shell interpolation."""
    # Prior guarded pilots put the zero-trip launches near 81--89 ms at
    # 10,000 iterations.  The R64 points are longer, so 6,500 iterations
    # keeps their predicted timed windows near 118--145 ms.
    iterations = 10_000 if point.repeats == 0 else 6_500
    return [
        python,
        str(acquire_script),
        "--batches",
        str(batches),
        "--warmup",
        "20",
        "--iters",
        str(iterations),
        "--cooldown-seconds",
        "30",
        "--max-start-temp-c",
        "50",
        "--max-temp-c",
        "58",
        "--sample-interval-seconds",
        "0.005",
        "--min-active-samples",
        "10",
        "--forbid-process-regex",
        FORBIDDEN_PROCESS_REGEX,
        "--output-dir",
        str(output_dir),
        "vmemloadshape",
        "--",
        "--vmem-width-dwords",
        "4",
        "--vmem-mode",
        point.mode,
        "--vmem-alias-lanes",
        "8",
        "--vmem-array-bytes",
        "8192",
        "--vmem-repeats",
        str(point.repeats),
        "--vmem-workgroups",
        str(point.workgroups),
    ]


def default_output_root() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = uuid.uuid4().hex[:8]
    return Path(tempfile.gettempdir()) / f"gfx90c-vmem-window-{timestamp}-{token}"


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def wait_for_stable_guard_clear(
    guard: acquire.ForbiddenProcessGuard,
    *,
    stable_seconds: float = STABLE_GUARD_CLEAR_SECONDS,
    timeout_seconds: float = GUARD_READY_TIMEOUT_SECONDS,
    poll_seconds: float = GUARD_POLL_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until the process guard has stayed clear for a full interval."""
    deadline = monotonic() + timeout_seconds
    clear_since: float | None = None
    while True:
        now = monotonic()
        match = guard.check()
        if match is None:
            if clear_since is None:
                clear_since = now
            if now - clear_since >= stable_seconds:
                return
        else:
            clear_since = None
        if now >= deadline:
            if match is None:
                detail = "guard did not remain continuously clear"
            else:
                detail = acquire.format_forbidden_process_match(match)
            raise TimeoutError(
                f"timed out waiting for a stable process guard: {detail}"
            )
        sleeper(min(poll_seconds, max(0.0, deadline - now)))


def run_sweep(
    *,
    output_root: Path,
    batches: int,
    execute: bool,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    guard_waiter: Callable[[], None] | None = None,
    points: Sequence[SweepPoint] | None = None,
    python: str = sys.executable,
    acquire_script: Path = ACQUIRE,
) -> int:
    points = tuple(sweep_points() if points is None else points)
    if not points:
        raise ValueError("sweep must contain at least one point")
    commands = [
        build_acquire_command(
            point,
            output_root / point.name,
            batches=batches,
            python=python,
            acquire_script=acquire_script,
        )
        for point in points
    ]

    if not execute:
        print(f"dry run; output root would be {output_root}")
        for command in commands:
            print(shlex.join(command))
        return 0

    # The exact run root must be new. Each child is intentionally left absent:
    # acquire_pinned.py creates it exclusively and owns every artifact within.
    output_root.mkdir(parents=True, exist_ok=False)
    if guard_waiter is None:
        process_guard = acquire.ForbiddenProcessGuard(
            [FORBIDDEN_PROCESS_REGEX]
        )
        guard_waiter = lambda: wait_for_stable_guard_clear(process_guard)
    manifest_path = output_root / "sweep_manifest.json"
    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "batches_per_point": batches,
        "inter_point_cooldown_seconds": INTER_POINT_COOLDOWN_SECONDS,
        "stable_guard_clear_seconds": STABLE_GUARD_CLEAR_SECONDS,
        "guard_ready_timeout_seconds": GUARD_READY_TIMEOUT_SECONDS,
        "forbidden_process_regex": FORBIDDEN_PROCESS_REGEX,
        "points": [asdict(point) for point in points],
        "completed": [],
        "failed_point": None,
    }
    write_manifest(manifest_path, manifest)

    completed = manifest["completed"]
    assert isinstance(completed, list)
    for index, (point, command) in enumerate(zip(points, commands)):
        print(
            f"waiting for {STABLE_GUARD_CLEAR_SECONDS:.0f} seconds of stable "
            f"guard clearance before {point.name}",
            flush=True,
        )
        try:
            guard_waiter()
        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
            manifest["status"] = "failed"
            manifest["failed_point"] = point.name
            manifest["guard_wait_failure"] = str(error)
            write_manifest(manifest_path, manifest)
            print(f"guard readiness failed: {error}", file=sys.stderr)
            return 1
        print(f"running {point.name}: {shlex.join(command)}", flush=True)
        result = runner(command, check=False)
        point_record = {
            "name": point.name,
            "returncode": result.returncode,
            "artifact_directory": str(output_root / point.name),
            "collector_manifest": str(
                output_root / point.name / "manifest.json"
            ),
        }
        completed.append(point_record)
        if result.returncode != 0:
            manifest["status"] = "failed"
            manifest["failed_point"] = point.name
            write_manifest(manifest_path, manifest)
            print(
                f"stopped after {point.name} failed with rc={result.returncode}; "
                f"artifacts remain under {output_root}",
                file=sys.stderr,
            )
            return result.returncode
        write_manifest(manifest_path, manifest)
        if index + 1 < len(points):
            print(
                f"cooling down {INTER_POINT_COOLDOWN_SECONDS:.0f} seconds "
                "before the next point",
                flush=True,
            )
            sleeper(INTER_POINT_COOLDOWN_SECONDS)

    manifest["status"] = "complete"
    write_manifest(manifest_path, manifest)
    print(f"completed sweep; artifacts are under {output_root}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the collector; without this flag only commands are printed",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="collect nine batches per point instead of a one-batch pilot",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="new run root (must not already exist)",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=("independent2", "independent8"),
        help="dependency mode to collect (repeatable; default: both)",
    )
    parser.add_argument(
        "--workgroups",
        action="append",
        type=int,
        choices=(16, 28),
        help="work-group count to collect (repeatable; default: both)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = (
        default_output_root()
        if args.output_root is None
        else args.output_root.resolve()
    )
    try:
        return run_sweep(
            output_root=output_root,
            batches=9 if args.production else 1,
            execute=args.execute,
            points=sweep_points(
                modes=(
                    ("independent2", "independent8")
                    if args.mode is None
                    else tuple(dict.fromkeys(args.mode))
                ),
                workgroup_counts=(
                    (16, 28)
                    if args.workgroups is None
                    else tuple(dict.fromkeys(args.workgroups))
                ),
            ),
        )
    except FileExistsError:
        print(f"output root already exists: {output_root}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"sweep setup failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
