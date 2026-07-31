#!/usr/bin/env bash
# Build and run the ISCA-10 HW timing harness on host RX 570 (gfx803, Polaris
# 20) via ROCm in podman. Mirrors the gfx90c harness but targets gfx803 and
# defaults to an UNPINNED clock policy because RX 570 hosts in this calibration
# loop run without root (no power_dpm_force_performance_level=high).
#
# IMPORTANT: with an unpinned clock, absolute hipEvent microsecond numbers are
# DIAGNOSTIC, not reference-quality. The amdgpu driver on Polaris idles the
# clock to 300 MHz and may not boost without the dpm override, so wall-clock
# times can be several times too long (see progress.md: the gfx90c unpinned
# re-run came back ~4x slower than the pinned reference, purely from
# throttling). Treat the µs column as a lower bound on real time.
#
# NOTE: this harness currently emits ONLY hipEvent µs. It does NOT yet collect
# s_memtime cycle-counter deltas — that probe is PENDING (see README.md "Open").
# Until the probe is added, the only HW data available is diagnostic µs, which
# is not safe to tune the timing model against without a pinned clock.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/build"
mkdir -p "$OUT"

IMG="${ROCM_IMAGE:-docker.io/rocm/dev-ubuntu-22.04:6.2.4}"
ARCH="${OFFLOAD_ARCH:-gfx803}"
GO_BIN="${GO_BIN:-$(command -v go || true)}"
if [[ -z "$GO_BIN" && -x /home/fedora/.local/go/bin/go ]]; then
  GO_BIN=/home/fedora/.local/go/bin/go
fi
if [[ -z "$GO_BIN" ]]; then
  echo "Go is required to generate deterministic calibration fixtures" >&2
  exit 2
fi

# Clock policy check. Default to allowing unpinned because the RX 570 host in
# this loop has no root; print a loud diagnostic banner so the output is never
# mistaken for a pinned-clock reference.
clock_policy_file=""
for candidate in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
  if [[ -r "$candidate" ]]; then
    clock_policy_file="$candidate"
    break
  fi
done
pinned=0
if [[ -n "$clock_policy_file" ]]; then
  clock_policy="$(<"$clock_policy_file")"
  echo "host GPU clock policy: $clock_policy ($clock_policy_file)" >&2
  if [[ "$clock_policy" == "high" ]]; then
    pinned=1
  fi
fi
if [[ "$pinned" != "1" && "${ALLOW_UNPINNED_CLOCK:-1}" != "1" ]]; then
  echo "refusing to run with an unpinned GPU clock; set the policy to 'high'" \
    "(needs root) or set ALLOW_UNPINNED_CLOCK=1 for a diagnostic run" >&2
  exit 2
fi
if [[ "$pinned" != "1" ]]; then
  echo "==============================================================" >&2
  echo "WARNING: unpinned GPU clock. µs numbers are DIAGNOSTIC, not"   >&2
  echo "         reference. Use cycle-counter columns for calibration."  >&2
  echo "==============================================================" >&2
fi

run_podman() {
  podman run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add=keep-groups \
    --security-opt=label=disable \
    -v "$ROOT:$ROOT:z" -w "$HERE" \
    "$IMG" "$@"
}

if [[ "${RX570_REUSE_BUILD:-0}" != "1" || ! -x "$OUT/isca10_bench" ]]; then
  echo "== generating deterministic benchmark fixtures =="
  "$GO_BIN" run "$HERE/generate_fixtures.go" -out "$OUT"

  # Derive the gfx803 harness source from the gfx90c one by rewriting HSACO and
  # fixture paths. This keeps a single source of truth and avoids a maintained
  # gfx803 source fork.
  GFX90C_BENCH="$ROOT/gpu_perf_scripts/calibration/gfx90c/isca10_bench.cpp"
  GFX803_BENCH="$OUT/isca10_bench_gfx803.cpp"
  sed \
    -e 's#kernels_gfx90c\.hsaco#kernels_gfx803.hsaco#g' \
    -e 's#calibration/gfx90c/build#calibration/rx570/build#g' \
    -e 's#cache_latency_gfx90c\.co#cache_latency_gfx803.co#g' \
    "$GFX90C_BENCH" > "$GFX803_BENCH"
  cp "$ROOT/gpu_perf_scripts/calibration/gfx90c/aes_tables.inc" \
    "$OUT/aes_tables.inc"

  echo "== building exact-HSACO calibration harness for $ARCH =="
  run_podman hipcc -O3 --offload-arch="$ARCH" -std=c++17 -I"$OUT" \
    "$GFX803_BENCH" -o "$OUT/isca10_bench"
else
  echo "== reusing exact-HSACO calibration harness =="
fi

echo "== running (clock $([ "$pinned" = "1" ] && echo pinned || echo UNPINNED/diagnostic)) =="
run_podman env MGPUSIM_ROOT="$ROOT" "$OUT/isca10_bench" "$@"
