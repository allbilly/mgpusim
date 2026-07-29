#!/usr/bin/env bash
# Build and run ISCA-10 HW timing harness on host gfx90c (ROCm via podman).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/build"
mkdir -p "$OUT"

IMG="${ROCM_IMAGE:-docker.io/rocm/dev-ubuntu-24.04:7.1.1}"
ARCH="${OFFLOAD_ARCH:-gfx90c}"
GO_BIN="${GO_BIN:-$(command -v go || true)}"
if [[ -z "$GO_BIN" && -x /home/fedora/.local/go/bin/go ]]; then
  GO_BIN=/home/fedora/.local/go/bin/go
fi
if [[ -z "$GO_BIN" ]]; then
  echo "Go is required to generate deterministic calibration fixtures" >&2
  exit 2
fi

clock_policy_file=""
for candidate in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
  if [[ -r "$candidate" ]]; then
    clock_policy_file="$candidate"
    break
  fi
done
if [[ -n "$clock_policy_file" ]]; then
  clock_policy="$(<"$clock_policy_file")"
  echo "host GPU clock policy: $clock_policy ($clock_policy_file)"
  if [[ "$clock_policy" != "high" &&
        "${ALLOW_UNPINNED_CLOCK:-0}" != "1" ]]; then
    echo "refusing calibration with an unpinned GPU clock; set the policy to" \
      "'high' or use ALLOW_UNPINNED_CLOCK=1 for a non-reference run" >&2
    exit 2
  fi
fi

run_podman() {
  podman run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add=video --group-add=render \
    --security-opt=label=disable \
    -v "$ROOT:$ROOT:z" -w "$HERE" \
    "$IMG" "$@"
}

echo "== generating deterministic benchmark fixtures =="
"$GO_BIN" run "$HERE/generate_fixtures.go" -out "$OUT"

echo "== building exact-HSACO calibration harness for $ARCH =="
run_podman hipcc -O3 --offload-arch="$ARCH" -std=c++17 -I"$HERE" \
  "$HERE/isca10_bench.cpp" -o "$OUT/isca10_bench"

echo "== running =="
run_podman env MGPUSIM_ROOT="$ROOT" "$OUT/isca10_bench" "$@"
