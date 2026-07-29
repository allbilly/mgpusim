#!/usr/bin/env bash
# Build and run ISCA-10 HW timing harness on host gfx90c (ROCm via podman).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="$ROOT/amd/benchmarks"
OUT="$HERE/build"
mkdir -p "$OUT"

IMG="${ROCM_IMAGE:-docker.io/rocm/dev-ubuntu-24.04:7.1.1}"
ARCH="${OFFLOAD_ARCH:-gfx90c}"

run_podman() {
  podman run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add=video --group-add=render \
    --security-opt=label=disable \
    -v "$ROOT:$ROOT:z" -w "$HERE" \
    "$IMG" "$@"
}

echo "== building isca10_bench for $ARCH =="
run_podman hipcc -O3 --offload-arch="$ARCH" -std=c++17 -I"$HERE" \
  "$HERE/isca10_bench.cpp" \
  "$BENCH/amdappsdk/matrixmultiplication/native/matrixmultiplication.cpp" \
  "$BENCH/amdappsdk/matrixtranspose/native/matrixtranspose_hip.cpp" \
  "$BENCH/heteromark/aes/native/kernels.cpp" \
  "$BENCH/heteromark/fir/native/fir.cpp" \
  "$BENCH/heteromark/kmeans/native/kmeans.cpp" \
  "$BENCH/heteromark/pagerank/native/pagerank.cpp" \
  "$BENCH/rodinia/nw/native/nw.cpp" \
  -o "$OUT/isca10_bench"

echo "== running =="
run_podman "$OUT/isca10_bench" "$@"
