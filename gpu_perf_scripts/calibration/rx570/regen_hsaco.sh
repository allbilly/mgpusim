#!/usr/bin/env bash
# Regenerate the exact optimized gfx803 (RX 570 / Polaris 20) code objects used
# by both MGPUSim and the hardware calibration harness. Mirrors the gfx90c
# regen_hsaco.sh; only the offload arch and destination filenames differ.
#
# The generated files are named kernels_gfx803.hsaco in each benchmark package.
# The hardware harness loads all of them by path, and every simulator benchmark
# embeds the same object, keeping all ten workloads on identical instruction
# streams.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${ROCM_IMAGE:-docker.io/rocm/dev-ubuntu-24.04:7.1.1}"
ARCH="${OFFLOAD_ARCH:-gfx803}"
OPT_LEVEL="${OPT_LEVEL:--O3}"
BUILD_DIR="$(mktemp -d /tmp/mgpusim-rx570-hsaco.XXXXXX)"

# benchmark-name|source-path|destination-path|exported-kernel-symbols
SPECS=(
  "vectoradd|amd/benchmarks/amdappsdk/vectoradd/native/vectoradd.cpp|amd/benchmarks/amdappsdk/vectoradd/kernels_gfx803.hsaco|_Z15vectoradd_floatPfPKfS1_ii"
  "relu|amd/benchmarks/dnn/layer_benchmarks/relu/native/relu.cpp|amd/benchmarks/dnn/layer_benchmarks/relu/kernels_gfx803.hsaco|ReLUForward"
  "matrixmult|amd/benchmarks/amdappsdk/matrixmultiplication/native/matrixmultiplication.cpp|amd/benchmarks/amdappsdk/matrixmultiplication/kernels_gfx803.hsaco|mmmKernel_local"
  "matrixtranspose|amd/benchmarks/amdappsdk/matrixtranspose/native/matrixtranspose_hip.cpp|amd/benchmarks/amdappsdk/matrixtranspose/kernels_gfx803.hsaco|matrixTranspose"
  "bitonicsort|amd/benchmarks/amdappsdk/bitonicsort/native/bitonicsort.cpp|amd/benchmarks/amdappsdk/bitonicsort/kernels_gfx803.hsaco|BitonicSort"
  "aes|amd/benchmarks/heteromark/aes/native/kernels.cpp|amd/benchmarks/heteromark/aes/kernels_gfx803.hsaco|Encrypt"
  "fir|amd/benchmarks/heteromark/fir/native/fir.cpp|amd/benchmarks/heteromark/fir/kernels_gfx803.hsaco|FIR"
  "kmeans|amd/benchmarks/heteromark/kmeans/native/kmeans.cpp|amd/benchmarks/heteromark/kmeans/kernels_gfx803.hsaco|kmeans_kernel_swap,kmeans_kernel_compute"
  "pagerank|amd/benchmarks/heteromark/pagerank/native/pagerank.cpp|amd/benchmarks/heteromark/pagerank/kernels_gfx803.hsaco|PageRankUpdateGpu"
  "nw|amd/benchmarks/rodinia/nw/native/nw.cpp|amd/benchmarks/rodinia/nw/kernels_gfx803.hsaco|nw_kernel1,nw_kernel2"
)

compiler_version="$(
  podman run --rm "$IMAGE" hipcc --version |
    sed -n '1p'
)"

manifest="$BUILD_DIR/hsaco_manifest.txt"
{
  echo "# rx570 (gfx803) calibration HSACO manifest"
  echo "architecture=$ARCH"
  echo "image=$IMAGE"
  echo "optimization=$OPT_LEVEL"
  echo "compiler=$compiler_version"
  echo "# benchmark sha256 source destination exported-symbols"
} >"$manifest"

for spec in "${SPECS[@]}"; do
  IFS='|' read -r name source destination symbols <<<"$spec"
  base="$(basename "$source" .cpp)"
  output="$BUILD_DIR/${base}-hip-amdgcn-amd-amdhsa-${ARCH}.out"

  echo "== building $name ($ARCH $OPT_LEVEL) =="
  podman run --rm \
    -v "$ROOT:$ROOT:ro,z" \
    -v "$BUILD_DIR:$BUILD_DIR:z" \
    -w "$BUILD_DIR" \
    "$IMAGE" \
    hipcc "$OPT_LEVEL" --save-temps -c --offload-arch="$ARCH" \
    "$ROOT/$source" -o "$name.o"

  if [[ ! -f "$output" ]]; then
    echo "expected device code object was not generated: $output" >&2
    exit 1
  fi

  install -m 0644 "$output" "$ROOT/$destination"
  digest="$(sha256sum "$ROOT/$destination" | awk '{print $1}')"
  echo "$name $digest $source $destination $symbols" >>"$manifest"
done

install -m 0644 "$manifest" "$HERE/hsaco_manifest.txt"
echo "Generated code objects and $HERE/hsaco_manifest.txt"
echo "Intermediate compiler output retained at $BUILD_DIR"
