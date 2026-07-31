#!/usr/bin/env bash
# Regenerate the exact optimized gfx90c code objects used by both MGPUSim and
# the hardware calibration harness.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${ROCM_IMAGE:-docker.io/rocm/dev-ubuntu-24.04:7.1.1}"
ARCH="${OFFLOAD_ARCH:-gfx90c}"
OPT_LEVEL="${OPT_LEVEL:--O3}"
BUILD_DIR="$(mktemp -d /tmp/mgpusim-gfx90c-hsaco.XXXXXX)"

# benchmark-name|source-path|destination-path|exported-kernel-symbols
SPECS=(
  "vectoradd|amd/benchmarks/amdappsdk/vectoradd/native/vectoradd.cpp|amd/benchmarks/amdappsdk/vectoradd/kernels_gfx90c.hsaco|_Z15vectoradd_floatPfPKfS1_ii"
  "relu|amd/benchmarks/dnn/layer_benchmarks/relu/native/relu.cpp|amd/benchmarks/dnn/layer_benchmarks/relu/kernels_gfx90c.hsaco|ReLUForward"
  "matrixmult|amd/benchmarks/amdappsdk/matrixmultiplication/native/matrixmultiplication.cpp|amd/benchmarks/amdappsdk/matrixmultiplication/kernels_gfx90c.hsaco|mmmKernel_local"
  "matrixtranspose|amd/benchmarks/amdappsdk/matrixtranspose/native/matrixtranspose_hip.cpp|amd/benchmarks/amdappsdk/matrixtranspose/kernels_gfx90c.hsaco|matrixTranspose"
  "bitonicsort|amd/benchmarks/amdappsdk/bitonicsort/native/bitonicsort.cpp|amd/benchmarks/amdappsdk/bitonicsort/kernels_gfx90c.hsaco|BitonicSort"
  "aes|amd/benchmarks/heteromark/aes/native/kernels.cpp|amd/benchmarks/heteromark/aes/kernels_gfx90c.hsaco|Encrypt"
  "fir|amd/benchmarks/heteromark/fir/native/fir.cpp|amd/benchmarks/heteromark/fir/kernels_gfx90c.hsaco|FIR"
  "cachelatency|amd/benchmarks/microbench/cachelatency/native/cache_latency.cpp|amd/benchmarks/microbench/cachelatency/kernels_gfx90c.hsaco|pointer_chase_kernel,vector_pointer_chase_kernel"
  "storestride|amd/benchmarks/microbench/storestride/native/store_stride.cpp|amd/benchmarks/microbench/storestride/kernels_gfx90c.hsaco|full_line_store_stride_kernel"
  "scratchspill|amd/benchmarks/microbench/scratchspill/native/scratch_spill.cpp|amd/benchmarks/microbench/scratchspill/kernels_gfx90c.hsaco|private_control4_kernel,private_scratch4_kernel"
  "kmeans|amd/benchmarks/heteromark/kmeans/native/kmeans.cpp|amd/benchmarks/heteromark/kmeans/kernels_gfx90c.hsaco|kmeans_kernel_swap,kmeans_kernel_compute"
  "pagerank|amd/benchmarks/heteromark/pagerank/native/pagerank.cpp|amd/benchmarks/heteromark/pagerank/kernels_gfx90c.hsaco|PageRankUpdateGpu"
  "nw|amd/benchmarks/rodinia/nw/native/nw.cpp|amd/benchmarks/rodinia/nw/kernels_gfx90c.hsaco|nw_kernel1,nw_kernel2"
)

compiler_version="$(
  podman run --rm "$IMAGE" hipcc --version |
    sed -n '1p'
)"

manifest="$BUILD_DIR/hsaco_manifest.txt"
{
  echo "# gfx90c calibration HSACO manifest"
  echo "architecture=$ARCH"
  echo "image=$IMAGE"
  echo "optimization=$OPT_LEVEL"
  echo "compiler=$compiler_version"
  echo "# benchmark sha256 source destination exported-symbols"
} >"$manifest"

for spec in "${SPECS[@]}"; do
  IFS='|' read -r name source destination symbols <<<"$spec"
  base="$(basename "$source" .cpp)"
  # hipcc --save-temps emits both a relocatable device object (.o) and the
  # linked loadable code object (.out). The latter is required by
  # hipModuleLoad and is also accepted by MGPUSim's code-object parser.
  output="$BUILD_DIR/${base}-hip-amdgcn-amd-amdhsa-${ARCH}.out"

  echo "== building $name ($ARCH $OPT_LEVEL) =="
  # A stable explicit CUID prevents the random temporary build path from
  # changing otherwise identical code-object hashes.
  podman run --rm \
    -v "$ROOT:$ROOT:ro,z" \
    -v "$BUILD_DIR:$BUILD_DIR:z" \
    -w "$BUILD_DIR" \
    "$IMAGE" \
    hipcc "$OPT_LEVEL" --save-temps -c --offload-arch="$ARCH" \
    -cuid="$name" \
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
