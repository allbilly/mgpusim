#!/usr/bin/env bash
# Run ISCA-10 suite on MGPUSim GCN5/gfx90c timing model; print kernel_time µs.
set -euo pipefail
export PATH="${HOME}/.local/go/bin:${PATH}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SAMPLES="$ROOT/amd/samples"
OUT_DIR="${1:-$ROOT/gpu_perf_scripts/calibration/gfx90c/sim_out}"
SIM_JOBS="${SIM_JOBS:-1}"
ONLY="${ONLY:-}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

COMMON=(-timing -arch gcn5 -gpu gfx90c -disable-rtm -verify)

run_one() {
  local name="$1"; shift
  local dir="$1"; shift
  local metric="$OUT_DIR/$name"
  echo "== sim $name ==" >&2
  rm -f \
    "$OUT_DIR/$name.bin" "$OUT_DIR/$name.log" "$OUT_DIR/$name.err" \
    "${metric}.sqlite3" "${metric}.sqlite3-shm" "${metric}.sqlite3-wal" \
    "${metric}.rc" "${metric}.sim.rc"
  # Initialize pessimistically so an interrupted or failed setup cannot leave
  # a stale success marker.
  printf '1\n' >"${metric}.rc"
  (
    cd "$SAMPLES/$dir"
    set +e
    go build -o "$OUT_DIR/$name.bin" .
    local build_rc=$?
    set -e
    if ((build_rc != 0)); then
      echo "$name BUILD_FAIL rc=$build_rc" >&2
      echo "$name nan"
      return 1
    fi
    set +e
    "$OUT_DIR/$name.bin" "${COMMON[@]}" -metric-file-name "$metric" "$@" \
      >"$OUT_DIR/$name.log" 2>"$OUT_DIR/$name.err"
    local rc=$?
    set -e
    printf '%s\n' "$rc" >"${metric}.sim.rc"
    if ((rc != 0)); then
      echo "$name FAIL rc=$rc" >&2
      tail -5 "$OUT_DIR/$name.err" >&2 || true
      echo "$name nan"
      return 1
    fi
    local db="${metric}.sqlite3"
    if [[ ! -f "$db" ]]; then
      echo "$name FAIL rc=$rc" >&2
      tail -5 "$OUT_DIR/$name.err" >&2 || true
      echo "$name nan"
      return 1
    fi
    local s
    s=$(sqlite3 "$db" "SELECT Value FROM mgpusim_metrics WHERE What='kernel_time' AND Location='Driver' LIMIT 1;" 2>/dev/null || true)
    if [[ -z "$s" ]]; then
      echo "$name NO_METRIC rc=$rc" >&2
      echo "$name nan"
      return 1
    fi
    local result
    if ! result=$(python3 -c \
      'import sys; print(sys.argv[1], float(sys.argv[2]) * 1e6)' \
      "$name" "$s"); then
      echo "$name BAD_METRIC rc=$rc" >&2
      echo "$name nan"
      return 1
    fi
    if ! printf '%s\n' "$result"; then
      return 1
    fi
    printf '0\n' >"${metric}.rc"
  )
}

selected() {
  local name="$1"
  [[ -z "$ONLY" || ",$ONLY," == *",$name,"* ]]
}

specs=(
  "vectoradd|vectoradd|-width 65536 -height 1"
  "relu|relu|-length 65536"
  "matrixmult|matrixmultiplication|-x 128 -y 128 -z 128"
  "matrixtranspose|matrixtranspose|-width 512"
  "bitonicsort|bitonicsort|-length 4096"
  "aes|aes|-length 4096"
  "fir|fir|-length 8192 -taps 16"
  "kmeans|kmeans|-points 4096 -features 16 -clusters 5 -max-iter 1"
  "pagerank|pagerank|-node 512 -sparsity 0.5 -iterations 2"
  "nw|nw|-length 128"
)

running=0
failed=0
for spec in "${specs[@]}"; do
  IFS='|' read -r name dir args <<<"$spec"
  selected "$name" || continue

  # shellcheck disable=SC2086 # args are the benchmark's intentional argv.
  run_one "$name" "$dir" $args &
  ((running += 1))
  if ((running >= SIM_JOBS)); then
    if ! wait -n; then
      failed=1
    fi
    running=$((running - 1))
  fi
done
while ((running > 0)); do
  if ! wait -n; then
    failed=1
  fi
  running=$((running - 1))
done
exit "$failed"
