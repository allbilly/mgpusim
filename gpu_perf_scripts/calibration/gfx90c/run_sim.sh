#!/usr/bin/env bash
# Run ISCA-10 suite on MGPUSim GCN5/gfx90c timing model; print kernel_time µs.
set -euo pipefail
export PATH="${HOME}/.local/go/bin:${PATH}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SAMPLES="$ROOT/amd/samples"
OUT_DIR="${1:-$ROOT/gpu_perf_scripts/calibration/gfx90c/sim_out}"
mkdir -p "$OUT_DIR"

COMMON=(-timing -arch gcn5 -gpu gfx90c -disable-rtm -verify)

run_one() {
  local name="$1"; shift
  local dir="$1"; shift
  local metric="$OUT_DIR/$name"
  echo "== sim $name ==" >&2
  (
    cd "$SAMPLES/$dir"
    go build -o "$OUT_DIR/$name.bin" .
    rm -f "${metric}.sqlite3" "${metric}.sqlite3-shm" "${metric}.sqlite3-wal"
    set +e
    "$OUT_DIR/$name.bin" "${COMMON[@]}" -metric-file-name "$metric" "$@" \
      >"$OUT_DIR/$name.log" 2>"$OUT_DIR/$name.err"
    local rc=$?
    set -e
    local db="${metric}.sqlite3"
    if [[ ! -f "$db" ]]; then
      echo "$name FAIL rc=$rc" >&2
      tail -5 "$OUT_DIR/$name.err" >&2 || true
      echo "$name nan"
      return
    fi
    local s
    s=$(sqlite3 "$db" "SELECT Value FROM mgpusim_metrics WHERE What='kernel_time' AND Location='Driver' LIMIT 1;" 2>/dev/null || true)
    if [[ -z "$s" ]]; then
      echo "$name NO_METRIC rc=$rc" >&2
      echo "$name nan"
      return
    fi
    python3 -c "print('$name', float('$s')*1e6)"
  )
}

run_one vectoradd vectoradd -width 65536 -height 1
run_one relu relu -length 65536
run_one matrixmult matrixmultiplication -x 128 -y 128 -z 128
run_one matrixtranspose matrixtranspose -width 512
run_one bitonicsort bitonicsort -length 4096
run_one aes aes -length 4096
run_one fir fir -length 8192 -taps 16
run_one kmeans kmeans -points 4096 -features 16 -clusters 5 -max-iter 1
run_one pagerank pagerank -node 512 -sparsity 0.5 -iterations 2
run_one nw nw -length 128
