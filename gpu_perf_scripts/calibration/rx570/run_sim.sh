#!/usr/bin/env bash
# Run the ISCA-10 suite on the MGPUSim RX 570 (gfx803) timing model; print
# kernel_time µs. Mirrors the gfx90c run_sim.sh but targets -arch gcn3 -gpu
# rx570 (gfx803 is classified as GCN3 in this codebase; the rx570 preset
# supplies the Polaris 20 microarchitecture).
#
# Every workload embeds the exact gfx803 object used by the hardware harness.
# A build failure is still reported as a skip so partial checkouts remain
# diagnosable.
set -euo pipefail
export PATH="${HOME}/.local/go/bin:${HOME}/go/bin:${PATH}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SAMPLES="$ROOT/amd/samples"
OUT_DIR="${1:-$ROOT/gpu_perf_scripts/calibration/rx570/sim_out}"
SIM_JOBS="${SIM_JOBS:-1}"
ONLY="${ONLY:-}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

COMMON=(-timing -arch gcn3 -gpu rx570 -verify)

run_one() {
  local name="$1"; shift
  local dir="$1"; shift
  local metric="$OUT_DIR/$name"
  echo "== sim $name ==" >&2
  (
    cd "$SAMPLES/$dir"
    if ! go build -o "$OUT_DIR/$name.bin" . 2>"$OUT_DIR/$name.build.err"; then
      echo "$name SKIP (build failed; likely missing gitignored HSACO)" >&2
      echo "$name nan"
      return
    fi
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
    s=$(python3 -c "
import sqlite3, sys
c = sqlite3.connect('$db')
r = c.execute(\"SELECT Value FROM mgpusim_metrics WHERE What='kernel_time' AND Location='Driver' LIMIT 1\").fetchone()
print(f'{r[0]*1e6:.3f}' if r else 'NO_METRIC')
" 2>/dev/null || true)
    if [[ -z "$s" ]]; then
      echo "$name NO_METRIC rc=$rc" >&2
      echo "$name nan"
      return
    fi
    echo "$name $s"
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
for spec in "${specs[@]}"; do
  IFS='|' read -r name dir args <<<"$spec"
  selected "$name" || continue

  # shellcheck disable=SC2086 # args are the benchmark's intentional argv.
  run_one "$name" "$dir" $args &
  ((running += 1))
  if ((running >= SIM_JOBS)); then
    wait -n
    running=$((running - 1))
  fi
done
wait
