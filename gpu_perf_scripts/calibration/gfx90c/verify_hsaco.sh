#!/usr/bin/env bash
# Verify that every embedded gfx90c code object matches the recorded digest
# and exports the kernel symbols used by the simulator/hardware harness.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${ROCM_IMAGE:-docker.io/rocm/dev-ubuntu-24.04:7.1.1}"
MANIFEST="$HERE/hsaco_manifest.txt"

failed=0
while read -r name expected _source destination symbols; do
  [[ -z "${name:-}" || "$name" == \#* || "$name" == *=* ]] && continue

  object="$ROOT/$destination"
  actual="$(sha256sum "$object" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "$name: digest mismatch ($actual != $expected)" >&2
    failed=1
    continue
  fi

  exports="$(
    podman run --rm -v "$ROOT:$ROOT:ro,z" "$IMAGE" \
      /opt/rocm/llvm/bin/llvm-readelf -s "$object" |
      awk '$4 == "FUNC" && $5 == "GLOBAL" {print $8}' |
      sort -u
  )"
  IFS=',' read -ra required_symbols <<<"$symbols"
  for symbol in "${required_symbols[@]}"; do
    if ! grep -Fxq "$symbol" <<<"$exports"; then
      echo "$name: missing exported kernel symbol $symbol" >&2
      failed=1
    fi
  done
  echo "$name: OK"
done <"$MANIFEST"

exit "$failed"
