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

  if [[ "$name" == "scratchspill" ]]; then
    notes="$(
      podman run --rm -v "$ROOT:$ROOT:ro,z" "$IMAGE" \
        /opt/rocm/llvm/bin/llvm-readelf --notes "$object"
    )"
    if [[ "$(grep -Fc '.private_segment_fixed_size: 0' <<<"$notes")" != 1 ||
          "$(grep -Fc '.private_segment_fixed_size: 20' <<<"$notes")" != 1 ||
          "$(grep -Fc '.vgpr_spill_count: 0' <<<"$notes")" != 1 ||
          "$(grep -Fc '.vgpr_spill_count: 4' <<<"$notes")" != 1 ]]; then
      echo "scratchspill: metadata does not describe the 0/4-spill pair" >&2
      failed=1
    fi

    control_disasm="$(
      podman run --rm -v "$ROOT:$ROOT:ro,z" "$IMAGE" \
        /opt/rocm/llvm/bin/llvm-objdump --mcpu=gfx90c \
        --disassemble-symbols=private_control4_kernel "$object"
    )"
    scratch_disasm="$(
      podman run --rm -v "$ROOT:$ROOT:ro,z" "$IMAGE" \
        /opt/rocm/llvm/bin/llvm-objdump --mcpu=gfx90c \
        --disassemble-symbols=private_scratch4_kernel "$object"
    )"
    if grep -Eq '^[[:space:]]*buffer_(load|store)' <<<"$control_disasm"; then
      echo "scratchspill: control4 unexpectedly contains MUBUF traffic" >&2
      failed=1
    fi
    store_count="$(
      grep -Ec '^[[:space:]]*buffer_store_dword[[:space:]]' \
        <<<"$scratch_disasm" || true
    )"
    load_count="$(
      grep -Ec '^[[:space:]]*buffer_load_dword[[:space:]]' \
        <<<"$scratch_disasm" || true
    )"
    if [[ "$store_count" != 4 || "$load_count" != 4 ]]; then
      echo "scratchspill: scratch4 has $store_count stores/$load_count loads, want 4/4" >&2
      failed=1
    fi
    for opcode in buffer_store_dword buffer_load_dword; do
      for offset in 0 4 8 12; do
        if [[ "$offset" == 0 ]]; then
          count="$(
            grep -Ec "^[[:space:]]*$opcode .*[, ]0[[:space:]]+//" \
              <<<"$scratch_disasm" || true
          )"
        else
          count="$(
            grep -Ec "^[[:space:]]*$opcode .*offset:$offset[[:space:]]+//" \
              <<<"$scratch_disasm" || true
          )"
        fi
        if [[ "$count" != 1 ]]; then
          echo "scratchspill: $opcode offset $offset occurs $count times, want 1" >&2
          failed=1
        fi
      done
    done
  fi

  if [[ "$name" == "fp32fma" ]]; then
    notes="$(
      podman run --rm -v "$ROOT:$ROOT:ro,z" "$IMAGE" \
        /opt/rocm/llvm/bin/llvm-readelf --notes "$object"
    )"
    if [[ "$(grep -Fc '.wavefront_size: 64' <<<"$notes")" != 1 ||
          "$(grep -Fc '.private_segment_fixed_size: 0' <<<"$notes")" != 1 ||
          "$(grep -Fc '.vgpr_spill_count: 0' <<<"$notes")" != 1 ]]; then
      echo "fp32fma: metadata does not describe a spill-free wave64 kernel" >&2
      failed=1
    fi

    disasm="$(
      podman run --rm -v "$ROOT:$ROOT:ro,z" "$IMAGE" \
        /opt/rocm/llvm/bin/llvm-objdump --mcpu=gfx90c \
        --disassemble-symbols=fp32_fma_kernel "$object"
    )"
    fma_count="$(
      grep -Ec '^[[:space:]]*v_fma_f32[[:space:]]' <<<"$disasm" || true
    )"
    store_count="$(
      grep -Ec '^[[:space:]]*global_store_dword[[:space:]]' \
        <<<"$disasm" || true
    )"
    if [[ "$fma_count" != 5 || "$store_count" != 1 ]]; then
      echo "fp32fma: disassembly has $fma_count FP32 FMAs/$store_count stores, want 5/1" >&2
      failed=1
    fi
  fi
  echo "$name: OK"
done <"$MANIFEST"

exit "$failed"
