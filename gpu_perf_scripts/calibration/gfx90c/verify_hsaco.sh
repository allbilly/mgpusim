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

  if [[ "$name" == "vmemloadshape" ]]; then
    notes="$(
      podman run --rm -v "$ROOT:$ROOT:ro,z" "$IMAGE" \
        /opt/rocm/llvm/bin/llvm-readelf --notes "$object"
    )"
    if [[ "$(grep -Fc '.wavefront_size: 64' <<<"$notes")" != 6 ||
          "$(grep -Fc '.private_segment_fixed_size: 0' <<<"$notes")" != 6 ||
          "$(grep -Fc '.vgpr_spill_count: 0' <<<"$notes")" != 6 ||
          "$(grep -Fc '.kernarg_segment_size: 32' <<<"$notes")" != 6 ]]; then
      echo "vmemloadshape: metadata does not describe six spill-free wave64 kernels" >&2
      failed=1
    fi

    for width in dword dwordx2 dwordx4; do
      opcode="global_load_$width"
      for mode in serial independent4; do
        symbol="vmem_load_${width}_${mode}"
        disasm="$(
          podman run --rm -v "$ROOT:$ROOT:ro,z" "$IMAGE" \
            /opt/rocm/llvm/bin/llvm-objdump --mcpu=gfx90c \
            --disassemble-symbols="$symbol" "$object"
        )"
        expected_loads=1
        [[ "$mode" == "independent4" ]] && expected_loads=4
        load_count="$(
          grep -Ec "^[[:space:]]*$opcode[[:space:]]" <<<"$disasm" || true
        )"
        store_count="$(
          grep -Ec '^[[:space:]]*global_store_dword[[:space:]]' \
            <<<"$disasm" || true
        )"
        all_stores="$(
          grep -Ec '^[[:space:]]*(buffer_store|flat_store|global_store_)' \
            <<<"$disasm" || true
        )"
        other_loads="$(
          grep -Ec '^[[:space:]]*(buffer_load|flat_load|global_load_)' \
            <<<"$disasm" || true
        )"
        if [[ "$load_count" != "$expected_loads" ||
              "$other_loads" != "$expected_loads" ||
              "$store_count" != 1 || "$all_stores" != 1 ]]; then
          echo "vmemloadshape: $symbol has $load_count expected loads, $other_loads total memory loads, $store_count expected stores, and $all_stores total memory stores" >&2
          failed=1
        fi

        first_load_line="$(grep -n -m1 "^[[:space:]]*$opcode[[:space:]]" <<<"$disasm" | cut -d: -f1 || true)"
        last_load_line="$(grep -n "^[[:space:]]*$opcode[[:space:]]" <<<"$disasm" | tail -1 | cut -d: -f1 || true)"
        first_drain_line="$(grep -n -m1 '^[[:space:]]*s_waitcnt vmcnt(0)' <<<"$disasm" | cut -d: -f1 || true)"
        if [[ -z "$first_load_line" || -z "$last_load_line" ||
              -z "$first_drain_line" ||
              "$first_load_line" -ge "$first_drain_line" ||
              "$last_load_line" -ge "$first_drain_line" ]]; then
          echo "vmemloadshape: $symbol does not issue its load set before vmcnt(0)" >&2
          failed=1
        fi
      done
    done
  fi
  echo "$name: OK"
done <"$MANIFEST"

exit "$failed"
