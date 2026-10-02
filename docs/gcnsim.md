# GCN simulator project location

Verified against the local workspace on 2026-10-02.

The GCN4/GCN5 simulator work lives in `/home/fedora/mgpusim/`, whose
Git remote is `https://github.com/allbilly/mgpusim`. This project is a fork
of the upstream `sarchlab/mgpusim` GPU simulator.

## Evidence in this project

- [`amd/arch/arch.go`](../amd/arch/arch.go) defines GCN3, GCN4
  (gfx804/gfx805 Polaris), GCN5 (gfx900/gfx90c Vega), and CDNA3 (gfx942).
- The [README](../README.md#isca-2019-application-suite-sim-vs-hardware-gfx90c)
  documents simulator calibration against a Renoir gfx90c APU.
- [`progress.md`](../progress.md) records the simulator development work.
- [`gpu_perf_scripts/calibration/gfx90c/`](../gpu_perf_scripts/calibration/gfx90c/)
  contains the gfx90c calibration harness and documentation.

Architecture definitions and documented calibration do not establish complete
instruction coverage or validation for every GPU in an architecture family.
Consult the project README and progress notes for the scope of reported results.
No simulation or hardware tests were rerun when updating this location note.

## Historical location

The original note, dated July 2026, identified a working copy at
`/home/fedora/amdgpu/mgpusim/` and a separate upstream clone at
`/home/fedora/mgpusim/`. That comparison no longer describes the workspace:
`/home/fedora/amdgpu/` is absent, and the current standalone `mgpusim`
checkout contains the GCN4/GCN5 work and uses the `allbilly/mgpusim` remote.

Use this project for the GCN simulator work. Keep this location note with its
simulator documentation rather than in the home directory.
