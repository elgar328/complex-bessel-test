# Usage Guide

## Prerequisites

- **`uv`** — Python package/environment manager
- **`cargo`** — Rust build tool
- **`gfortran`** — Fortran compiler

## Pipeline Overview

```
Step 1: build       Build Rust binaries + Fortran binaries
Step 2: grid        Generate test grid (results/test_grid.json)
Step 3: compute     Run all 4 implementations (rust, fortran, scipy, mpmath)
        bench       Benchmark Rust/Fortran/scipy
Step 4: compare     Compare results (results/comparison.json)
Step 5: dashboard   Generate SVG visualizations (images/*.svg)
```

## Running

Multiple commands can be chained in one invocation: `./run build grid compute`.

### Full pipeline

```bash
./run all                # build → grid → compute → bench → compare → dashboard
./run all-heavy          # build → grid → compute-mpmath → compute → bench → compare → dashboard
```

> **Note:** `all` does not include `compute-mpmath`. Use `all-heavy` when you need fresh reference values.

### Commands

| Command | Description |
|---|---|
| `build` | Build all (= `build-rust` + `build-fortran`) |
| `build-rust` | `cargo build --release` |
| `build-fortran` | Compile Fortran binaries with gfortran |
| `grid` | Generate test grid (`results/test_grid.json`) |
| `compute` | Compute all (= `compute-rust` + `compute-fortran` + `compute-scipy`) |
| `compute-rust` | Compute Rust results |
| `compute-fortran` | Compute Fortran results |
| `compute-scipy` | Compute SciPy results |
| `compute-mpmath` | Compute mpmath reference values (slowest step, not included in `compute` or `all`) |
| `bench` | Benchmark all (= `bench-rust` + `bench-fortran` + `bench-scipy`) |
| `bench-rust` | Benchmark Rust |
| `bench-fortran` | Benchmark Fortran |
| `bench-scipy` | Benchmark SciPy |
| `compare` | Compare results against mpmath (`results/comparison.json`) |
| `dashboard` | Generate SVG visualizations (`images/*.svg`) |
| `all-heavy` | Full pipeline including `compute-mpmath` (= `build` + `grid` + `compute-mpmath` + `compute` + `bench` + `compare` + `dashboard`) |
| `clean` | Delete `results/*.json` + `images/*.{svg,pdf}` + build artifacts |

### Common patterns

After modifying Rust code (reuses existing mpmath results):

```bash
./run compute-rust bench-rust compare dashboard
```

Regenerate comparison + charts only:

```bash
./run compare dashboard
```

First-time setup (including mpmath reference values):

```bash
./run all-heavy
```