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
| `publish` | Amend everything into the single root commit and force-push (see [Publishing](#publishing)) |
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

## Updating for a New complex-bessel Release

After a new complex-bessel version is published to crates.io:

1. Bump the dependency in `Cargo.toml` (`complex-bessel = "X.Y.Z"`).
2. `./run build` — downloads the new version and rebuilds.
3. `./run compute-rust bench-rust compare dashboard`
   - mpmath reference values and Fortran/SciPy results are reused (the grid
     is unchanged and those implementations did not change).
   - **Run benchmarks on an idle machine.** Background load skews the
     timings noticeably (observed 19% ↔ 26% swings in the Rust-vs-Fortran
     speedup).
4. Review the README diff (accuracy numbers should normally be unchanged;
   only benchmark timings move), then `./run publish`.

The updated `images/*.svg` are hotlinked from the complex-bessel README, so
its graphs refresh automatically once pushed (GitHub's image cache may take
a few minutes).

## Publishing

This repo intentionally keeps **exactly one commit**. History has no value
here (results are regenerated wholesale) and the tracked results/images are
large, so accumulating history would only bloat the repo.

`./run publish` runs:

```bash
git add -A
git commit --amend --no-edit   # rewrites the single root commit
git push --force-with-lease origin main
```

Git hooks live in the tracked `githooks/` directory. After a fresh clone,
activate them once with:

```bash
git config core.hooksPath githooks
```