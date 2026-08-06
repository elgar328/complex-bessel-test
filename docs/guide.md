# Usage Guide

## Prerequisites

- **`uv`** — Python package/environment manager
- **`cargo`** — Rust build tool
- **`gfortran`** — Fortran compiler
- **`julia`** — for SpecialFunctions.jl (the `julia/` project; packages are installed and updated automatically)

SciPy and SpecialFunctions.jl are always compared at their latest releases: `compute-scipy`, `bench-scipy`, `compute-julia`, and `bench-julia` upgrade them first (`uv lock --upgrade-package scipy`, `Pkg.update()`), so `python/uv.lock` and `julia/Manifest.toml` change when a new release is out.

## Pipeline Overview

```
Step 1: build       Build Rust binaries + Fortran binaries
Step 2: grid        Generate test grid (results/test_grid.json)
Step 3: compute     Run the implementations (rust, fortran, scipy, julia; mpmath separately)
        bench       Benchmark Rust/Fortran/SciPy/Julia
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
| `compute` | Compute all (= `compute-rust` + `compute-fortran` + `compute-scipy` + `compute-julia` + `compute-f32`) |
| `compute-rust` | Compute Rust results |
| `compute-fortran` | Compute Fortran results |
| `compute-scipy` | Upgrade SciPy, compute SciPy results |
| `compute-julia` | Update SpecialFunctions.jl, compute Julia results |
| `compute-mpmath` | Compute mpmath reference values (slowest step, not included in `compute` or `all`) |
| `compute-f32` | Compute single-precision results: Rust f32 and Fortran CBSUBS (see [f32 Report](#f32-report)) |
| `bench` | Benchmark all (= `bench-rust` + `bench-fortran` + `bench-scipy` + `bench-julia`) |
| `bench-rust` | Benchmark Rust |
| `bench-fortran` | Benchmark Fortran |
| `bench-scipy` | Upgrade SciPy, benchmark SciPy |
| `bench-julia` | Update SpecialFunctions.jl, benchmark Julia |
| `compare` | Compare results against mpmath (`results/comparison.json`) |
| `dashboard` | Generate SVG visualizations (`images/*.svg`) |
| `all-heavy` | Full pipeline including `compute-mpmath` (= `build` + `grid` + `compute-mpmath` + `compute` + `bench` + `compare` + `dashboard`) |
| `publish` | Amend everything into the single root commit and force-push (see [Publishing](#publishing)) |
| `edge` | Edge test vs Fortran in extreme regions (see [Edge Test](#edge-test)); not part of `all` |
| `differences` | Implementation-differences table and chart in the README (see [Implementation Differences](#implementation-differences)); part of `all` |
| `clean` | Delete `results/*.json` + `images/*.{svg,pdf}` + build artifacts |

### Common patterns

After modifying Rust code (reuses existing mpmath results):

```bash
./run compute-rust compute-f32 bench-rust compare dashboard
```

Regenerate comparison + charts only:

```bash
./run compare dashboard
```

First-time setup (including mpmath reference values):

```bash
./run all-heavy
```

Testing unreleased complex-bessel changes: every command uses the `complex-bessel` dependency in `Cargo.toml`, so point it at the dev checkout (`complex-bessel = { path = "../complex-bessel" }`) and rebuild. Switch back to the released version before publishing (see [Updating for a New complex-bessel Release](#updating-for-a-new-complex-bessel-release)).

## Edge Test

```bash
./run build edge
```

Runs about 1.4M cases (~40 s) in extreme regions the accuracy grid does not reach: overflow/underflow thresholds, long `_seq` runs, large negative orders, the ZUOIK and ZSERI underflow windows, scaled Y at large |Im z|, long `_seq_into` runs across its stack-chunk boundaries, and Airy limits (`REGIONS` in `python/edge.py`). Rust is compared with ZBSUBS (f64) and CBSUBS (f32) on status, NZ, and every sequence element; negative orders use Fortran single values per order plus the reflection formulas.

Deliberate deviations from Fortran (`intentional_zuoik`, `intentional_zbesy`), chunked `_seq_into` runs that succeed where the whole Fortran call overflows (`intentional_chunk`), negative-order `_seq_into` runs identical to `_seq` whose limit decisions differ from the single-value reference (`call_dependent`), J and Y on the positive real axis returned exactly real where Fortran leaves an imaginary rounding residue (`intentional_real`), and rounding-level underflow decisions on an axis (`axis_noise`) are reported but do not fail; anything else does (see the docstring in `python/edge.py`). The report is written to `results/edge_report.json`.

## Implementation Differences

```bash
./run build differences
```

Evaluates a fixed set of inputs where the implementations differ (`TABLE` in `python/differences.py`), a sample of values on the real and imaginary axes, and scaled functions near the axes at large |z|, with all four libraries (complex-argument calls only). References come from mpmath, built from the real functions J, Y, I, K of |z| on the axes (mpmath's complex evaluation does not resolve a tiny component), at increasing precision until two evaluations agree in each component. They are cached in `results/differences_mpmath.json`. The script writes `results/differences.json`, `images/componentwise_accuracy.{svg,pdf}`, and the README table between the `differences` markers.

## f32 Report

```bash
./run build compute-f32 compare
```

Evaluates the accuracy grid in f32 with Rust and with CBSUBS and adds an `f32` summary to `results/comparison.json` (error vs mpmath per function, and status agreement with CBSUBS); no charts. Only inputs exact in f32 are used, and references below 1e-28 are skipped. The raw f32 result files are git-ignored.

## Updating for a New complex-bessel Release

After a new complex-bessel version is published to crates.io:

1. Bump the dependency in `Cargo.toml` (`complex-bessel = "X.Y.Z"`).
2. `./run build` — downloads the new version and rebuilds.
3. `./run compute bench compare dashboard`
   - mpmath reference values are reused (the grid is unchanged); SciPy and SpecialFunctions.jl are upgraded to their latest releases, and all benchmarks run in the same session so the timings are comparable.
   - **Run benchmarks on an idle machine.** Background load skews the timings noticeably (observed 19% ↔ 26% swings in the Rust-vs-Fortran speedup).
4. Review the README diff (accuracy numbers should normally be unchanged; only benchmark timings move), then `./run publish`.

The updated `images/*.svg` are hotlinked from the complex-bessel README, so its graphs refresh automatically once pushed (GitHub's image cache may take a few minutes).

## Publishing

This repo intentionally keeps **exactly one commit** on the remote. History has no value here (results are regenerated wholesale) and the tracked results/images are large, so accumulating history would only bloat the repo. Local commits during development are fine; publishing squashes them.

`./run publish` runs:

```bash
git reset --soft <root>        # squash local commits into the root
git add -A
git commit --amend --no-edit   # rewrites the single root commit
git push --force-with-lease origin main
```

Git hooks live in the tracked `githooks/` directory. After a fresh clone, activate them once with:

```bash
git config core.hooksPath githooks
```