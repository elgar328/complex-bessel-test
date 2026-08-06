"""Pipeline orchestration CLI — replaces Makefile.

Usage:
    uv run --project python python/run.py <command> [<command> ...]
    ./run <command> [<command> ...]

Commands can be chained: ./run build grid compute-rust bench-rust compare dashboard
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"
FORTRAN = ROOT / "fortran"
FORTRAN_BUILD = FORTRAN / ".build"
PYTHON = ROOT / "python"
JULIA = ROOT / "julia"


def run_cmd(args, **kwargs):
    """Run a subprocess command; exit on failure."""
    print(f"  $ {' '.join(str(a) for a in args)}", flush=True)
    proc = subprocess.run(args, **kwargs)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def run_python_script(name, *args):
    """Run a sibling Python script using the same interpreter."""
    script = Path(__file__).parent / name
    run_cmd([sys.executable, str(script), *args])


# ── Build ──


def cmd_build_rust():
    print("Building Rust...", flush=True)
    run_cmd(["cargo", "build", "--release"], cwd=ROOT)


def cmd_build_fortran():
    print("Building Fortran...", flush=True)
    toms644_src = FORTRAN / "toms644"
    if not toms644_src.exists():
        print("ERROR: fortran/toms644 not found", file=sys.stderr)
        sys.exit(1)

    FORTRAN_BUILD.mkdir(parents=True, exist_ok=True)
    zbsubs = FORTRAN_BUILD / "zbsubs.f"

    # Extract zbsubs.f from combined toms644 file
    run_cmd(
        [
            "awk",
            '/^C\\*\\*\\* / { fname = $2; if (fname == "Readme") { skip = 1; next } skip = 0; next }\n'
            'skip == 0 && fname == "zbsubs.f" { print }',
            str(toms644_src),
        ],
        stdout=open(zbsubs, "w"),
    )

    # Fix separator lines for gfortran compatibility
    run_cmd(["sed", "-i", "", "s/^--*/C ---/", str(zbsubs)])

    machcon = FORTRAN / "machcon_ieee.f"

    # Compile compute_grid
    run_cmd(
        [
            "gfortran",
            "-O2",
            "-std=legacy",
            "-o",
            str(FORTRAN / "compute_grid"),
            str(FORTRAN / "compute_grid.f"),
            str(machcon),
            str(zbsubs),
        ]
    )

    # Compile bench_fortran
    run_cmd(
        [
            "gfortran",
            "-O2",
            "-std=legacy",
            "-o",
            str(FORTRAN / "bench_fortran"),
            str(FORTRAN / "bench.f"),
            str(machcon),
            str(zbsubs),
        ]
    )

    cmd_build_fortran_edge(toms644_src, zbsubs, machcon)


# ZUOIK picks the ZN rotation from the unreflected ZI; CUOIK (single
# precision) uses AIMAG(ZR). complex-bessel follows CUOIK, so the edge test
# builds a patched ZBSUBS to tell that deliberate difference from bugs.
ZUOIK_ZN_TEST = "      IF (ZI.GT.0.0D0) GO TO 40\n"
ZUOIK_ZN_TEST_ZRI = "      IF (ZRI.GT.0.0D0) GO TO 40\n"


def cmd_build_fortran_edge(toms644_src, zbsubs, machcon):
    """Build the edge-test drivers: ZBSUBS, ZBSUBS with ZRI, and CBSUBS."""
    cbsubs = FORTRAN_BUILD / "cbsubs.f"
    run_cmd(
        [
            "awk",
            '/^C\\*\\*\\* / { fname = $2; next }\nfname == "cbsubs.f" { print }',
            str(toms644_src),
        ],
        stdout=open(cbsubs, "w"),
    )
    run_cmd(["sed", "-i", "", "s/^--*/C ---/", str(cbsubs)])

    src = zbsubs.read_text()
    start = src.index("      SUBROUTINE ZUOIK(")
    end = src.index("      SUBROUTINE ", start + 1)
    body = src[start:end]
    if body.count(ZUOIK_ZN_TEST) != 1:
        print("ERROR: ZUOIK ZN test not found exactly once", file=sys.stderr)
        sys.exit(1)
    zbsubs_zri = FORTRAN_BUILD / "zbsubs_zri.f"
    zbsubs_zri.write_text(
        src[:start] + body.replace(ZUOIK_ZN_TEST, ZUOIK_ZN_TEST_ZRI) + src[end:]
    )

    # -O0: no FMA contraction, so rounding matches the source as written
    # (some edge decisions compare rounding-level quantities)
    flags = ["gfortran", "-O0", "-std=legacy", "-w"]
    for exe, driver, lib, extra in (
        ("edge64", "edge_f64.f", zbsubs, []),
        ("edge64_zri", "edge_f64.f", zbsubs_zri, []),
        ("edge32", "edge_f32.f", cbsubs, [FORTRAN / "r1mach_ieee.f"]),
    ):
        run_cmd(
            flags
            + ["-o", str(FORTRAN / exe), str(FORTRAN / driver), str(machcon), str(lib)]
            + [str(p) for p in extra]
        )


def cmd_build():
    cmd_build_rust()
    cmd_build_fortran()


# ── Grid ──


def cmd_grid():
    print("Generating test grid...", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_python_script("generate_grid.py")


# ── Compute ──


def cmd_compute_rust():
    print("Computing Rust results...", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    grid = RESULTS / "test_grid.json"
    out = RESULTS / "rust_results.json"
    with open(grid) as fin, open(out, "w") as fout:
        run_cmd(
            [str(ROOT / "target" / "release" / "compute")],
            stdin=fin,
            stdout=fout,
            stderr=subprocess.DEVNULL,
        )


def cmd_compute_fortran():
    print("Computing Fortran results...", flush=True)
    run_python_script("compute_fortran.py", "--mode", "compute")


# SciPy and SpecialFunctions.jl are compared at their latest releases:
# each compute/bench step upgrades them first.


def upgrade_scipy():
    print("Upgrading SciPy to the latest release...", flush=True)
    run_cmd(["uv", "lock", "--upgrade-package", "scipy"], cwd=PYTHON)
    run_cmd(["uv", "sync"], cwd=PYTHON)


def upgrade_julia():
    print("Updating SpecialFunctions.jl to the latest release...", flush=True)
    run_cmd(["julia", f"--project={JULIA}", "-e", "using Pkg; Pkg.update()"])


def run_julia(mode):
    run_cmd(["julia", f"--project={JULIA}", str(JULIA / "compute.jl"), mode])


def cmd_compute_scipy():
    print("Computing SciPy results...", flush=True)
    upgrade_scipy()
    run_python_script("compute_scipy.py", "--mode", "compute")


def cmd_compute_julia():
    print("Computing Julia results...", flush=True)
    upgrade_julia()
    run_julia("compute")


def cmd_compute_mpmath():
    print("Computing mpmath results...", flush=True)
    run_python_script("compute_mpmath.py")


def cmd_compute_f32():
    """Single precision (Rust f32 and Fortran CBSUBS), for the f32 report."""
    print("Computing Rust f32 results...", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    grid = RESULTS / "test_grid.json"
    out = RESULTS / "rust_f32_results.json"
    with open(grid) as fin, open(out, "w") as fout:
        run_cmd(
            [str(ROOT / "target" / "release" / "compute"), "--f32"],
            stdin=fin,
            stdout=fout,
            stderr=subprocess.DEVNULL,
        )
    run_python_script("compute_fortran.py", "--mode", "compute-f32")


def cmd_compute():
    cmd_compute_rust()
    cmd_compute_fortran()
    cmd_compute_scipy()
    cmd_compute_julia()
    cmd_compute_f32()


# ── Bench ──


def cmd_bench_rust():
    print("Benchmarking Rust...", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    grid = RESULTS / "test_grid.json"
    out = RESULTS / "rust_bench.json"
    with open(grid) as fin, open(out, "w") as fout:
        run_cmd(
            [str(ROOT / "target" / "release" / "bench")],
            stdin=fin,
            stdout=fout,
            stderr=subprocess.DEVNULL,
        )


def cmd_bench_fortran():
    print("Benchmarking Fortran...", flush=True)
    run_python_script("compute_fortran.py", "--mode", "bench")


def cmd_bench_scipy():
    print("Benchmarking SciPy...", flush=True)
    upgrade_scipy()
    run_python_script("compute_scipy.py", "--mode", "bench")


def cmd_bench_julia():
    print("Benchmarking Julia...", flush=True)
    upgrade_julia()
    run_julia("bench")


def cmd_bench():
    cmd_bench_rust()
    cmd_bench_fortran()
    cmd_bench_scipy()
    cmd_bench_julia()


# ── Edge test ──


def cmd_edge():
    """Extreme-region test vs Fortran (needs build; see docs/guide.md)."""
    print("Running edge test...", flush=True)
    run_python_script("edge.py")


# ── Implementation differences ──


def cmd_differences():
    """Implementation-differences table and chart (needs build; reuses cached references)."""
    print("Computing implementation differences...", flush=True)
    upgrade_scipy()
    upgrade_julia()
    run_python_script("differences.py")


# ── Compare & Dashboard ──


def cmd_compare():
    print("Comparing results...", flush=True)
    run_python_script("compare.py")


def cmd_dashboard():
    print("Generating dashboard...", flush=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    run_python_script("dashboard.py")


# ── All ──


def cmd_all():
    cmd_build()
    cmd_grid()
    cmd_compute()
    cmd_bench()
    cmd_compare()
    cmd_dashboard()
    cmd_differences()


def cmd_all_heavy():
    cmd_build()
    cmd_grid()
    cmd_compute_mpmath()
    cmd_compute()
    cmd_bench()
    cmd_compare()
    cmd_dashboard()
    cmd_differences()


# ── Clean ──


def cmd_publish():
    """Fold everything into the single root commit and force-push.

    This repo keeps exactly one commit: history has no value here and the
    tracked results/images are large, so each publish amends the root
    commit in place and force-pushes it. Local commits made during
    development are squashed into the root commit first.
    """
    print("Publishing (single-commit amend + force push)...", flush=True)
    root = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if len(root) != 1:
        sys.exit(f"expected exactly one root commit, found {len(root)}")
    run_cmd(["git", "reset", "--soft", root[0]], cwd=ROOT)
    run_cmd(["git", "add", "-A"], cwd=ROOT)
    run_cmd(["git", "commit", "--amend", "--no-edit"], cwd=ROOT)
    run_cmd(["git", "push", "--force-with-lease", "origin", "main"], cwd=ROOT)


def cmd_clean():
    print("Cleaning...", flush=True)
    for f in RESULTS.glob("*.json"):
        f.unlink()
        print(f"  rm {f}")
    for ext in ("*.svg", "*.pdf"):
        for f in IMAGES.glob(ext):
            f.unlink()
            print(f"  rm {f}")
    run_cmd(["cargo", "clean"], cwd=ROOT)
    for name in ["compute_grid", "bench_fortran", "edge64", "edge64_zri", "edge32"]:
        p = FORTRAN / name
        if p.exists():
            p.unlink()
            print(f"  rm {p}")
    if FORTRAN_BUILD.exists():
        shutil.rmtree(FORTRAN_BUILD)
        print(f"  rm -rf {FORTRAN_BUILD}")


# ── CLI ──

COMMANDS = {
    "build": cmd_build,
    "build-rust": cmd_build_rust,
    "build-fortran": cmd_build_fortran,
    "grid": cmd_grid,
    "compute": cmd_compute,
    "compute-rust": cmd_compute_rust,
    "compute-fortran": cmd_compute_fortran,
    "compute-scipy": cmd_compute_scipy,
    "compute-julia": cmd_compute_julia,
    "compute-mpmath": cmd_compute_mpmath,
    "compute-f32": cmd_compute_f32,
    "bench": cmd_bench,
    "bench-rust": cmd_bench_rust,
    "bench-fortran": cmd_bench_fortran,
    "bench-scipy": cmd_bench_scipy,
    "bench-julia": cmd_bench_julia,
    "compare": cmd_compare,
    "dashboard": cmd_dashboard,
    "edge": cmd_edge,
    "differences": cmd_differences,
    "all": cmd_all,
    "all-heavy": cmd_all_heavy,
    "publish": cmd_publish,
    "clean": cmd_clean,
}


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline orchestration for complex-bessel-test",
    )
    parser.add_argument(
        "commands",
        nargs="+",
        choices=list(COMMANDS.keys()),
        metavar="command",
        help=f"Commands to run: {', '.join(COMMANDS.keys())}",
    )
    args = parser.parse_args()

    for cmd_name in args.commands:
        print(f"\n=== {cmd_name} ===", flush=True)
        COMMANDS[cmd_name]()


if __name__ == "__main__":
    main()
