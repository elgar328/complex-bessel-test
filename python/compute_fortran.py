"""Run the Fortran compute_grid binary with grid data and collect results.

Generates text input from test_grid.json, pipes to Fortran binary,
parses output, and saves as fortran_results.json.
Also generates bench input and runs bench_fortran.
"""

import argparse
import json
import os
import re
import subprocess
import sys

# Map function names to Fortran IDs
FUNC_ID = {
    "besselj": 1,
    "bessely": 2,
    "besseli": 3,
    "besselk": 4,
    "hankel1": 5,
    "hankel2": 6,
    "airy": 7,
    "airyprime": 8,
    "biry": 9,
    "biryprime": 10,
    "besselj_scaled": 1,
    "bessely_scaled": 2,
    "besseli_scaled": 3,
    "besselk_scaled": 4,
    "hankel1_scaled": 5,
    "hankel2_scaled": 6,
    "airy_scaled": 7,
    "airyprime_scaled": 8,
    "biry_scaled": 9,
    "biryprime_scaled": 10,
}

AIRY_FUNCS = {
    "airy",
    "airyprime",
    "biry",
    "biryprime",
    "airy_scaled",
    "airyprime_scaled",
    "biry_scaled",
    "biryprime_scaled",
}

# Fortran IERR -> status mapping
IERR_STATUS = {
    0: "ok",
    1: "invalid_input",
    2: "overflow",
    3: "reduced_precision",
    4: "total_precision_loss",
    5: "convergence_failure",
}


def generate_fortran_input(grid, grid_name, grid_key):
    """Generate Fortran input lines. Returns list of (line_str, metadata_dict)."""
    spec = grid[grid_key]
    nu_values = spec["nu_values"]
    re_values = spec["re_values"]
    im_values = spec["im_values"]
    functions = spec["functions"]
    n_re = len(re_values)
    n_im = len(im_values)

    kode = 1 if "unscaled" in grid_key else 2

    lines = []
    meta = []

    for func in functions:
        fid = FUNC_ID[func]
        is_airy_fn = func in AIRY_FUNCS

        # Fortran only supports nu >= 0
        airy_seen_z = set()

        for ni, nu in enumerate(nu_values):
            if nu < 0:
                continue  # Skip negative nu for Fortran

            for ri, re_val in enumerate(re_values):
                for ii, im_val in enumerate(im_values):
                    idx = ni * n_re * n_im + ri * n_im + ii

                    if is_airy_fn:
                        z_key = (re_val, im_val)
                        if z_key in airy_seen_z:
                            continue
                        airy_seen_z.add(z_key)
                        # Airy: nu is ignored by Fortran, pass 0
                        line = f"{fid} 0.0 {re_val} {im_val} {kode}"
                    else:
                        line = f"{fid} {nu} {re_val} {im_val} {kode}"

                    lines.append(line)
                    meta.append(
                        {
                            "grid": grid_name,
                            "function": func,
                            "idx": idx,
                            "nu": nu,
                            "z_re": re_val,
                            "z_im": im_val,
                        }
                    )

    return lines, meta


def fix_fortran_float(s):
    """Fix gfortran output that drops 'E' for 3-digit exponents.

    e.g. '-0.7869812510210358-104' -> '-0.7869812510210358E-104'
    """
    # Match: digits followed by +/- and 3+ digit exponent, without E/D
    fixed = re.sub(r"(\d)([+-])(\d{3,})$", r"\1E\2\3", s)
    return float(fixed)


def parse_fortran_output(output_lines):
    """Parse Fortran output lines into result dicts."""
    results = []
    for line in output_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        fid = int(parts[0])
        fnu = fix_fortran_float(parts[1])
        z_re = fix_fortran_float(parts[2])
        z_im = fix_fortran_float(parts[3])
        kode = int(parts[4])
        res_re = fix_fortran_float(parts[5])
        res_im = fix_fortran_float(parts[6])
        ierr = int(parts[7])
        results.append(
            {
                "fid": fid,
                "fnu": fnu,
                "z_re": z_re,
                "z_im": z_im,
                "kode": kode,
                "re": res_re,
                "im": res_im,
                "ierr": ierr,
            }
        )
    return results


def run_compute(grid, results_dir, fortran_dir):
    """Run Fortran compute_grid and save results."""
    print("Generating Fortran input...", flush=True)
    all_lines = []
    all_meta = []

    for grid_name, grid_key in [
        ("unscaled", "unscaled_grid"),
        ("scaled", "scaled_grid"),
    ]:
        lines, meta = generate_fortran_input(grid, grid_name, grid_key)
        all_lines.extend(lines)
        all_meta.extend(meta)

    print(f"  {len(all_lines)} Fortran calls", flush=True)

    input_text = "\n".join(all_lines) + "\n"

    compute_bin = os.path.join(fortran_dir, "compute_grid")
    print("Running Fortran compute_grid...", flush=True)
    proc = subprocess.run(
        [compute_bin],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        print(f"Fortran error: {proc.stderr}", file=sys.stderr)
        sys.exit(1)

    output_lines = proc.stdout.strip().split("\n")
    parsed = parse_fortran_output(output_lines)

    if len(parsed) != len(all_meta):
        print(
            f"WARNING: expected {len(all_meta)} results, got {len(parsed)}",
            file=sys.stderr,
        )

    # Build result records
    fortran_results = []
    # Airy cache for expansion
    airy_cache = {}  # (grid, func, re, im) -> parsed result

    for i, (m, p) in enumerate(zip(all_meta, parsed)):
        func = m["function"]
        is_airy_fn = func in AIRY_FUNCS

        record = {
            "grid": m["grid"],
            "function": func,
            "idx": m["idx"],
            "nu": m["nu"],
            "z_re": m["z_re"],
            "z_im": m["z_im"],
            "re": p["re"] if p["ierr"] == 0 else None,
            "im": p["im"] if p["ierr"] == 0 else None,
            "ierr": p["ierr"],
        }

        if is_airy_fn:
            key = (m["grid"], func, m["z_re"], m["z_im"])
            airy_cache[key] = record
        else:
            fortran_results.append(record)

    # Expand Airy results for all (non-negative) nu values
    for grid_name, grid_key in [
        ("unscaled", "unscaled_grid"),
        ("scaled", "scaled_grid"),
    ]:
        spec = grid[grid_key]
        nu_values = spec["nu_values"]
        re_values = spec["re_values"]
        im_values = spec["im_values"]
        functions = spec["functions"]
        n_re = len(re_values)
        n_im = len(im_values)

        for func in functions:
            if func not in AIRY_FUNCS:
                continue
            for ni, nu in enumerate(nu_values):
                if nu < 0:
                    continue
                for ri, re_val in enumerate(re_values):
                    for ii, im_val in enumerate(im_values):
                        idx = ni * n_re * n_im + ri * n_im + ii
                        key = (grid_name, func, re_val, im_val)
                        base = airy_cache.get(key)
                        if base:
                            fortran_results.append(
                                {
                                    "grid": grid_name,
                                    "function": func,
                                    "idx": idx,
                                    "nu": nu,
                                    "z_re": re_val,
                                    "z_im": im_val,
                                    "re": base["re"],
                                    "im": base["im"],
                                    "ierr": base["ierr"],
                                }
                            )

    out_path = os.path.join(results_dir, "fortran_results.json")
    with open(out_path, "w") as f:
        json.dump(fortran_results, f)

    ok_count = sum(1 for r in fortran_results if r["ierr"] == 0)
    print(f"Fortran compute done: {ok_count}/{len(fortran_results)} ok")


def run_bench(grid, results_dir, fortran_dir):
    """Run Fortran benchmark and save results."""
    print("Generating Fortran bench input...", flush=True)
    bench_lines = []
    bench_meta = []

    for grid_name, grid_key in [
        ("unscaled", "unscaled_grid"),
        ("scaled", "scaled_grid"),
    ]:
        spec = grid[grid_key]
        nu_values = spec["nu_values"]
        re_values = spec["re_values"]
        im_values = spec["im_values"]
        functions = spec["functions"]
        kode = 1 if "unscaled" in grid_key else 2

        for func in functions:
            fid = FUNC_ID[func]
            for nu in nu_values:
                if nu < 0:
                    continue
                for re_val in re_values:
                    for im_val in im_values:
                        if func in AIRY_FUNCS:
                            line = f"{fid} 0.0 {re_val} {im_val} {kode}"
                        else:
                            line = f"{fid} {nu} {re_val} {im_val} {kode}"
                        bench_lines.append(line)
                        bench_meta.append(
                            {
                                "grid": grid_name,
                                "function": func,
                                "nu": nu,
                                "z_re": re_val,
                                "z_im": im_val,
                            }
                        )

    bench_input = "\n".join(bench_lines) + "\n"
    bench_bin = os.path.join(fortran_dir, "bench_fortran")

    print(f"Running Fortran bench ({len(bench_lines)} points)...", flush=True)
    proc = subprocess.run(
        [bench_bin],
        input=bench_input,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    bench_results = []
    if proc.returncode == 0:
        for line, m in zip(proc.stdout.strip().split("\n"), bench_meta):
            parts = line.strip().split()
            if len(parts) >= 5:
                time_ns = int(parts[4])
                bench_results.append(
                    {
                        "grid": m["grid"],
                        "function": m["function"],
                        "nu": m["nu"],
                        "z_re": m["z_re"],
                        "z_im": m["z_im"],
                        "time_ns": time_ns,
                    }
                )

    bench_path = os.path.join(results_dir, "fortran_bench.json")
    with open(bench_path, "w") as f:
        json.dump(bench_results, f)

    print(f"Fortran bench done: {len(bench_results)} points")


def main():
    parser = argparse.ArgumentParser(description="Fortran compute/benchmark")
    parser.add_argument("--mode", choices=["compute", "bench", "all"], default="all")
    args = parser.parse_args()

    project_dir = os.path.join(os.path.dirname(__file__), "..")
    results_dir = os.path.join(project_dir, "results")
    fortran_dir = os.path.join(project_dir, "fortran")
    grid_path = os.path.join(results_dir, "test_grid.json")

    with open(grid_path) as f:
        grid = json.load(f)

    if args.mode in ("compute", "all"):
        run_compute(grid, results_dir, fortran_dir)
    if args.mode in ("bench", "all"):
        run_bench(grid, results_dir, fortran_dir)


if __name__ == "__main__":
    main()
