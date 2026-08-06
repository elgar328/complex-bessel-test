"""Compute Bessel/Airy values using scipy and measure performance."""

import argparse
import json
import os
import time
import warnings

import numpy as np
from scipy import special

warnings.filterwarnings("ignore")


FUNC_MAP_UNSCALED = {
    "besselj": lambda nu, z: special.jv(nu, z),
    "bessely": lambda nu, z: special.yv(nu, z),
    "besseli": lambda nu, z: special.iv(nu, z),
    "besselk": lambda nu, z: special.kv(nu, z),
    "hankel1": lambda nu, z: special.hankel1(nu, z),
    "hankel2": lambda nu, z: special.hankel2(nu, z),
    "airy": lambda nu, z: special.airy(z)[0],
    "airyprime": lambda nu, z: special.airy(z)[1],
    "biry": lambda nu, z: special.airy(z)[2],
    "biryprime": lambda nu, z: special.airy(z)[3],
}

FUNC_MAP_SCALED = {
    "besselj_scaled": lambda nu, z: special.jve(nu, z),
    "bessely_scaled": lambda nu, z: special.yve(nu, z),
    "besseli_scaled": lambda nu, z: special.ive(nu, z),
    "besselk_scaled": lambda nu, z: special.kve(nu, z),
    "hankel1_scaled": lambda nu, z: special.hankel1e(nu, z),
    "hankel2_scaled": lambda nu, z: special.hankel2e(nu, z),
    "airy_scaled": lambda nu, z: special.airye(z)[0],
    "airyprime_scaled": lambda nu, z: special.airye(z)[1],
    "biry_scaled": lambda nu, z: special.airye(z)[2],
    "biryprime_scaled": lambda nu, z: special.airye(z)[3],
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


def compute_scipy_value(func, nu, z):
    """Compute a single scipy value. Returns (re, im, status)."""
    func_map = {**FUNC_MAP_UNSCALED, **FUNC_MAP_SCALED}
    fn = func_map.get(func)
    if fn is None:
        return None, None, "unknown_function"

    try:
        val = fn(nu, z)
        val = complex(val)
        if np.isinf(val.real) or np.isinf(val.imag):
            return None, None, "inf"
        if np.isnan(val.real) or np.isnan(val.imag):
            return None, None, "nan"
        return val.real, val.imag, "ok"
    except Exception:
        return None, None, "error"


def run_compute(grid, results_dir):
    """Compute scipy values and save results."""
    results = []

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

        # Pre-compute Airy cache
        airy_cache = {}  # (func, re_bits, im_bits) -> (re, im, status)

        for func in functions:
            is_airy_fn = func in AIRY_FUNCS
            print(f"  scipy: {grid_name}/{func}...", flush=True)

            for ni, nu in enumerate(nu_values):
                for ri, re_val in enumerate(re_values):
                    for ii, im_val in enumerate(im_values):
                        idx = ni * n_re * n_im + ri * n_im + ii
                        z = complex(re_val, im_val)

                        if is_airy_fn:
                            cache_key = (func, re_val, im_val)
                            if cache_key not in airy_cache:
                                res_re, res_im, status = compute_scipy_value(
                                    func, 0.0, z
                                )
                                airy_cache[cache_key] = (res_re, res_im, status)
                            res_re, res_im, status = airy_cache[cache_key]
                        else:
                            res_re, res_im, status = compute_scipy_value(func, nu, z)

                        results.append(
                            {
                                "grid": grid_name,
                                "function": func,
                                "idx": idx,
                                "nu": nu,
                                "z_re": re_val,
                                "z_im": im_val,
                                "re": res_re,
                                "im": res_im,
                                "status": status,
                            }
                        )

    out_path = os.path.join(results_dir, "scipy_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"scipy compute done: {ok_count}/{len(results)} ok")


def run_bench(grid, results_dir):
    """Run scipy benchmark and save results."""
    print("Benchmarking scipy...", flush=True)

    # Warmup
    for _ in range(100):
        special.jv(0.5, 1.0 + 2.0j)
        special.kv(0.5, 1.0 + 2.0j)

    bench_results = []

    for grid_name, grid_key in [
        ("unscaled", "unscaled_grid"),
        ("scaled", "scaled_grid"),
    ]:
        spec = grid[grid_key]
        nu_values = spec["nu_values"]
        re_values = spec["re_values"]
        im_values = spec["im_values"]
        functions = spec["functions"]

        func_map = {**FUNC_MAP_UNSCALED, **FUNC_MAP_SCALED}

        for func in functions:
            fn = func_map.get(func)
            if fn is None:
                continue
            for nu in nu_values:
                for re_val in re_values:
                    for im_val in im_values:
                        z = complex(re_val, im_val)
                        # Warmup
                        try:
                            fn(nu, z)
                        except Exception:
                            pass
                        # Time (1000 repetitions)
                        nrep = 1000
                        t0 = time.perf_counter_ns()
                        for _rep in range(nrep):
                            try:
                                fn(nu, z)
                            except Exception:
                                pass
                        t1 = time.perf_counter_ns()
                        bench_results.append(
                            {
                                "grid": grid_name,
                                "function": func,
                                "nu": nu,
                                "z_re": re_val,
                                "z_im": im_val,
                                "time_ns": (t1 - t0) // nrep,
                            }
                        )

    bench_path = os.path.join(results_dir, "scipy_bench.json")
    with open(bench_path, "w") as f:
        json.dump(bench_results, f)

    print(f"scipy bench done: {len(bench_results)} points")


def main():
    parser = argparse.ArgumentParser(description="SciPy compute/benchmark")
    parser.add_argument("--mode", choices=["compute", "bench", "all"], default="all")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    grid_path = os.path.join(results_dir, "test_grid.json")

    with open(grid_path) as f:
        grid = json.load(f)

    if args.mode in ("compute", "all"):
        run_compute(grid, results_dir)
    if args.mode in ("bench", "all"):
        run_bench(grid, results_dir)


if __name__ == "__main__":
    main()
