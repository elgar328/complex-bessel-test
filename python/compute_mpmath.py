"""Compute reference values using mpmath (50+ digit precision).

Uses multiprocessing for parallelization.
"""

import json
import os
import multiprocessing

import math

import mpmath

mpmath.mp.dps = 50  # 50 decimal digits


def _extra_dps(func, nu, z_im):
    """Extra decimal digits needed to compensate for internal cancellation.

    For complex z with large |Im(z)|, mpmath's Bessel computations suffer
    catastrophic cancellation:

    - H1/H2: mpmath computes H = J ± iY.  |J|,|Y| ~ exp(|Im(z)|) but
      |H| ~ exp(-|Im(z)|), losing ~2|Im(z)|/ln(10) digits.
    - J/Y: for integer order, Y_n involves a limit with similar cancellation.
      Large order nu adds ~nu/ln(10) digits of cancellation.

    Other scaled functions (I, K, Ai, Bi) don't need extra dps because:
    - I series has all-positive terms (no cancellation)
    - K is computed directly for small values
    - Ai/Bi use specialized algorithms without cancellation
    """
    if func in ("besselj_scaled", "bessely_scaled", "hankel1_scaled", "hankel2_scaled"):
        # digits lost ≈ (2|Im(z)| + |nu|) / ln(10)
        return int((2 * abs(z_im) + abs(nu)) / math.log(10)) + 15
    return 0


def compute_one_bessel(args):
    """Compute a single Bessel function value.

    Args: (grid, function, idx, nu, z_re, z_im)
    Returns: dict with results
    """
    grid, func, idx, nu, z_re, z_im = args
    mpmath.mp.dps = 50 + _extra_dps(func, nu, z_im)

    z = mpmath.mpc(z_re, z_im)
    nu_mp = mpmath.mpf(nu)

    try:
        val = _compute_value(func, nu_mp, z)
        if val is None:
            return {
                "grid": grid,
                "function": func,
                "idx": idx,
                "nu": nu,
                "z_re": z_re,
                "z_im": z_im,
                "re": None,
                "im": None,
                "status": "error",
            }

        return {
            "grid": grid,
            "function": func,
            "idx": idx,
            "nu": nu,
            "z_re": z_re,
            "z_im": z_im,
            "re": mpmath.nstr(val.real, 20),
            "im": mpmath.nstr(val.imag, 20),
            "status": "ok",
        }
    except Exception as e:
        return {
            "grid": grid,
            "function": func,
            "idx": idx,
            "nu": nu,
            "z_re": z_re,
            "z_im": z_im,
            "re": None,
            "im": None,
            "status": f"error:{type(e).__name__}",
        }


def _compute_value(func, nu, z):
    """Dispatch to appropriate mpmath function."""

    # Unscaled functions
    if func == "besselj":
        return mpmath.besselj(nu, z)
    elif func == "bessely":
        return mpmath.bessely(nu, z)
    elif func == "besseli":
        return mpmath.besseli(nu, z)
    elif func == "besselk":
        return mpmath.besselk(nu, z)
    elif func == "hankel1":
        return mpmath.hankel1(nu, z)
    elif func == "hankel2":
        return mpmath.hankel2(nu, z)
    elif func == "airy":
        return mpmath.airyai(z)
    elif func == "airyprime":
        return mpmath.airyai(z, derivative=1)
    elif func == "biry":
        return mpmath.airybi(z)
    elif func == "biryprime":
        return mpmath.airybi(z, derivative=1)

    # Scaled functions: compute unscaled then apply scaling factor
    elif func == "besselj_scaled":
        val = mpmath.besselj(nu, z)
        return val * mpmath.exp(-abs(z.imag))
    elif func == "bessely_scaled":
        val = mpmath.bessely(nu, z)
        return val * mpmath.exp(-abs(z.imag))
    elif func == "besseli_scaled":
        val = mpmath.besseli(nu, z)
        return val * mpmath.exp(-abs(z.real))
    elif func == "besselk_scaled":
        val = mpmath.besselk(nu, z)
        return val * mpmath.exp(z)
    elif func == "hankel1_scaled":
        val = mpmath.hankel1(nu, z)
        return val * mpmath.exp(-1j * z)
    elif func == "hankel2_scaled":
        val = mpmath.hankel2(nu, z)
        return val * mpmath.exp(1j * z)
    elif func == "airy_scaled":
        val = mpmath.airyai(z)
        zeta = mpmath.mpf(2) / 3 * z * mpmath.sqrt(z)
        return val * mpmath.exp(zeta)
    elif func == "airyprime_scaled":
        val = mpmath.airyai(z, derivative=1)
        zeta = mpmath.mpf(2) / 3 * z * mpmath.sqrt(z)
        return val * mpmath.exp(zeta)
    elif func == "biry_scaled":
        val = mpmath.airybi(z)
        zeta = mpmath.mpf(2) / 3 * z * mpmath.sqrt(z)
        return val * mpmath.exp(-abs(zeta.real))
    elif func == "biryprime_scaled":
        val = mpmath.airybi(z, derivative=1)
        zeta = mpmath.mpf(2) / 3 * z * mpmath.sqrt(z)
        return val * mpmath.exp(-abs(zeta.real))

    return None


def is_airy(func):
    return func in {
        "airy",
        "airyprime",
        "biry",
        "biryprime",
        "airy_scaled",
        "airyprime_scaled",
        "biry_scaled",
        "biryprime_scaled",
    }


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    grid_path = os.path.join(results_dir, "test_grid.json")

    with open(grid_path) as f:
        grid = json.load(f)

    tasks = []

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
            if is_airy(func):
                # Airy: only unique z points, use nu=0 placeholder
                seen_z = set()
                for ni, nu in enumerate(nu_values):
                    for ri, re in enumerate(re_values):
                        for ii, im in enumerate(im_values):
                            idx = ni * n_re * n_im + ri * n_im + ii
                            z_key = (re, im)
                            if z_key not in seen_z:
                                seen_z.add(z_key)
                                tasks.append((grid_name, func, idx, 0.0, re, im))
            else:
                for ni, nu in enumerate(nu_values):
                    for ri, re in enumerate(re_values):
                        for ii, im in enumerate(im_values):
                            idx = ni * n_re * n_im + ri * n_im + ii
                            tasks.append((grid_name, func, idx, nu, re, im))

    # Shuffle so heavy/light tasks are interleaved, giving accurate ETA.
    # (imap_unordered + chunksize=1 ensures good load balancing regardless.)
    import random

    random.shuffle(tasks)

    total = len(tasks)
    n_heavy = sum(1 for t in tasks if _extra_dps(t[1], t[3], t[5]) > 0)
    ncpu = os.cpu_count() or 4
    print(
        f"Computing {total} mpmath tasks with {ncpu} cores "
        f"({n_heavy} need extra dps)..."
    )

    import time

    results = []
    t0 = time.time()
    last_print = t0
    with multiprocessing.Pool(ncpu) as pool:
        for i, r in enumerate(pool.imap_unordered(compute_one_bessel, tasks), 1):
            results.append(r)
            now = time.time()
            if now - last_print >= 60 or i == total:
                elapsed = now - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(
                    f"  {i}/{total} ({100 * i / total:.1f}%) "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )
                last_print = now

    # For Airy functions, expand results to all nu values
    expanded = []
    airy_cache = {}  # (grid, func, re, im) -> result

    for r in results:
        func = r["function"]
        if is_airy(func):
            key = (r["grid"], func, r["z_re"], r["z_im"])
            airy_cache[key] = r
        else:
            expanded.append(r)

    # Now expand Airy results for all nu values
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
            if not is_airy(func):
                continue
            for ni, nu in enumerate(nu_values):
                for ri, re in enumerate(re_values):
                    for ii, im in enumerate(im_values):
                        idx = ni * n_re * n_im + ri * n_im + ii
                        key = (grid_name, func, re, im)
                        base = airy_cache.get(key)
                        if base:
                            expanded.append(
                                {
                                    "grid": grid_name,
                                    "function": func,
                                    "idx": idx,
                                    "nu": nu,
                                    "z_re": re,
                                    "z_im": im,
                                    "re": base["re"],
                                    "im": base["im"],
                                    "status": base["status"],
                                }
                            )

    out_path = os.path.join(results_dir, "mpmath_results.json")
    with open(out_path, "w") as f:
        json.dump(expanded, f)

    ok_count = sum(1 for r in expanded if r["status"] == "ok")
    print(f"mpmath done: {ok_count}/{len(expanded)} ok, saved to {out_path}")


if __name__ == "__main__":
    main()
