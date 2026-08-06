"""Compare Rust, Fortran, SciPy and Julia results against mpmath ground truth.

Produces comparison.json with accuracy statistics and error distributions.
"""

import json
import math
import os
import struct
from collections import defaultdict

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

NEGATIVE_NU_VALUES = {-0.5, -1.5, -2.0}

# Implementations compared against mpmath, in display order
IMPLS = ["rust", "fortran", "scipy", "julia"]

# Functions that display as short names in results
DISPLAY_NAMES = {
    "besselj": "J",
    "bessely": "Y",
    "besseli": "I",
    "besselk": "K",
    "hankel1": "H1",
    "hankel2": "H2",
    "airy": "Ai",
    "airyprime": "Ai'",
    "biry": "Bi",
    "biryprime": "Bi'",
    "besselj_scaled": "J_s",
    "bessely_scaled": "Y_s",
    "besseli_scaled": "I_s",
    "besselk_scaled": "K_s",
    "hankel1_scaled": "H1_s",
    "hankel2_scaled": "H2_s",
    "airy_scaled": "Ai_s",
    "airyprime_scaled": "Ai'_s",
    "biry_scaled": "Bi_s",
    "biryprime_scaled": "Bi'_s",
}


# Below about tiny/eps, Amos flushes results to zero or computes them with
# reduced precision near underflow, so relative error is meaningless there
# (the f64 counterpart of F32_FLOOR below). Smaller results above it are
# compared like any other.
F64_FLOOR = 1e-290

# Grid points where the function is exactly zero, so relative error is
# undefined: K_{±3/2}(z) ∝ 1 + 1/z vanishes at z = -1, H(1)_{±3/2}(z) ∝ 1 + i/z
# at z = -i, and H(2)_{±3/2}(z) ∝ 1 - i/z at z = i. Keyed by (function without
# "_scaled", nu, z_re, z_im).
ANALYTIC_ZEROS = {
    ("besselk", 1.5, -1.0, 0.0),
    ("besselk", -1.5, -1.0, 0.0),
    ("hankel1", 1.5, 0.0, -1.0),
    ("hankel1", -1.5, 0.0, -1.0),
    ("hankel2", 1.5, 0.0, 1.0),
    ("hankel2", -1.5, 0.0, 1.0),
}


def is_analytic_zero(key):
    """True if the (grid, function, nu, z_re, z_im) key is in ANALYTIC_ZEROS."""
    _, func, nu, z_re, z_im = key
    return (func.removesuffix("_scaled"), nu, z_re, z_im) in ANALYTIC_ZEROS


# complex-bessel deliberately differs from ZBESY for scaled Y when
# 2|Im z| >= ELIM: ZBESY drops the K term there as an underflow although it
# can dominate (see the complex-bessel CHANGELOG). Where the two differ by more
# than DEVIATION_TOL in that region, the point is counted as intentional
# instead of entering the Rust-vs-Fortran distribution; Rust's accuracy there
# is still measured against mpmath.
F64_ELIM = 700.92
DEVIATION_TOL = 1e-8


def in_deviation_region(key):
    """True if Rust deliberately differs from Fortran at this key."""
    _, func, _, _, z_im = key
    return func == "bessely_scaled" and 2 * abs(z_im) >= F64_ELIM


def relative_error(computed_re, computed_im, ref_re, ref_im):
    """Compute relative error between computed and reference complex values.

    Returns None when both the reference and the difference are below
    F64_FLOOR (comparison meaningless). math.hypot is used to avoid overflow
    on large intermediate values.
    """
    diff_re = computed_re - ref_re
    diff_im = computed_im - ref_im
    diff_mag = math.hypot(diff_re, diff_im)

    ref_mag = math.hypot(ref_re, ref_im)

    if ref_mag < F64_FLOOR:
        # Underflow region. A difference above the floor is a real problem,
        # reported as an absolute error.
        return None if diff_mag < F64_FLOOR else diff_mag

    return diff_mag / ref_mag


def load_results(path):
    """Load a results JSON file. Returns list of dicts."""
    with open(path) as f:
        return json.load(f)


def index_by_key(records, include_negative_nu=True):
    """Build dict keyed by (grid, function, nu, z_re, z_im) -> record."""
    idx = {}
    for r in records:
        nu = r["nu"]
        if not include_negative_nu and nu in NEGATIVE_NU_VALUES:
            continue
        key = (r["grid"], r["function"], r["nu"], r["z_re"], r["z_im"])
        idx[key] = r
    return idx


# ── Single precision (f32) ──
#
# Summary only (no distributions, no charts). Only grid points whose ν, Re z,
# and Im z are exact in f32 are used, so the error is that of the algorithm,
# not of rounding the input. References below F32_FLOOR are skipped: there
# Amos's routines may legitimately return zero (underflow handling starts
# near exp(-ALIM) ≈ 2e-28 in single precision).

F32_FLOOR = 1e-28

RUST_IERR = {
    "ok": 0,
    "invalid_input": 1,
    "overflow": 2,
    "reduced_precision": 3,
    "total_precision_loss": 4,
    "convergence_failure": 5,
}


def f32_exact(x):
    return struct.unpack("f", struct.pack("f", x))[0] == x


def relative_error_f32(c_re, c_im, ref_re, ref_im):
    """Relative error, or None if the reference is below F32_FLOOR."""
    ref_mag = math.hypot(ref_re, ref_im)
    if ref_mag < F32_FLOOR:
        return None
    return math.hypot(c_re - ref_re, c_im - ref_im) / ref_mag


def summarize_f32(entries):
    """entries: list of (err, record) -> summary with the worst point."""
    if not entries:
        return {"count": 0}
    entries = sorted(entries, key=lambda e: e[0])
    errs = [e[0] for e in entries]
    n = len(errs)
    worst = entries[-1][1]
    return {
        "count": n,
        "median_err": errs[n // 2],
        "p99_err": errs[int(n * 0.99)],
        "p999_err": errs[int(n * 0.999)],
        "max_err": errs[-1],
        "max_at": {k: worst[k] for k in ("nu", "z_re", "z_im")},
    }


def compare_f32(results_dir, mpmath_idx):
    """Rust f32 and Fortran CBSUBS vs mpmath, and Rust f32 vs CBSUBS status."""
    rust_path = os.path.join(results_dir, "rust_f32_results.json")
    fortran_path = os.path.join(results_dir, "fortran_f32_results.json")
    if not (os.path.exists(rust_path) and os.path.exists(fortran_path)):
        print("f32 results not found; skipping f32 report", flush=True)
        return None

    rust = load_results(rust_path)
    fortran_idx = index_by_key(load_results(fortran_path))

    def reference(key):
        mp = mpmath_idx.get(key)
        if not mp or mp["status"] != "ok" or mp["re"] is None:
            return None
        return float(mp["re"]), float(mp["im"])

    rust_all = defaultdict(list)  # every exact point, including ν < 0
    rust_common = defaultdict(list)  # ν >= 0, both implementations ok
    fortran_common = defaultdict(list)
    status = defaultdict(lambda: {"match": 0, "mismatch": 0, "examples": []})
    skipped = defaultdict(lambda: {"inexact_input": 0, "below_floor": 0})

    for r in rust:
        key = (r["grid"], r["function"], r["nu"], r["z_re"], r["z_im"])
        func = r["function"]
        fr = fortran_idx.get(key)

        if fr is not None:
            rust_ierr = RUST_IERR.get(r["status"], -1)
            st = status[func]
            if rust_ierr == fr["ierr"]:
                st["match"] += 1
            else:
                st["mismatch"] += 1
                if len(st["examples"]) < 20:
                    st["examples"].append(
                        {
                            "nu": r["nu"],
                            "z_re": r["z_re"],
                            "z_im": r["z_im"],
                            "rust_status": r["status"],
                            "fortran_ierr": fr["ierr"],
                        }
                    )

        if not all(f32_exact(float(r[k])) for k in ("nu", "z_re", "z_im")):
            skipped[func]["inexact_input"] += 1
            continue
        if r["re"] is None:
            continue
        ref = reference(key)
        if ref is None:
            continue
        err = relative_error_f32(r["re"], r["im"], *ref)
        if err is None:
            skipped[func]["below_floor"] += 1
            continue
        rust_all[func].append((err, r))

        if fr is not None and fr["re"] is not None:
            f_err = relative_error_f32(fr["re"], fr["im"], *ref)
            rust_common[func].append((err, r))
            fortran_common[func].append((f_err, fr))

    funcs = sorted(set(rust_all) | set(status))
    return {
        "note": (
            "Points with nu, Re z, Im z exact in f32; references below "
            f"{F32_FLOOR:g} skipped. rust: all such points (incl. nu < 0); "
            "rust_common/fortran_common: nu >= 0 where both succeed."
        ),
        "functions": {
            func: {
                "rust": summarize_f32(rust_all[func]),
                "rust_common": summarize_f32(rust_common[func]),
                "fortran_common": summarize_f32(fortran_common[func]),
                "status_vs_fortran": status[func],
                "skipped": skipped[func],
            }
            for func in funcs
        },
    }


def print_f32_summary(f32):
    print("\n=== f32 accuracy vs mpmath (exact-f32 inputs) ===")
    print(
        f"  {'':6s}  {'rust median':>11s} {'p99':>9s} {'max':>9s}"
        f"  | {'common: rust max':>16s} {'CBSUBS max':>10s}  | status mismatch"
    )
    for func, d in f32["functions"].items():
        r, rc, fc = d["rust"], d["rust_common"], d["fortran_common"]
        if not r.get("count"):
            continue
        dname = DISPLAY_NAMES.get(func, func)
        rc_max = f"{rc['max_err']:.2e}" if rc.get("count") else "N/A"
        fc_max = f"{fc['max_err']:.2e}" if fc.get("count") else "N/A"
        print(
            f"  {dname:6s}  {r['median_err']:11.2e} {r['p99_err']:9.2e} {r['max_err']:9.2e}"
            f"  | {rc_max:>16s} {fc_max:>10s}  | {d['status_vs_fortran']['mismatch']}"
        )


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")

    print("Loading results...", flush=True)
    rust = load_results(os.path.join(results_dir, "rust_results.json"))
    fortran = load_results(os.path.join(results_dir, "fortran_results.json"))
    mpmath_data = load_results(os.path.join(results_dir, "mpmath_results.json"))
    scipy_data = load_results(os.path.join(results_dir, "scipy_results.json"))
    julia_data = load_results(os.path.join(results_dir, "julia_results.json"))

    # Index mpmath by key
    mpmath_idx = index_by_key(mpmath_data)

    # Index Fortran by key (nu >= 0 only)
    fortran_idx = index_by_key(fortran, include_negative_nu=False)

    # ── 1. Accuracy vs mpmath ──
    print("Computing accuracy vs mpmath...", flush=True)
    accuracy = defaultdict(lambda: {impl: [] for impl in IMPLS})
    # Points left out by relative_error (underflow) or ANALYTIC_ZEROS
    skipped = defaultdict(lambda: {impl: 0 for impl in IMPLS})

    # Rust vs mpmath
    for r in rust:
        key = (r["grid"], r["function"], r["nu"], r["z_re"], r["z_im"])
        if r["status"] not in ("ok", "reduced_precision") or r["re"] is None:
            continue
        mp = mpmath_idx.get(key)
        if not mp or mp["status"] != "ok" or mp["re"] is None:
            continue
        ref_re = float(mp["re"])
        ref_im = float(mp["im"])
        err = (
            None
            if is_analytic_zero(key)
            else relative_error(r["re"], r["im"], ref_re, ref_im)
        )
        if err is None:
            skipped[r["function"]]["rust"] += 1
        else:
            accuracy[r["function"]]["rust"].append(err)

    # Fortran vs mpmath (nu >= 0 only)
    for r in fortran:
        key = (r["grid"], r["function"], r["nu"], r["z_re"], r["z_im"])
        if r["ierr"] not in (0, 3) or r["re"] is None:
            continue
        mp = mpmath_idx.get(key)
        if not mp or mp["status"] != "ok" or mp["re"] is None:
            continue
        ref_re = float(mp["re"])
        ref_im = float(mp["im"])
        err = (
            None
            if is_analytic_zero(key)
            else relative_error(r["re"], r["im"], ref_re, ref_im)
        )
        if err is None:
            skipped[r["function"]]["fortran"] += 1
        else:
            accuracy[r["function"]]["fortran"].append(err)

    # SciPy and Julia vs mpmath (same record format)
    for impl_name, data in (("scipy", scipy_data), ("julia", julia_data)):
        for r in data:
            key = (r["grid"], r["function"], r["nu"], r["z_re"], r["z_im"])
            if r["status"] != "ok" or r["re"] is None:
                continue
            mp = mpmath_idx.get(key)
            if not mp or mp["status"] != "ok" or mp["re"] is None:
                continue
            ref_re = float(mp["re"])
            ref_im = float(mp["im"])
            err = (
                None
                if is_analytic_zero(key)
                else relative_error(r["re"], r["im"], ref_re, ref_im)
            )
            if err is None:
                skipped[r["function"]][impl_name] += 1
            else:
                accuracy[r["function"]][impl_name].append(err)

    # Summarize accuracy
    accuracy_summary = {}
    for func, data in accuracy.items():
        accuracy_summary[func] = {}
        for impl_name, errors in data.items():
            if errors:
                accuracy_summary[func][impl_name] = {
                    "count": len(errors),
                    "max_err": max(errors),
                    "median_err": sorted(errors)[len(errors) // 2],
                    "mean_err": sum(errors) / len(errors),
                    "p99_err": sorted(errors)[int(len(errors) * 0.99)],
                    "skipped": skipped[func][impl_name],
                }
            else:
                accuracy_summary[func][impl_name] = {
                    "count": 0,
                    "max_err": None,
                    "median_err": None,
                    "mean_err": None,
                    "p99_err": None,
                    "skipped": skipped[func][impl_name],
                }

    # ── 2. Rust vs Fortran direct comparison (nu >= 0 only) ──
    print("Computing Rust vs Fortran...", flush=True)
    rust_vs_fortran = defaultdict(
        lambda: {
            "errors": [],
            "skipped": 0,
            "intentional": 0,
            "status_match": 0,
            "status_mismatch": 0,
            "mismatches": [],
        }
    )

    rust_idx = index_by_key(rust, include_negative_nu=False)

    for key, fr in fortran_idx.items():
        rr = rust_idx.get(key)
        if rr is None:
            continue

        func = key[1]

        # Status comparison
        rust_ok = rr["status"] in ("ok", "reduced_precision")
        fortran_ok = fr["ierr"] in (0, 3)

        # Map Rust status to Fortran IERR equivalent
        rust_ierr_equiv = {
            "ok": 0,
            "invalid_input": 1,
            "overflow": 2,
            "reduced_precision": 3,
            "total_precision_loss": 4,
            "convergence_failure": 5,
        }.get(rr["status"], -1)

        if rust_ierr_equiv == fr["ierr"]:
            rust_vs_fortran[func]["status_match"] += 1
        else:
            rust_vs_fortran[func]["status_mismatch"] += 1
            if len(rust_vs_fortran[func]["mismatches"]) < 20:
                rust_vs_fortran[func]["mismatches"].append(
                    {
                        "nu": key[2],
                        "z_re": key[3],
                        "z_im": key[4],
                        "rust_status": rr["status"],
                        "fortran_ierr": fr["ierr"],
                    }
                )

        # Value comparison (only when both ok)
        if rust_ok and fortran_ok and rr["re"] is not None and fr["re"] is not None:
            err = (
                None
                if is_analytic_zero(key)
                else relative_error(rr["re"], rr["im"], fr["re"], fr["im"])
            )
            if err is None:
                rust_vs_fortran[func]["skipped"] += 1
            elif err > DEVIATION_TOL and in_deviation_region(key):
                rust_vs_fortran[func]["intentional"] += 1
            else:
                rust_vs_fortran[func]["errors"].append(err)

    # Summarize Rust vs Fortran
    rvf_summary = {}
    for func, data in rust_vs_fortran.items():
        errors = data["errors"]
        rvf_summary[func] = {
            "count": len(errors),
            "max_err": max(errors) if errors else None,
            "median_err": sorted(errors)[len(errors) // 2] if errors else None,
            "skipped": data["skipped"],
            "intentional": data["intentional"],
            "status_match": data["status_match"],
            "status_mismatch": data["status_mismatch"],
            "mismatches": data["mismatches"],
        }

    # ── 3. Error summary ──
    print("Computing error summary...", flush=True)
    error_summary = defaultdict(lambda: defaultdict(int))

    for r in rust:
        error_summary[r["function"]]["total"] += 1
        error_summary[r["function"]][r["status"]] += 1

    # ── Save ──
    # Save all error distributions for dashboard
    error_distributions = {"accuracy_vs_mpmath": {}}

    for func, data in accuracy.items():
        error_distributions["accuracy_vs_mpmath"][func] = {}
        for impl_name, errors in data.items():
            # Filter out inf/nan for violin plot rendering, clamp zeros to 1e-17
            finite = [e for e in errors if math.isfinite(e)]
            clamped = [max(e, 1e-17) for e in finite]
            error_distributions["accuracy_vs_mpmath"][func][impl_name] = clamped

    # ── 4. Single precision ──
    print("Computing f32 report...", flush=True)
    f32 = compare_f32(results_dir, mpmath_idx)

    comparison = {
        "accuracy_vs_mpmath": accuracy_summary,
        "rust_vs_fortran": rvf_summary,
        "error_summary": dict(error_summary),
        "error_distributions": error_distributions,
    }
    if f32 is not None:
        comparison["f32"] = f32

    out_path = os.path.join(results_dir, "comparison.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nComparison saved to {out_path}")

    # Print summary
    print("\n=== Accuracy vs mpmath (max / p99 relative error) ===")
    for func in sorted(accuracy_summary.keys()):
        data = accuracy_summary[func]
        parts = []
        for impl_name in IMPLS:
            d = data.get(impl_name, {})
            if d.get("max_err") is not None:
                parts.append(
                    f"{impl_name}: max={d['max_err']:.2e} p99={d['p99_err']:.2e} (n={d['count']})"
                )
        dname = DISPLAY_NAMES.get(func, func)
        print(f"  {dname:6s}: {', '.join(parts)}")

    print("\n=== Rust vs Fortran (max relative error, nu >= 0 only) ===")
    for func in sorted(rvf_summary.keys()):
        d = rvf_summary[func]
        dname = DISPLAY_NAMES.get(func, func)
        max_e = f"{d['max_err']:.2e}" if d["max_err"] is not None else "N/A"
        print(
            f"  {dname:6s}: max={max_e}, status_match={d['status_match']}, mismatch={d['status_mismatch']}"
        )

    print("\n=== Error summary (Rust) ===")
    for func in sorted(error_summary.keys()):
        counts = error_summary[func]
        dname = DISPLAY_NAMES.get(func, func)
        ok = counts.get("ok", 0)
        total = counts.get("total", 0)
        overflow = counts.get("overflow", 0)
        invalid = counts.get("invalid_input", 0)
        print(f"  {dname:6s}: {ok}/{total} ok, {overflow} overflow, {invalid} invalid")

    if f32 is not None:
        print_f32_summary(f32)


if __name__ == "__main__":
    main()
