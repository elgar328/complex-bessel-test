"""Edge-case test: complex-bessel vs TOMS 644 Fortran in extreme regions.

Complements the accuracy grid (which covers ordinary arguments) with the
regions where port bugs have actually hidden: overflow/underflow thresholds,
long `_seq` runs across the uniform-expansion switch (FNUL), large negative
orders, the ZUOIK marginal-underflow window, power-series underflow retries,
scaled Y beyond 2|Im z| = ELIM, long `_seq_into` runs across its stack-chunk
boundaries, and Airy thresholds. Both f64 (vs ZBSUBS) and f32 (vs CBSUBS) are
checked.

Compared per case: status (Fortran IERR), underflow count (NZ), and every
element of `_seq` (or `_seq_into` in the `seq_into` region) (relative
difference). Negative orders have no Fortran counterpart: their reference is
Fortran single values (N = 1) per order plus the reflection formulas, with NZ
the number of zero elements. No mpmath: the question here is "does the port
behave like the original", not absolute accuracy.

Each disagreement is classified:
  intentional_zuoik  Rust matches ZBSUBS with ZUOIK's ZN test patched to use
                     ZRI (as single-precision CUOIK does); complex-bessel
                     follows CUOIK deliberately.
  intentional_zbesy  scaled Y (of negative non-half-integer order too, and J of
                     negative non-integer order, which uses it) with
                     2|Im z| >= ELIM: ZBESY drops its K term there as an
                     underflow, which is wrong for large nu; complex-bessel
                     keeps it. Recognised when Rust matches exp(-|Im z|) times
                     the unscaled double-precision ZBESY.
  intentional_chunk  `seq_into` only: the chunked `_seq_into` succeeds where
                     the whole Fortran call fails, and it matches Fortran
                     single values at every order. Amos's overflow tests use
                     the last order and length of a call (K: max(fnu+n-1, n)),
                     so the shorter chunk calls are more lenient.
  intentional_real   J or Y on the positive real axis: complex-bessel returns
                     them exactly real, Fortran with an imaginary rounding
                     residue, which matters next to a zero of the function.
                     Recognised when Rust matches Fortran with that residue
                     dropped.
  intentional_axis   H, K or Y with z exactly on an axis: complex-bessel
                     recomputes a part that is much smaller than the other
                     from two real functions (e.g. Re Y_nu(-x) = cos(nu pi)
                     Y_nu(x) next to half-integer orders), where Amos has
                     only a normwise accurate rounding residue. Recognised
                     when Rust matches Fortran with the smaller part of each
                     element taken from Rust, where that part is below 0.1
                     times the larger one and differs by at most twice the
                     tolerance (relative to the value).
  axis_noise         z exactly on an axis and only the underflow decision
                     (NZ / zero-vs-value) differs: ZUCHK then compares a
                     component that is pure rounding noise.
  call_dependent     `seq_into` with nu < 0 only: `_seq_into` is bitwise equal
                     to `_seq` (the chunks change nothing), and only NZ,
                     zero-vs-value, or status differs from the single-value
                     reference: near the limits a whole call decides
                     differently from single values (Amos's tests depend on
                     the orders of the call).
  mismatch           anything else -> test fails.
Additionally `single_vs_seq` counts N = 1 cases where the single-value
function disagrees with `_seq` (also a failure).

Requires the complex-bessel dependency to point at the code under test
(e.g. a path dependency on the dev checkout).
"""

import argparse
import collections
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FORTRAN = ROOT / "fortran"
RUST_EDGE = ROOT / "target" / "release" / "edge"

# ELIM, ALIM, FNUL as computed by Amos's routines (I1MACH/D1MACH/R1MACH).
PARAMS = {
    "f64": {"elim": 700.92, "alim": 664.87, "fnul": 85.92},
    "f32": {"elim": 79.75, "alim": 63.80, "fnul": 33.55},
}

# Relative tolerance for element values (complex-norm difference over the
# local sequence magnitude), indexed by [reduced precision (IERR=3)].
# Measured Rust-vs-Fortran maxima (2026-11): f64 3.8e-11 / 1.2e-7,
# f32 1.0e-3 / 0.5. Port bugs found so far were off by 1e-3 (f64) up to
# factors of 1/TOL; f32 3e-3 still flags 0.3.0's ZDIV-era drift. f32 with
# IERR=3 (|z| beyond ~160 for Airy) can have no correct digits at all, so
# only its status is compared.
TOL = {"f64": (1e-9, 1e-5), "f32": (3e-3, math.inf)}

BESSEL = ("J", "Y", "I", "K", "H1", "H2")
AIRY = ("AI", "AP", "BI", "BP")
ERROR_CODES = {1, 2, 4, 5}


# ── Grid ──


def quantize(x, prec):
    """Round to a value exactly representable in the target precision."""
    q = 8.0 if prec == "f32" else 1024.0
    return round(x * q) / q


def linspace(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def polar(r, deg, prec):
    t = math.radians(deg)
    return quantize(r * math.cos(t), prec), quantize(r * math.sin(t), prec)


def bessel_cases(region, prec, nus, ns, zs, fns=BESSEL):
    for fn in fns:
        for kode in (1, 2):
            for nu in nus:
                for n in ns:
                    for zr, zi in zs:
                        yield (prec, region, fn, kode, n, nu, zr, zi)


def region_threshold(prec):
    """|Re z| or |Im z| around ALIM..ELIM (scaling/overflow/underflow tests)."""
    p = PARAMS[prec]
    a_vals = [quantize(a, prec) for a in linspace(p["alim"] - 20, p["elim"] + 10, 25)]
    b_vals = [
        quantize(b, prec) for b in linspace(-4.5 * p["elim"], 4.5 * p["elim"], 25)
    ]
    zs = sorted(
        {z for a in a_vals for b in b_vals for z in ((a, b), (b, a), (-a, b), (b, -a))}
    )
    if prec == "f64":
        nus = [0, 0.3, 1, 2.5, 20, 45.5, 85.5, 90.5, 150, 400]
    else:
        nus = [0, 0.3, 1, 2.5, 7, 20.5, 33.5, 38.5, 60, 150]
    # n = 4 runs the recurrences that only start from the third member
    return bessel_cases("threshold", prec, nus, (1, 4), zs)


def region_long_n(prec):
    """nu below FNUL but nu + n - 1 above it (dispatch and recurrences)."""
    p = PARAMS[prec]
    fnul = p["fnul"]
    n = int(fnul) + 15
    nus = [0.5, 2, 10.25, quantize(fnul - 10, prec)]
    radii = [0.5, 20, fnul, 2 * fnul, p["alim"] + 10]
    zs = sorted(
        {polar(r, d, prec) for r in radii for d in range(0, 360, 22)} | {(2.0, 1.0)}
    )
    return bessel_cases("long_n", prec, nus, (n,), zs)


def region_negative_nu(prec):
    """Large negative orders, all-negative and crossing zero."""
    # Half-integer orders need one function (cos(nu pi) = 0; for Y it can
    # overflow where the result is finite: -200.5 at small |z|), other
    # non-integer orders two
    if prec == "f64":
        nus = [-10, -10.25, -10.5, -50.5, -100, -200, -200.5]
        special = [(710.0, 0.0), (0.0, 710.0)]
    else:
        nus = [-10, -10.25, -10.5, -30.5, -60]
        special = [(85.0, 0.0), (0.0, 85.0)]
    zs = sorted({polar(r, d, prec) for r in (0.5, 5, 30) for d in range(0, 360, 45)})
    yield from bessel_cases("negative_nu", prec, nus, (1, 3, 15), zs)
    # orders below the requested ones overflow here (I_0, I_1 at |z| ~ ELIM)
    yield from bessel_cases("negative_nu", prec, nus, (2,), special)
    # the top orders underflow after the orders cross zero (the underflow
    # count once covered only leading zeros); off the axes, where ZUCHK is
    # not noise-driven
    # (f32 inputs are multiples of 1/8, so |z| = 0.5 to stay off the axes)
    n_under, r_under = (130, 0.1) if prec == "f64" else (40, 0.5)
    zs = sorted({polar(r_under, d, prec) for d in (20, 75, 110, 200, 290)})
    yield from bessel_cases(
        "negative_nu", prec, [-0.5, -3, -10.25, -10.5], (n_under,), zs, fns=("J", "I")
    )


def region_zuoik(prec):
    """ZUOIK marginal-underflow window for I and J (large nu, moderate |z|)."""
    if prec == "f64":
        nus = list(range(300, 451, 25))
        radii = [18 + 4 * i for i in range(12)]
    else:
        nus = list(range(40, 101, 10))
        radii = [4 + 2.5 * i for i in range(13)]
    zs = sorted({polar(r, d, prec) for r in radii for d in range(0, 360, 15)})
    return bessel_cases("zuoik", prec, nus, (1, 3), zs, fns=("J", "I"))


def region_series(prec):
    """Power-series path (|z|^2/4 <= nu+1) with the top orders near underflow:
    ZSERI retries lower orders after an underflow, keeping its scaling."""
    p = PARAMS[prec]
    target = -(p["alim"] + p["elim"]) / 2

    def log_i(nu, r):
        # leading term of ln I_nu(r) for the power series
        return nu * math.log(r / 2) - math.lgamma(nu + 1)

    cases = []
    for r in (1.0, 3.0, 6.5, 12.0):
        lo, hi = 1.0, 5000.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if log_i(mid, r) > target else (lo, mid)
        # integer and half-integer orders: with z on an axis the phase then
        # sits on an axis too, which is where ZUCHK flags the top orders
        nus = [round(lo) + d / 2 for d in range(-16, 17)]
        zs = sorted({polar(r, d, prec) for d in range(0, 360, 30)})
        cases.extend(bessel_cases("series", prec, nus, (2, 3, 5), zs))
    return cases


def region_zbesy(prec):
    """Scaled Y with 2|Im z| >= ELIM and orders up to a few times |Im z|,
    where ZBESY drops a term that dominates for large nu."""
    if prec == "f64":
        ims = [340, 360, 400, 500, 650]
        res = [-300, -50, 0, 50, 300]
        # Negative orders: J of half-integer order uses Y alone; Y of
        # half-integer order does not use Y, of other non-integer orders does
        nus = [200, 400, 575, 800, 1200, -575.5, -575.25]
    else:
        ims = [38, 41, 45, 55, 70]
        res = [-40, -6, 0, 6, 40]
        nus = [25, 45, 65, 90, 130, -65.5, -65.25]
    for fn in ("Y", "J"):
        for nu in nus:
            if fn == "J" and nu >= 0:
                continue  # J uses Y only through negative-order reflection
            for n in (1, 3):
                for im in ims:
                    for re in res:
                        for sign in (1, -1):
                            yield (
                                prec,
                                "zbesy",
                                fn,
                                2,
                                n,
                                nu,
                                float(re),
                                sign * float(im),
                            )


def region_seq_into(prec):
    """Long sequences through `_seq_into`, whose stack workspace holds 64
    elements: Y's K values go in chunks of 64, and the reflection formulas
    of non-integer nu < 0 in chunks of 64 (I) or 32 (J, Y; half-integer nu
    needs only one function, so Y's K values go in chunks of 64), so these
    runs cross chunk boundaries. Negative integer nu covers both mirror
    branches of a crossing (largest |order| below or above zero)."""
    p = PARAMS[prec]
    if prec == "f64":
        nus = [
            0.4,
            7.5,
            45.5,
            150.5,
            -12.25,
            -40.75,
            -80.25,
            -80.5,
            -150.75,
            -40,
            -70,
            -100,
        ]
        ns = (65, 100, 150)
        radii = [0.5, 6.5, 30, p["fnul"], p["alim"], 690, 705]
    else:
        # nu = 0.4, z = 34, n = 100 (scaled Y): the whole Fortran call
        # overflows, the chunked one does not (intentional_chunk)
        nus = [0.4, 7.5, 20.5, -12.25, -40.5, -40.75, -40, -70]
        ns = (65, 100)
        radii = [0.5, 6.5, 20, 34, p["alim"], 75, 85]
    zs = sorted({polar(r, d, prec) for r in radii for d in range(0, 360, 40)})
    return bessel_cases("seq_into", prec, nus, ns, zs)


def region_airy(prec):
    """Airy overflow/underflow thresholds and the |z| range limits."""
    if prec == "f64":
        radii = [1, 5, 50] + list(range(95, 113)) + [500, 1000, 1030, 1.0e6, 1.1e6]
    else:
        radii = (
            [1, 5, 15]
            + [19 + 0.5 * i for i in range(17)]
            + [100, 160, 165, 2.6e4, 2.7e4]
        )
    zs = sorted({polar(r, d * 7.5, prec) for r in radii for d in range(48)})
    for fn in AIRY:
        for kode in (1, 2):
            for zr, zi in zs:
                yield (prec, "airy", fn, kode, 1, 0, zr, zi)


REGIONS = {
    "threshold": region_threshold,
    "long_n": region_long_n,
    "negative_nu": region_negative_nu,
    "zuoik": region_zuoik,
    "series": region_series,
    "zbesy": region_zbesy,
    "seq_into": region_seq_into,
    "airy": region_airy,
}


# ── Running implementations ──


def fmt(x):
    return repr(float(x))


def run_lines(exe, lines):
    proc = subprocess.run(
        [str(exe)], input="".join(lines), capture_output=True, text=True, check=True
    )
    out = proc.stdout.splitlines()
    if len(out) != len(lines):
        sys.exit(f"{exe.name}: {len(out)} output lines for {len(lines)} inputs")
    return out


def parse_result(line, rust=False):
    """-> (code, nz, single_flag, [complex values])"""
    f = line.split()
    code, nz = int(f[0]), int(f[1])
    single = int(f[2]) if rust else 0
    nums = [float(x) for x in f[3 if rust else 2 :]]
    vals = [complex(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
    return code, nz, single, vals


def sincospi(x):
    """(sin(pi x), cos(pi x)), exact at multiples of 1/2."""
    r = math.fmod(x, 2.0)
    if r * 2 == math.floor(r * 2):
        k = int(round(r * 2)) % 4
        return ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0))[k]
    return math.sin(math.pi * r), math.cos(math.pi * r)


def reflection_parts(fn, a):
    """Positive-order functions needed for order -a, primary first. For
    half-integer a, cos(a pi) = 0: J needs only Y, and Y only J."""
    integer = a == math.floor(a)
    half = a - 0.5 == math.floor(a - 0.5)
    if fn == "J":
        return ("J",) if integer else ("Y",) if half else ("J", "Y")
    if fn == "Y":
        return ("Y",) if integer else ("J",) if half else ("J", "Y")
    if fn == "I":
        return ("I",) if integer else ("I", "K")
    return (fn,)


def reflect(fn, kode, a, z, parts):
    """Value of fn_{-a}(z) from positive-order values (DLMF 10.2.3, 10.4.6-7,
    10.27.2, 10.27.3). parts maps function name -> value at order a."""
    s, c = sincospi(a)
    if fn == "J":
        return (parts["J"] * c if c else 0) - (parts["Y"] * s if s else 0)
    if fn == "Y":
        return (parts["Y"] * c if c else 0) + (parts["J"] * s if s else 0)
    if fn == "I":
        if not s:
            return parts["I"]
        k = parts["K"]
        if kode == 2:
            # K_scaled = exp(z) K, I_scaled = exp(-|Re z|) I
            k = k * complex(math.cos(-z.imag), math.sin(-z.imag))
            if z.real > 0:
                k = k * math.exp(-2 * z.real)
        return parts["I"] + 2 / math.pi * s * k
    if fn == "K":
        return parts["K"]
    sign = 1 if fn == "H1" else -1
    return parts[fn] * complex(c, sign * s)


def fortran_requests(case):
    """Fortran input lines needed for one case (list of (key, line))."""
    prec, _, fn, kode, n, nu, zr, zi = case
    if nu >= 0:
        return [
            (
                (fn, kode, n, nu, zr, zi),
                f"{fn} {kode} {n} {fmt(nu)} {fmt(zr)} {fmt(zi)}\n",
            )
        ]
    return per_order_requests(case)


def per_order_requests(case):
    """Fortran single-value (N = 1) requests for every order of a case."""
    prec, _, fn, kode, n, nu, zr, zi = case
    reqs = []
    for j in range(n):
        order = nu + j
        a = abs(order)
        names = reflection_parts(fn, a) if order < 0 else (fn,)
        for name in names:
            key = (name, kode, 1, a, zr, zi)
            reqs.append((key, f"{name} {kode} 1 {fmt(a)} {fmt(zr)} {fmt(zi)}\n"))
    return reqs


def reference(case, results):
    """Fortran reference (code, nz, values) for a case."""
    prec, _, fn, kode, n, nu, zr, zi = case
    if nu >= 0:
        code, nz, _, vals = results[(fn, kode, n, nu, zr, zi)]
        # The Airy API exposes no NZ; an underflow still shows as a zero value
        return code, (0 if fn in AIRY else nz), vals
    return per_order_reference(case, results)


def per_order_reference(case, results):
    """Reference built from Fortran single values, one per order (negative
    orders through the reflection formulas)."""
    prec, _, fn, kode, n, nu, zr, zi = case
    z = complex(zr, zi)
    vals, worst = [], 0
    for j in range(n):
        order = nu + j
        a = abs(order)
        names = reflection_parts(fn, a) if order < 0 else (fn,)
        parts = {}
        for name in names:
            code, _, _, v = results[(name, kode, 1, a, zr, zi)]
            if code in ERROR_CODES:
                return code, 0, []
            worst = max(worst, code)
            parts[name] = v[0]
        vals.append(reflect(fn, kode, a, z, parts) if order < 0 else parts[fn])
    # complex-bessel counts every zero of a reflected sequence: underflow can
    # be at large |orders| below zero or at the high orders above it
    return worst, sum(1 for v in vals if v == 0), vals


def compare(rust, ref, tol):
    """-> (kind, max_rel): kind is None when they agree."""
    rcode, rnz, rvals = rust
    fcode, fnz, fvals = ref
    if rcode != fcode:
        return "status", 0.0
    if rcode in ERROR_CODES:
        return None, 0.0
    if rnz != fnz:
        return "nz", 0.0
    worst = 0.0
    n = len(rvals)
    for j, (r, f) in enumerate(zip(rvals, fvals)):
        if r == 0 and f == 0:
            continue
        if r == 0 or f == 0:
            return "zero", 0.0
        # Scale by the local magnitude of the sequence (this member and its
        # neighbours): an oscillating member near a zero is computed as a
        # small difference of larger terms, so its error scales with them.
        lo, hi = max(0, j - 1), min(n, j + 2)
        scale = max(abs(x) for x in rvals[lo:hi] + fvals[lo:hi])
        worst = max(worst, abs(r - f) / scale)
    return ("value" if worst > tol[rcode == 3] else None), worst


def zbesy_dropped(case):
    """Scaled Y (or J via negative-order reflection) where ZBESY drops its
    K term (2|Im z| >= ELIM)."""
    prec, _, fn, kode, _, _, _, zi = case
    return fn in ("Y", "J") and kode == 2 and 2 * abs(zi) >= PARAMS[prec]["elim"]


def unscaled(case):
    return case[:3] + (1,) + case[4:]


def real_part_only(ref):
    """Fortran reference with the imaginary parts dropped."""
    code, nz, vals = ref
    return code, nz, [complex(v.real, 0.0) for v in vals]


def small_part_from(rust_vals, ref, tol):
    """Fortran reference with the smaller part of each element from Rust, or
    None unless that part is below 0.1 times the larger one in Fortran and
    the two differ by at most 2 tol times |value| (a residue of a normwise
    accurate value, not a different formula)."""
    code, nz, vals = ref
    out = []
    for v, r in zip(vals, rust_vals):
        re_small = abs(v.real) < abs(v.imag)
        small, large = (v.real, v.imag) if re_small else (v.imag, v.real)
        rs = r.real if re_small else r.imag
        if rs != small and (
            abs(small) >= 0.1 * abs(large) or abs(rs - small) > 2 * tol * abs(v)
        ):
            return None
        out.append(complex(rs, v.imag) if re_small else complex(v.real, rs))
    return code, nz, out


def scaled_from_unscaled(case, results):
    """exp(-|Im z|) times the unscaled double-precision reference."""
    code, _, vals = reference(unscaled(case), results)
    if code in ERROR_CODES:
        return code, 0, []
    half = math.exp(-abs(case[7]) / 2)
    vals = [v * half * half for v in vals]
    return code, sum(1 for v in vals if v == 0), vals


# ── Main ──


def run(regions, precs, max_examples):
    cases = [c for prec in precs for name in regions for c in REGIONS[name](prec)]
    print(f"{len(cases)} cases", flush=True)

    out = run_lines(
        RUST_EDGE,
        [
            f"{c[0]} {c[2]} {c[3]} {c[4]} {fmt(c[5])} {fmt(c[6])} {fmt(c[7])}"
            + (" into\n" if c[1] == "seq_into" else "\n")
            for c in cases
        ],
    )
    rust = [parse_result(line, rust=True) for line in out]
    # The same `_seq_into` cases through `_seq` (one call per function)
    into_idx = [i for i, c in enumerate(cases) if c[1] == "seq_into"]
    out = run_lines(
        RUST_EDGE,
        [
            f"{c[0]} {c[2]} {c[3]} {c[4]} {fmt(c[5])} {fmt(c[6])} {fmt(c[7])}\n"
            for c in (cases[i] for i in into_idx)
        ],
    )
    whole = {i: parse_result(line, rust=True) for i, line in zip(into_idx, out)}

    refs = {}
    for prec in precs:
        exes = {"orig": FORTRAN / ("edge64" if prec == "f64" else "edge32")}
        if prec == "f64":
            exes["zri"] = FORTRAN / "edge64_zri"
        reqs = {}
        for c in cases:
            if c[0] == prec:
                reqs.update(fortran_requests(c))
        keys = list(reqs)
        for tag, exe in exes.items():
            res = run_lines(exe, [reqs[k] for k in keys])
            refs[(prec, tag)] = {k: parse_result(line) for k, line in zip(keys, res)}
        # Unscaled references in double precision, where ZBESY keeps both terms
        ureqs = {}
        for c in cases:
            if c[0] == prec and zbesy_dropped(c):
                ureqs.update(fortran_requests(unscaled(c)))
        if ureqs:
            keys = list(ureqs)
            res = run_lines(FORTRAN / "edge64", [ureqs[k] for k in keys])
            refs[(prec, "unscaled64")] = {
                k: parse_result(line) for k, line in zip(keys, res)
            }
        # Single values per order where a chunked `_seq_into` succeeds but
        # the whole Fortran call fails
        sreqs = {}
        for c, (rcode, *_) in zip(cases, rust):
            if (
                c[0] == prec
                and c[1] == "seq_into"
                and rcode not in ERROR_CODES
                and reference(c, refs[(prec, "orig")])[0] in ERROR_CODES
            ):
                sreqs.update(per_order_requests(c))
        if sreqs:
            keys = list(sreqs)
            res = run_lines(exes["orig"], [sreqs[k] for k in keys])
            refs[(prec, "single")] = {
                k: parse_result(line) for k, line in zip(keys, res)
            }

    buckets = collections.Counter()
    maxrel = collections.defaultdict(float)
    examples = collections.defaultdict(list)
    for i, (case, (rcode, rnz, single, rvals)) in enumerate(zip(cases, rust)):
        prec, region, fn, kode, n, nu, zr, zi = case
        group = (prec, region)
        if single:
            buckets[group + ("single_vs_seq",)] += 1
            if len(examples["single_vs_seq"]) < max_examples:
                examples["single_vs_seq"].append({"case": case})
        r = (rcode, rnz, rvals)
        kind, rel = compare(r, reference(case, refs[(prec, "orig")]), TOL[prec])
        if kind is None:
            bucket = "pass"
            maxrel[group] = max(maxrel[group], rel)
        elif (prec, "zri") in refs and compare(
            r, reference(case, refs[(prec, "zri")]), TOL[prec]
        )[0] is None:
            bucket = "intentional_zuoik"
        elif (
            zbesy_dropped(case)
            and compare(
                r, scaled_from_unscaled(case, refs[(prec, "unscaled64")]), TOL[prec]
            )[0]
            is None
        ):
            bucket = "intentional_zbesy"
        elif (
            region == "seq_into"
            and kind == "status"
            and rcode not in ERROR_CODES
            and (prec, "single") in refs
            and compare(
                r, per_order_reference(case, refs[(prec, "single")]), TOL[prec]
            )[0]
            is None
        ):
            bucket = "intentional_chunk"
        elif (
            fn in ("J", "Y")
            and zi == 0
            and zr > 0
            and compare(
                r, real_part_only(reference(case, refs[(prec, "orig")])), TOL[prec]
            )[0]
            is None
        ):
            bucket = "intentional_real"
        elif (
            fn in ("Y", "K", "H1", "H2")
            and (zi == 0 or zr == 0)
            and kind == "value"
            and (
                axis_ref := small_part_from(
                    rvals, reference(case, refs[(prec, "orig")]), TOL[prec][0]
                )
            )
            is not None
            and compare(r, axis_ref, TOL[prec])[0] is None
        ):
            bucket = "intentional_axis"
        elif (zr == 0 or zi == 0) and kind in ("nz", "zero"):
            bucket = "axis_noise"
        elif (
            region == "seq_into"
            and nu < 0
            and kind in ("nz", "zero", "status")
            and whole[i][:2] == (rcode, rnz)
            and whole[i][3] == rvals
        ):
            bucket = "call_dependent"
        else:
            bucket = "mismatch"
        buckets[group + (bucket,)] += 1
        if bucket != "pass" and len(examples[bucket]) < max_examples:
            ref = reference(case, refs[(prec, "orig")])
            examples[bucket].append(
                {
                    "case": case,
                    "kind": kind,
                    "rust": [rcode, rnz],
                    "fortran": [ref[0], ref[1]],
                }
            )
    return cases, buckets, maxrel, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--regions", nargs="+", choices=list(REGIONS), default=list(REGIONS)
    )
    parser.add_argument(
        "--precs", nargs="+", choices=["f64", "f32"], default=["f64", "f32"]
    )
    parser.add_argument(
        "--examples", type=int, default=20, help="examples kept per bucket"
    )
    args = parser.parse_args()

    cases, buckets, maxrel, examples = run(args.regions, args.precs, args.examples)

    names = [
        "pass",
        "intentional_zuoik",
        "intentional_zbesy",
        "intentional_chunk",
        "call_dependent",
        "intentional_real",
        "intentional_axis",
        "axis_noise",
        "mismatch",
        "single_vs_seq",
    ]
    print(
        f"\n{'prec':4s} {'region':12s} "
        + " ".join(f"{b:>17s}" for b in names)
        + "  max_rel"
    )
    summary = {}
    for prec in args.precs:
        for region in args.regions:
            g = (prec, region)
            row = {b: buckets[g + (b,)] for b in names}
            row["max_rel_passing"] = maxrel[g]
            summary[f"{prec}/{region}"] = row
            print(
                f"{prec:4s} {region:12s} "
                + " ".join(f"{row[b]:17d}" for b in names)
                + f"  {maxrel[g]:.1e}"
            )
    failures = sum(buckets[k] for k in buckets if k[2] in ("mismatch", "single_vs_seq"))
    for bucket in ("mismatch", "single_vs_seq"):
        for ex in examples.get(bucket, [])[:5]:
            print(f"  {bucket}: {ex}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {
        "cases": len(cases),
        "tolerance": TOL,
        "summary": summary,
        "examples": dict(examples),
        "failures": failures,
    }
    (RESULTS / "edge_report.json").write_text(json.dumps(report, indent=1))
    print(
        f"\n{'FAIL' if failures else 'PASS'}: {failures} failing cases "
        f"(report: results/edge_report.json)"
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
