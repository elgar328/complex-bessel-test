"""Implementation differences: inputs where the implementations differ.

Collects sensitive or unusual inputs (cancellation on the axes, branch cuts,
range limits, special orders, scaling) and compares complex-bessel (Rust),
AMOS/TOMS 644 (Fortran), SciPy, and SpecialFunctions.jl (Julia) against
mpmath, through each library's complex-argument functions. Values on the
real and imaginary axes, and next to them, are sampled for a chart.

Outputs:
- results/differences.json       all values, references and errors
- results/differences_mpmath.json reference cache (reused between runs)
- images/componentwise_accuracy.{svg,pdf} parts on the axes, and of scaled functions near them
- README.md                      the table between the differences markers

References: values on the real and imaginary axes are built from the real
functions J, Y, I, K of the real argument |z| (DLMF 10.11, 10.27, 10.34):
mpmath's complex-argument evaluation does not resolve a component that is
far smaller than the other. Elsewhere mpmath is used directly. Either way
the precision is raised until two evaluations agree in each component.
"""

import json
import math
import os
import subprocess
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy import special

import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from dashboard import (
    COLORS,
    IMPLS,
    VIOLIN_WIDTH,
    _format_error_ticks,
    _get_versions,
    _legend_labels,
    _offset,
)

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
IMAGES = os.path.join(ROOT, "images")

# ── Cases ──

# Table rows: (group, label, function, nu, z_re, z_im, part, note).
# `part` is the error measured: "value" (relative to |f|), or "re" / "im"
# (relative to that component).
TABLE = [
    (
        "Cancellation on the axes",
        "H<sup>(1)</sup><sub>10</sub>(1), real part",
        "hankel1",
        10.0,
        1.0,
        0.0,
        "re",
        "Re H = J₁₀(1) ≪ Y₁₀(1)",
    ),
    (
        "Cancellation on the axes",
        "H<sup>(1)</sup><sub>1</sub>(2·10⁻³²), real part",
        "hankel1",
        1.0,
        2e-32,
        0.0,
        "re",
        "openspecfun#45",
    ),
    (
        "Cancellation on the axes",
        "K<sub>20</sub>(5i), imaginary part",
        "besselk",
        20.0,
        0.0,
        5.0,
        "im",
        "Im K = −(π/2)·J₂₀(5)",
    ),
    (
        "Cancellation on the axes",
        "H<sup>(2)</sup><sub>1</sub>(20i), real part",
        "hankel2",
        1.0,
        0.0,
        20.0,
        "re",
        "Re H = (2/π)·K₁(20) ≪ I₁(20)",
    ),
    (
        "Cancellation on the axes",
        "Y<sub>0.500000000001</sub>(−5), real part",
        "bessely",
        0.500000000001,
        -5.0,
        0.0,
        "re",
        "Re Y = cos(νπ)·Y<sub>ν</sub>(5)",
    ),
    (
        "Cancellation on the axes",
        "J<sub>0.500000000001</sub>(−5), real part",
        "besselj",
        0.500000000001,
        -5.0,
        0.0,
        "re",
        "Re J = cos(νπ)·J<sub>ν</sub>(5)",
    ),
    (
        "Branch cut",
        "Ai(−2 − 0i)",
        "airy",
        0.0,
        -2.0,
        -0.0,
        "value",
        "Ai is entire; −0 must act like +0",
    ),
    (
        "Range limit",
        "H<sup>(1)</sup><sub>86</sub>(10⁹)",
        "hankel1",
        86.0,
        1e9,
        0.0,
        "value",
        "Airy argument beyond its own limit",
    ),
    (
        "Range limit",
        "J<sub>100000</sub>(10⁹)",
        "besselj",
        1e5,
        1e9,
        0.0,
        "value",
        "",
    ),
    (
        "Special orders",
        "Y<sub>−519.5</sub>(100)",
        "bessely",
        -519.5,
        100.0,
        0.0,
        "value",
        "cos(νπ) = 0 removes the overflowing Y<sub>519.5</sub>",
    ),
    (
        "Scaling",
        "H<sup>(2)</sup><sub>0.5</sub>(−10⁸), scaled",
        "hankel2_scaled",
        0.5,
        -1e8,
        0.0,
        "value",
        "phase of exp(−2z) in the continuation",
    ),
    (
        "Scaling",
        "Y<sub>1000</sub>(300 + 500i), scaled",
        "bessely_scaled",
        1000.0,
        300.0,
        500.0,
        "value",
        "a dominant term dropped as underflow",
    ),
]

# Axis sample: (function, half-axis) rows whose two components are real
# functions of very different size.
AXIS_ROWS = [
    ("hankel1", "+x"),
    ("hankel2", "+x"),
    ("hankel1", "-x"),
    ("hankel2", "-x"),
    ("bessely", "-x"),
    ("besselk", "+ix"),
    ("hankel2", "+ix"),
    ("hankel1", "-ix"),
]
AXIS_ORDERS = [
    0.3,
    1.0,
    2.5,
    5.7,
    10.0,
    20.5,
    40.2,
    85.0,
    150.5,
    300.0,
    -0.3,
    -1.0,
    -2.5,
    -5.7,
    -10.0,
    -20.5,
    -40.2,
    2.0001,
    4.4999999,
    -7.5000001,
    12.000001,
]
# x as fractions of |ν| (J·Y rows) or absolute (I·K rows)
AXIS_FRACS = [1e-8, 1e-4, 1e-2, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.5, 3.0]
AXIS_IK_X = [0.1, 1.0, 5.0, 20.0, 60.0, 200.0]


def axis_points():
    """(function, ν, Re z, Im z, chart category) on the axes."""
    pts = []
    for func, axis in AXIS_ROWS:
        ik = axis.endswith("ix") and func != "besselk"
        for nu in AXIS_ORDERS:
            m = abs(nu)
            xs = AXIS_IK_X if ik else [m * f for f in AXIS_FRACS]
            for x in xs:
                z = {
                    "+x": (x, 0.0),
                    "-x": (-x, 0.0),
                    "+ix": (0.0, x),
                    "-ix": (0.0, -x),
                }[axis]
                pts.append((func, nu, z[0], z[1], (func, axis)))
    return pts


# Scaled functions continued across the imaginary axis (Amos ZACON), next
# to the axis where their K term is not negligible: H near the real axis,
# K and I (negative non-integer ν) near the imaginary axis. Amos rounds the
# phase of exp(−2z) there to an ulp of 2·Im z, a relative error up to ε·|z|.
SCALED_ROWS = ["hankel1_scaled", "hankel2_scaled", "besselk_scaled", "besseli_scaled"]
SCALED_ORDERS = [0.3, 0.5, 1.7, 5.5, 20.3, 60.5, 85.0]
SCALED_R = [1e2, 1e4, 1e6, 1e8, 1e9]
SCALED_D = [1e-3, 1.0, 5.0]


def scaled_points():
    """(function, ν, Re z, Im z, chart category) next to the axes."""
    pts = []
    for func in SCALED_ROWS:
        for nu in SCALED_ORDERS:
            if func == "besseli_scaled":
                nu = -nu
            for r in SCALED_R:
                for d in SCALED_D:
                    for sign in (1.0, -1.0):
                        z = {
                            # below the real axis for H1, above for H2
                            "hankel1_scaled": (sign * r, -d),
                            "hankel2_scaled": (sign * r, d),
                            # left of the imaginary axis for K and I
                            "besselk_scaled": (-d, sign * r),
                            "besseli_scaled": (-d, sign * r),
                        }[func]
                        pts.append((func, nu, z[0], z[1], (func, "near")))
    return pts


# ── References (mpmath) ──

AIRY = {"airy", "airyprime", "biry", "biryprime"}


def _airy(base, z):
    if base == "airy":
        return mp.airyai(z)
    if base == "airyprime":
        return mp.airyai(z, derivative=1)
    if base == "biry":
        return mp.airybi(z)
    return mp.airybi(z, derivative=1)


def _real_axis(base, nu, x, neg):
    """f_ν(±x) from the real functions at x > 0 (signed ν)."""
    if base in ("besseli", "besselk"):
        i, k = mp.besseli(nu, x), mp.besselk(nu, x)
        if not neg:
            return mp.mpc(i) if base == "besseli" else mp.mpc(k)
        e = mp.expjpi(nu)
        return e * i if base == "besseli" else mp.conj(e) * k - 1j * mp.pi * i
    j, y = mp.besselj(nu, x), mp.bessely(nu, x)
    if neg:
        e = mp.expjpi(nu)
        j, y = e * j, mp.conj(e) * y + 2j * mp.cospi(nu) * j
    return {
        "besselj": j,
        "bessely": y,
        "hankel1": j + 1j * y,
        "hankel2": j - 1j * y,
    }[base] * mp.mpc(1)


def _imag_axis(base, nu, x):
    """f_ν(ix) for x > 0 from the real functions at x."""
    e = mp.expjpi(nu / 2)
    if base == "besselj":
        return e * mp.besseli(nu, x)
    if base == "besseli":
        return e * mp.besselj(nu, x)
    k, i = mp.besselk(nu, x), mp.besseli(nu, x)
    h1 = 2 / (1j * mp.pi) * mp.conj(e) * k
    k_neg = mp.expjpi(-nu) * k - 1j * mp.pi * i
    h2 = 2j / mp.pi * e * k_neg
    if base == "hankel1":
        return h1
    if base == "hankel2":
        return h2
    if base == "bessely":
        return (h1 - h2) / 2j
    # K(ix) = −(iπ/2)·e^{−iνπ/2}·H2(x)
    h2x = mp.besselj(nu, x) - 1j * mp.bessely(nu, x)
    return -0.5j * mp.pi * mp.conj(e) * h2x


def _direct(base, nu, z):
    return {
        "besselj": mp.besselj,
        "bessely": mp.bessely,
        "besseli": mp.besseli,
        "besselk": mp.besselk,
        "hankel1": mp.hankel1,
        "hankel2": mp.hankel2,
    }[base](nu, z)


def _value(base, nu, re, im):
    if base in AIRY:
        # A real argument on the real axis keeps the imaginary part exactly 0
        return mp.mpc(_airy(base, mp.mpf(re) if im == 0 else mp.mpc(re, im)))
    if im == 0:
        return _real_axis(base, nu, mp.mpf(abs(re)), re < 0)
    if re == 0:
        v = _imag_axis(base, nu, mp.mpf(abs(im)))
        if im > 0:
            return v
        # f(conj z) = conj f(z) for real ν; H1 and H2 swap
        swap = {"hankel1": "hankel2", "hankel2": "hankel1"}.get(base, base)
        return mp.conj(_imag_axis(swap, nu, mp.mpf(-im)))
    return _direct(base, nu, mp.mpc(re, im))


def _scale(base, re, im):
    """The factor of the `_scaled` variant (Amos KODE=2)."""
    z = mp.mpc(re, im)
    if base in ("besselj", "bessely"):
        return mp.exp(-abs(mp.mpf(im)))
    if base == "besseli":
        return mp.exp(-abs(mp.mpf(re)))
    if base == "besselk":
        return mp.exp(z)
    if base == "hankel1":
        return mp.exp(-1j * z)
    if base == "hankel2":
        return mp.exp(1j * z)
    zeta = mp.mpf(2) / 3 * z * mp.sqrt(z)
    if base in ("airy", "airyprime"):
        return mp.exp(zeta)
    return mp.exp(-abs(zeta.real))


def _agree(a, b, componentwise):
    """The components of b if a and b agree to 20 digits: in each component
    (an exact zero, as on the real axis for a real function, agrees with
    itself), or normwise."""
    tol = mp.mpf(10) ** -20
    if componentwise:
        if any(
            abs(ca - cb) > tol * abs(cb)
            for ca, cb in ((a.real, b.real), (a.imag, b.imag))
        ):
            return None
    elif abs(a - b) > tol * abs(b):
        return None
    return (float(b.real), float(b.imag))


def reference(func, nu, re, im, componentwise=True):
    """(re, im) of the true value, each component (or the value) to ~20 digits.

    The formulas can cancel far below |f| (one component can be 10⁻¹⁰⁰ of
    the other), so the precision is doubled until two evaluations agree
    componentwise. None if they never do.
    """
    base = func.removesuffix("_scaled")
    prev = None
    for dps in (50, 100, 200, 400, 800):
        with mp.workdps(dps):
            v = _value(base, mp.mpf(nu), re, im)
            if func.endswith("_scaled"):
                v *= _scale(base, re, im)
            v = mp.mpc(v)
            if prev is not None:
                got = _agree(prev, v, componentwise)
                if got is not None:
                    return got
        prev = v
    return None


def key(func, nu, re, im, componentwise=True):
    return f"{func}|{nu!r}|{re!r}|{im!r}|{'c' if componentwise else 'n'}"


def references(points):
    path = os.path.join(RESULTS, "differences_mpmath.json")
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            cache = json.load(f)
    todo = [p for p in points if key(*p) not in cache]
    print(f"  mpmath: {len(todo)} new references ({len(cache)} cached)", flush=True)
    for p in todo:
        cache[key(*p)] = reference(*p)
    # Keep only the current points
    used = {key(*p): cache[key(*p)] for p in points}
    with open(path, "w") as f:
        json.dump(used, f)
    return [used[key(*p)] for p in points]


# ── Implementations ──

SCIPY = {
    "besselj": special.jv,
    "bessely": special.yv,
    "besseli": special.iv,
    "besselk": special.kv,
    "hankel1": special.hankel1,
    "hankel2": special.hankel2,
    "besselj_scaled": special.jve,
    "bessely_scaled": special.yve,
    "besseli_scaled": special.ive,
    "besselk_scaled": special.kve,
    "hankel1_scaled": special.hankel1e,
    "hankel2_scaled": special.hankel2e,
    "airy": lambda nu, z: special.airy(z)[0],
    "airyprime": lambda nu, z: special.airy(z)[1],
    "biry": lambda nu, z: special.airy(z)[2],
    "biryprime": lambda nu, z: special.airy(z)[3],
}


def _scipy_one(func, nu, x):
    try:
        v = complex(SCIPY[func](nu, x))
    except Exception:
        return (None, None, "error")
    if math.isnan(v.real) or math.isnan(v.imag):
        return (None, None, "nan")
    if math.isinf(v.real) or math.isinf(v.imag):
        return (None, None, "inf")
    return (v.real, v.imag, "ok")


def run_scipy(points):
    out = []
    for func, nu, re, im, *_ in points:
        out.append(
            dict(zip(("re", "im", "status"), _scipy_one(func, nu, complex(re, im))))
        )
    return out


def run_rust(points):
    data = [{"function": p[0], "nu": p[1], "z_re": p[2], "z_im": p[3]} for p in points]
    proc = subprocess.run(
        [os.path.join(ROOT, "target", "release", "compute"), "--points"],
        input=json.dumps(data),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


FORTRAN_ID = {
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
}
IERR_STATUS = {
    1: "invalid_input",
    2: "overflow",
    4: "total_precision_loss",
    5: "convergence_failure",
}


def run_fortran(points):
    """ZBSUBS for ν ≥ 0 (Amos has no negative orders)."""
    from compute_fortran import fix_fortran_float

    lines, idx = [], []
    for i, (func, nu, re, im, *_) in enumerate(points):
        if nu < 0:
            continue
        base = func.removesuffix("_scaled")
        kode = 2 if func.endswith("_scaled") else 1
        lines.append(f"{FORTRAN_ID[base]} {nu!r} {re!r} {im!r} {kode}")
        idx.append(i)
    proc = subprocess.run(
        [os.path.join(ROOT, "fortran", "compute_grid")],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    out = [{"re": None, "im": None, "status": "n/a"} for _ in points]
    rows = [ln.split() for ln in proc.stdout.splitlines() if ln.strip()]
    for i, parts in zip(idx, rows):
        ierr = int(parts[7])
        if ierr in (0, 3):
            out[i] = {
                "re": fix_fortran_float(parts[5]),
                "im": fix_fortran_float(parts[6]),
                "status": "ok",
            }
        else:
            out[i] = {"re": None, "im": None, "status": IERR_STATUS.get(ierr, "error")}
    return out


def run_julia(points):
    data = [{"function": p[0], "nu": p[1], "z_re": p[2], "z_im": p[3]} for p in points]
    with open(os.path.join(RESULTS, "differences_points.json"), "w") as f:
        json.dump(data, f)
    julia_dir = os.path.join(ROOT, "julia")
    subprocess.run(
        [
            "julia",
            f"--project={julia_dir}",
            os.path.join(julia_dir, "compute.jl"),
            "points",
        ],
        check=True,
    )
    with open(os.path.join(RESULTS, "differences_julia.json")) as f:
        out = json.load(f)
    os.remove(os.path.join(RESULTS, "differences_points.json"))
    os.remove(os.path.join(RESULTS, "differences_julia.json"))
    return out


# ── Errors ──


def error(got_re, got_im, ref, part):
    """Relative error of `part` ("value", "re", "im"); None if no value."""
    if got_re is None or ref is None:
        return None
    if part == "value":
        d = math.hypot(got_re - ref[0], got_im - ref[1])
        n = math.hypot(*ref)
    else:
        k = 0 if part == "re" else 1
        d = abs((got_re, got_im)[k] - ref[k])
        n = abs(ref[k])
    if n == 0:
        return 0.0 if d == 0 else math.inf
    return d / n


# Result columns: (implementation, label)
COLUMNS = [
    ("rust", "Rust"),
    ("fortran", "Fortran"),
    ("scipy", "SciPy"),
    ("julia", "Julia"),
]

STATUS_TEXT = {
    "overflow": "Overflow",
    "invalid_input": "InvalidInput",
    "total_precision_loss": "PrecisionLoss",
    "convergence_failure": "NoConvergence",
    "domain_error": "DomainError",
    "nan": "NaN",
    "inf": "Inf",
    "error": "error",
    "n/a": "—",
}


# Below this, Amos (and the accuracy chart) treat a value as underflowed
UNDERFLOW = 1e-290


def cell(res, ref, part):
    """Table text for one implementation: the error, or what came back."""
    status = res["status"]
    if status not in ("ok", "reduced_precision"):
        return STATUS_TEXT.get(status, status)
    got = (res["re"], res["im"])
    if ref is not None and math.hypot(*ref) < UNDERFLOW and got == (0.0, 0.0):
        return "0 (underflow)"
    e = error(res["re"], res["im"], ref, part)
    if e is None:
        return ""
    if e == 0:
        return "0"
    if e < 1e-16:
        return "<1e-16"
    text = f"{e:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"**{text}**" if e > 1e-6 else text


def fmt_value(ref, part):
    if ref is None:
        return "?"
    if part == "re":
        return f"{ref[0]:.4g}"
    if part == "im":
        return f"{ref[1]:.4g}"
    re, im = ref
    # Components at the level of the reference's own noise are not shown
    if abs(im) < 1e-15 * abs(re):
        im = 0.0
    if abs(re) < 1e-15 * abs(im):
        return f"{im:.4g}i"
    if im == 0:
        return f"{re:.4g}"
    sign = "−" if im < 0 else "+"
    return f"{re:.4g} {sign} {abs(im):.4g}i"


def table_markdown(table_res):
    header = (
        "| Input | Note | True value | "
        + " | ".join(label for _, label in COLUMNS)
        + " |"
    )
    sep = "|:--|:--|--:|" + "--:|" * len(COLUMNS)
    lines = []
    group = None
    for row, res_by_impl, ref in table_res:
        g, label, _, _, _, _, part, note = row
        if g != group:
            if lines:
                lines.append("")
            lines += [f"**{g}**", "", header, sep]
            group = g
        cells = [cell(res_by_impl[impl], ref, part) for impl, _ in COLUMNS]
        lines.append(
            f"| {label} | {note} | {fmt_value(ref, part)} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "> Relative error of the value, or of the named part, against mpmath. "
        "**Bold**: above 10⁻⁶. Error names: the status the call returned "
        "(an Amos error, NaN, Inf, or a Julia exception); 0 (underflow): 0 for a "
        "true value below 10⁻²⁹⁰, as in the accuracy chart. — : Fortran has no "
        "negative orders. All calls take a complex argument (SciPy and Julia also "
        "have real-argument functions, which use other methods).",
    ]
    return "\n".join(lines)


# ── Axis chart ──

AXIS_LABELS = {
    ("hankel1", "+x"): "H$^{(1)}$(+x)",
    ("hankel2", "+x"): "H$^{(2)}$(+x)",
    ("hankel1", "-x"): "H$^{(1)}$(−x)",
    ("hankel2", "-x"): "H$^{(2)}$(−x)",
    ("bessely", "-x"): "Y(−x)",
    ("besselk", "+ix"): "K(+ix)",
    ("hankel2", "+ix"): "H$^{(2)}$(+ix)",
    ("hankel1", "-ix"): "H$^{(1)}$(−ix)",
    ("hankel1_scaled", "near"): "H$^{(1)}_{s}$(±x − iδ)",
    ("hankel2_scaled", "near"): "H$^{(2)}_{s}$(±x + iδ)",
    ("besselk_scaled", "near"): "K$_{s}$(−δ ± iy)",
    ("besseli_scaled", "near"): "I$_{s}$(−δ ± iy), ν < 0",
}


def parts_error(got_re, got_im, ref):
    """The larger relative error of the real and imaginary parts. Parts below
    the underflow level (or exactly zero) are left out."""
    errs = [
        abs(g - r) / abs(r)
        for g, r in ((got_re, ref[0]), (got_im, ref[1]))
        if abs(r) >= UNDERFLOW
    ]
    return max(errs) if errs else None


def _swarm_offsets(ys, half_width, bin_size=0.2):
    """Horizontal offsets that spread points with similar log10 y."""
    bins = {}
    for i, y in enumerate(ys):
        bins.setdefault(round(y / bin_size), []).append(i)
    widest = max(len(v) for v in bins.values())
    step = min(half_width * 2 / max(widest, 1), half_width / 4)
    off = [0.0] * len(ys)
    for members in bins.values():
        n = len(members)
        for k, i in enumerate(members):
            off[i] = (k - (n - 1) / 2) * step
    return off


def plot_axis(axis_res, labels):
    """Two stacked swarm panels in the style of accuracy.svg."""
    print("  componentwise_accuracy.svg...", flush=True)
    # Errors are clipped to [10^lo, 10^hi]; hi is on the 2-decade tick grid
    lo, hi = -17, 4
    cats = [c for c in AXIS_LABELS if any(a[0] == c for a in axis_res)]
    groups = [
        (
            "Componentwise Accuracy — On the Real and Imaginary Axes",
            [c for c in cats if c[1] != "near"],
        ),
        (
            "Componentwise Accuracy — Scaled Functions Near the Axes",
            [c for c in cats if c[1] == "near"],
        ),
    ]
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 1, hspace=0.25)
    rng = np.random.default_rng(42)
    for gi, (title, group) in enumerate(groups):
        ax = fig.add_subplot(gs[gi])
        for ci, cat in enumerate(group):
            per_impl = {}
            for impl in IMPLS:
                ys, statuses = [], set()
                for c, ref, res in axis_res:
                    if c != cat:
                        continue
                    v = res[impl]
                    statuses.add(v["status"])
                    if v["status"] not in ("ok", "reduced_precision"):
                        continue
                    e = parts_error(v["re"], v["im"], ref)
                    if e is not None:
                        ys.append(min(max(math.log10(max(e, 1e-300)), lo), hi))
                per_impl[impl] = (ys, statuses)
            for vi, (impl, color) in enumerate(zip(IMPLS, COLORS)):
                ys, statuses = per_impl[impl]
                if statuses == {"n/a"}:
                    # Not supported by this implementation (Amos: ν < 0);
                    # placed at the middle of the nearest column to the right
                    # that has points (else to the left)
                    order = IMPLS[vi + 1 :] + IMPLS[:vi][::-1]
                    near = next((per_impl[o][0] for o in order if per_impl[o][0]), None)
                    ax.text(
                        ci + _offset(vi),
                        (min(near) + max(near)) / 2 if near else (lo + hi) / 2,
                        "N/A",
                        color=color,
                        fontsize=9,
                        ha="center",
                        va="center",
                    )
                if not ys:
                    continue
                ys = [y + rng.uniform(-0.02, 0.02) for y in ys]
                offs = _swarm_offsets(ys, VIOLIN_WIDTH * 0.45)
                ax.scatter(
                    [ci + _offset(vi) + o for o in offs],
                    ys,
                    s=4,
                    color=color,
                    alpha=0.7,
                    linewidth=0,
                )
        handles = [
            mpatches.Patch(facecolor=c, alpha=0.7, label=lab)
            for c, lab in zip(COLORS, labels)
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=9,
            ncol=len(handles),
        )
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("Relative Error (worse of Re, Im)", fontsize=11)
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels([AXIS_LABELS[c] for c in group], fontsize=10)
        ax.set_xlim(-0.8, len(group) - 0.2)
        ax.grid(axis="y", alpha=0.3)
        # As in accuracy.svg: 80% data, 5% below, 15% above (for the legend)
        unit = (hi - lo) / 0.80
        ax.set_ylim(lo - unit * 0.05, hi + unit * 0.15)
        _format_error_ticks(ax)
        # No ticks above the clipping row, whose values are clipped there
        ax.yaxis.set_major_locator(FixedLocator(list(range(-16, hi + 1, 4))))
        ax.yaxis.set_minor_locator(NullLocator())
        fmt = ax.yaxis.get_major_formatter()
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: "≥" + fmt(x, pos) if x >= hi else fmt(x, pos))
        )
    for ext in ("svg", "pdf"):
        fig.savefig(
            os.path.join(IMAGES, f"componentwise_accuracy.{ext}"),
            format=ext,
            bbox_inches="tight",
        )
    plt.close(fig)


# ── Main ──


def update_readme(md):
    path = os.path.join(ROOT, "README.md")
    start, end = "<!-- differences-start -->", "<!-- differences-end -->"
    with open(path) as f:
        readme = f.read()
    si, ei = readme.find(start), readme.find(end)
    if si == -1 or ei == -1:
        print("    WARNING: differences markers not found in README.md", flush=True)
        return
    with open(path, "w") as f:
        f.write(readme[: si + len(start)] + "\n" + md + "\n" + readme[ei:])


def main():
    # (function, ν, Re z, Im z, componentwise reference?)
    table_pts = [(r[2], r[3], r[4], r[5], r[6] != "value") for r in TABLE]
    axis_pts = axis_points() + scaled_points()
    points = table_pts + [(*p[:4], True) for p in axis_pts]

    print("Implementation differences: references...", flush=True)
    refs = references(points)
    print("Implementation differences: implementations...", flush=True)
    results = {
        "rust": run_rust(points),
        "fortran": run_fortran(points),
        "scipy": run_scipy(points),
        "julia": run_julia(points),
    }

    def res_at(i):
        return {impl: results[impl][i] for impl in IMPLS}

    table_res = [(TABLE[i], res_at(i), refs[i]) for i in range(len(TABLE))]

    # Axis sample: every value with a usable reference
    n_t = len(TABLE)
    axis_res = [
        (p[4], refs[n_t + j], res_at(n_t + j))
        for j, p in enumerate(axis_pts)
        if refs[n_t + j] is not None and UNDERFLOW <= math.hypot(*refs[n_t + j]) < 1e290
    ]
    print(f"  axis sample: {len(axis_res)} values", flush=True)

    with open(os.path.join(RESULTS, "differences.json"), "w") as f:
        json.dump(
            {
                "points": [list(p) for p in points],
                "references": refs,
                "results": results,
            },
            f,
        )

    plot_axis(axis_res, _legend_labels(_get_versions()))
    update_readme(table_markdown(table_res))

    # Summary on stdout
    for on_axis in (True, False):
        for impl in IMPLS:
            errs = []
            for c, ref, res in axis_res:
                r = res[impl]
                if (c[1] != "near") == on_axis and r["status"] in (
                    "ok",
                    "reduced_precision",
                ):
                    e = parts_error(r["re"], r["im"], ref)
                    if e is not None:
                        errs.append(e)
            if errs:
                errs.sort()
                where = "on  " if on_axis else "near"
                print(
                    f"  {where} {impl:8s} n={len(errs):5d} "
                    f"median={errs[len(errs) // 2]:.1e} "
                    f"p90={errs[int(len(errs) * 0.9)]:.1e} max={errs[-1]:.1e}"
                )
    if any(r is None for r in refs):
        print(f"  WARNING: {sum(r is None for r in refs)} references unresolved")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
