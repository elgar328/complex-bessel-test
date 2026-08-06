"""Generate SVG visualizations and summary table from comparison results.

Produces 2 SVG files in images/ directory:
1. accuracy.svg  — Precision: each impl vs mpmath
2. eval_time.svg — Performance comparison

Also updates README.md with a markdown summary table.
"""

import json
import math
import os
import platform
import subprocess as _subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")


def _get_versions():
    """Collect library versions for legend labels."""
    import subprocess
    import scipy
    import mpmath

    # complex-bessel version via cargo metadata
    try:
        meta = subprocess.check_output(
            ["cargo", "metadata", "--format-version=1"],
            cwd=os.path.join(os.path.dirname(__file__), ".."),
            text=True,
        )
        packages = json.loads(meta)["packages"]
        rust_ver = next(p["version"] for p in packages if p["name"] == "complex-bessel")
    except (FileNotFoundError, subprocess.CalledProcessError, StopIteration, KeyError):
        rust_ver = None

    # gfortran version (e.g. "14.2.0" from "GNU Fortran ... 14.2.0")
    try:
        out = subprocess.check_output(["gfortran", "--version"], text=True)
        gfortran_ver = out.splitlines()[0].split()[-1]
    except (FileNotFoundError, subprocess.CalledProcessError):
        gfortran_ver = None

    # Julia and SpecialFunctions.jl versions from the julia/ project
    julia_dir = os.path.join(os.path.dirname(__file__), "..", "julia")
    try:
        out = subprocess.check_output(
            [
                "julia",
                f"--project={julia_dir}",
                "-e",
                'using Pkg; print(VERSION, " ", '
                'Pkg.dependencies()[Base.UUID("276daf66-3868-5448-9aa4-cd146d93841b")].version)',
            ],
            text=True,
        )
        julia_ver, specfun_ver = out.split()
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        julia_ver, specfun_ver = None, None

    return {
        "rust": rust_ver,
        "gfortran": gfortran_ver,
        "scipy": scipy.__version__,
        "mpmath": mpmath.__version__,
        "julia": julia_ver,
        "specialfunctions": specfun_ver,
    }


def _legend_labels(versions):
    """Short legend names; the versions go into a caption (versions_text)."""
    return ["complex-bessel", "Amos (Fortran)", "SciPy", "SpecialFunctions.jl"]


def versions_text(versions, references=True):
    """One line naming each implementation with its version and language
    (and the mpmath version of the references)."""

    def fmt(name, ver):
        return f"{name} {ver}" if ver else name

    gf = versions.get("gfortran")
    jl = versions.get("julia")
    py = platform.python_version()
    return (
        fmt("complex-bessel", versions.get("rust"))
        + " (Rust), Amos/TOMS 644 (Fortran"
        + (f", gfortran {gf}" if gf else "")
        + "), "
        + fmt("SciPy", versions.get("scipy"))
        + f" (Python {py}), "
        + fmt("SpecialFunctions.jl", versions.get("specialfunctions"))
        + (f" (Julia {jl})" if jl else " (Julia)")
        + (
            "; references: " + fmt("mpmath", versions.get("mpmath"))
            if references
            else ""
        )
    )


def _update_versions(readme_path, versions):
    """Insert the versions sentence between markers in README."""
    start_marker = "<!-- versions-start -->"
    end_marker = "<!-- versions-end -->"
    with open(readme_path) as f:
        readme = f.read()
    si, ei = readme.find(start_marker), readme.find(end_marker)
    if si == -1 or ei == -1:
        return
    line = "Versions: " + versions_text(versions) + "."
    with open(readme_path, "w") as f:
        f.write(readme[: si + len(start_marker)] + line + readme[ei:])


# One entry per implementation, in the order of _legend_labels
IMPLS = ["rust", "fortran", "scipy", "julia"]
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
# Width of one violin in a group of len(IMPLS)
VIOLIN_WIDTH = 0.8 / len(IMPLS)


def _offset(i):
    """x offset of implementation i within a function's group."""
    return (i - (len(IMPLS) - 1) / 2) * VIOLIN_WIDTH


# Display order for functions
UNSCALED_FUNCS = [
    "besselj",
    "bessely",
    "besseli",
    "besselk",
    "hankel1",
    "hankel2",
    "airy",
    "airyprime",
    "biry",
    "biryprime",
]
SCALED_FUNCS = [
    "besselj_scaled",
    "bessely_scaled",
    "besseli_scaled",
    "besselk_scaled",
    "hankel1_scaled",
    "hankel2_scaled",
    "airy_scaled",
    "airyprime_scaled",
    "biry_scaled",
    "biryprime_scaled",
]
ALL_FUNCS = UNSCALED_FUNCS + SCALED_FUNCS

SHORT_NAMES = {
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
    "besselj_scaled": "J",
    "bessely_scaled": "Y",
    "besseli_scaled": "I",
    "besselk_scaled": "K",
    "hankel1_scaled": "H1",
    "hankel2_scaled": "H2",
    "airy_scaled": "Ai",
    "airyprime_scaled": "Ai'",
    "biry_scaled": "Bi",
    "biryprime_scaled": "Bi'",
}


def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path) as f:
        return json.load(f)


def safe_log10(errors):
    """Convert errors to log10, clamping zeros to -17."""
    return [math.log10(max(e, 1e-17)) for e in errors]


OUTLIER_THRESHOLD_UNSCALED = -12  # log10(1e-12) → displays as 10⁻¹²
OUTLIER_THRESHOLD_SCALED = -10  # log10(1e-10) → displays as 10⁻¹⁰


def auto_ylim(datasets_list, pad=1.0):
    """Compute y-axis limits from the full range of all data."""
    all_vals = []
    for datasets in datasets_list:
        for d in datasets:
            all_vals.extend(d)
    if not all_vals:
        return (-17, 0)
    lo = min(all_vals)
    hi = max(all_vals)
    return (lo - pad, hi + pad)


def make_violin(ax, positions, datasets, colors, labels, title, ylabel):
    """Create a grouped violin plot (no outlier handling)."""
    width = VIOLIN_WIDTH
    all_parts = []

    for i, (data_list, color, label) in enumerate(zip(datasets, colors, labels)):
        valid_pos = []
        valid_data = []
        for j, d in enumerate(data_list):
            if d and len(d) > 1:
                valid_pos.append(positions[j] + _offset(i))
                valid_data.append(d)

        if valid_data:
            parts = ax.violinplot(
                valid_data,
                positions=valid_pos,
                widths=width * 0.9,
                showmedians=True,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            parts["cmedians"].set_color(color)
            all_parts.append((mpatches.Patch(facecolor=color, alpha=0.7), label))

    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=11)
    if all_parts:
        ax.legend(*zip(*all_parts), loc="upper right", fontsize=9, ncol=len(all_parts))
    ax.grid(axis="y", alpha=0.3)


def _format_error_ticks(ax, exp_offset=0):
    """Format y-axis ticks as 10⁻ⁿ instead of raw log10 values.

    exp_offset shifts the displayed exponent.
    """
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    major = 5 if span > 30 else 2
    ax.yaxis.set_major_locator(MultipleLocator(major))
    ax.yaxis.set_minor_locator(MultipleLocator(major // 2 if major > 2 else 1))

    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

    def fmt(x, pos):
        exp = int(x) + exp_offset
        return f"10{str(exp).translate(sup)}"

    ax.yaxis.set_major_formatter(FuncFormatter(fmt))


def _draw_violin_with_outliers(
    ax, positions, x_labels, datasets, colors, labels, title, threshold=None
):
    """Draw grouped violin with outlier scatter dots.

    Data above threshold is excluded from the violin body
    and shown as individual scatter points with jitter.
    """
    width = VIOLIN_WIDTH
    if threshold is None:
        threshold = OUTLIER_THRESHOLD_UNSCALED
    all_parts = []
    rng = np.random.default_rng(42)

    for i, (data_list, color, label) in enumerate(zip(datasets, colors, labels)):
        valid_pos = []
        valid_data = []

        for j, d in enumerate(data_list):
            if not d or len(d) <= 1:
                continue
            pos = positions[j] + _offset(i)
            main = [v for v in d if v <= threshold]
            outliers = [v for v in d if v > threshold]

            if main and len(main) > 1:
                valid_pos.append(pos)
                valid_data.append(main)

            if outliers:
                jitter = rng.uniform(-width * 0.3, width * 0.3, len(outliers))
                ax.scatter(
                    [pos + j for j in jitter],
                    outliers,
                    s=20,
                    color=color,
                    alpha=0.7,
                    zorder=6,
                    edgecolors="white",
                    linewidth=0.5,
                )

        if valid_data:
            parts = ax.violinplot(
                valid_data,
                positions=valid_pos,
                widths=width * 0.9,
                showmedians=True,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            parts["cmedians"].set_color(color)
            all_parts.append((mpatches.Patch(facecolor=color, alpha=0.7), label))

    # Main axis
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Relative Error", fontsize=11)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=10)
    if all_parts:
        ax.legend(*zip(*all_parts), loc="upper right", fontsize=9, ncol=len(all_parts))
    ax.grid(axis="y", alpha=0.3)

    # y-limits: 5% padding below, 10% padding above
    all_vals = [v for dl in datasets for d in dl for v in d]
    data_lo = min(all_vals) if all_vals else -17
    data_hi = max(all_vals) if all_vals else 0
    span = data_hi - data_lo if data_hi > data_lo else 1.0
    # total = 80% data + 5% bottom + 15% top
    unit = span / 0.80
    ax.set_ylim(data_lo - unit * 0.05, data_hi + unit * 0.15)

    # Format ticks after ylim is set (so span calculation is accurate)
    _format_error_ticks(ax)

    # Shared x-range
    xlim = (min(positions) - 0.8, max(positions) + 0.8)
    ax.set_xlim(xlim)


# ── 1. Accuracy violin ──
def plot_accuracy_violin(labels):
    print("  accuracy.svg...", flush=True)
    comparison = load_json("comparison.json")
    dist = comparison["error_distributions"]["accuracy_vs_mpmath"]

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 1, hspace=0.25)

    # ── Unscaled ──
    ax0 = fig.add_subplot(gs[0])

    funcs = [f for f in UNSCALED_FUNCS if f in dist]
    x_labels = [SHORT_NAMES.get(f, f) for f in funcs]
    positions = list(range(len(funcs)))

    _draw_violin_with_outliers(
        ax0,
        positions,
        x_labels,
        [[safe_log10(dist[f].get(impl, [])) for f in funcs] for impl in IMPLS],
        COLORS,
        labels,
        "Accuracy — Unscaled Functions",
        threshold=OUTLIER_THRESHOLD_UNSCALED,
    )

    # ── Scaled ──
    ax1 = fig.add_subplot(gs[1])

    s_funcs = [f for f in SCALED_FUNCS if f in dist]
    s_labels = [SHORT_NAMES.get(f, f) for f in s_funcs]
    s_positions = list(range(len(s_funcs)))

    _draw_violin_with_outliers(
        ax1,
        s_positions,
        s_labels,
        [[safe_log10(dist[f].get(impl, [])) for f in s_funcs] for impl in IMPLS],
        COLORS,
        labels,
        "Accuracy — Scaled Functions",
        threshold=OUTLIER_THRESHOLD_SCALED,
    )

    fig.savefig(
        os.path.join(IMAGES_DIR, "accuracy.svg"), format="svg", bbox_inches="tight"
    )
    fig.savefig(
        os.path.join(IMAGES_DIR, "accuracy.pdf"), format="pdf", bbox_inches="tight"
    )
    plt.close(fig)


# ── 2. Performance violin ──
def plot_performance_violin(labels, headline=None):
    print("  eval_time.svg...", flush=True)

    benches = [load_json(f"{impl}_bench.json") for impl in IMPLS]

    # Group by base function (merge scaled + unscaled), convert ns to us
    MERGE_MAP = {
        "besselj": "J",
        "besselj_scaled": "J",
        "bessely": "Y",
        "bessely_scaled": "Y",
        "besseli": "I",
        "besseli_scaled": "I",
        "besselk": "K",
        "besselk_scaled": "K",
        "hankel1": "H1",
        "hankel1_scaled": "H1",
        "hankel2": "H2",
        "hankel2_scaled": "H2",
        "airy": "Ai",
        "airy_scaled": "Ai",
        "airyprime": "Ai'",
        "airyprime_scaled": "Ai'",
        "biry": "Bi",
        "biry_scaled": "Bi",
        "biryprime": "Bi'",
        "biryprime_scaled": "Bi'",
    }
    BASE_ORDER = ["J", "Y", "I", "K", "H1", "H2", "Ai", "Ai'", "Bi", "Bi'"]

    def group_by_base(records):
        groups = {}
        for r in records:
            base = MERGE_MAP.get(r["function"])
            if base is None:
                continue
            if base not in groups:
                groups[base] = []
            groups[base].append(r["time_ns"] / 1000.0)
        return groups

    groups = [group_by_base(b) for b in benches]

    funcs = [f for f in BASE_ORDER if any(f in g for g in groups)]
    x_labels = funcs
    positions = list(range(len(funcs)))

    datasets = [
        [np.log10(np.array(g.get(f, [0.1])) + 0.001).tolist() for f in funcs]
        for g in groups
    ]

    fig, ax = plt.subplots(figsize=(11, 5))

    # The average-speedup takeaway lives in the chart title, so it stays
    # up to date everywhere the SVG is hotlinked (e.g. complex-bessel README).
    if headline:
        title = f"Evaluation Time  |  {headline}"
    else:
        title = "Evaluation Time (lower is better)"

    make_violin(
        ax,
        positions,
        datasets,
        COLORS,
        labels,
        title,
        "Time per call",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=10)

    # y-axis limits: 5% padding below, 15% padding above
    all_vals = [v for dl in datasets for d in dl for v in d]
    data_lo, data_hi = min(all_vals), max(all_vals)
    span = data_hi - data_lo if data_hi > data_lo else 1.0
    unit = span / 0.80
    ax.set_ylim(data_lo - unit * 0.05, data_hi + unit * 0.15)

    # y-axis ticks: display actual time units
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    ax.yaxis.set_major_locator(MultipleLocator(1))

    def fmt_time(x, pos):
        us = 10**x
        if us >= 1000:
            return f"{us / 1000:.0f}ms"
        elif us >= 1:
            return f"{us:.0f}μs"
        elif us >= 0.001:
            return f"{us * 1000:.0f}ns"
        else:
            return f"{us * 1e6:.0f}ps"

    ax.yaxis.set_major_formatter(FuncFormatter(fmt_time))

    fig.tight_layout()
    fig.savefig(
        os.path.join(IMAGES_DIR, "eval_time.svg"), format="svg", bbox_inches="tight"
    )
    fig.savefig(
        os.path.join(IMAGES_DIR, "eval_time.pdf"), format="pdf", bbox_inches="tight"
    )
    plt.close(fig)


# ── 4. Summary table (markdown) ──
def plot_summary_table(versions):
    print("  summary table (README.md)...", flush=True)
    comparison = load_json("comparison.json")
    acc = comparison["accuracy_vs_mpmath"]
    rvf = comparison["rust_vs_fortran"]
    err_sum = comparison["error_summary"]

    bench_rust = load_json("rust_bench.json")

    # Group bench by function
    bench_groups = {}
    for r in bench_rust:
        func = r["function"]
        if func not in bench_groups:
            bench_groups[func] = []
        bench_groups[func].append(r["time_ns"] / 1000.0)

    def build_rows(func_list):
        rows = []
        for func in func_list:
            if func not in acc and func not in err_sum:
                continue
            dname = SHORT_NAMES.get(func, func)

            es = err_sum.get(func, {})
            total = es.get("total", 0)
            ok = es.get("ok", 0)
            warn = es.get("reduced_precision", 0)
            err = total - ok - warn
            owe = f"{ok} / {warn} / {err}"

            rvf_data = rvf.get(func, {})
            sm = rvf_data.get("status_match", 0)
            smm = rvf_data.get("status_mismatch", 0)
            sm_total = sm + smm
            match_pct = f"{100 * sm / sm_total:.1f}" if sm_total > 0 else "N/A"

            acc_rust = acc.get(func, {}).get("rust", {})
            med_err = acc_rust.get("median_err")
            med_err_str = f"{med_err:.1e}" if med_err is not None else "N/A"

            times = bench_groups.get(func, [])
            median_time = f"{np.median(times):.2f}" if times else "N/A"

            rows.append(
                f"| {dname} | {total} | {owe} | {match_pct} | {med_err_str} | {median_time} |"
            )
        return rows

    header = "| Func | Points | Ok / Warn / Err | Match (%) | Rel Err | Time (μs) |"
    sep = "|:----:|:------:|:---------------:|:---------:|:-------:|:---------:|"

    lines = []
    lines.append("**Unscaled Functions**")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    lines.extend(build_rows(UNSCALED_FUNCS))
    lines.append("")
    lines.append("**Scaled Functions**")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    lines.extend(build_rows(SCALED_FUNCS))
    lines.append("")
    lines.append(
        "> **Points**: test grid points per function.  \n"
        "> **Ok / Warn / Err**: result counts by status (Warn: reduced precision, Fortran IERR=3).  \n"
        "> **Match**: Rust–Fortran agreement on Ok/Warn/Err status.  \n"
        "> **Rel Err**: median relative error vs mpmath (50+ digit precision).  \n"
        "> **Time**: median evaluation time per call."
    )

    md_table = "\n".join(lines)

    # Insert into README.md between markers
    readme_path = os.path.join(os.path.dirname(__file__), "..", "README.md")
    start_marker = "<!-- summary-table-start -->"
    end_marker = "<!-- summary-table-end -->"

    with open(readme_path) as f:
        readme = f.read()

    start_idx = readme.find(start_marker)
    end_idx = readme.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        print("    WARNING: summary table markers not found in README.md", flush=True)
        return

    new_readme = (
        readme[: start_idx + len(start_marker)]
        + "\n"
        + md_table
        + "\n"
        + readme[end_idx:]
    )

    with open(readme_path, "w") as f:
        f.write(new_readme)


def _get_cpu_name():
    """Return a human-readable CPU name, e.g. 'Apple M4 Max'."""
    system = platform.system()
    try:
        if system == "Darwin":
            return _subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if system == "Windows":
            return (
                _subprocess.check_output(["wmic", "cpu", "get", "Name"], text=True)
                .strip()
                .split("\n")[-1]
                .strip()
            )
    except Exception:
        pass
    return platform.processor() or platform.machine()


# Names of the other implementations in the speedup sentence
SPEEDUP_NAMES = {"fortran": "Amos", "scipy": "SciPy", "julia": "SpecialFunctions.jl"}


def _compute_speedups():
    """Average time ratio of each other implementation to Rust.

    Per function the median times are compared; returns {impl: mean ratio}
    for the implementations whose bench data exists.
    """

    def median_by_func(bench_data):
        groups = {}
        for r in bench_data:
            groups.setdefault(r["function"], []).append(r["time_ns"])
        return {f: np.median(ts) for f, ts in groups.items()}

    rust_med = median_by_func(load_json("rust_bench.json"))
    ratios = {}
    for impl in SPEEDUP_NAMES:
        path = os.path.join(RESULTS_DIR, f"{impl}_bench.json")
        if not os.path.exists(path):
            continue
        med = median_by_func(load_json(f"{impl}_bench.json"))
        r = [med[f] / rust_med[f] for f in ALL_FUNCS if f in rust_med and f in med]
        if r:
            ratios[impl] = float(np.mean(r))
    return ratios


def _speedup_phrases(ratios, bold=False):
    """E.g. ["31% faster than Fortran", "8× faster than SciPy"]."""
    phrases = []
    for impl, ratio in ratios.items():
        if ratio >= 2:
            amount = f"{ratio:.0f}× faster"
        elif ratio >= 1:
            amount = f"{(ratio - 1) * 100:.0f}% faster"
        else:
            amount = f"{(1 - ratio) * 100:.0f}% slower"
        if bold:
            amount = f"**{amount}**"
        phrases.append(f"{amount} than {SPEEDUP_NAMES[impl]}")
    return phrases


def _join(phrases):
    if len(phrases) <= 2:
        return " and ".join(phrases)
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


def _update_bench_speedup(readme_path, ratios):
    """Update the average-speedup sentence in README."""
    phrases = _speedup_phrases(ratios, bold=True)
    if not phrases:
        return

    line = "On average across all functions, complex-bessel is " + _join(phrases) + "."

    start_marker = "<!-- bench-speedup-start -->"
    end_marker = "<!-- bench-speedup-end -->"
    with open(readme_path) as f:
        readme = f.read()
    si = readme.find(start_marker)
    ei = readme.find(end_marker)
    if si == -1 or ei == -1:
        return
    new = readme[: si + len(start_marker)] + line + readme[ei:]
    with open(readme_path, "w") as f:
        f.write(new)


def _update_bench_env(readme_path):
    """Insert benchmark environment info between markers in README."""
    start_marker = "<!-- bench-env-start -->"
    end_marker = "<!-- bench-env-end -->"

    with open(readme_path) as f:
        readme = f.read()

    start_idx = readme.find(start_marker)
    end_idx = readme.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        return

    cpu = _get_cpu_name()
    env_line = f"> Measured on {cpu}."

    new_readme = (
        readme[: start_idx + len(start_marker)]
        + "\n"
        + env_line
        + "\n"
        + readme[end_idx:]
    )

    with open(readme_path, "w") as f:
        f.write(new_readme)


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    versions = _get_versions()
    labels = _legend_labels(versions)

    ratios = _compute_speedups()
    headline = ", ".join(_speedup_phrases(ratios)) or None

    print("Generating dashboard SVGs...", flush=True)
    plot_accuracy_violin(labels)
    plot_performance_violin(labels, headline)
    plot_summary_table(versions)

    readme_path = os.path.join(os.path.dirname(__file__), "..", "README.md")
    _update_bench_speedup(readme_path, ratios)
    _update_bench_env(readme_path)
    _update_versions(readme_path, versions)

    print("Done! SVGs and PDFs saved to images/, summary table updated in README.md")


if __name__ == "__main__":
    main()
