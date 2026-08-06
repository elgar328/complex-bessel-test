# Compute Bessel/Airy values with SpecialFunctions.jl and measure performance.
#
# Usage: julia --project=julia julia/compute.jl compute|bench
# Reads results/test_grid.json; writes results/julia_results.json (same
# records as scipy_results.json) or results/julia_bench.json.

using JSON
using SpecialFunctions

const RESULTS = joinpath(@__DIR__, "..", "results")

const FUNCS = Dict{String,Function}(
    "besselj" => besselj,
    "bessely" => bessely,
    "besseli" => besseli,
    "besselk" => besselk,
    "hankel1" => hankelh1,
    "hankel2" => hankelh2,
    "airy" => (nu, z) -> airyai(z),
    "airyprime" => (nu, z) -> airyaiprime(z),
    "biry" => (nu, z) -> airybi(z),
    "biryprime" => (nu, z) -> airybiprime(z),
    "besselj_scaled" => besseljx,
    "bessely_scaled" => besselyx,
    "besseli_scaled" => besselix,
    "besselk_scaled" => besselkx,
    "hankel1_scaled" => hankelh1x,
    "hankel2_scaled" => hankelh2x,
    "airy_scaled" => (nu, z) -> airyaix(z),
    "airyprime_scaled" => (nu, z) -> airyaiprimex(z),
    "biry_scaled" => (nu, z) -> airybix(z),
    "biryprime_scaled" => (nu, z) -> airybiprimex(z),
)

const GRIDS = (("unscaled", "unscaled_grid"), ("scaled", "scaled_grid"))

"""(re, im, status) of f(nu, z); errors (Amos IERR 1, 2, 4, 5) and
non-finite values become a status, as in compute_scipy.py."""
function evaluate(f, nu, z)
    v = try
        ComplexF64(f(nu, z))
    catch e
        e isa SpecialFunctions.AmosException && return (nothing, nothing, "error")
        e isa DomainError && return (nothing, nothing, "error")
        rethrow()
    end
    (isinf(real(v)) || isinf(imag(v))) && return (nothing, nothing, "inf")
    (isnan(real(v)) || isnan(imag(v))) && return (nothing, nothing, "nan")
    return (real(v), imag(v), "ok")
end

function run_compute(grid)
    results = Dict{String,Any}[]
    for (gname, gkey) in GRIDS
        spec = grid[gkey]
        nus, res, ims = Float64.(spec["nu_values"]), Float64.(spec["re_values"]), Float64.(spec["im_values"])
        nre, nim = length(res), length(ims)
        for func in spec["functions"]
            println("  julia: $gname/$func...")
            f = FUNCS[func]
            for (ni, nu) in enumerate(nus), (ri, re) in enumerate(res), (ii, im) in enumerate(ims)
                idx = (ni - 1) * nre * nim + (ri - 1) * nim + (ii - 1)
                vre, vim, status = evaluate(f, nu, complex(re, im))
                push!(results, Dict(
                    "grid" => gname, "function" => func, "idx" => idx,
                    "nu" => nu, "z_re" => re, "z_im" => im,
                    "re" => vre, "im" => vim, "status" => status,
                ))
            end
        end
    end
    open(joinpath(RESULTS, "julia_results.json"), "w") do io
        JSON.json(io, results)
    end
    ok = count(r -> r["status"] == "ok", results)
    println("julia compute done: $ok/$(length(results)) ok")
end

# Time per call over `nrep` calls, as in the Rust/Fortran/SciPy benchmarks.
# A point where the call throws is timed with a try block around each call
# (the error is part of the cost there); elsewhere without, as try/catch
# itself costs time.
@noinline function time_calls(f::F, nu, z, nrep) where {F}
    throws = try
        f(nu, z)
        false
    catch
        true
    end
    acc = 0.0
    t0 = time_ns()
    if throws
        for _ in 1:nrep
            try
                acc += real(f(nu, z))
            catch
            end
        end
    else
        for _ in 1:nrep
            acc += real(f(nu, z))
        end
    end
    t1 = time_ns()
    return (t1 - t0) ÷ nrep, acc
end

function run_bench(grid)
    println("Benchmarking julia...")
    records = Dict{String,Any}[]
    sink = 0.0
    for (gname, gkey) in GRIDS
        spec = grid[gkey]
        nus, res, ims = Float64.(spec["nu_values"]), Float64.(spec["re_values"]), Float64.(spec["im_values"])
        for func in spec["functions"]
            f = FUNCS[func]
            # Warm up (compiles the method)
            for _ in 1:100
                sink += time_calls(f, 0.5, complex(1.0, 2.0), 1)[2]
            end
            for nu in nus, re in res, im in ims
                z = complex(re, im)
                time_calls(f, nu, z, 1)
                t, a = time_calls(f, nu, z, 1000)
                sink += a
                push!(records, Dict(
                    "grid" => gname, "function" => func, "nu" => nu,
                    "z_re" => re, "z_im" => im, "time_ns" => t,
                ))
            end
        end
    end
    open(joinpath(RESULTS, "julia_bench.json"), "w") do io
        JSON.json(io, records)
    end
    println("julia bench done: $(length(records)) points (checksum $(isfinite(sink)))")
end

const AMOS_STATUS = Dict(1 => "invalid_input", 2 => "overflow", 4 => "total_precision_loss",
    5 => "convergence_failure")

"""(re, im, status) of f(nu, x); the status names the error (Amos error,
DomainError) instead of a generic "error"."""
function evaluate_named(f, nu, x)
    v = try
        ComplexF64(f(nu, x))
    catch e
        e isa SpecialFunctions.AmosException &&
            return (nothing, nothing, get(AMOS_STATUS, e.id, "error"))
        e isa DomainError && return (nothing, nothing, "domain_error")
        rethrow()
    end
    (isinf(real(v)) || isinf(imag(v))) && return (nothing, nothing, "inf")
    (isnan(real(v)) || isnan(imag(v))) && return (nothing, nothing, "nan")
    return (real(v), imag(v), "ok")
end

# Implementation differences: results/differences_points.json →
# results/differences_julia.json
# (complex arguments, as everywhere in the comparison).
function run_points()
    points = JSON.parsefile(joinpath(RESULTS, "differences_points.json"))
    out = map(points) do p
        f = FUNCS[p["function"]]
        nu, re, im = Float64(p["nu"]), Float64(p["z_re"]), Float64(p["z_im"])
        cre, cim, cst = evaluate_named(f, nu, complex(re, im))
        Dict{String,Any}("re" => cre, "im" => cim, "status" => cst)
    end
    open(joinpath(RESULTS, "differences_julia.json"), "w") do io
        JSON.json(io, out)
    end
    println("julia points done: $(length(out)) points")
end

function main()
    mode = isempty(ARGS) ? "compute" : ARGS[1]
    mode == "points" && return run_points()
    grid = JSON.parsefile(joinpath(RESULTS, "test_grid.json"))
    mode == "compute" && return run_compute(grid)
    mode == "bench" && return run_bench(grid)
    error("unknown mode $mode (compute, bench or points)")
end

main()
