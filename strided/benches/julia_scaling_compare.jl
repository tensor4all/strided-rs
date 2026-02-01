# Scaling benchmarks matching Strided.jl's benchtests.jl.
#
# Usage:
#   JULIA_NUM_THREADS=1 julia --project=. benches/julia_scaling_compare.jl
#
# Measures performance across exponentially-scaled array sizes.
# Uses manual warmup + median timing (no BenchmarkTools) for speed.

using Strided
using Printf
using Statistics

println("Julia scaling benchmarks (cf. Strided.jl benchtests.jl)")
println("Julia Threads: ", Threads.nthreads())
println()

"""
Adaptive benchmark: warmup, calibrate iteration count, collect samples, return median.
Mirrors the Rust bench_adaptive() function.
"""
function bench_adaptive(f)
    # Warmup
    for _ in 1:3
        f()
    end

    # Calibrate: find how many iters fit in ~100ms
    t0 = time_ns()
    f()
    single_ns = time_ns() - t0
    if single_ns == 0
        iters = 10000
    else
        iters = clamp(div(100_000_000, single_ns), 3, 10000)
    end

    # Collect samples
    samples = Vector{Float64}(undef, iters)
    for i in 1:iters
        t0 = time_ns()
        f()
        samples[i] = Float64(time_ns() - t0)
    end

    return median(samples)  # in nanoseconds
end

function benchmark_sum()
    sizes = ceil.(Int, 2 .^ (2:1.5:20))
    println("=== benchmark_sum (1D) ===")
    @printf("%10s %12s %12s %12s %8s\n", "size", "base (us)", "strided_1t (us)", "strided_mt (us)", "ratio")

    for s in sizes
        A = randn(Float64, s)

        t_base = bench_adaptive(() -> sum(A))

        Strided.disable_threads()
        t_strided_1t = bench_adaptive(() -> @strided sum(A))

        Strided.enable_threads()
        t_strided_mt = bench_adaptive(() -> @strided sum(A))

        ratio = t_strided_1t / max(t_base, 1.0)
        @printf("%10d %12.3f %12.3f %12.3f %8.2fx\n",
                s, t_base / 1e3, t_strided_1t / 1e3, t_strided_mt / 1e3, ratio)
    end
    println()
end

function benchmark_permute(p, label)
    sizes = [4, 8, 12, 16, 24, 32, 48, 64]
    println("=== benchmark_permute $label ===")
    @printf("%6s %10s %12s %12s %12s %8s\n",
            "s", "s^4", "copy (us)", "strided_1t (us)", "strided_mt (us)", "ratio")

    for s in sizes
        total = s^4
        A = randn(Float64, s, s, s, s)
        B = similar(A)

        t_copy = bench_adaptive(() -> copy!(B, A))

        Strided.disable_threads()
        t_strided_1t = bench_adaptive(() -> @strided permutedims!(B, A, p))

        Strided.enable_threads()
        t_strided_mt = bench_adaptive(() -> @strided permutedims!(B, A, p))

        ratio = t_strided_1t / max(t_copy, 1.0)
        @printf("%6d %10d %12.3f %12.3f %12.3f %8.2fx\n",
                s, total, t_copy / 1e3, t_strided_1t / 1e3, t_strided_mt / 1e3, ratio)
    end
    println()
end

benchmark_sum()
benchmark_permute((4, 3, 2, 1), "(4,3,2,1)")
benchmark_permute((2, 3, 4, 1), "(2,3,4,1)")
benchmark_permute((3, 4, 1, 2), "(3,4,1,2)")
