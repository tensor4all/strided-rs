using Strided
using BenchmarkTools
using Statistics

println("Rank-25 tensor permutation benchmarks (Julia)")
println("Julia Threads: ", Threads.nthreads())
println()

function bench_permute_rank25(perm, label)
    dims = ntuple(_ -> 2, 25)
    A = randn(Float64, dims...)
    B = similar(A)

    println("=== permute_rank25_$label ===")
    println("Permutation (1-based): $perm")

    Strided.disable_threads()
    t = @benchmark @strided permutedims!($B, $A, $perm)
    println("julia_strided_1t: ", mean(t.times) / 1e6, " ms")

    Strided.enable_threads()
    t = @benchmark @strided permutedims!($B, $A, $perm)
    println("julia_strided_mt: ", mean(t.times) / 1e6, " ms")
    println()
end

# Full reversal: (25, 24, ..., 1) in 1-based
bench_permute_rank25(ntuple(i -> 26 - i, 25), "reverse")

# Cyclic shift by 1: (2, 3, ..., 25, 1) in 1-based
bench_permute_rank25(ntuple(i -> i == 25 ? 1 : i + 1, 25), "cyclic_shift_1")

# Pairwise swap: (2,1, 4,3, ..., 24,23, 25) in 1-based
bench_permute_rank25(
    ntuple(i -> i <= 24 ? (iseven(i) ? i - 1 : i + 1) : 25, 25),
    "pairwise_swap",
)
