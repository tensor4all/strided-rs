using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_outer(T::DataType)
    println("outer (OMEinsum): $T")
    m = rand(T, 100, 100)
    t = @benchmark ein"ij,kl -> ijkl"($m, $m)
    println("outer (OMEinsum): ", mean(t.times) / 1e6, " ms")
    println()
end

bench_outer(Float64)
bench_outer(ComplexF64)
