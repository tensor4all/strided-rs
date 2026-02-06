using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_indexsum(T::DataType)
    println("indexsum (OMEinsum): $T")
    m = rand(T, 100, 100, 100)
    t = @benchmark ein"ijk -> ik"($m)
    println("indexsum (OMEinsum): ", mean(t.times) / 1e6, " ms")
    println()
end

bench_indexsum(Float64)
bench_indexsum(ComplexF64)
