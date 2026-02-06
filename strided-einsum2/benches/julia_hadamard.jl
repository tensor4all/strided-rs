using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_hadamard(T::DataType)
    println("hadamard (OMEinsum): $T")
    m = rand(T, 100, 100, 100)
    t = @benchmark ein"ijk,ijk -> ijk"($m, $m)
    println("hadamard (OMEinsum): ", mean(t.times) / 1e6, " ms")
    println()
end

bench_hadamard(Float64)
bench_hadamard(ComplexF64)
