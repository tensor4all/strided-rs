using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_starandcontract(T::DataType)
    println("starandcontract (OMEinsum): $T")
    m = rand(T, 50, 50)
    t = @benchmark ein"ij,ik,ik -> j"($m, $m, $m)
    println("starandcontract (OMEinsum): ", mean(t.times) / 1e6, " ms")
    println()
end

bench_starandcontract(Float64)
bench_starandcontract(ComplexF64)
