using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_star(T::DataType)
    println("star (OMEinsum): $T")
    m = rand(T, 100, 100)
    t = @benchmark ein"ij,ik,il -> jkl"($m, $m, $m)
    println("star (OMEinsum): ", mean(t.times) / 1e6, " ms")
    println()
end

bench_star(Float64)
bench_star(ComplexF64)
