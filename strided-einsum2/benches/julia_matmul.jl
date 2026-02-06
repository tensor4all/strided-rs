using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_matmul(T::DataType)
    println("matmul (OMEinsum): $T")
    for m in [
        rand(T, 1000, 1000),
    ]
        t = @benchmark ein"ij,jk -> ik"($m, $m)
        println("matmul (OMEinsum): ", mean(t.times) / 1e6, " ms")
    end
    println()
end

bench_matmul(Float64)
bench_matmul(ComplexF64)
