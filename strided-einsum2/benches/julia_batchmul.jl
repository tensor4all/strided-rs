using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_batchmul(T::DataType)
    println("matmul (OMEinsum): $T")
    for m in [
        rand(T, 1000, 1000, 3),
    ]
        t = @benchmark ein"ijk,jlk -> ilk"($m,$m)
        println("batchmul (OMEinsum): ", mean(t.times) / 1e6, " ms")
    end
    println()
end

bench_batchmul(Float64)
bench_batchmul(ComplexF64)
