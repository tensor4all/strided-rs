using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_diag(T::DataType)
    println("diag (OMEinsum): $T")
    for m in [
        rand(T, 100, 100, 100)
    ]
        t = @benchmark  ein"ijj -> ij"($m)
        println("diag (OMEinsum): ", mean(t.times) / 1e6, " ms")
    end
    println()
end

bench_diag(Float64)
bench_diag(ComplexF64)
