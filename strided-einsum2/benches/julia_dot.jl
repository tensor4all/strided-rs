using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_dot(T::DataType)
    println("dot (OMEinsum): $T")
    for m in [
        rand(T,100, 100, 100)
    ]
        t = @benchmark ein"ijk,ijk -> "($m, $m)
        println("dot (OMEinsum): ", mean(t.times) / 1e6, " ms")
    end
    println()
end

bench_dot(Float64)
bench_dot(ComplexF64)
