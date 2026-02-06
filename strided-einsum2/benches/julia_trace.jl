using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_trace(T::DataType)
    println("trace (OMEinsum): $T")
    for m in [
        rand(T,1000, 1000)
    ]
        t = @benchmark ein"ii -> "($m)
        println("trace (OMEinsum): ", mean(t.times) / 1e6, " ms")
    end
    println()
end

bench_trace(Float64)
bench_trace(ComplexF64)
