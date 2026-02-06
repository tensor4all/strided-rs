using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_tcontract(T::DataType)
    println("tcontract (OMEinsum): $T")
    for m in [
        rand(T,30, 30, 30)
    ]
        t = @benchmark ein"ijk, jlk -> il"($m,$m)
        println("tcontract (OMEinsum): ", mean(t.times) / 1e6, " ms")
    end
    println()
end

bench_tcontract(Float64)
bench_tcontract(ComplexF64)
