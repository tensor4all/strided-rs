using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_perm(T::DataType)
    println("perm (OMEinsum): $T")
    for m in [
        rand(T,30, 30, 30, 30)
    ]
        t = @benchmark ein"ijkl -> ljki"($m)
        println("perm (OMEinsum): ", mean(t.times) / 1e6, " ms")
    end
    println()
end

bench_perm(Float64)
bench_perm(ComplexF64)
