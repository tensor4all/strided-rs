using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_ptrace(T::DataType)
    println("ptrace (OMEinsum): $T")
    for m in [
        rand(T, 100, 100, 100)
    ]
        t = @benchmark  ein"iij -> j"($m)
        println("ptrace (OMEinsum): ", mean(t.times) / 1e6, " ms")
    end
    println()
end

bench_ptrace(Float64)
bench_ptrace(ComplexF64)
