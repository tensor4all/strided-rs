using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_matmul(T::DataType)
    println("matmul (OMEinsum): $T")
    # (1) square: n1 = n2 = n3
    m1 = rand(T, 1000, 1000)
    t1 = @benchmark ein"ij,jk -> ik"($m1, $m1)
    println("  (1) square 1000×1000: ", mean(t1.times) / 1e6, " ms")
    # (2) n1 = n3 >> n2
    m2a = rand(T, 2000, 50)
    m2b = rand(T, 50, 2000)
    t2 = @benchmark ein"ij,jk -> ik"($m2a, $m2b)
    println("  (2) (2000,50)*(50,2000): ", mean(t2.times) / 1e6, " ms")
    # (3) n1 = n3 << n2
    m3a = rand(T, 50, 2000)
    m3b = rand(T, 2000, 50)
    t3 = @benchmark ein"ij,jk -> ik"($m3a, $m3b)
    println("  (3) (50,2000)*(2000,50): ", mean(t3.times) / 1e6, " ms")
    println()
end

bench_matmul(Float64)
bench_matmul(ComplexF64)
