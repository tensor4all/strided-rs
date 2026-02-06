using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_outer(T::DataType)
    println("outer (OMEinsum): $T")
    # (1) square: n1 = n2 = n3 = n4 = 40 (output 40^4)
    m1 = rand(T, 40, 40)
    t1 = @benchmark ein"ij,kl -> ijkl"($m1, $m1)
    println("  (1) square (40,40)x(40,40): ", mean(t1.times) / 1e6, " ms")
    # (2) n1=n2 >> n3=n4: (80,20) x (20,80)
    m2a = rand(T, 80, 20)
    m2b = rand(T, 20, 80)
    t2 = @benchmark ein"ij,kl -> ijkl"($m2a, $m2b)
    println("  (2) (80,20)x(20,80): ", mean(t2.times) / 1e6, " ms")
    # (3) n1=n2 << n3=n4: (20,80) x (80,20)
    m3a = rand(T, 20, 80)
    m3b = rand(T, 80, 20)
    t3 = @benchmark ein"ij,kl -> ijkl"($m3a, $m3b)
    println("  (3) (20,80)x(80,20): ", mean(t3.times) / 1e6, " ms")
    println()
end

bench_outer(Float64)
bench_outer(ComplexF64)
