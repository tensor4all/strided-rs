using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_dot(T::DataType)
    println("dot (OMEinsum): $T")
    # (1) square: n1 = n2 = n3 = 100
    m1 = rand(T, 100, 100, 100)
    t1 = @benchmark ein"ijk,ijk -> "($m1, $m1)
    println("  (1) square (100,100,100): ", mean(t1.times) / 1e6, " ms")
    # (2) n1 = n3 >> n2
    m2 = rand(T, 2000, 50, 2000)
    t2 = @benchmark ein"ijk,ijk -> "($m2, $m2)
    println("  (2) (2000,50,2000): ", mean(t2.times) / 1e6, " ms")
    # (3) n1 = n3 << n2
    m3 = rand(T, 50, 2000, 50)
    t3 = @benchmark ein"ijk,ijk -> "($m3, $m3)
    println("  (3) (50,2000,50): ", mean(t3.times) / 1e6, " ms")
    println()
end

bench_dot(Float64)
bench_dot(ComplexF64)
