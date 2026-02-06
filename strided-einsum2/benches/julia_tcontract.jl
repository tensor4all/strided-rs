using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_tcontract(T::DataType)
    println("tcontract (OMEinsum): $T")
    # (1) square: n_i = n_j = n_k = n_l = 30
    m1 = rand(T, 30, 30, 30)
    t1 = @benchmark ein"ijk, jlk -> il"($m1, $m1)
    println("  (1) square (30,30,30): ", mean(t1.times) / 1e6, " ms")
    # (2) n_i = n_l >> n_j: A(2000,50,50), B(50,2000,50)
    m2a = rand(T, 2000, 50, 50)
    m2b = rand(T, 50, 2000, 50)
    t2 = @benchmark ein"ijk, jlk -> il"($m2a, $m2b)
    println("  (2) (2000,50,50)*(50,2000,50): ", mean(t2.times) / 1e6, " ms")
    # (3) n_i = n_l << n_j: A(50,2000,2000), B(2000,50,2000)
    m3a = rand(T, 50, 2000, 2000)
    m3b = rand(T, 2000, 50, 2000)
    t3 = @benchmark ein"ijk, jlk -> il"($m3a, $m3b)
    println("  (3) (50,2000,2000)*(2000,50,2000): ", mean(t3.times) / 1e6, " ms")
    println()
end

bench_tcontract(Float64)
bench_tcontract(ComplexF64)
