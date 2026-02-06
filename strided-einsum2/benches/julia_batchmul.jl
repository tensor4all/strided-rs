using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_batchmul(T::DataType)
    println("batchmul (OMEinsum): $T")
    batch = 3
    # (1) square: n1 = n2 = n3
    m1 = rand(T, 1000, 1000, batch)
    t1 = @benchmark ein"ijk,jlk -> ilk"($m1, $m1)
    println("  (1) square (1000,1000,batch): ", mean(t1.times) / 1e6, " ms")
    # (2) n1 = n3 >> n2
    m2a = rand(T, 2000, 50, batch)
    m2b = rand(T, 50, 2000, batch)
    t2 = @benchmark ein"ijk,jlk -> ilk"($m2a, $m2b)
    println("  (2) (2000,50,b)*(50,2000,b): ", mean(t2.times) / 1e6, " ms")
    # (3) n1 = n3 << n2
    m3a = rand(T, 50, 2000, batch)
    m3b = rand(T, 2000, 50, batch)
    t3 = @benchmark ein"ijk,jlk -> ilk"($m3a, $m3b)
    println("  (3) (50,2000,b)*(2000,50,b): ", mean(t3.times) / 1e6, " ms")
    println()
end

bench_batchmul(Float64)
bench_batchmul(ComplexF64)
