using Strided
using BenchmarkTools
using Statistics

println("Julia Threads: ", Threads.nthreads())
println("Strided threads: ", Strided.get_num_threads())
println()

function bench_symmetrize_4000()
    A = rand(4000, 4000)
    B = similar(A)
    t = @benchmark @strided $B .= ($A .+ $A') ./ 2
    println("symmetrize_4000 (Strided): ", mean(t.times) / 1e6, " ms")
end

function bench_scale_transpose_1000()
    A = rand(1000, 1000)
    B = similar(A)
    t = @benchmark @strided $B .= 3 .* $A'
    println("scale_transpose_1000 (Strided): ", mean(t.times) / 1e6, " ms")
end

function bench_mwe_stridedview_scale_transpose_1000()
    n = 1000
    A = rand(n, n)
    B = similar(A)
    svA = StridedView(A)
    svB = StridedView(B)
    t = @benchmark $svB .= 3 .* $svA'
    println("mwe_stridedview_scale_transpose_1000 (StridedView): ", mean(t.times) / 1e6, " ms")
end

function bench_complex_elementwise_1000()
    A = rand(1000, 1000)
    B = similar(A)
    t = @benchmark @strided $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A)
    println("complex_elementwise_1000 (Strided): ", mean(t.times) / 1e6, " ms")
end

function bench_permute_32_4d()
    A = randn(32, 32, 32, 32)
    B = similar(A)
    t = @benchmark @strided permutedims!($B, $A, (4,3,2,1))
    println("permute_32_4d (Strided): ", mean(t.times) / 1e6, " ms")
end

function bench_multiple_permute_sum_32_4d()
    A = randn(32, 32, 32, 32)
    B = similar(A)
    t = @benchmark @strided $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3))
    println("multiple_permute_sum_32_4d (Strided): ", mean(t.times) / 1e6, " ms")
end

function bench_sum_1m()
    A = randn(1_000_000)
    svA = StridedView(A)
    t = @benchmark sum($svA)
    println("sum_1m (StridedView): ", mean(t.times) / 1e6, " ms")
end

bench_symmetrize_4000()
bench_mwe_stridedview_scale_transpose_1000()
bench_scale_transpose_1000()
bench_complex_elementwise_1000()
bench_permute_32_4d()
bench_multiple_permute_sum_32_4d()
bench_sum_1m()
