using Strided
using BenchmarkTools
using Statistics

# Julia must run with 1 thread for fair comparison if Rust is single-threaded (mostly)
# or just compare as-is. strided-rs is single-threaded by default in these benches.
println("Julia Threads: ", Threads.nthreads())

function bench_symmetrize_4000()
    A = rand(4000, 4000)
    B = similar(A)
    # This is what Rust's benches/readme_examples.rs does
    t = @benchmark @strided $B .= ($A .+ $A') ./ 2
    println("symmetrize_4000 (Strided): ", mean(t.times) / 1e6, " ms")
end

function bench_scale_transpose_1000()
    A = rand(1000, 1000)
    B = similar(A)
    t = @benchmark @strided $B .= 3 .* $A'
    println("scale_transpose_1000 (Strided): ", mean(t.times) / 1e6, " ms")
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

bench_symmetrize_4000()
bench_scale_transpose_1000()
bench_complex_elementwise_1000()
bench_permute_32_4d()
bench_multiple_permute_sum_32_4d()
