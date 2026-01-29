using Strided
using BenchmarkTools
using Statistics

function bench_permute_naive()
    A = randn(32, 32, 32, 32)
    B = similar(A)
    # Naive nested loops for permutedims!(B, A, (4,3,2,1))
    t = @benchmark begin
        for i4 in 1:32, i3 in 1:32, i2 in 1:32, i1 in 1:32
            $B[i1, i2, i3, i4] = $A[i4, i3, i2, i1]
        end
    end
    println("permute_32_4d (Naive Julia): ", mean(t.times) / 1e6, " ms")
end

function bench_permute_builtin()
    A = randn(32, 32, 32, 32)
    B = similar(A)
    t = @benchmark permutedims!($B, $A, (4,3,2,1))
    println("permute_32_4d (Built-in Julia): ", mean(t.times) / 1e6, " ms")
end

function bench_permute_strided()
    A = randn(32, 32, 32, 32)
    B = similar(A)
    t = @benchmark @strided permutedims!($B, $A, (4,3,2,1))
    println("permute_32_4d (Strided Julia): ", mean(t.times) / 1e6, " ms")
end

bench_permute_naive()
bench_permute_builtin()
bench_permute_strided()
