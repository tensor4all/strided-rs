using BenchmarkTools
using Strided

function run_readme_examples_single_thread()
    println("Threads: ", Threads.nthreads())

    A = randn(4000, 4000)
    B = similar(A)
    @btime $B .= ($A .+ $A') ./ 2
    @btime @strided $B .= ($A .+ $A') ./ 2

    A = randn(1000, 1000)
    B = similar(A)
    @btime $B .= 3 .* $A'
    @btime @strided $B .= 3 .* $A'
    @btime $B .= $A .* exp.(-2 .* $A) .+ sin.($A .* $A)
    @btime @strided $B .= $A .* exp.(-2 .* $A) .+ sin.($A .* $A)

    A = randn(32, 32, 32, 32)
    B = similar(A)
    @btime permutedims!($B, $A, (4, 3, 2, 1))
    @btime @strided permutedims!($B, $A, (4, 3, 2, 1))
    @btime $B .= permutedims($A, (1, 2, 3, 4)) .+
                permutedims($A, (2, 3, 4, 1)) .+
                permutedims($A, (3, 4, 1, 2)) .+
                permutedims($A, (4, 1, 2, 3))
    @btime @strided $B .= permutedims($A, (1, 2, 3, 4)) .+
                         permutedims($A, (2, 3, 4, 1)) .+
                         permutedims($A, (3, 4, 1, 2)) .+
                         permutedims($A, (4, 1, 2, 3))
end

run_readme_examples_single_thread()
