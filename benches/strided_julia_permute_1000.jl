using BenchmarkTools
using Strided

function bench_permute_1000()
    size = 1000
    a = reshape(collect(0.0:(size * size - 1)), size, size)
    b = similar(a)
    perm = (2, 1)

    println("Threads: ", Threads.nthreads())

    @btime permutedims!($b, $a, $perm)
    @btime @strided permutedims!($b, $a, $perm)
end

bench_permute_1000()
