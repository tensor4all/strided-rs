using BenchmarkTools
using Strided

function bench_transpose_4000()
    a = randn(4000, 4000)
    b = zeros(4000, 4000)
    
    println("Julia Native Transpose (4000x4000):")
    @btime $b .= $a'
    
    println("Julia Strided Transpose (4000x4000):")
    @btime @strided $b .= $a'
end

bench_transpose_4000()