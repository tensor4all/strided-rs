using OMEinsum
using BenchmarkTools
using Statistics
using Random

@info "Julia Threads: $(Threads.nthreads())"

function bench_manyinds(T::DataType)
    println("manyinds (OMEinsum): $T")
    code = ein"abcdefghijklmnop,flnqrcipstujvgamdwxyz->bcdeghkmnopqrstuvwxyz"
    arr1 = rand(T, map(i->2, OMEinsum.getixs(code)[1])...)
    arr2 = rand(T, map(i->2, OMEinsum.getixs(code)[2])...)
    t = @benchmark $code($arr1,$arr2)
    println("manyinds (OMEinsum): ", mean(t.times) / 1e6, " ms")
    println()
end

bench_manyinds(Float64)
bench_manyinds(ComplexF64)
