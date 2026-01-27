using Pkg
Pkg.activate(temp=true)
Pkg.add("BenchmarkTools")
Pkg.add(path="extern/Strided.jl")

using BenchmarkTools
using Strided

println("Julia version: ", VERSION)
println("Number of threads: ", Threads.nthreads())
println()

# Benchmark 1: B = (A + A') / 2 for 4000x4000
println("=== Benchmark 1: Symmetrize 4000x4000 ===")
A = randn(4000, 4000)
B = similar(A)

t1 = @belapsed $B .= ($A .+ $A') ./ 2
println("Base: ", round(t1 * 1000, digits=3), " ms")

t2 = @belapsed @strided $B .= ($A .+ $A') ./ 2
println("Strided: ", round(t2 * 1000, digits=3), " ms")
println("Speedup: ", round(t1/t2, digits=2), "x")
println()

# Benchmark 2: B = 3 * A' for 1000x1000
println("=== Benchmark 2: Scale transpose 1000x1000 ===")
A = randn(1000, 1000)
B = similar(A)

t1 = @belapsed $B .= 3 .* $A'
println("Base: ", round(t1 * 1e6, digits=1), " µs")

t2 = @belapsed @strided $B .= 3 .* $A'
println("Strided: ", round(t2 * 1e6, digits=1), " µs")
println("Speedup: ", round(t1/t2, digits=2), "x")
println()

# Benchmark 3: Complex elementwise for 1000x1000
println("=== Benchmark 3: Complex elementwise 1000x1000 ===")
A = randn(1000, 1000)
B = similar(A)

t1 = @belapsed $B .= $A .* exp.(-2 .* $A) .+ sin.($A .* $A)
println("Base: ", round(t1 * 1000, digits=3), " ms")

t2 = @belapsed @strided $B .= $A .* exp.(-2 .* $A) .+ sin.($A .* $A)
println("Strided: ", round(t2 * 1000, digits=3), " ms")
println("Speedup: ", round(t1/t2, digits=2), "x")
println()

# Benchmark 4: permutedims! for 32x32x32x32
println("=== Benchmark 4: Permute 32x32x32x32 ===")
A = randn(32, 32, 32, 32)
B = similar(A)

t1 = @belapsed permutedims!($B, $A, (4,3,2,1))
println("Base: ", round(t1 * 1000, digits=3), " ms")

t2 = @belapsed @strided permutedims!($B, $A, (4,3,2,1))
println("Strided: ", round(t2 * 1000, digits=3), " ms")
println("Speedup: ", round(t1/t2, digits=2), "x")
println()

# Benchmark 5: Multiple permutedims sum
println("=== Benchmark 5: Multiple permute sum 32x32x32x32 ===")
A = randn(32, 32, 32, 32)
B = similar(A)

t1 = @belapsed $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3))
println("Base: ", round(t1 * 1000, digits=3), " ms")

t2 = @belapsed @strided $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3))
println("Strided: ", round(t2 * 1000, digits=3), " ms")
println("Speedup: ", round(t1/t2, digits=2), "x")
