# Strided Array Design Document

This document analyzes the implementations of Julia’s `StridedViews.jl` and `Strided.jl` packages and summarizes design insights for a Rust implementation.

## Table of Contents

1. [Overview](#overview)
2. [StridedView Type Design](#stridedview-type-design)
3. [Memory Layout and Indexing](#memory-layout-and-indexing)
4. [Operation Composition](#operation-composition)
5. [View Transformations](#view-transformations)
6. [Cache Optimization](#cache-optimization)
7. [Multithreading](#multithreading)
8. [Broadcasting](#broadcasting)
9. [Linear Algebra Integration](#linear-algebra-integration)
10. [Implications for Rust Implementation](#implications-for-rust-implementation)

---

## Overview

### Package Structure

```
StridedViews.jl  - StridedView type definition (data structure only, no computation)
    ↓
Strided.jl       - Efficient computation implementations (map/reduce, broadcast, BLAS integration)
```

### Design Principles

1. **Zero-copy**: `permutedims`, `transpose`, `adjoint`, and slicing return views
2. **Lazy evaluation**: operations only modify metadata (strides, offsets)
3. **Cache-friendly**: blocking and loop-order optimization
4. **Multithreading**: parallelization via divide-and-conquer

---

## StridedView Type Design

### Type Definition

```julia
struct StridedView{T,N,A<:DenseArray,F<:Union{FN,FC,FA,FT}} <: AbstractArray{T,N}
    parent::A              # parent array (stores data)
    size::NTuple{N,Int}    # size for each dimension
    strides::NTuple{N,Int} # stride for each dimension
    offset::Int            # start offset
    op::F                  # element-wise operation
end
```

### Type Parameters

| Parameter | Description |
|-----------|-------------|
| `T` | Element type (after applying `op`) |
| `N` | Rank (known at compile time) |
| `A` | Parent array type (`DenseArray` subtype) |
| `F` | Operation function type (one of four variants) |

### Rust Mapping

```rust
pub struct StridedView<'a, T, const N: usize, Op = Identity> {
    data: &'a [T],           // or &'a mut [T]
    size: [usize; N],
    strides: [isize; N],     // supports negative strides
    offset: usize,
    _op: PhantomData<Op>,
}
```

---

## Memory Layout and Indexing

### Linear Index Calculation

Conversion from multidimensional index `(i₁, i₂, ..., iₙ)` to linear index:

```
linear_index = offset + 1 + Σ(iₖ - 1) * strideₖ
```

### Julia Implementation

```julia
@inline function _computeind(indices::NTuple{N,Int}, strides::NTuple{N,Int}) where {N}
    return (indices[1] - 1) * strides[1] + _computeind(tail(indices), tail(strides))
end
_computeind(indices::Tuple{}, strides::Tuple{}) = 1
```

### Rust Sketch

```rust
#[inline]
fn compute_index<const N: usize>(indices: &[usize; N], strides: &[isize; N]) -> usize {
    indices.iter()
        .zip(strides.iter())
        .map(|(&idx, &stride)| idx as isize * stride)
        .sum::<isize>() as usize
}
```

### Stride Normalization

Strides for size-1 dimensions can be arbitrary, so normalization is required:

```julia
function _normalizestrides(size::Dims{N}, strides::Dims{N}) where {N}
    for i in 1:N
        if size[i] == 1
            # set stride for size-1 dimension based on the previous dimension
            newstride = i == 1 ? 1 : strides[i - 1] * size[i - 1]
            strides = setindex(strides, newstride, i)
        elseif size[i] == 0
            # if any dimension has size 0, set all strides to 1
            return (1, cumprod(size)[1:end-1]...)
        end
    end
    return strides
end
```

---

## Operation Composition

### Four Element Operations

| Function | Description | Type Alias |
|----------|-------------|------------|
| `identity` | Identity | `FN` |
| `conj` | Complex conjugate | `FC` |
| `transpose` | Transpose (element-wise) | `FT` |
| `adjoint` | Adjoint (conjugate transpose) | `FA` |

### Composition Rules (Group Structure)

These operations are closed under composition and form a group:

```julia
# operation after applying conj
_conj(::FN) = conj       # identity ∘ conj = conj
_conj(::FC) = identity   # conj ∘ conj = identity
_conj(::FA) = transpose  # adjoint ∘ conj = transpose
_conj(::FT) = adjoint    # transpose ∘ conj = adjoint

# operation after applying transpose
_transpose(::FN) = transpose
_transpose(::FC) = adjoint
_transpose(::FA) = conj
_transpose(::FT) = identity

# operation after applying adjoint
_adjoint(::FN) = adjoint
_adjoint(::FC) = transpose
_adjoint(::FA) = identity
_adjoint(::FT) = conj
```

### Cayley Table

|  ∘  | id | conj | trans | adj |
|-----|-----|------|-------|-----|
| **id** | id | conj | trans | adj |
| **conj** | conj | id | adj | trans |
| **trans** | trans | adj | id | conj |
| **adj** | adj | trans | conj | id |

### Rust Sketch

```rust
pub trait ElementOp: Copy + Default {
    fn apply<T: ComplexFloat>(value: T) -> T;
    type Composed<Other: ElementOp>: ElementOp;
}

#[derive(Copy, Clone, Default)]
pub struct Identity;

#[derive(Copy, Clone, Default)]
pub struct Conj;

#[derive(Copy, Clone, Default)]
pub struct Transpose;

#[derive(Copy, Clone, Default)]
pub struct Adjoint;

impl ElementOp for Identity {
    fn apply<T: ComplexFloat>(value: T) -> T { value }
    type Composed<Other: ElementOp> = Other;
}

impl ElementOp for Conj {
    fn apply<T: ComplexFloat>(value: T) -> T { value.conj() }
    type Composed<Other: ElementOp> = /* ... */;
}
```

---

## View Transformations

### permutedims (dimension permutation)

Reorder strides and sizes without copying data:

```julia
@inline function Base.permutedims(a::StridedView{T,N}, p) where {T,N}
    _isperm(N, p) || throw(ArgumentError("Invalid permutation"))
    newsize = ntuple(n -> size(a, p[n]), Val(N))
    newstrides = ntuple(n -> stride(a, p[n]), Val(N))
    return StridedView{T}(a.parent, newsize, newstrides, a.offset, a.op)
end
```

### transpose / adjoint (2D arrays)

```julia
LinearAlgebra.transpose(a::StridedView{<:Number,2}) = permutedims(a, (2, 1))
LinearAlgebra.adjoint(a::StridedView{<:Number,2}) = permutedims(conj(a), (2, 1))
```

### sview (slicing)

```julia
@inline function Base.getindex(a::StridedView{T,N}, I::Vararg{SliceIndex,N}) where {T,N}
    return StridedView{T}(a.parent,
                          _computeviewsize(a.size, I),
                          _computeviewstrides(a.strides, I),
                          a.offset + _computeviewoffset(a.strides, I),
                          a.op)
end
```

#### View Size Calculation

```julia
@inline function _computeviewsize(oldsize::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}
    if isa(I[1], Int)
        return _computeviewsize(tail(oldsize), tail(I))  # dimension reduction
    elseif isa(I[1], Colon)
        return (oldsize[1], _computeviewsize(tail(oldsize), tail(I))...)
    else  # Range
        return (length(I[1]), _computeviewsize(tail(oldsize), tail(I))...)
    end
end
```

#### View Stride Calculation

```julia
@inline function _computeviewstrides(oldstrides::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}
    if isa(I[1], Integer)
        return _computeviewstrides(tail(oldstrides), tail(I))  # dimension reduction
    elseif isa(I[1], Colon)
        return (oldstrides[1], _computeviewstrides(tail(oldstrides), tail(I))...)
    else  # Range
        return (oldstrides[1] * step(I[1]), _computeviewstrides(tail(oldstrides), tail(I))...)
    end
end
```

---

## Cache Optimization

### Background

Contiguous memory access is critical for performance. Strided access can be improved by reordering loops and blocking to improve cache locality.

### Julia Strategy

The core pipeline is:

```
_mapreduce_fuse! → _mapreduce_order! → _mapreduce_block! → _mapreduce_kernel!
```

### Blocking

Block sizes are chosen based on element size and cache capacity to reduce cache misses.

### Loop Order Optimization

Dimensions are ordered by stride magnitude to maximize contiguous access.

---

## Multithreading

### Divide-and-Conquer Parallelization

Dimensions are split into independent tasks, and each task is processed in parallel.

### Reduction Considerations

- Do not split reduction dimensions (dest stride 0) to avoid contention
- For full reductions, use per-thread outputs to avoid false sharing

```julia
if op !== nothing && _length(dims, strides[1]) == 1  # full reduction
    T = eltype(arrays[1])
    spacing = isbitstype(T) ? max(1, div(64, sizeof(T))) : 1  # avoid false sharing
    threadedout = similar(arrays[1], spacing * get_num_threads())
    ...
end
```

---

## Broadcasting

### BroadcastStyle

```julia
struct StridedArrayStyle{N} <: AbstractArrayStyle{N}
end

Broadcast.BroadcastStyle(::Type{<:StridedView{<:Any,N}}) where {N} = StridedArrayStyle{N}()
```

### Size Promotion

Broadcast size-1 dimensions using stride 0:

```julia
function promoteshape1(sz::Dims{N}, a::StridedView) where {N}
    newstrides = ntuple(Val(N)) do d
        if size(a, d) == sz[d]
            stride(a, d)
        elseif size(a, d) == 1
            0  # size-1 dimensions broadcast with stride 0
        else
            throw(DimensionMismatch(...))
        end
    end
    return StridedView(a.parent, sz, newstrides, a.offset, a.op)
end
```

#### Rust Implementation Notes (strided-rs)

- Use `promote_strides_to_shape` to promote size-1 dimensions to stride-0
- `broadcast_capture_into` is the main broadcast evaluation entry point
- **Mixed-op is removed**. Apply element ops on views (e.g., `conj()`) to keep a single `Op`

### CaptureArgs (Broadcast Expression Capture)

A structure for efficiently evaluating broadcast expressions:

```julia
struct CaptureArgs{F,Args<:Tuple}
    f::F
    args::Args
end

struct Arg  # placeholder for StridedView
end

# Broadcasted → CaptureArgs conversion
@inline function make_capture(bc::Broadcasted)
    args = make_tcapture(bc.args)
    return CaptureArgs(bc.f, args)
end

@inline make_capture(a::StridedView) = Arg()  # StridedView becomes placeholder
@inline make_capture(a) = a                   # others kept as-is

# Replace Arg with actual values at evaluation time
(c::CaptureArgs)(vals...) = consume(c, vals)[1]
```

#### Rust Implementation Notes

- Lazy evaluation via `CaptureArgs` + `Consume`
- Unified API: `broadcast_capture_into(dest, capture, sources)` where `sources` is `&[&StridedArrayView]`
- 2/3/4-arity helpers removed (single API surface)
- `mapreducedim_capture_views_into` is the mapreducedim-side entry point

---

## Linear Algebra Integration

### BLAS-like Operations

```julia
LinearAlgebra.rmul!(dst::StridedView, α::Number) = mul!(dst, dst, α)
LinearAlgebra.lmul!(α::Number, dst::StridedView) = mul!(dst, α, dst)
LinearAlgebra.axpy!(a, X::StridedView, Y::StridedView) = Y .= a .* X .+ Y
LinearAlgebra.axpby!(a, X, b, Y) = Y .= a .* X .+ b .* Y
```

### Matrix Multiplication

#### BLAS Matrix Check

```julia
function isblasmatrix(A::StridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}
    if A.op == identity
        return stride(A, 1) == 1 || stride(A, 2) == 1  # contiguous row or column
    elseif A.op == conj
        return stride(A, 2) == 1  # column-major only
    else
        return false
    end
end
```

#### Conversion to BLAS Form

```julia
function getblasmatrix(A::StridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}
    if A.op == identity
        if stride(A, 1) == 1
            return A, 'N'  # column-major
        else
            return transpose(A), 'T'  # row-major → transpose
        end
    else
        return adjoint(A), 'C'  # conjugate transpose
    end
end
```

#### Generic Matrix Multiplication (mapreduce-based)

```julia
function __mul!(C, A, B, α, β)
    m, n = size(C)
    k = size(A, 2)

    # Convert matrix multiplication to a 3D mapreduce
    # C[i,j] = Σ_k A[i,k] * B[k,j]
    A2 = sreshape(A, (m, 1, k))
    B2 = sreshape(permutedims(B, (2, 1)), (1, n, k))
    C2 = sreshape(C, (m, n, 1))

    _mapreducedim!(*, +, initop, (m, n, k), (C2, A2, B2))
end
```

---

## Implications for Rust Implementation

### 1. Type Design (Current Names)

```rust
/// Immutable strided view
pub struct StridedArrayView<'a, T, const N: usize, Op: ElementOp = Identity> {
    data: &'a [T],
    size: [usize; N],
    strides: [isize; N],
    offset: usize,
    _op: PhantomData<Op>,
}

/// Mutable strided view
pub struct StridedArrayViewMut<'a, T, const N: usize, Op: ElementOp = Identity> {
    data: &'a mut [T],
    size: [usize; N],
    strides: [isize; N],
    offset: usize,
    _op: PhantomData<Op>,
}
```

### 2. Operation Traits

```rust
pub trait ElementOp: Copy + Default + 'static {
    fn apply<T: Num + Copy>(value: T) -> T;
}

// Composition is represented at the type level via ElementOp + Compose
pub trait Compose<Other: ElementOp>: ElementOp {
    type Result: ElementOp;
}
```

### 3. Iterator Design

```rust
/// Efficient multidimensional iterator
pub struct StridedIter<'a, T, const N: usize, Op: ElementOp> {
    view: &'a StridedView<'a, T, N, Op>,
    indices: [usize; N],
    linear_index: usize,
    exhausted: bool,
}

impl<'a, T, const N: usize, Op: ElementOp> Iterator for StridedIter<'a, T, N, Op> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let value = unsafe { self.view.get_unchecked(self.linear_index) };
        self.advance();
        Some(Op::apply(value))
    }
}
```

---

## References

- [StridedViews.jl](https://github.com/Jutho/StridedViews.jl)
- [Strided.jl](https://github.com/Jutho/Strided.jl)
- [mdarray (Rust)](https://github.com/fkastner/mdarray)
