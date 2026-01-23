# Strided Array 設計ドキュメント

このドキュメントは、Julia の `StridedViews.jl` および `Strided.jl` パッケージの実装を分析し、Rust での実装に向けた設計知見をまとめたものです。

## 目次

1. [概要](#概要)
2. [StridedView 型の設計](#stridedview-型の設計)
3. [メモリレイアウトとインデックス計算](#メモリレイアウトとインデックス計算)
4. [操作関数の合成](#操作関数の合成)
5. [ビュー変換操作](#ビュー変換操作)
6. [キャッシュ最適化](#キャッシュ最適化)
7. [マルチスレッド実装](#マルチスレッド実装)
8. [ブロードキャスト](#ブロードキャスト)
9. [線形代数との統合](#線形代数との統合)
10. [Rust 実装への示唆](#rust-実装への示唆)

---

## 概要

### パッケージ構成

```
StridedViews.jl  - StridedView 型の定義（データ構造のみ、計算なし）
    ↓
Strided.jl       - 効率的な計算実装（map/reduce、ブロードキャスト、BLAS連携）
```

### 設計思想

1. **ゼロコピー**: `permutedims`, `transpose`, `adjoint`, スライシングはすべてビューを返す
2. **遅延評価**: 操作はメタデータ（ストライド、オフセット）の変更のみ
3. **キャッシュフレンドリー**: ブロッキングとループ順序の最適化
4. **マルチスレッド**: 分割統治法による並列化

---

## StridedView 型の設計

### 型定義

```julia
struct StridedView{T,N,A<:DenseArray,F<:Union{FN,FC,FA,FT}} <: AbstractArray{T,N}
    parent::A              # 親配列（実データを保持）
    size::NTuple{N,Int}    # 各次元のサイズ
    strides::NTuple{N,Int} # 各次元のストライド
    offset::Int            # 開始オフセット
    op::F                  # 要素に適用する操作
end
```

### 型パラメータ

| パラメータ | 説明 |
|-----------|------|
| `T` | 要素型（`op` 適用後の型） |
| `N` | 次元数（コンパイル時既知） |
| `A` | 親配列の型（`DenseArray` のサブタイプ） |
| `F` | 操作関数の型（4種類のいずれか） |

### Rust での対応

```rust
pub struct StridedView<'a, T, const N: usize, Op = Identity> {
    data: &'a [T],           // または &'a mut [T]
    size: [usize; N],
    strides: [isize; N],     // 負のストライドをサポート
    offset: usize,
    _op: PhantomData<Op>,
}
```

---

## メモリレイアウトとインデックス計算

### 線形インデックス計算

多次元インデックス `(i₁, i₂, ..., iₙ)` から線形インデックスへの変換：

```
linear_index = offset + 1 + Σ(iₖ - 1) * strideₖ
```

### Julia 実装

```julia
@inline function _computeind(indices::NTuple{N,Int}, strides::NTuple{N,Int}) where {N}
    return (indices[1] - 1) * strides[1] + _computeind(tail(indices), tail(strides))
end
_computeind(indices::Tuple{}, strides::Tuple{}) = 1
```

### Rust 実装案

```rust
#[inline]
fn compute_index<const N: usize>(indices: &[usize; N], strides: &[isize; N]) -> usize {
    indices.iter()
        .zip(strides.iter())
        .map(|(&idx, &stride)| idx as isize * stride)
        .sum::<isize>() as usize
}
```

### ストライドの正規化

サイズ1の次元のストライドは任意の値を取れるため、正規化が必要：

```julia
function _normalizestrides(size::Dims{N}, strides::Dims{N}) where {N}
    for i in 1:N
        if size[i] == 1
            # サイズ1の次元のストライドを前の次元に基づいて設定
            newstride = i == 1 ? 1 : strides[i - 1] * size[i - 1]
            strides = setindex(strides, newstride, i)
        elseif size[i] == 0
            # サイズ0の次元がある場合、全てのストライドを1に
            return (1, cumprod(size)[1:end-1]...)
        end
    end
    return strides
end
```

---

## 操作関数の合成

### 4つの操作関数

| 関数 | 説明 | 型エイリアス |
|------|------|-------------|
| `identity` | 恒等関数 | `FN` |
| `conj` | 複素共役 | `FC` |
| `transpose` | 転置（要素レベル） | `FT` |
| `adjoint` | 随伴（共役転置） | `FA` |

### 合成規則（群構造）

これらの操作は合成に関して閉じており、群を形成します：

```julia
# conj を適用した後の操作
_conj(::FN) = conj       # identity ∘ conj = conj
_conj(::FC) = identity   # conj ∘ conj = identity
_conj(::FA) = transpose  # adjoint ∘ conj = transpose
_conj(::FT) = adjoint    # transpose ∘ conj = adjoint

# transpose を適用した後の操作
_transpose(::FN) = transpose
_transpose(::FC) = adjoint
_transpose(::FA) = conj
_transpose(::FT) = identity

# adjoint を適用した後の操作
_adjoint(::FN) = adjoint
_adjoint(::FC) = transpose
_adjoint(::FA) = identity
_adjoint(::FT) = conj
```

### 群の乗積表

|  ∘  | id | conj | trans | adj |
|-----|-----|------|-------|-----|
| **id** | id | conj | trans | adj |
| **conj** | conj | id | adj | trans |
| **trans** | trans | adj | id | conj |
| **adj** | adj | trans | conj | id |

### Rust での実装案

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

## ビュー変換操作

### permutedims（次元の並べ替え）

ストライドとサイズを並べ替えるだけで、データコピーなし：

```julia
@inline function Base.permutedims(a::StridedView{T,N}, p) where {T,N}
    _isperm(N, p) || throw(ArgumentError("Invalid permutation"))
    newsize = ntuple(n -> size(a, p[n]), Val(N))
    newstrides = ntuple(n -> stride(a, p[n]), Val(N))
    return StridedView{T}(a.parent, newsize, newstrides, a.offset, a.op)
end
```

### transpose / adjoint（2次元配列）

```julia
LinearAlgebra.transpose(a::StridedView{<:Number,2}) = permutedims(a, (2, 1))
LinearAlgebra.adjoint(a::StridedView{<:Number,2}) = permutedims(conj(a), (2, 1))
```

### sview（スライシング）

```julia
@inline function Base.getindex(a::StridedView{T,N}, I::Vararg{SliceIndex,N}) where {T,N}
    return StridedView{T}(a.parent,
                          _computeviewsize(a.size, I),
                          _computeviewstrides(a.strides, I),
                          a.offset + _computeviewoffset(a.strides, I),
                          a.op)
end
```

#### ビューサイズの計算

```julia
@inline function _computeviewsize(oldsize::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}
    if isa(I[1], Int)
        return _computeviewsize(tail(oldsize), tail(I))  # 次元削減
    elseif isa(I[1], Colon)
        return (oldsize[1], _computeviewsize(tail(oldsize), tail(I))...)
    else  # Range
        return (length(I[1]), _computeviewsize(tail(oldsize), tail(I))...)
    end
end
```

#### ビューストライドの計算

```julia
@inline function _computeviewstrides(oldstrides::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}
    if isa(I[1], Integer)
        return _computeviewstrides(tail(oldstrides), tail(I))  # 次元削減
    elseif isa(I[1], Colon)
        return (oldstrides[1], _computeviewstrides(tail(oldstrides), tail(I))...)
    else  # Range
        return (oldstrides[1] * step(I[1]), _computeviewstrides(tail(oldstrides), tail(I))...)
    end
end
```

### sreshape（ストライド保持リシェイプ）

すべての `reshape` がストライド構造を保持できるわけではない：

**成功条件**:
- 次元の分割は常に可能
- 次元の結合は `stride(A, i+1) == size(A, i) * stride(A, i)` の場合のみ

```julia
function _computereshapestrides(newsize::Dims, oldsize::Dims{N}, strides::Dims{N}) where {N}
    d, r = divrem(oldsize[1], newsize[1])
    if r == 0
        s1 = strides[1]
        if d == 1
            # 次元をそのまま進める
            oldsize = (tail(oldsize)..., 1)
            strides = (tail(strides)..., 1)
            stail = _computereshapestrides(tail(newsize), oldsize, strides)
            return isnothing(stail) ? nothing : (s1, stail...)
        else
            # 次元を分割
            oldsize = (d, tail(oldsize)...)
            strides = (newsize[1] * s1, tail(strides)...)
            stail = _computereshapestrides(tail(newsize), oldsize, strides)
            return isnothing(stail) ? nothing : (s1, stail...)
        end
    else
        # 分割できない場合
        return prod(newsize) != prod(oldsize) ? throw(DimensionMismatch()) : nothing
    end
end
```

---

## キャッシュ最適化

### 定数

```julia
const BLOCKMEMORYSIZE = 1 << 15  # L1キャッシュサイズ（32KB）
const _cachelinelength = 64      # キャッシュライン長（64バイト）
```

### 次元の融合

連続する次元を融合してループオーバーヘッドを削減：

```julia
function _mapreduce_fuse!(f, op, initop, dims, arrays)
    allstrides = map(strides, arrays)
    @inbounds for i in length(dims):-1:2
        merge = true
        for s in allstrides
            # 全配列で連続している場合のみ融合可能
            if s[i] != dims[i - 1] * s[i - 1]
                merge = false
                break
            end
        end
        if merge
            dims = setindex(dims, dims[i - 1] * dims[i], i - 1)
            dims = setindex(dims, 1, i)
        end
    end
    return _mapreduce_order!(f, op, initop, dims, allstrides, arrays)
end
```

### ループ順序の最適化

各次元の「重要度」を計算し、ストライドが小さい次元を内側に配置：

```julia
function _mapreduce_order!(f, op, initop, dims, strides, arrays)
    M = length(arrays)
    N = length(dims)

    # 重要度計算：ストライドが小さいほど重要
    g = 8 * sizeof(Int) - leading_zeros(M + 1)

    # 出力配列（最初の配列）は2倍の重み
    importance = 2 .* (1 .<< (g .* (N .- indexorder(strides[1]))))
    for k in 2:M
        importance = importance .+ (1 .<< (g .* (N .- indexorder(strides[k]))))
    end

    # サイズ1の次元は後回し
    importance = importance .* (dims .> 1)

    # 重要度の降順でソート
    p = sortperm(importance; rev=true)
    dims = getindices(dims, p)
    strides = broadcast(getindices, strides, (p,))
    ...
end
```

#### indexorder 関数

ストライドの相対的な順序を計算：

```julia
function indexorder(strides::NTuple{N,Int}) where {N}
    return ntuple(Val(N)) do i
        si = abs(strides[i])
        si == 0 && return 1  # ゼロストライドは順序1
        k = 1
        for s in strides
            if s != 0 && abs(s) < si
                k += 1
            end
        end
        return k
    end
end
```

### ブロックサイズ計算

L1キャッシュに収まるようにブロックサイズを計算：

```julia
function _computeblocks(dims, costs, bytestrides, strideorders, blocksize=BLOCKMEMORYSIZE)
    # 全体がキャッシュに収まる場合はそのまま
    if totalmemoryregion(dims, bytestrides) <= blocksize
        return dims
    end

    # 最小ストライド順が全配列で一致する場合、最初の次元はそのまま
    minstrideorder = minimum(minimum.(strideorders))
    if all(isequal(minstrideorder), first.(strideorders))
        d1 = dims[1]
        dr = _computeblocks(tail(dims), ...)
        return (d1, dr...)
    end

    # ブロックサイズを縮小
    blocks = dims
    while totalmemoryregion(blocks, bytestrides) >= 2 * blocksize
        i = _lastargmax((blocks .- 1) .* costs)
        blocks = setindex(blocks, (blocks[i] + 1) >> 1, i)  # 半分に
    end
    while totalmemoryregion(blocks, bytestrides) > blocksize
        i = _lastargmax((blocks .- 1) .* costs)
        blocks = setindex(blocks, blocks[i] - 1, i)  # 1ずつ減少
    end
    return blocks
end
```

### メモリ領域計算

キャッシュライン単位でメモリ領域を計算：

```julia
function totalmemoryregion(dims, bytestrides)
    memoryregion = 0
    for i in 1:length(bytestrides)
        strides = bytestrides[i]
        numcontigeouscachelines = 0
        numcachelineblocks = 1
        for (d, s) in zip(dims, strides)
            if s < _cachelinelength
                # 連続領域：キャッシュライン内
                numcontigeouscachelines += (d - 1) * s
            else
                # 非連続領域：別のキャッシュラインブロック
                numcachelineblocks *= d
            end
        end
        numcontigeouscachelines = div(numcontigeouscachelines, _cachelinelength) + 1
        memoryregion += _cachelinelength * numcontigeouscachelines * numcachelineblocks
    end
    return memoryregion
end
```

---

## マルチスレッド実装

### 定数

```julia
const MINTHREADLENGTH = 1 << 15  # スレッド化の最小要素数（32768）
```

### 分割統治法

```julia
function _mapreduce_threaded!(f, op, initop, dims, blocks, strides, offsets,
                              costs, arrays, nthreads, spacing, taskindex)
    if nthreads == 1 || prod(dims) <= MINTHREADLENGTH
        # シングルスレッド実行
        _mapreduce_kernel!(f, op, initop, dims, blocks, arrays, strides, offsets)
    else
        # 最大コストの次元を選択して分割
        i = _lastargmax((dims .- 1) .* costs)

        if costs[i] == 0 || dims[i] <= min(blocks[i], 1024)
            # 分割不可能
            _mapreduce_kernel!(...)
        else
            # 次元を半分に分割
            di = dims[i]
            ndi = di >> 1
            nnthreads = nthreads >> 1

            newdims = setindex(dims, ndi, i)

            # 前半を別スレッドで実行
            t = Threads.@spawn _mapreduce_threaded!(
                f, op, initop, newdims, blocks, strides,
                offsets, costs, arrays, nnthreads, spacing, taskindex
            )

            # 後半を現在のスレッドで実行
            stridesi = getindex.(strides, i)
            newoffsets2 = offsets .+ ndi .* stridesi
            newdims2 = setindex(dims, di - ndi, i)
            nnthreads2 = nthreads - nnthreads

            _mapreduce_threaded!(
                f, op, initop, newdims2, blocks, strides,
                newoffsets2, costs, arrays, nnthreads2, spacing, taskindex + nnthreads
            )

            wait(t)
        end
    end
end
```

### リダクション時の考慮事項

- リダクション次元（出力配列のストライドが0）は分割しない（競合回避）
- 完全リダクションの場合、各スレッドに独立した出力領域を割り当て

```julia
if op !== nothing && _length(dims, strides[1]) == 1  # 完全リダクション
    T = eltype(arrays[1])
    spacing = isbitstype(T) ? max(1, div(64, sizeof(T))) : 1  # False Sharing回避
    threadedout = similar(arrays[1], spacing * get_num_threads())
    ...
end
```

---

## ブロードキャスト

### BroadcastStyle

```julia
struct StridedArrayStyle{N} <: AbstractArrayStyle{N}
end

Broadcast.BroadcastStyle(::Type{<:StridedView{<:Any,N}}) where {N} = StridedArrayStyle{N}()
```

### サイズのプロモーション

サイズ1の次元をストライド0でブロードキャスト：

```julia
function promoteshape1(sz::Dims{N}, a::StridedView) where {N}
    newstrides = ntuple(Val(N)) do d
        if size(a, d) == sz[d]
            stride(a, d)
        elseif size(a, d) == 1
            0  # サイズ1の次元はストライド0でブロードキャスト
        else
            throw(DimensionMismatch(...))
        end
    end
    return StridedView(a.parent, sz, newstrides, a.offset, a.op)
end
```

### CaptureArgs（ブロードキャスト式のキャプチャ）

ブロードキャスト式を効率的に評価するための構造：

```julia
struct CaptureArgs{F,Args<:Tuple}
    f::F
    args::Args
end

struct Arg  # StridedView のプレースホルダー
end

# Broadcasted → CaptureArgs 変換
@inline function make_capture(bc::Broadcasted)
    args = make_tcapture(bc.args)
    return CaptureArgs(bc.f, args)
end

@inline make_capture(a::StridedView) = Arg()  # StridedView はプレースホルダーに
@inline make_capture(a) = a                   # その他はそのまま

# 評価時に Arg を実際の値で置換
(c::CaptureArgs)(vals...) = consume(c, vals)[1]
```

---

## 線形代数との統合

### BLAS風操作

```julia
LinearAlgebra.rmul!(dst::StridedView, α::Number) = mul!(dst, dst, α)
LinearAlgebra.lmul!(α::Number, dst::StridedView) = mul!(dst, α, dst)
LinearAlgebra.axpy!(a, X::StridedView, Y::StridedView) = Y .= a .* X .+ Y
LinearAlgebra.axpby!(a, X, b, Y) = Y .= a .* X .+ b .* Y
```

### 行列乗算

#### BLAS行列の判定

```julia
function isblasmatrix(A::StridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}
    if A.op == identity
        return stride(A, 1) == 1 || stride(A, 2) == 1  # 行または列が連続
    elseif A.op == conj
        return stride(A, 2) == 1  # 列優先のみ
    else
        return false
    end
end
```

#### BLAS形式への変換

```julia
function getblasmatrix(A::StridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}
    if A.op == identity
        if stride(A, 1) == 1
            return A, 'N'  # 列優先
        else
            return transpose(A), 'T'  # 行優先 → 転置
        end
    else
        return adjoint(A), 'C'  # 共役転置
    end
end
```

#### 汎用行列乗算（mapreduce ベース）

```julia
function __mul!(C, A, B, α, β)
    m, n = size(C)
    k = size(A, 2)

    # 行列乗算を3次元のmapreduce問題に変換
    # C[i,j] = Σ_k A[i,k] * B[k,j]
    A2 = sreshape(A, (m, 1, k))
    B2 = sreshape(permutedims(B, (2, 1)), (1, n, k))
    C2 = sreshape(C, (m, n, 1))

    _mapreducedim!(*, +, initop, (m, n, k), (C2, A2, B2))
end
```

---

## Rust 実装への示唆

### 1. 型設計

```rust
/// ストライドビュー
pub struct StridedView<'a, T, const N: usize, Op: ElementOp = Identity> {
    data: &'a [T],
    size: [usize; N],
    strides: [isize; N],
    offset: usize,
    _op: PhantomData<Op>,
}

/// 可変ストライドビュー
pub struct StridedViewMut<'a, T, const N: usize, Op: ElementOp = Identity> {
    data: &'a mut [T],
    size: [usize; N],
    strides: [isize; N],
    offset: usize,
    _op: PhantomData<Op>,
}
```

### 2. 操作トレイト

```rust
pub trait ElementOp: Copy + Default + 'static {
    fn apply<T: Num + Copy>(value: T) -> T;
}

// 合成を型レベルで表現
pub trait Compose<Other: ElementOp>: ElementOp {
    type Result: ElementOp;
}
```

### 3. イテレータ設計

```rust
/// 効率的な多次元イテレータ
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

### 4. キャッシュ最適化

```rust
const L1_CACHE_SIZE: usize = 32 * 1024;  // 32KB
const CACHE_LINE_SIZE: usize = 64;        // 64 bytes

pub fn compute_blocks<const N: usize>(
    dims: &[usize; N],
    costs: &[usize; N],
    byte_strides: &[Vec<usize>],
) -> [usize; N] {
    // Julia実装と同様のロジック
}
```

### 5. 並列化（rayon）

```rust
use rayon::prelude::*;

pub fn parallel_map<T, U, F>(
    src: &StridedView<T, N>,
    dst: &mut StridedViewMut<U, N>,
    f: F,
) where
    T: Sync,
    U: Send,
    F: Fn(T) -> U + Sync,
{
    if dst.len() < MIN_PARALLEL_SIZE {
        // シングルスレッド
        sequential_map(src, dst, f);
    } else {
        // 分割統治
        let split_dim = find_best_split_dimension(dst);
        let (left, right) = split_at_dimension(dst, split_dim);
        rayon::join(
            || parallel_map(src_left, left, &f),
            || parallel_map(src_right, right, &f),
        );
    }
}
```

### 6. SIMD最適化

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 連続メモリ領域の場合にSIMD最適化
#[inline]
pub fn simd_add_contiguous(dst: &mut [f64], src: &[f64]) {
    #[cfg(target_feature = "avx2")]
    unsafe {
        // AVX2 実装
        for i in (0..dst.len()).step_by(4) {
            let a = _mm256_loadu_pd(dst.as_ptr().add(i));
            let b = _mm256_loadu_pd(src.as_ptr().add(i));
            let c = _mm256_add_pd(a, b);
            _mm256_storeu_pd(dst.as_mut_ptr().add(i), c);
        }
    }
}
```

### 7. ndarray との統合

```rust
use ndarray::{ArrayView, ArrayViewMut, Dimension};

impl<'a, T, const N: usize, Op: ElementOp> From<ArrayView<'a, T, Dim<[usize; N]>>>
    for StridedView<'a, T, N, Op>
{
    fn from(arr: ArrayView<'a, T, Dim<[usize; N]>>) -> Self {
        let shape = arr.shape();
        let strides = arr.strides();
        // ...
    }
}
```

---

## 参考資料

- [StridedViews.jl](https://github.com/Jutho/StridedViews.jl)
- [Strided.jl](https://github.com/Jutho/Strided.jl)
- [mdarray (Rust)](https://github.com/fkastner/mdarray)
