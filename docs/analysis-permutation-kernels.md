# Permutation カーネル分析: tensornetwork_permutation_light_415

## 1. テンソルサイズ分布

ベンチマークインスタンス `tensornetwork_permutation_light_415` の分析:

| 項目 | 値 |
|------|-----|
| テンソル数 | 415 |
| dtype | float64 (8 bytes) |
| Shape `[2]` | 155 テンソル (2要素) |
| Shape `[2, 2]` | 260 テンソル (4要素) |
| 最大テンソルサイズ | 4要素 (32 bytes) |
| 収縮パス数 | 2 |

**重要な発見**: すべてのテンソルが極めて小さい (2〜4要素)。これは **setup overhead が計算時間を支配する** ワークロードである。415個のテンソルに対する数百回の contraction ステップにおいて、各操作の「準備コスト」が実際の演算コストの数倍〜数十倍になりうる。

## 2. Julia の Permutation 実装

### 2.1 OMEinsum.jl の Permutedims ルール

Julia 側では `matchrule.jl:28-29` でunary operationが `Permutedims` ルールにマッチする:

```julia
# matchrule.jl:28-29
if Nx == Ny
    if all(i -> i in iy, ix)
        return Permutedims()
    end
end
```

`unaryrules.jl:125-128` で `Permutedims` ルールが実行される:

```julia
function unary_einsum!(::Permutedims, ix, iy, x::AbstractArray, y::AbstractArray, sx, sy)
    perm = ntuple(i -> findfirst(==(iy[i]), ix)::Int, length(iy))
    return tensorpermute!(y, x, perm, sx, sy)
end
```

### 2.2 tensorpermute! の次元グルーピング最適化

`utils.jl:122-156` で `tensorpermute!` は連続する permutation インデックスをグルーピングして次元を縮小する:

```julia
function tensorpermute!(C::AbstractArray{T,N}, A::AbstractArray{T,N}, perm, sx, sy) where {T,N}
    # perm = [2, 3, 1] で size=(a,b,c) のとき
    # [2,3] が連続 → グルーピングして reshape → 2D の permutedims に縮小
    newshape_slots = fill(-1, N)
    dk = 1
    @inbounds begin
        permk = perm[1]
        newperm = [permk]
        newshape_slots[permk] = size(A, permk)
    end
    @inbounds for i = 2:N
        permi = perm[i]
        if permi == permk + dk  # 連続グループ
            newshape_slots[permk] *= size(A, permi)
            dk += 1
        else
            # ...新グループ
        end
    end
    # reshape して次元を削減してから permutedims!
    A_ = reshape(A, newshape...)
    permutedims!(reshape(C, permed_shape), A_, newperm)
end
```

この最適化により、例えば 3D の permutation が 2D に縮小されることがある。

### 2.3 Strided.jl の `_mapreduce_kernel!` (@generated)

Julia の `Base.permutedims!` は StridedView に対して `copy!(dst, permutedims(src, p))` を呼び、これは `map!(identity, dst, src)` に帰着する (`mapreduce.jl:1-3`)。

**`_mapreduce_kernel!`** (`mapreduce.jl:229-425`) は Julia の `@generated` マクロにより、**コンパイル時に (N, M) の各組み合わせに対して完全に特殊化されたコードを生成する**:

```julia
@generated function _mapreduce_kernel!((f), (op), (initop),
    dims::NTuple{N,Int}, blocks::NTuple{N,Int},
    arrays::NTuple{M,StridedView}, strides::NTuple{M,NTuple{N,Int}},
    offsets::NTuple{M,Int}) where {N,M}
    # コンパイル時にループ構造を生成
    # 最内ループに @simd を付与
    # stride 変数はすべてシンボルとして展開
end
```

**コンパイル時生成の結果** (N=2, M=2 の copy の場合):

```julia
@inbounds begin
    # stride 変数がすべてローカル変数に展開
    stride_1_1 = strides[1][1]  # dst の dim 0 stride
    stride_1_2 = strides[2][1]  # src の dim 0 stride
    stride_2_1 = strides[1][2]  # dst の dim 1 stride
    stride_2_2 = strides[2][2]  # src の dim 1 stride

    # 外側ブロックループ
    for J2 in 1:blocks[2]:dims[2]
        d2 = min(blocks[2], dims[2] - J2 + 1)
        for J1 in 1:blocks[1]:dims[1]
            d1 = min(blocks[1], dims[1] - J1 + 1)
            # 内側要素ループ
            for j2 in Base.OneTo(d2)
                @simd for j1 in Base.OneTo(d1)
                    A1[ParentIndex(I1)] = A2[ParentIndex(I2)]  # identity copy
                    I1 += stride_1_1
                    I2 += stride_1_2
                end
                I1 -= d1 * stride_1_1
                I2 -= d1 * stride_1_2
                I1 += stride_2_1
                I2 += stride_2_2
            end
            # ... offset advancement
        end
    end
end
```

#### Julia の key advantages:
1. **`NTuple{N,Int}` (スタック割当)**: dims, strides, offsets がすべてタプル → ヒープ割当ゼロ
2. **`@generated` 完全特殊化**: ループネストがコンパイル時に展開され、動的ディスパッチなし
3. **`@simd`**: 最内ループで LLVM の自動ベクトル化を保証
4. **`@inbounds`**: 境界チェック完全除去
5. **`ParentIndex` 直接アクセス**: 1つの整数オフセットでデータにアクセス

## 3. strided-rs の Permutation 実装

### 3.1 permute() (view.rs:219)

```rust
pub fn permute(&self, perm: &[usize]) -> Result<StridedView<'a, T, Op>> {
    let rank = self.dims.len();
    // ...validation...
    let mut seen = vec![false; rank];  // ← ヒープ割当 #1
    // ...
    let new_dims: Vec<usize> = perm.iter().map(|&p| self.dims[p]).collect();  // ← ヒープ割当 #2
    let new_strides: Vec<isize> = perm.iter().map(|&p| self.strides[p]).collect();  // ← ヒープ割当 #3
    // ... 新しい StridedView を返す (dims, strides は Vec)
}
```

**rank=2 のテンソルに対して permute() 1回で 3つの Vec がヒープ割当される。**

### 3.2 copy_into (ops_view.rs:198)

```rust
pub fn copy_into<T: Copy, Op: ElementOp<T>>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T, Op>,
) -> Result<()> {
    // 1. contiguous fast path チェック
    if sequential_contiguous_layout(dst_dims, &[dst_strides, src_strides]).is_some() {
        // ptr::copy_nonoverlapping で直接コピー → 最適
        return Ok(());
    }
    // 2. fallback to map_into
    map_into(dest, src, |x| x)  // ← full pipeline
}
```

### 3.3 map_into (map_view.rs:210)

```rust
pub fn map_into<D, A, Op>(dest, src, f) -> Result<()> {
    // 1. contiguous fast path チェック (again)
    if sequential_contiguous_layout(...).is_some() { ... return }

    // 2. Full pipeline (非 contiguous の場合)
    let strides_list: [&[isize]; 2] = [dst_strides, src_strides];
    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size);
    //                   ↑ 内部で多数の Vec 割当

    // 3. Block iteration
    let initial_offsets = vec![0isize; ordered_strides.len()];  // ← ヒープ割当
    for_each_inner_block_preordered(&fused_dims, &plan.block, &ordered_strides,
        &initial_offsets, |offsets, len, strides| {
            inner_loop_map1::<D, A, Op>(dp, strides[0], sp, strides[1], len, &f);
        },
    )
}
```

### 3.4 build_plan_fused の内部割当 (kernel.rs:44)

```rust
pub fn build_plan_fused(dims, strides_list, dest_index, elem_size)
    -> (Vec<usize>, Vec<Vec<isize>>, KernelPlan)
{
    // 1. compute_order → Vec<usize> (order)
    //    内部: index_order() × N配列 → Vec<usize> × N
    //    compute_importance() → Vec<u64>
    //    sort_by_importance() → Vec<usize>
    let order = order::compute_order(dims, strides_list, dest_index);

    // 2. ordered_dims/strides → Vec<usize>, Vec<Vec<isize>>
    let ordered_dims: Vec<usize> = order.iter().map(|&d| dims[d]).collect();
    let ordered_strides: Vec<Vec<isize>> = ...;

    // 3. fuse_dims → Vec<usize>
    let fused_dims = fuse_dims(&ordered_dims, &ordered_strides_refs);

    // 4. compress_dims → (Vec<usize>, Vec<Vec<isize>>)
    let (compressed_dims, compressed_strides) = compress_dims(&fused_dims, &ordered_strides);

    // 5. compute_block_sizes → Vec<usize>
    //    内部: byte_strides → Vec<Vec<isize>>
    //    stride_orders → Vec<Vec<usize>>
    //    compute_costs → Vec<isize>
    let block = block::compute_block_sizes(...);

    // 返値: 3つの Vec + KernelPlan
}
```

**1回の map_into 呼び出しで概算 15〜20 回のヒープ割当が発生する。**

### 3.5 einsum2_into パイプライン (lib.rs:291)

binary contraction 経由で permutation を実行する場合:

```rust
pub fn einsum2_into<T, OpA, OpB, ID>(c, a, b, ic, ia, ib, alpha, beta) {
    // 1. Einsum2Plan::new() → 複数の HashMap, Vec
    let plan = Einsum2Plan::new(ia, ib, ic)?;

    // 2. validate_dimensions
    validate_dimensions(&plan, a.dims(), b.dims(), c.dims(), ...)?;

    // 3. Trace reduction (if any)

    // 4. Permute to canonical order
    let a_perm = a_view.permute(&plan.left_perm)?;   // 3 Vec
    let b_perm = b_view.permute(&plan.right_perm)?;   // 3 Vec
    let mut c_perm = c.permute(&plan.c_to_internal_perm)?;  // 3 Vec

    // 5. Tiny path dispatch
    if should_use_tiny_strided_path(m, n, k, batch_elems) {
        bgemm_strided_into_with_map(...)?;
        // 内部: MultiIndex::new() × 4 (各 Vec 2つ)
    }
}
```

**1回の einsum2_into で概算 20〜30 回のヒープ割当。**

### 3.6 bgemm_naive の MultiIndex オーバーヘッド (util.rs:15)

```rust
pub struct MultiIndex {
    dims: Vec<usize>,    // ← ヒープ割当
    current: Vec<usize>, // ← ヒープ割当
    total: usize,
    count: usize,
}

pub fn offset(&self, strides: &[isize]) -> isize {
    self.current.iter().zip(strides.iter())
        .map(|(&i, &s)| i as isize * s).sum()
    // ↑ 2要素でも iterator + sum のオーバーヘッド
}
```

tiny tensors (m=n=k=2) では、`MultiIndex` の iterator ベースの offset 計算自体がスカラー演算より高コスト。

## 4. 実装上の差異と性能差の関係

### 4.1 ヒープ割当の累積コスト

| 操作 | Julia | Rust |
|------|-------|------|
| permute() metadata | 0 (タプル/スタック) | 3 Vec |
| build_plan | 0 (compile-time) | 15-20 Vec |
| kernel dispatch | 0 (compile-time) | match + Vec |
| MultiIndex | N/A (unrolled) | 4 Vec |
| **合計 / operation** | **~0 alloc** | **~20-30 alloc** |

415テンソルの contraction (~400+ ステップ) × 20-30 alloc/step = **8,000-12,000 回のヒープ割当**

### 4.2 Julia の compile-time specialization vs Rust の runtime dispatch

**Julia**: `@generated _mapreduce_kernel!` は `(N, M)` の各組み合わせに対してコンパイル時にループ構造を完全生成。
2要素ベクトルの copy では実質的に:
```julia
@inbounds @simd for j in 1:2
    dst[I1] = src[I2]
    I1 += s1; I2 += s2
end
```

**Rust**: `for_each_inner_block_preordered` → `match rank` → `kernel_1d_inner` / `kernel_2d_inner` のランタイムディスパッチ。
さらにコールバッククロージャを経由:
```rust
for_each_inner_block_preordered(
    &fused_dims, &plan.block, &ordered_strides, &initial_offsets,
    |offsets, len, strides| {  // ← closure indirect call
        inner_loop_map1::<D, A, Op>(dp, strides[0], sp, strides[1], len, &f);
    },
)
```

### 4.3 contiguous fast path の効果

`copy_into` は contiguous layout を検出した場合 `ptr::copy_nonoverlapping` を使用する (ops_view.rs:226)。
permuted view はストライドが変わるため contiguous にならず、full pipeline に fallback する。

Julia も同様に full pipeline を通すが、`@generated` による特殊化のおかげで overhead が最小限。

### 4.4 simd::dispatch_if_large のスキップ (simd.rs:14-22)

```rust
pub(crate) fn dispatch_if_large<R>(len: usize, f: impl FnOnce() -> R) -> R {
    if len >= 64 {
        dispatch(f)  // SIMD path
    } else {
        f()  // fallback
    }
}
```

2-4 要素では常に fallback (非SIMD) パスを使用。Julia の `@simd` は LLVM に対するヒントであり、小さなループでもベクトル化の機会がある。

## 5. strided-rs で改善できる具体的な点

### 5.1 小テンソル専用 fast path (最重要)

**問題**: 2-4 要素のテンソルに対して build_plan_fused の full pipeline が実行される
**改善案**: total_len ≤ 16 の場合、fuse/order/block をスキップして直接コピー

```rust
// map_view.rs の map_into に追加
pub fn map_into<D, A, Op>(...) -> Result<()> {
    let total = total_len(dst_dims);

    // Tiny tensor fast path: skip plan building entirely
    if total <= 16 {
        return tiny_map_into(dest, src, f);
    }
    // ... existing code
}

fn tiny_map_into<D, A, Op>(dest, src, f) -> Result<()> {
    // 直接 pointer arithmetic で全要素を走査
    // Vec 割当ゼロ
}
```

### 5.2 SmallVec / ArrayVec の活用 (重要)

**問題**: `permute()` で `Vec<usize>`, `Vec<isize>` が毎回ヒープ割当される
**改善案**: `SmallVec<[usize; 8]>` または `ArrayVec` に置き換え

```rust
// view.rs の permute() を変更
pub fn permute(&self, perm: &[usize]) -> Result<StridedView<'a, T, Op>> {
    let rank = self.dims.len();
    let mut seen = SmallVec::<[bool; 8]>::from_elem(false, rank);
    let new_dims: SmallVec<[usize; 8]> = perm.iter().map(|&p| self.dims[p]).collect();
    let new_strides: SmallVec<[isize; 8]> = perm.iter().map(|&p| self.strides[p]).collect();
    // ...
}
```

これにより rank ≤ 8 で **ヒープ割当がゼロ** になる。

### 5.3 Einsum2Plan のキャッシュ (重要)

**問題**: 同じ index pattern の contraction が繰り返し呼ばれるたびに Plan を再構築
**改善案**: `(ia, ib, ic)` のハッシュで Plan をキャッシュ

```rust
// 例: thread-local cache
thread_local! {
    static PLAN_CACHE: RefCell<HashMap<u64, Arc<Einsum2Plan<char>>>> = ...;
}
```

### 5.4 MultiIndex の最適化 (中程度)

**問題**: tiny tensors で `MultiIndex` の iterator overhead が相対的に大きい
**改善案**: dims が空か 1次元の場合の特殊化

```rust
impl MultiIndex {
    pub fn offset(&self, strides: &[isize]) -> isize {
        match self.dims.len() {
            0 => 0,
            1 => self.current[0] as isize * strides[0],
            2 => self.current[0] as isize * strides[0]
               + self.current[1] as isize * strides[1],
            _ => self.current.iter().zip(strides).map(|(&i, &s)| i as isize * s).sum()
        }
    }
}
```

### 5.5 単項 permutation の専用パス (中程度)

**問題**: opteinsum の単項 permutation が `copy_into` → `map_into` → full pipeline を通る
**改善案**: `single_tensor.rs` で 2×2 テンソルの permutation を直接実行

```rust
// 2×2 transpose の特殊化
fn tiny_transpose_2x2(dest: *mut f64, src: *const f64,
                       src_strides: [isize; 2], dest_strides: [isize; 2]) {
    unsafe {
        *dest = *src;
        *dest.offset(dest_strides[1]) = *src.offset(src_strides[0]);
        *dest.offset(dest_strides[0]) = *src.offset(src_strides[1]);
        *dest.offset(dest_strides[0] + dest_strides[1]) =
            *src.offset(src_strides[0] + src_strides[1]);
    }
}
```

### 5.6 tensorpermute! の次元グルーピング移植

Julia の `tensorpermute!` (`utils.jl:122-156`) が行う連続インデックスのグルーピングにより次元数を削減する最適化は、strided-rs の einsum2 レイヤーでは実装されていない。

## 6. まとめ

`tensornetwork_permutation_light_415` で Julia が Rust より速い主因は:

1. **ヒープ割当の差**: Julia はタプル (スタック) ベース、Rust は Vec (ヒープ) ベースのメタデータ管理。415テンソルの数百回の contraction で ~10,000回のヒープ割当が累積。

2. **compile-time specialization の差**: Julia の `@generated` マクロにより `_mapreduce_kernel!` はコンパイル時に完全特殊化。Rust はランタイム match + クロージャ間接呼び出し。

3. **tiny tensor での setup overhead**: 2-4要素のテンソルでは、Rust の `build_plan_fused` (order → fuse → compress → block) の setup コストが実際のコピー/計算コストを大幅に上回る。

4. **1スレッドでの 12% 差 (315ms vs 353ms)** はこれらの overhead の累積で説明できる。4スレッドでの差拡大はスレッディングオーバーヘッドの問題 (Task #1 の調査範囲) と思われる。

**最も効果的な改善策は 5.1 (小テンソル fast path) と 5.2 (SmallVec 化) であり、これにより 1スレッド性能で Julia と同等レベルに到達できる可能性がある。**
