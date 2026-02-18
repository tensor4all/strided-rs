# GEMM戦略分析: tensornetwork_permutation_light_415

## 1. ベンチマークインスタンスの特性

### 1.1 テンソル構成

`tensornetwork_permutation_light_415.json` の構成:

- **テンソル数**: 415
- **行列テンソル (2x2)**: 260個 (tensor 0-259)
- **ベクトルテンソル (2)**: 155個 (tensor 260-414)
- **全ボンド次元**: 2 (全インデックスのサイズが2)
- **contraction step数**: 414
- **出力**: スカラー (全インデックスが縮約される)

これは典型的な量子テンソルネットワーク (qubit系) のインスタンスである。
全てのボンド次元が2であるため、各contractionステップの計算量は極めて小さい。

### 1.2 Contraction Pattern 統計

414 ステップの分類結果:

| パターン | 件数 | 割合 | 説明 | Julia分類 |
|----------|------|------|------|-----------|
| lo=1,ro=0,k=1 | 128 | 30.9% | 行列ベクトル積 (M=2,N=1,K=2) | `mul!(y, x1, x2)` |
| lo=1,ro=1,k=1 | 50 | 12.1% | 2x2行列積 (M=2,N=2,K=2) | `mul!(y, x1, x2)` |
| lo=0,ro=0,k=1 | 41 | 9.9% | 内積 (M=1,N=1,K=2) | `reshape(x1,1,n)*x2` |
| lo=0,ro=1,k=0 | 22 | 5.3% | スカラー×ベクトル (外積) | `@addmul!` broadcast |
| lo=1,ro=2,k=1 | 20 | 4.8% | matmul → 3Dテンソル出力 | batched GEMM |
| lo=2,ro=1,k=0 | 16 | 3.9% | 外積 (行列×ベクトル、縮約なし) | `@addmul!` broadcast |
| lo=1,ro=2,k=0 | 14 | 3.4% | 外積 → 3Dテンソル | `@addmul!` broadcast |
| lo=1,ro=1,k=0 | 11 | 2.7% | 外積 (ベクトル×ベクトル → 行列) | `@addmul!` broadcast |
| その他 (大規模merge) | 112 | 27.0% | 高次元テンソルの結合・縮約 | 後述 |

### 1.3 融合GEMM次元の分布

| M | N | K | 件数 | 操作種別 |
|---|---|---|------|----------|
| 2 | 1 | 2 | 128 | matvec (2×2 行列 × 2ベクトル) |
| 2 | 2 | 2 | 50 | 2×2 GEMM |
| 1 | 1 | 2 | 41 | dot product |
| 1 | 2 | 1 | 22 | scalar × vector |
| 2 | 4 | 2 | 20 | batched matvec |
| 4 | 2 | 1 | 16 | outer product |

**Tiny GEMM (M,N,K 全て ≤ 8): 356/414 = 86.0%**

残り14%は後半のmergeステップで、出力テンソルのランクが最大30次元
(2^30 ≈ 10億要素) に達するが、これらも実質的には「1つの縮約インデックス(dim=2)で
2つの大テンソルを結合する」操作であり、fused M や N は大きくても K は通常2以下。

### 1.4 出力テンソル次元数の分布

| 出力次元 | 件数 | 割合 |
|----------|------|------|
| 0D (スカラー) | 43 | 10.4% |
| 1D | 155 | 37.4% |
| 2D | 83 | 20.0% |
| 3D | 55 | 13.3% |
| 4D以上 | 78 | 18.8% |
| (うち10D以上) | 26 | 6.3% |

後半のステップでは出力が20-30次元に成長し、各次元がサイズ2のため
総要素数は 2^20 ～ 2^30 に達する。

## 2. OMEinsum.jl の contraction routing 実装

### 2.1 分類システム (`matchrule.jl`)

OMEinsum.jl は `match_rule_binary()` (matchrule.jl:70-82) で二項演算を分類する:

```julia
function match_rule_binary(ix1, ix2, iy)
    if !_isunique(ix1) || !_isunique(ix2) || !_isunique(iy)
        DefaultRule()
    elseif (Nx1 + Nx2 + Ny) % 2 == 0  # no batch
        _match_simple2(ix1,ix2,iy,Nx1,Nx2,Ny)
    elseif ix1[Nx1]==ix2[Nx2]==iy[Ny]  # batch = last index
        rule = _match_simple2(...)
        _add_batch(rule)
    else
        DefaultRule()
    end
end
```

**重要**: `_isunique()` (matchrule.jl:83-94) は **4個以上のインデックスを持つ
テンソルに対して常に `false` を返す**:

```julia
@inline function _isunique(ix)
    if length(ix) <= 1; return true
    elseif length(ix) == 2; return ix[1] != ix[2]
    elseif length(ix) == 3; ...
    else; return false  # ← 4インデックス以上は常にfalse
    end
end
```

しかし、これは `match_rule` による最適化ルート選択にのみ影響し、
実際の二項演算は `einsum!` (einsum.jl:104-124) で `analyze_binary` を通じて
常に実行される。

### 2.2 二項演算の実行パイプライン (`einsum.jl:104-124`)

```julia
function einsum!(ixs, iy, xs::NTuple{2,Any}, y, sx, sy, size_dict)
    # 1. インデックス分類
    c1, c2, cy, s1, s2, s3, i1, i2, iyb = analyze_binary(ix1v, ix2v, iyv, size_dict)

    # 2. SimpleBinaryRule を構築 (型パラメータで分岐)
    rule = SimpleBinaryRule{(i1...,),(i2...,),(iyb...,)}()

    # 3. 入力を正規化順序に並べ替え (必要な場合)
    xs1 = simplifyto(ix1v, c1, x1, size_dict)  # Permutedims → tensorpermute!
    xs2 = simplifyto(ix2v, c2, x2, size_dict)

    # 4. 融合次元にreshape
    x1_ = safe_reshape(xs1, s1)
    x2_ = safe_reshape(xs2, s2)

    # 5. 演算実行 (BLASまたはbroadcast)
    binary_einsum!(rule, x1_, x2_, y_, sx, sy)

    # 6. 出力の並べ替え (必要な場合)
    if cy != iyv
        einsum!((cy,), iyv, (y_,), y, sx, sy, size_dict)
    end
end
```

### 2.3 `analyze_binary` の分類 (einsum.jl:139-210)

```julia
function analyze_binary(ix1, ix2, iy, size_dict)
    ix_inner, ix1_outer, ix2_outer, batch = _analyze_binary_input(ix1, ix2, iy)
    c1 = vcat(ix1_outer, ix_inner, batch)  # A: [lo, sum, batch]
    c2 = vcat(ix_inner, ix2_outer, batch)  # B: [sum, ro, batch]
    cy = vcat(ix1_outer, ix2_outer, batch) # C: [lo, ro, batch]
    si = prod(size_dict[x] for x in ix1_outer)  # M
    sj = prod(size_dict[x] for x in ix_inner)   # K
    sk = prod(size_dict[x] for x in ix2_outer)   # N
    sl = prod(size_dict[x] for x in batch)       # batch_size
    # → SimpleBinaryRule の 'i','j','k','l' に対応する融合次元を返す
end
```

### 2.4 各パターンの実行 (`binaryrules.jl`)

小さいパターンに対する特殊化:

| パターン | Julia実装 | 操作 |
|----------|-----------|------|
| `(i,),(j,),(i,j,)` (外積) | `@addmul! sy*y + sx*x1*reshape(x2,1,n)` | broadcast |
| `(i,j),(j,),(i,)` (matvec) | `mul!(y, x1, x2, sx, sy)` | BLAS gemv |
| `(i,j),(j,k),(i,k)` (matmul) | `mul!(y, x1, x2, sx, sy)` | BLAS gemm |
| `(j,),(j,),()` (dot) | `reshape(x1,1,n)*x2` | BLAS dot |
| `(i,),(),(i,)` (scalar×vec) | `@addmul! sy*y + sx*x1*Ref(scalar)` | broadcast |
| batched版 | `_batched_gemm!(...)` | batched BLAS |

**重要**: 次元が2のBLAS呼び出し (`gemm`, `gemv`) は、BLASのセットアップオーバーヘッド
(~200-500ns) が実際の計算 (~数ns) を大幅に上回る。
Julia は `@addmul!` マクロによる broadcast で小次元ケースを効率的に処理する。

### 2.5 `simplifyto` による並べ替え

```julia
function simplifyto(ix1, c1, x1, size_dict)
    if c1 != ix1
        xs1 = similar(x1, ([size_dict[l] for l in c1]...,))
        return einsum!((_collect(LT, ix1),), c1, (x1,), xs1, true, false, size_dict)
    else
        return x1
    end
end
```

`simplifyto` は並べ替えが必要な場合、再帰的に `einsum!` の unary パイプラインを
呼び出す。Unary パイプラインでは `Permutedims()` → `tensorpermute!()` が使われ、
これは **Strided.jl のキャッシュ最適化カーネル** を使用する。

## 3. strided-rs の contraction routing 実装

### 3.1 計画構築 (`plan.rs:41-156`)

```rust
impl Einsum2Plan {
    pub fn new(ia: &[ID], ib: &[ID], ic: &[ID]) -> Result<Self> {
        // 線形スキャンでインデックス分類
        for id in ia {
            if ib.contains(id) {
                if ic.contains(id) { batch.push(id); }
                else { sum.push(id); }
            } else if ic.contains(id) { lo.push(id); }
            else { left_trace.push(id); }
        }
        // 並べ替え計算
        // left_perm: ia → [lo, sum, batch] 順序
        // right_perm: ib → [sum, ro, batch] 順序
        // c_to_internal_perm: ic → [lo, ro, batch] 順序
    }
}
```

### 3.2 実行パイプライン (`lib.rs:291-386`)

```rust
pub fn einsum2_into<T: Scalar, OpA, OpB, ID: AxisId>(
    c: StridedViewMut<T>, a: &StridedView<T, OpA>, b: &StridedView<T, OpB>,
    ic: &[ID], ia: &[ID], ib: &[ID], alpha: T, beta: T,
) -> Result<()> {
    // 1. Plan構築
    let plan = Einsum2Plan::new(ia, ib, ic)?;

    // 2. 次元検証
    validate_dimensions(&plan, a.dims(), b.dims(), c.dims(), ia, ib, ic)?;

    // 3. trace軸の削減 (あれば)

    // 4. Permute to canonical order (metadata-only, zero-copy)
    let a_perm = a.permute(&plan.left_perm)?;
    let b_perm = b.permute(&plan.right_perm)?;
    let mut c_perm = c.permute(&plan.c_to_internal_perm)?;

    // 5. Tiny path判定 (lib.rs:186-189)
    // TINY_STRIDED_DIM_LIMIT = 8
    if m <= 8 && n <= 8 && k <= 8 {
        bgemm_naive::bgemm_strided_into(...);  // ← 直接strided演算
        return Ok(());
    }

    // 6. 大きい場合: contiguous bufferにコピー → GEMM backend
    let a_op = contiguous::prepare_input_view(&a_perm, ...);
    let b_op = contiguous::prepare_input_view(&b_perm, ...);
    let mut c_op = contiguous::prepare_output_view(&mut c_perm, ...);
    ActiveBackend::bgemm_contiguous_into(&mut c_op, &a_op, &b_op, ...);
    c_op.finalize_into(&mut c_perm)?;
}
```

### 3.3 Tiny path の実装 (`bgemm_naive.rs`)

`TINY_STRIDED_DIM_LIMIT = 8` により、**このベンチマークの86%のステップ** が
tiny path を通る。Tiny path は `MultiIndex` を使った汎用ループで、
BLAS呼び出しのオーバーヘッドを回避する。

### 3.4 opteinsum の実行フロー (`expr.rs`)

```rust
fn eval_pair_alloc<T: PoolOps>(
    ld: StridedData<T>, left_ids: &[char],
    rd: StridedData<T>, right_ids: &[char],
    output_ids: &[char], pool: &mut BufferPool,
) -> Result<EinsumOperand<'static>> {
    let out_dims = out_dims_from_ids(...);
    let mut c_arr = T::pool_acquire(pool, &out_dims);  // バッファプール
    einsum2_into(c_arr.view_mut(), &a_view, &b_view, ...);
    T::pool_release(pool, ld);  // 入力バッファをプールに返却
    T::pool_release(pool, rd);
    Ok(T::wrap_array(c_arr))
}
```

バッファプール (`BufferPool`) は `HashMap<usize, Vec<Vec<T>>>` で管理され、
要素数をキーとして再利用可能なバッファを保持する。

## 4. routing の差異と性能差の関係

### 4.1 per-step オーバーヘッドの比較

414ステップにおけるper-stepオーバーヘッドの影響は甚大である。
1ステップあたり1μsの差が全体で414μsの差になる。

| 処理 | Julia (OMEinsum.jl) | Rust (strided-rs) |
|------|---------------------|-------------------|
| インデックス分類 | `analyze_binary`: Vectorの動的生成 | `Einsum2Plan::new`: Vec の動的生成 |
| 入力並べ替え | `simplifyto` → `tensorpermute!` (実データコピー) | `view.permute()` (zero-copy metadata) |
| 出力バッファ確保 | `similar(y, dims)` → GCアロケータ | `pool_acquire` → HashMap lookup |
| GEMM実行 | `mul!` (BLAS) or `@addmul!` (broadcast) | `bgemm_strided_into` (汎用ループ) |
| 出力並べ替え | `einsum!` → unary pipeline | 不要 (viewベースで事前permute) |

### 4.2 核心的な差異

#### (A) 入力の並べ替え戦略

**Julia**: `simplifyto` は入力テンソルを正規化順序に **実データコピー** して
並べ替える。これによりGEMM呼び出し時にデータが連続メモリに配置される。
しかし2x2行列に対してこのコピーは無駄。

**Rust**: `view.permute()` は **zero-copy** でメタデータのみ並べ替える。
Tiny path では strided アクセスで直接計算する。
大きいテンソルの場合のみ `prepare_input_view` でコピーが発生。

→ **Rust の方が理論的には有利** (小テンソルでコピー不要)

#### (B) 出力の canonical order

**Julia**: `analyze_binary` は出力を `[lo, ro, batch]` 順で生成し、
呼び出し元が期待する順序と異なる場合、追加の unary einsum で並べ替える。

**Rust**: `c_to_internal_perm` で出力viewを事前にpermute し、GEMM結果が
直接正しい位置に書き込まれる。追加コピー不要。

→ **Rust の方が理論的には有利** (出力並べ替え不要)

#### (C) 後半ステップの大テンソル処理

後半のmergeステップでは 2^10 ～ 2^30 要素のテンソルが扱われる。
この場合の典型的な操作は:

```
左: [idx1, idx2, ...idx_m, shared]  (m+1 次元)
右: [shared, idx_m+1, ..., idx_n]   (n-m+1 次元)
出力: [idx1, idx2, ..., idx_n]      (n 次元)
```

M = 2^m, N = 2^(n-m), K = 2 の GEMM に融合される。

**Julia**: `simplifyto` による並べ替え → reshape → `_batched_gemm!` (BLAS)
- 並べ替えに `tensorpermute!` (Strided.jl カーネル) を使用
- BLAS gemm は M,N が大きくても K=2 なので実質的に axpy 操作

**Rust**: `view.permute()` (zero-copy) → `prepare_input_view` でコピー判定
- グループが fusable なら zero-copy、そうでなければ col-major にコピー
- K=2 だが M,N が大きいため GEMM backend を使用

→ 大テンソルの場合は **Julia の Strided.jl tensorpermute! vs Rust の
strided-rs copy_into の性能** が支配的。

#### (D) バッファ管理

**Julia**: `similar(y, dims)` で毎回新規アロケーション。
Julia の GC は世代別GCで、短寿命オブジェクトは nursery (bump allocator) で
高速に確保・回収される。実質的に O(1) アロケーション。

**Rust**: `BufferPool` による HashMap ベースの再利用。
`HashMap::get_mut` + `Vec::pop` のオーバーヘッドがある。
ただしシステムアロケータ呼び出しを避けられるメリットも。

→ 414ステップでは Julia の GC 戦略の方が低オーバーヘッドの可能性。

## 5. strided-rs が見逃している最適化

### 5.1 K=1 (外積/要素演算) の特殊化

**現状**: 86ステップ (20.8%) が K=1 (縮約なし) の操作。
これらは `lo=*, ro=*, k=0` パターンで、外積またはスカラー乗算。

`einsum2_into` は `plan.sum.is_empty() && plan.lo.is_empty() && plan.ro.is_empty()`
の場合のみ element-wise fast path を使う (lib.rs:360-368, 647-666)。
しかし lo または ro が非空の外積は GEMM パスに流れてしまう。

**提案**: `plan.sum.is_empty()` (K=1) の場合の専用 fast path を追加。
外積は `zip_map2_into` の broadcastバージョンで処理可能。

```rust
// 提案: K=1 fast path
if plan.sum.is_empty() && beta == T::zero() {
    // 外積: C[lo,ro,batch] = alpha * A[lo,batch] * B[ro,batch]
    // broadcast経由で実装可能
}
```

### 5.2 1x1 GEMM (スカラー乗算) の直接処理

41ステップ (9.9%) が M=1, N=1, K=2 の内積 (全インデックスが縮約)。
結果はスカラーなので、GEMM パイプラインのセットアップ全体が無駄。

**提案**: `m == 1 && n == 1` の場合、直接 dot product を計算。

### 5.3 tiny path の M*N*K 制限の緩和

**現状**: `TINY_STRIDED_DIM_LIMIT = 8` で M,N,K **全て** ≤ 8 の場合のみ
tiny path を使用 (lib.rs:186-189)。

後半ステップの M=128, N=1, K=2 のようなケースでは tiny path に入らないが、
K=2 なので contiguous packing のコストが支配的。

**提案**: K が小さい場合 (K ≤ 4) は M,N が大きくても GEMM packing を避ける
条件を追加。

```rust
fn should_use_tiny_strided_path(m: usize, n: usize, k: usize, _batch: usize) -> bool {
    (m <= 8 && n <= 8 && k <= 8)
    || k <= 2  // K が極小なら packing コストが支配的
}
```

### 5.4 BufferPool の高速化

`HashMap<usize, Vec<Vec<T>>>` の代わりに、頻出サイズ (2, 4, 8, 16, ...) に対する
固定スロットを持つ arena-style allocator を検討。

### 5.5 Einsum2Plan の構築コスト削減

各ステップで `Einsum2Plan::new` が `Vec` の確保と線形スキャンを行う。
414ステップで繰り返されるため、pre-allocated な scratch buffer を持つ
plan builder を検討。

## 6. "permutation" ラベルの由来

このベンチマークが "permutation" と呼ばれる理由は以下の通り:

1. **テンソルネットワークの構造**: 量子回路のテンソルネットワーク表現で、
   多くのゲートが「置換ゲート (permutation gate)」に相当する。
   SWAP ゲートなどは2量子ビット間の状態を入れ替える操作であり、
   テンソル的には permutedims 操作に対応する。

2. **contraction の特性**: 全ボンド次元が2で、ほとんどの操作が:
   - 行列ベクトル積 (30.9%) — 実質的にインデックスの再ラベリング
   - 外積/スカラー積 (21.7%) — テンソルの結合 (permutation的)
   - 2×2行列積 (12.1%) — 非常に軽い計算

   **計算量は極めて小さく、データの並べ替え (permutation) が支配的**。

3. **opt_flops path の特性**: 最適化された contraction path は多くの
   「free index を保持しつつ1つだけ縮約する」ステップを含む。
   出力テンソルのランクが徐々に増加し (最大30次元)、
   各ステップの主な作業は「正しい位置にデータを配置する」こと (= permutation)。

4. **"light"の意味**: 全ボンド次元が2のため計算負荷が軽い (light)。
   同じネットワーク構造でボンド次元が大きい場合は "heavy" になる。

## 7. 結論

### 7.1 性能差の主要因

faer と OpenBLAS で Rust の差がほぼゼロという観察と合致する結論:

1. **BLAS/GEMM は無関係**: 86%のステップが M,N,K ≤ 8 の tiny path。
   残り14%も K ≤ 2 が大半。GEMM backend の選択は性能に影響しない。

2. **per-step オーバーヘッドが支配的**: 414ステップの各ステップにおける
   plan構築、バッファ管理、次元検証のオーバーヘッドが累積する。

3. **大テンソルの permutation/copy**: 後半ステップでの 2^20 以上の要素を持つ
   テンソルの並べ替え・コピーが実行時間に影響。

4. **K=1 外積パスの最適化不足**: 約21%のステップが外積/スカラー乗算だが、
   GEMM パイプラインのセットアップコストを払っている。

### 7.2 推奨アクション (優先度順)

1. **K=1 fast path の追加** — 外積・スカラー乗算の21%のステップを最適化
2. **スカラー出力 (M=1,N=1) の直接計算** — 10%のステップを最適化
3. **K ≤ 2 での packing 回避** — 後半の大テンソルステップの packing コスト削減
4. **BufferPool の arena 化** — per-step アロケーションオーバーヘッド削減
5. **Plan builder の scratch buffer** — Einsum2Plan 構築コスト削減
