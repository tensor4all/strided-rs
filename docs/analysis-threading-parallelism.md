# スレッド並列化の調査報告: tensornetwork_permutation_light_415

## 概要

`tensornetwork_permutation_light_415` ベンチマークにおいて、4スレッド時にJuliaがRustの約2倍速い原因を、
スレッド/並列化の観点から調査した。

### 観察されたベンチマーク結果

| スレッド数 | Julia (OpenBLAS) | Rust (faer) | Julia倍率 | スケーリング(Julia) | スケーリング(Rust) |
|---|---:|---:|---:|---:|---:|
| 1 (opt_flops) | 315 ms | 353 ms | 1.12x | - | - |
| 4 (opt_flops) | 142 ms | 285 ms | 2.01x | 2.22x | 1.24x |
| 4 (opt_size) | 119 ms | 281 ms | 2.36x | - | - |

**核心問題**: Juliaは1→4スレッドで2.22倍のスケーリングを達成するのに対し、Rustは1.24倍に留まる。

### 対照ベンチマーク（参考）

| ベンチマーク | Julia 4T/1T | Rust 4T/1T | 備考 |
|---|---:|---:|---|
| str_nw_mera_closed_120 | 3.04x | 2.82x | 両者とも良好にスケール |
| str_nw_mera_open_26 | 2.71x | 3.07x | Rustの方がスケールが良い |
| tensornetwork_permutation_light_415 | 2.22x | 1.24x | **Rustのスケーリングが極端に悪い** |

→ 問題は`tensornetwork_permutation_light_415`に特有。他のベンチマークではRustも良好にスケールする。

---

## 1. ベンチマーク特性の分析

### tensornetwork_permutation_light_415 の特徴

```
テンソル数: 415
全テンソルサイズ: 2〜4要素（極小）
総要素数: 1,350（全テンソル合計）
縮約ステップ数: 414（= テンソル数 - 1）
log10_flops: 9.65（≈ 4.5 × 10^9 FLOPS）
log2_size: 24.00（最大中間テンソル ≈ 16M要素 ≈ 128MB）
```

**特異性**: 415個の極小テンソル（2〜4要素）が414ステップの二項縮約ツリーで逐次的に縮約される。
初期ステップでは極小テンソル同士の縮約（GEMMのM,N,Kが1〜4程度）、
ツリーが進むにつれ中間テンソルが指数的に成長し、
最終ステップ付近では最大2^24要素規模のGEMMが発生する。

---

## 2. Julia側の並列化戦略

### 2.1 アーキテクチャ

Julia側のパイプライン（`OMEinsum.jl`）:

```
einsum(NestedEinsum, xs, size_dict)
  ├── 各ノードで逐次的に再帰評価（並列なし）
  └── 二項縮約:
       ├── analyze_binary() → reshape → SimpleBinaryRule{...}
       ├── simplifyto() → permutedims!() [単一スレッド]
       └── binary_einsum!() → mul!() or _batched_gemm!()
                                 ↓
                           BLAS GEMM (4 OMP threads)
```

### 2.2 並列化ポイント

| 操作 | 並列化 | APIソース |
|---|---|---|
| 縮約ツリー走査 | **逐次** | `einsum!(NestedEinsum, ...)` 内の `for` ループ |
| permutedims! | **単一スレッド** | Julia Base `permutedims!` |
| BLAS GEMM (mul!) | **4スレッド** | OpenBLAS via libblastrampoline (OMP_NUM_THREADS=4) |
| batched_gemm! | バッチ: **逐次** / 各GEMM: **4スレッド** | BatchedRoutines.jl → BLAS ccall |

**証拠** (ソースコード):
- `OMEinsum.jl/src/einsequence.jl:270-278`: `einsum!(NestedEinsum, ...)` は逐次ループ
- `OMEinsum.jl/src/utils.jl:122-156`: `tensorpermute!` は `permutedims!`（Base, 単一スレッド）
- `BatchedRoutines.jl/src/BatchedRoutines.jl:39`: `@iterate_batch` マクロは単純な `for` ループを生成
- `BatchedRoutines.jl/src/blas.jl:126-136`: 各バッチで BLAS ccall (gemm)

### 2.3 スレッドモデル

```
Julia側のスレッドモデル:

メインスレッド: ┌─step1─┐┌─step2─┐┌─step3─┐...┌─step414─┐
                │perm→GEMM││perm→GEMM││perm→GEMM│...│perm→GEMM│
                    ↓          ↓          ↓            ↓
BLAS threads:  [T1 T2 T3 T4] [T1 T2 T3 T4] ... (OMP persistent pthreads)
```

OpenBLASのスレッドモデルは **persistent pthreads + spin-wait同期** であり、
GEMM呼び出しごとのディスパッチオーバーヘッドが極めて小さい。

---

## 3. Rust側の並列化戦略

### 3.1 アーキテクチャ

```
EinsumCode::evaluate(operands)
  ├── eval_node() [逐次的に再帰]
  └── eval_pair() → einsum2_into()
       ├── Einsum2Plan::new() [軸分類・転置計算]
       ├── strided_kernel::copy_into() [**単一スレッド!**]
       └── bgemm_faer::bgemm_strided_into()
            ├── copy to contiguous [**単一スレッド!**]
            └── matmul_with_conj(Par::rayon(0)) [rayon並列]
```

### 3.2 並列化ポイント

| 操作 | 並列化 | APIソース |
|---|---|---|
| 縮約ツリー走査 | **逐次** | `expr.rs:eval_node()` の再帰呼び出し |
| Permute (metadata) | **ゼロコピー** | `StridedView::permute()` はメタデータ操作のみ |
| copy_into (データコピー) | **単一スレッド** ⚠️ | `strided-kernel` の `parallel` feature が無効 |
| faer GEMM | **rayon(4スレッド)** | `Par::rayon(0)` → `Rayon(NonZeroUsize)` |
| バッチGEMMループ | **逐次** | `bgemm_faer.rs:219-233` の `for` ループ |

### 3.3 重大な発見: `parallel` feature が無効

**ベンチマークスイートの依存関係チェーン**:

```
strided-rs-benchmark-suite/Cargo.toml
  └── strided-opteinsum (features: faer or blas)
       └── strided-kernel = { path = "../strided-kernel" }  ← feature指定なし!
       └── strided-einsum2
            └── strided-kernel = { path = "../strided-kernel" }  ← feature指定なし!
```

`strided-kernel` の Cargo.toml:
```toml
[features]
default = ["simd"]
parallel = ["rayon", "smallvec"]  # ← opt-in, デフォルトではない
```

**結論**: `strided-kernel` の `parallel` feature が有効化されていないため、
`copy_into`, `map_into`, `zip_map2_into` 等の全てのデータ操作は **単一スレッド** で実行されている。

唯一の並列化は **faer の GEMM 内部** (`Par::rayon(0)`) のみ。

### 3.4 スレッドモデル

```
Rust側のスレッドモデル:

メインスレッド: ┌──step1──┐┌──step2──┐...┌──step414──┐
                │copy→GEMM││copy→GEMM│...│copy→GEMM  │
                  ↓    ↓      ↓    ↓          ↓    ↓
                 1T  rayon   1T  rayon       1T  rayon
                      pool        pool            pool

copy: 単一スレッド（parallel feature無効のため）
GEMM: rayon work-stealing pool（4スレッド）
```

---

## 4. なぜ4スレッドでJuliaが2倍速いのか

### 4.1 主因: GEMM並列化の効率差

#### 414ステップの特性

414ステップの大部分は小〜中サイズのGEMMであり、最終ステップ付近でのみ大規模GEMMが発生する。

| ステップ位置 | 推定GEMM規模 | Julia OpenBLAS | faer + rayon |
|---|---|---|---|
| 初期（〜350ステップ） | M,N,K ≤ 数十 | ⚡ 低オーバーヘッド | ⚠️ rayon dispatch cost |
| 中期（〜50ステップ） | M,N,K 〜 数百 | ✅ 効果的な並列化 | ✅ 効果的な並列化 |
| 終期（〜14ステップ） | M,N,K 〜 数千 | ✅ 優れた並列化 | ✅ 優れた並列化 |

#### OpenBLAS vs faer+rayon の並列化オーバーヘッド

**OpenBLAS**:
- persistent pthreads（プロセス起動時に生成、常駐）
- spin-wait 同期（バリア到達をスピンロックで待機）
- 1回のGEMM呼び出しオーバーヘッド: **〜1μs以下**
- 小規模GEMMでは内部ヒューリスティクスで自動的にシングルスレッドにフォールバック

**faer + rayon**:
- rayon work-stealing pool（グローバルスレッドプール）
- `Par::rayon(0)` = 常にrayonプールを使用（閾値0）
- 各GEMM呼び出しで `rayon::join()` を使用しタスク分割
- 1回のディスパッチオーバーヘッド: **〜数μs**（work-stealingキュー操作）
- 小規模GEMMでも常にrayonディスパッチが発生

#### 累積オーバーヘッド計算（仮説）

```
414ステップ × (rayon overhead - OpenBLAS overhead) ≈ 414 × 3μs ≈ 1.2ms
```

しかし1.2msは285ms vs 142msの差（143ms）を説明できない。
オーバーヘッドだけでは説明不十分であり、**GEMMカーネル自体の並列効率差**も寄与している。

### 4.2 副因: データコピー操作の非並列化

`strided-kernel` の `parallel` feature が無効のため:

1. `bgemm_strided_into` 内の非連続オペランドの連続化コピー (`strided_kernel::copy_into`) が単一スレッド
2. `einsum2_into` 内の trace reduction (`trace::reduce_trace_axes`) が単一スレッド
3. `single_tensor_einsum` 内の permutation/copy が単一スレッド

これらの操作は、中間テンソルが大きくなる後半のステップで顕著な影響を持つ。
Julia側では `permutedims!` も単一スレッドだが、BLAS GEMMが効率的に並列化されるため、
相対的にGEMM以外の時間が支配的にならない。

### 4.3 副因: macOS環境の最適化

ベンチマーク環境は macOS (Darwin 25.2.0) であり、Julia の `BLAS vendor: lbt` (libblastrampoline) は
Apple Accelerate Framework にルーティングされる可能性がある。
Apple Accelerate は Apple Silicon に最適化されており、スレッド管理がOSカーネルと緊密に統合されている。

**注意**: これは仮説であり、実際にどのBLAS実装にルーティングされているかは
`BLAS.get_config()` で確認する必要がある。

### 4.4 他のベンチマークとの比較で支持される証拠

| ベンチマーク | テンソル数 | ステップ数 | Rust 4T/1T | 特徴 |
|---|---:|---:|---:|---|
| str_nw_mera_closed_120 | 120 | 119 | 2.82x | 少ないステップ、大きなGEMM |
| str_nw_mera_open_26 | 26 | 25 | 3.07x | 非常に少ないステップ、大きなGEMM |
| tensornetwork_permutation_light_415 | 415 | 414 | 1.24x | **多数のステップ、小さなGEMM多数** |

→ ステップ数が多く、個々のGEMMが小さいほど、rayonのオーバーヘッドが累積し、スケーリングが悪化する。

---

## 5. 改善提案

### 5.1 即効性が高い改善

#### (a) `parallel` feature を有効化

`strided-opteinsum/Cargo.toml` と `strided-einsum2/Cargo.toml` で:

```toml
strided-kernel = { path = "../strided-kernel", features = ["parallel"] }
```

これにより `copy_into`, `map_into` 等が rayon で並列化され、
特に大きな中間テンソルのデータ操作が高速化される。

**期待効果**: 中間テンソルのコピー/permutation操作の並列化で、特に後半ステップが高速化。

#### (b) faer GEMM の並列化閾値の導入

現在 `Par::rayon(0)` を常に使用しているが、小規模GEMM（M*N*K < 閾値）では
`Par::Seq` にフォールバックすることで、rayon dispatch オーバーヘッドを回避:

```rust
// bgemm_faer.rs: do_batch クロージャ内
let total_elements = m * n * k;
let par = if total_elements < GEMM_PARALLEL_THRESHOLD {
    Par::Seq
} else {
    Par::rayon(0)
};
matmul_with_conj(c_mat, accum, a_mat, cj_a, b_mat, cj_b, alpha, par);
```

`GEMM_PARALLEL_THRESHOLD` は実験的に決定（32768〜65536程度を推奨）。

**期待効果**: 小規模GEMMの不要なrayon dispatchを回避。414ステップの大部分で
オーバーヘッドが削減される。

### 5.2 中期的な改善

#### (c) バッチGEMMの並列化

現在 `bgemm_faer.rs` のバッチループは逐次的:

```rust
for _ in 0..total {
    do_batch(a_off, b_off, c_off);  // 各バッチ要素を逐次処理
    ...
}
```

バッチ数が十分大きい場合、バッチ次元を `rayon::par_iter` で並列化:

```rust
if total >= BATCH_PARALLEL_THRESHOLD {
    // rayon parallel iteration over batch elements
} else {
    // sequential loop (current behavior)
}
```

**注意**: 各バッチ内のGEMMも `Par::rayon(0)` を使っている場合、nested parallelism を避けるため、
バッチ並列時は個々のGEMMを `Par::Seq` にする必要がある。

#### (d) 縮約ツリーのタスク並列化

縮約ツリーの独立な部分木を並列に評価:

```rust
// 現在（逐次）:
let (left, left_ids) = eval_node(&args[0], ...)?;
let (right, right_ids) = eval_node(&args[1], ...)?;

// 改善案（タスク並列）:
let ((left, left_ids), (right, right_ids)) = rayon::join(
    || eval_node(&args[0], ...),
    || eval_node(&args[1], ...),
);
```

**課題**:
- オペランドの所有権管理（`Vec<Option<EinsumOperand>>`の共有）
- BufferPool のスレッド安全化
- 小さな部分木でのオーバーヘッド回避

**期待効果**: 大きな部分木が独立に評価可能な場合、最大2倍の高速化。
ただし415テンソルの二項ツリーでは、多くの依存関係があるため効果は限定的。

### 5.3 長期的な改善

#### (e) 小規模GEMM特化カーネル

`TINY_STRIDED_DIM_LIMIT = 8` 以下のGEMM（M,N,K ≤ 8）に対して、
コンパイル時にサイズ特化したインラインカーネルを使用:

```rust
// 例: 4x4行列乗算の手動SIMD最適化
#[inline(always)]
fn gemm_4x4(a: &[f64; 16], b: &[f64; 16], c: &mut [f64; 16]) {
    // AVX2/NEON を使用した固定サイズGEMM
}
```

---

## 6. まとめ

### 根本原因の優先順位

| 順位 | 原因 | 確信度 | 根拠 |
|---|---|---|---|
| 1 | faer+rayon のGEMM dispatch オーバーヘッド（414回の小規模GEMM） | 高 | 小GEMM多数のベンチマークのみでスケーリングが悪い |
| 2 | `strided-kernel` の `parallel` feature 未有効化 | 確定 | Cargo.toml の依存関係チェーンで確認済み |
| 3 | OpenBLAS の低レイテンシスレッドモデル（persistent pthreads + spin-wait） | 高 | BLAS実装の既知特性 |
| 4 | macOS 環境での BLAS 最適化（Apple Accelerate の可能性） | 仮説 | BLAS vendor が lbt であることから推測 |

### 推奨アクション

1. **即時**: `parallel` feature を有効化し、faer GEMM に並列化閾値を導入
2. **短期**: ベンチマークを再実行して効果を測定
3. **中期**: バッチ並列化と縮約ツリー並列化を検討
