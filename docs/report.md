# mdarray-strided ベンチマークレポート

**実行日**: 2026-01-23
**環境**: Rust release build, optimized

## 概要

`mdarray-strided` クレートのパフォーマンスを、naive実装（ネストしたループ）と比較したベンチマーク結果です。各ベンチマークは、連続配列（contiguous）とストライド配列（permuted/transposed）の両方のケースを評価しています。

## ベンチマーク結果

### 1. Copy Operations

#### 1.1 Copy Permuted (転置配列のコピー)

**対応ベンチ**: `bench_copy_permuted` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- copy_permuted`

| Size | Method | Time (mean) | Throughput | Speedup vs Naive |
|------|--------|--------------|------------|------------------|
| 100×100 | naive | 3.86 µs | 2.59 Gelem/s | 1.00× |
| 100×100 | strided | 5.20 µs | 1.92 Gelem/s | **0.74×** (slower) |
| 500×500 | naive | 83.77 µs | 2.98 Gelem/s | 1.00× |
| 500×500 | strided | 82.92 µs | 3.01 Gelem/s | **1.01×** (同等) |
| 1000×1000 | naive | 366.50 µs | 2.73 Gelem/s | 1.00× |
| 1000×1000 | strided | 381.35 µs | 2.62 Gelem/s | **0.96×** (slower) |

**分析**: タイル化の閾値を32×32に下げたことで、500×500ではnaiveと同等の性能を達成しました。100×100でも改善が見られますが、まだnaiveより遅いです。1000×1000ではタイル化の効果により以前より大幅に改善し、naiveに近い水準まで縮まりました。

#### 1.2 Copy Contiguous (連続配列のコピー)

**対応ベンチ**: `bench_copy_contiguous` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- copy_contiguous`

| Size | Method | Time (mean) | Throughput | Speedup vs Naive |
|------|--------|--------------|------------|------------------|
| 100×100 | naive | 2.01 µs | 4.98 Gelem/s | 1.00× |
| 100×100 | strided | 1.45 µs | 6.88 Gelem/s | **1.38×** (faster) |
| 500×500 | naive | 37.85 µs | 6.60 Gelem/s | 1.00× |
| 500×500 | strided | 36.41 µs | 6.87 Gelem/s | **1.04×** (faster) |
| 1000×1000 | naive | 147.52 µs | 6.78 Gelem/s | 1.00× |
| 1000×1000 | strided | 146.71 µs | 6.82 Gelem/s | **1.01×** (faster) |

**分析**: `strided` を `copy_into_uninit` に変更したことで、zero-initialization のコストを回避し、naive を上回る性能を達成しました。小さいサイズ（100×100）では約1.4倍、大きいサイズでも同等〜わずかに高速です。

### 2. Zip Map Operations (要素ごとの二項演算)

#### 2.1 Zip Map Mixed Strides (混合ストライド)

**対応ベンチ**: `bench_zip_map_mixed_strides` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- zip_map_mixed`

| Size | Method | Time (mean) | Speedup vs Naive |
|------|--------|--------------|------------------|
| 100×100 | naive | 12.33 µs | 1.00× |
| 100×100 | strided | 5.66 µs | **2.18×** (faster) |
| 500×500 | naive | 299.94 µs | 1.00× |
| 500×500 | strided | 117.10 µs | **2.56×** (faster) |
| 1000×1000 | naive | 1.29 ms | 1.00× |
| 1000×1000 | strided | 550.37 µs | **2.34×** (faster) |

**分析**: 2D混合ストライド向けの専用パスにより、naiveより 2.2-2.6x 高速になりました。書き込み連続の内側ループを優先し、ポインタ増分でオーバーヘッドを削減しています。

#### 2.2 Zip Map Contiguous (連続配列)

**対応ベンチ**: `bench_zip_map_contiguous` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- zip_map_contiguous`

| Size | Method | Time (mean) | Speedup vs Naive |
|------|--------|--------------|------------------|
| 100×100 | naive | 10.56 µs | 1.00× |
| 100×100 | strided | 3.24 µs | **3.26×** (faster) |
| 500×500 | naive | 256.67 µs | 1.00× |
| 500×500 | strided | 68.38 µs | **3.75×** (faster) |
| 1000×1000 | naive | 1.02 ms | 1.00× |
| 1000×1000 | strided | 342.86 µs | **2.98×** (faster) |

**分析**: 連続配列では、strided実装はnaive実装より**約3.0-3.8倍高速**です。これは、連続アクセスパターンで最適化されたループが効果的に動作しているためです。500×500サイズで最大の高速化（3.75倍）を示しています。

### 3. Reduce Operations

#### 3.1 Reduce Transposed (転置配列のリダクション)

**対応ベンチ**: `bench_reduce_transposed` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- reduce_transposed`

| Size | Method | Time (mean) | Speedup vs Naive |
|------|--------|--------------|------------------|
| 100×100 | naive | 9.18 µs | 1.00× |
| 100×100 | strided | 10.05 µs | **0.91×** (slower) |
| 500×500 | naive | 232.43 µs | 1.00× |
| 500×500 | strided | 231.26 µs | **1.01×** (同等) |
| 1000×1000 | naive | 964.46 µs | 1.00× |
| 1000×1000 | strided | 986.34 µs | **0.98×** (slower) |

**分析**: 転置配列のリダクションでは、小さいサイズではオーバーヘッドが影響しますが、中〜大サイズはほぼ同等の性能です。

#### 3.2 Reduce Contiguous (連続配列のリダクション)

**対応ベンチ**: `bench_reduce_contiguous` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- reduce_contiguous`

| Size | Method | Time (mean) | Speedup vs Naive |
|------|--------|--------------|------------------|
| 100×100 | naive | 9.19 µs | 1.00× |
| 100×100 | strided | 9.42 µs | **0.98×** (同等) |
| 500×500 | naive | 230.55 µs | 1.00× |
| 500×500 | strided | 230.35 µs | **1.00×** (同等) |
| 1000×1000 | naive | 927.92 µs | 1.00× |
| 1000×1000 | strided | 924.77 µs | **1.00×** (同等) |

**分析**: 連続配列のリダクションでは、naive実装とほぼ同等の性能です。連続アクセスパターンでは、最適化の効果が限定的です。

### 4. Specialized Operations

#### 4.1 Symmetrize (対称化: `B = (A + Aᵀ) / 2`)

**対応ベンチ**: `bench_symmetrize_aat` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- symmetrize_aat`

**サイズ**: 4000×4000

| Method | Time (mean) | Throughput | Speedup vs Naive |
|--------|--------------|------------|------------------|
| naive | 62.28 ms | 257 Melem/s | 1.00× |
| strided | 35.69 ms | 448 Melem/s | **1.74×** (faster) |

**分析**: 特化カーネル `symmetrize_into` は、タイルベースの反復と `(i,j)/(j,i)` の同時更新により、naive実装より**約1.74倍高速**です。連続メモリの高速パス（row-major/col-major）により、キャッシュ効率が大幅に改善されています。

#### 4.2 Scale Transpose (スケール転置: `B = α * Aᵀ`)

**対応ベンチ**: `bench_scale_transpose` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- scale_transpose`

**サイズ**: 1000×1000

| Method | Time (mean) | Throughput | Speedup vs Naive |
|--------|--------------|------------|------------------|
| naive | 458.80 µs | 2.18 Gelem/s | 1.00× |
| strided | 398.38 µs | 2.51 Gelem/s | **1.15×** (faster) |

**分析**: `strided` は `copy_transpose_scale_into_fast` を使用しており、naive より**約1.15倍高速**です。4x4マイクロカーネルによる最適化が効果的です。

#### 4.3 Nonlinear Map (非線形マップ: `B = A .* exp(-2A) .+ sin(A²)`)

**対応ベンチ**: `bench_nonlinear_map` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- nonlinear_map`

**サイズ**: 1000×1000

| Method | Time (mean) | Throughput | Speedup vs Naive |
|--------|--------------|------------|------------------|
| naive | 13.61 ms | 73.5 Melem/s | 1.00× |
| strided | 13.20 ms | 75.7 Melem/s | **1.03×** (slightly faster) |

**分析**: 非線形マップでは、strided実装はnaive実装とほぼ同等か、わずかに高速です。計算コストが高いため、メモリアクセスパターンの影響が相対的に小さくなっています。

### 5. 4D Array Operations (Strided.jl README ベンチマーク)

Julia の Strided.jl README.md から移植したベンチマークです。

#### 5.1 Permutedims 4D (4次元配列の順列: `B = permutedims(A, (4,3,2,1))`)

**対応ベンチ**: `bench_permutedims_4d` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- permutedims_4d`

**サイズ**: 32×32×32×32 (1,048,576 要素)

| Method | Time (mean) | Throughput | Speedup vs Naive |
|--------|--------------|------------|------------------|
| naive | 1.11 ms | 0.94 Gelem/s | 1.00× |
| strided | 1.08 ms | 0.97 Gelem/s | **1.03×** (faster) |

**分析**: 4Dで出力が連続メモリの場合に、ブロッキングを回避してポインタ増分の4重ループで処理するパスを追加したことで、strided は **naive をわずかに上回る**性能まで改善しました。入力側のストライドを使ったタイル化や内側次元の連続性を活かす最適化は依然として余地があります。

#### 5.2 Multi-Permute Sum (複数順列の和: `zip_map4_into`)

**対応ベンチ**: `bench_multi_permute_sum` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- multi_permute_sum`

```
B = permutedims(A, (1,2,3,4)) + permutedims(A, (2,3,4,1)) +
    permutedims(A, (3,4,1,2)) + permutedims(A, (4,1,2,3))
```

**サイズ**: 32×32×32×32 (1,048,576 要素)

| Method | Time (mean) | Throughput | Speedup vs Naive |
|--------|--------------|------------|------------------|
| naive | 4.99 ms | 210 Melem/s | 1.00× |
| strided_fused (`zip_map4_into`) | 2.31 ms | 453 Melem/s | **2.16×** (faster) |

**分析**: 4Dで出力が連続メモリの場合に、ブロッキングを回避してポインタ増分の4重ループで処理するパスを導入したことで、`zip_map4_into` が **約2.16倍高速**になりました。ブロック計画や多配列ストライドのオーバーヘッドが削減されています。

**改善提案**:
- 出力が非連続のケースでも部分的な連続性を検出して高速パスへ寄せる
- 4D permute など他のパターンにも特化パスを追加する

## 総合評価

### 強み

1. **連続配列での zip_map**: 約3.0-3.8倍の高速化を達成
2. **対称化操作**: 特化カーネルにより約1.7倍の高速化
3. **混合ストライドのzip_map**: 2.2-2.6x 高速化
4. **転置スケール**: naiveより約1.15倍高速
5. **転置コピー**: 500×500はほぼ同等、他サイズはやや遅い
6. **`zip_map4_into`**: 4Dの多順列和でも contig 出力なら高速（約2.16倍）

### 改善の余地

1. **4D配列の操作**: contig 出力では同等〜わずかに高速だが、非連続出力の最適化は未検証
2. **転置配列のコピー**: サイズによっては遅い（0.72-0.86×）
3. **小さいサイズでのオーバーヘッド**: 100×100程度の小さい配列では、オーバーヘッドが支配的

### 推奨事項

1. **4D配列のブロッキング戦略最適化**: 現在の戦略は2D配列に最適化されているため、高次元配列向けの改善が必要（[Issue #5](https://github.com/AtelierArith/strided-rs-private/issues/5)）
2. **小サイズ向けのオーバーヘッド削減**: 100×100程度の小さい配列では、ブロッキングをスキップする閾値の調整が有効
3. **特化カーネルの拡張**: `symmetrize_into` のように、特定のパターンに特化したカーネルを追加することで、さらなる高速化が期待できる

## 技術的詳細

### 実装の特徴

- **ループ順序最適化**: ストライドパターンに基づいてループ順序を最適化
- **キャッシュブロッキング**: L1キャッシュサイズ（32KB）を考慮したブロックサイズの計算
- **連続アクセスパスの最適化**: 連続配列では `ptr::add(1)` による線形反復を使用
- **特化カーネル**: 対称化や転置スケールなどの特定パターンに特化したカーネル

### ベンチマーク設定

- **サンプル数**: 通常100サンプル、大きなベンチマーク（symmetrize_aat, scale_transpose, nonlinear_map）は10サンプル
- **ウォームアップ時間**: 3秒
- **測定時間**: 通常5秒、大きなベンチマークは10秒
- **ビルド設定**: `--release` フラグで最適化ビルド

## 結論

`mdarray-strided` は、**2D配列での要素ごとの演算**や**特定の操作（対称化）**で優れたパフォーマンスを示しています：

- **zip_map (連続)**: 3.0-3.8倍高速
- **zip_map (混合ストライド)**: 2.2-2.6倍高速
- **symmetrize**: 1.7倍高速

一方、**4D配列での複雑な順列操作**では、contig 出力では同等〜わずかに高速になりましたが、非連続出力向けの最適化は今後の課題です。

`zip_map4_into` の追加により、4配列の単一パス処理が可能になりました。2D配列では効果的ですが、4D配列では追加の最適化が必要です。

### API の成熟度

| 機能 | 2D配列 | 4D配列 |
|------|--------|--------|
| `zip_map2_into` | **推奨** | 要検証 |
| `zip_map4_into` | **推奨** | 要最適化 |
| `symmetrize_into` | **推奨** | N/A |
| `copy_into` (転置) | 同等 | 要最適化 |
