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
| 100×100 | naive | 4.44 µs | 2.25 Gelem/s | 1.00× |
| 100×100 | strided | 5.30 µs | 1.89 Gelem/s | **0.84×** (slower) |
| 500×500 | naive | 82.23 µs | 3.04 Gelem/s | 1.00× |
| 500×500 | strided | 103.19 µs | 2.42 Gelem/s | **0.80×** (slower) |
| 1000×1000 | naive | 359.74 µs | 2.78 Gelem/s | 1.00× |
| 1000×1000 | strided | 431.26 µs | 2.32 Gelem/s | **0.83×** (slower) |

**分析**: 2Dの書き込み連続パスを追加したことで、以前より大幅に改善し、naiveに近い水準まで縮まりました。依然として読み取り側が非連続アクセスになるため、わずかに遅い傾向は残ります。

#### 1.2 Copy Contiguous (連続配列のコピー)

**対応ベンチ**: `bench_copy_contiguous` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- copy_contiguous`

| Size | Method | Time (mean) | Throughput | Speedup vs Naive |
|------|--------|--------------|------------|------------------|
| 100×100 | naive | 1.98 µs | 5.06 Gelem/s | 1.00× |
| 100×100 | strided | 3.03 µs | 3.30 Gelem/s | **0.65×** (slower) |
| 100×100 | uninit | 1.34 µs | 7.46 Gelem/s | **1.48×** (faster) |
| 500×500 | naive | 37.70 µs | 6.63 Gelem/s | 1.00× |
| 500×500 | strided | 52.35 µs | 4.78 Gelem/s | **0.72×** (slower) |
| 500×500 | uninit | 33.37 µs | 7.49 Gelem/s | **1.13×** (faster) |
| 1000×1000 | naive | 149.48 µs | 6.69 Gelem/s | 1.00× |
| 1000×1000 | strided | 209.34 µs | 4.78 Gelem/s | **0.71×** (slower) |
| 1000×1000 | uninit | 153.15 µs | 6.53 Gelem/s | **0.98×** (同等) |

**分析**: zero-initialization を避けられる `copy_into_uninit` は、連続コピーで naive を上回る速度が出ます。`strided` は `Tensor::zeros` の初期化コスト分が残るため、差が出やすいです。

### 2. Zip Map Operations (要素ごとの二項演算)

#### 2.1 Zip Map Mixed Strides (混合ストライド)

**対応ベンチ**: `bench_zip_map_mixed_strides` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- zip_map_mixed`

| Size | Method | Time (mean) | Speedup vs Naive |
|------|--------|--------------|------------------|
| 100×100 | naive | 12.51 µs | 1.00× |
| 100×100 | strided | 5.69 µs | **2.20×** (faster) |
| 500×500 | naive | 321.82 µs | 1.00× |
| 500×500 | strided | 115.19 µs | **2.79×** (faster) |
| 1000×1000 | naive | 1.24 ms | 1.00× |
| 1000×1000 | strided | 566.42 µs | **2.19×** (faster) |

**分析**: 2D混合ストライド向けの専用パスを導入し、naiveより 2.2-2.8x 高速になりました。書き込み連続の内側ループを優先し、ポインタ増分でオーバーヘッドを削減しています。

#### 2.2 Zip Map Contiguous (連続配列)

**対応ベンチ**: `bench_zip_map_contiguous` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- zip_map_contiguous`

| Size | Method | Time (mean) | Speedup vs Naive |
|------|--------|--------------|------------------|
| 100×100 | naive | 10.88 µs | 1.00× |
| 100×100 | strided | 3.36 µs | **3.24×** (faster) |
| 500×500 | naive | 264.17 µs | 1.00× |
| 500×500 | strided | 66.03 µs | **4.00×** (faster) |
| 1000×1000 | naive | 1.05 ms | 1.00× |
| 1000×1000 | strided | 444.27 µs | **2.36×** (faster) |

**分析**: 連続配列では、strided実装はnaive実装より**約2.4-4.0倍高速**です。これは、連続アクセスパターンで最適化されたループが効果的に動作しているためです。500×500サイズで最大の高速化（4.0倍）を示しています。

### 3. Reduce Operations

#### 3.1 Reduce Transposed (転置配列のリダクション)

**対応ベンチ**: `bench_reduce_transposed` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- reduce_transposed`

| Size | Method | Time (mean) | Speedup vs Naive |
|------|--------|--------------|------------------|
| 100×100 | naive | 9.21 µs | 1.00× |
| 100×100 | strided | 10.64 µs | **0.87×** (slower) |
| 500×500 | naive | 231.76 µs | 1.00× |
| 500×500 | strided | 238.48 µs | **0.97×** (slower) |
| 1000×1000 | naive | 954.44 µs | 1.00× |
| 1000×1000 | strided | 942.37 µs | **1.01×** (同等) |

**分析**: 転置配列のリダクションでは、小さいサイズではオーバーヘッドが影響しますが、大きいサイズ（1000×1000）ではほぼ同等の性能です。

#### 3.2 Reduce Contiguous (連続配列のリダクション)

**対応ベンチ**: `bench_reduce_contiguous` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- reduce_contiguous`

| Size | Method | Time (mean) | Speedup vs Naive |
|------|--------|--------------|------------------|
| 100×100 | naive | 9.15 µs | 1.00× |
| 100×100 | strided | 9.30 µs | **0.98×** (同等) |
| 500×500 | naive | 226.36 µs | 1.00× |
| 500×500 | strided | 230.89 µs | **0.98×** (同等) |
| 1000×1000 | naive | 918.41 µs | 1.00× |
| 1000×1000 | strided | 925.97 µs | **0.99×** (同等) |

**分析**: 連続配列のリダクションでは、naive実装とほぼ同等の性能です。連続アクセスパターンでは、最適化の効果が限定的です。

### 4. Specialized Operations

#### 4.1 Symmetrize (対称化: `B = (A + Aᵀ) / 2`)

**対応ベンチ**: `bench_symmetrize_aat` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- symmetrize_aat`

**サイズ**: 4000×4000

| Method | Time (mean) | Throughput | Speedup vs Naive |
|--------|--------------|------------|------------------|
| naive | 61.14 ms | 262 Melem/s | 1.00× |
| strided | 34.88 ms | 459 Melem/s | **1.75×** (faster) |

**分析**: 特化カーネル `symmetrize_into` は、タイルベースの反復と `(i,j)/(j,i)` の同時更新により、naive実装より**約1.75倍高速**です。連続メモリの高速パス（row-major/col-major）により、キャッシュ効率が大幅に改善されています。

#### 4.2 Scale Transpose (スケール転置: `B = α * Aᵀ`)

**対応ベンチ**: `bench_scale_transpose` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- scale_transpose`

**サイズ**: 1000×1000

| Method | Time (mean) | Throughput | Speedup vs Naive |
|--------|--------------|------------|------------------|
| naive | 475.10 µs | 2.10 Gelem/s | 1.00× |
| strided | 378.34 µs | 2.64 Gelem/s | **1.26×** (faster) |
| strided (tile=16) | 601.18 µs | 1.66 Gelem/s | **0.79×** (slower) |
| strided (tile=24) | 598.87 µs | 1.67 Gelem/s | **0.79×** (slower) |
| strided (tile=32) | 618.86 µs | 1.62 Gelem/s | **0.77×** (slower) |

**分析**: `strided` は `copy_transpose_scale_into_fast` を使用しており、naive より**約1.26倍高速**です。4x4マイクロカーネルによる最適化が効果的です。タイル版は naive より遅く、デフォルトの `strided` が最適です。

#### 4.3 Nonlinear Map (非線形マップ: `B = A .* exp(-2A) .+ sin(A²)`)

**対応ベンチ**: `bench_nonlinear_map` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- nonlinear_map`

**サイズ**: 1000×1000

| Method | Time (mean) | Throughput | Speedup vs Naive |
|--------|--------------|------------|------------------|
| naive | 13.70 ms | 73.0 Melem/s | 1.00× |
| strided | 13.34 ms | 75.0 Melem/s | **1.03×** (slightly faster) |

**分析**: 非線形マップでは、strided実装はnaive実装とほぼ同等か、わずかに高速です。計算コストが高いため、メモリアクセスパターンの影響が相対的に小さくなっています。

### 5. 4D Array Operations (Strided.jl README ベンチマーク)

Julia の Strided.jl README.md から移植したベンチマークです。

#### 5.1 Permutedims 4D (4次元配列の順列: `B = permutedims(A, (4,3,2,1))`)

**対応ベンチ**: `bench_permutedims_4d` (`benches/strided_bench.rs`)
**実行コマンド**: `cargo bench -- permutedims_4d`

**サイズ**: 32×32×32×32 (1,048,576 要素)

| Method | Time (mean) | Throughput | Speedup vs Naive |
|--------|--------------|------------|------------------|
| naive | 1.02 ms | 1.03 Gelem/s | 1.00× |
| strided | 4.73 ms | 222 Melem/s | **0.22×** (slower) |

**分析**: 4D配列の複雑な順列では、現在のブロッキング戦略が最適化されておらず、naiveより遅くなっています。4Dでは次元数が多いため、ブロッキングのオーバーヘッドが大きくなる傾向があります。

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
| naive | 5.16 ms | 203 Melem/s | 1.00× |
| strided_fused (`zip_map4_into`) | 13.69 ms | 76.6 Melem/s | **0.38×** (slower) |

**分析**: `zip_map4_into` を使った単一パス実装は、4D配列では naive の4重ループより**約2.65倍遅く**なっています。主な要因は以下の通りです：

1. **メモリブロックサイズの制約**: 5つの配列（出力+4入力）があるため、`BLOCK_MEMORY_SIZE` (32KB) を5で割ると各配列あたり約6.4KBしか使えず、キャッシュ効率が低下
2. **ブロッキングオーバーヘッド**: 4D配列では4レベルのループネストとオフセット計算のコストが大きい
3. **ストライド計算の複雑さ**: 5つの配列のストライドを毎回計算する必要があり、計算コストが高い
4. **順列パターンの最適化不足**: 特定の順列パターン（identity, [1,2,3,0], [2,3,0,1], [3,0,1,2]）に特化した最適化が未実装

**改善提案**:
- 4D配列や多配列操作では、ブロックサイズ計算を動的に調整（配列数に応じて、または出力配列を優先）
- 小ブロックサイズの場合はブロッキングをスキップする閾値を導入
- ストライドを事前計算してキャッシュ、または共通パターンを最適化
- 特定の順列パターンに特化したカーネルを追加（順列インデックスの直接計算）
- 部分的に連続な次元を検出して最適化（内側次元が連続ならその部分で高速パス）

## 総合評価

### 強み

1. **連続配列での zip_map**: 約3-4倍の高速化を達成
2. **対称化操作**: 特化カーネルにより約1.5倍の高速化
3. **混合ストライドのzip_map**: 2-2.6x 高速化
4. **転置スケール**: naiveと同等〜わずかに高速
5. **転置コピー**: naiveに近い水準まで改善
6. **`zip_map4_into`**: 4配列の単一パス処理が可能（2D配列で効果的）

### 改善の余地

1. **4D配列の操作**: 現在のブロッキング戦略が4D配列に最適化されておらず、naiveより遅い
2. **転置配列のコピー**: 依然としてやや遅い（0.83-0.85×）
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

- **zip_map (連続)**: 3-4倍高速
- **zip_map (混合ストライド)**: 2-2.6倍高速
- **symmetrize**: 1.5倍高速

一方、**4D配列での複雑な順列操作**では、現在のブロッキング戦略が最適化されておらず、naiveより遅くなっています。これは今後の最適化課題です。

`zip_map4_into` の追加により、4配列の単一パス処理が可能になりました。2D配列では効果的ですが、4D配列では追加の最適化が必要です。

### API の成熟度

| 機能 | 2D配列 | 4D配列 |
|------|--------|--------|
| `zip_map2_into` | **推奨** | 要検証 |
| `zip_map4_into` | **推奨** | 要最適化 |
| `symmetrize_into` | **推奨** | N/A |
| `copy_into` (転置) | 同等 | 要最適化 |
