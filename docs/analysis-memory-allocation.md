# メモリアロケーション分析: tensornetwork_permutation_light_415

## 概要

`tensornetwork_permutation_light_415` ベンチマークにおいて、Julia が Rust より高速である原因をメモリアロケーションパターンの観点から調査した。415個のテンソルから成るネットワークの縮約において、414回の中間テンソル生成が発生し、その大部分が極小サイズ（4要素=32バイト）であることが判明した。このワークロードでは、カーネル計算時間よりもアロケーションのオーバーヘッドが支配的になる。

## 1. テンソルサイズ統計

### 入力テンソル（415個）

| 項目 | 値 |
|------|-----|
| テンソル数 | 415 |
| dtype | float64 |
| 出力 | スカラー（空の出力インデックス） |
| 最小要素数 | 2 |
| 最大要素数 | 4 |
| 平均要素数 | 3.3 |
| 中央値要素数 | 4 |
| 全テンソル合計要素数 | 1,350 |
| 全テンソル合計バイト数 | 10,800 bytes |
| 次元数 | 1〜2次元 |
| 全次元サイズ | 2（すべて dim=2） |

**特徴**: 全入力テンソルは極小サイズ（2x2行列または長さ2のベクトル）。

### 中間テンソル（opt_flops / opt_size パス共通）

両パスの結果は同一であった。

| 項目 | 値 |
|------|-----|
| 縮約ステップ数 | 414 |
| log10(FLOPS) | 9.65 |
| log2(最大中間サイズ) | 24.00 |
| 最小中間要素数 | 1 |
| 最大中間要素数 | 16,777,216 (2^24) |
| 平均中間要素数 | 66,374.4 |
| 中央値中間要素数 | 4 |
| 合計中間バイト数 | 219,832,072 bytes (~210 MB) |

### 中間テンソルのサイズ分布

| 要素数 | バイト数 | 出現回数 | 累積割合 |
|--------|----------|----------|----------|
| 1 | 8 | 1 | 0.2% |
| 2 | 16 | 22 | 5.6% |
| 4 | 32 | 193 | **52.2%** |
| 8 | 64 | 90 | 73.9% |
| 16 | 128 | 34 | 82.1% |
| 32 | 256 | 18 | 86.5% |
| 64 | 512 | 10 | 88.9% |
| 128 | 1,024 | 10 | 91.3% |
| 256〜512 | 2K〜4K | 10 | 93.7% |
| 1K〜8K | 8K〜64K | 9 | 95.9% |
| 16K〜64K | 128K〜512K | 7 | 97.6% |
| 128K〜256K | 1M〜2M | 4 | 98.6% |
| 1M〜16M | 8M〜128M | 4 | 99.5% |
| 16,777,216 | 128M | 1 | 100% |

**重要な発見**: 414回のアロケーションのうち **305回（73.7%）が8要素（64バイト）以下**。中央値は4要素（32バイト）。ユニークサイズは22種類のみで、392回のバッファ再利用が理論的に可能。

## 2. タイム計測範囲の比較

### Rust (`strided-rs-benchmark-suite/src/main.rs`)

```rust
// タイム外: 入力テンソルの生成
let operands = create_operands(&instance.shapes_colmajor, &instance.dtype);
// タイム開始
let t0 = Instant::now();
let result = code.evaluate(operands, None)?;  // 全中間アロケーション含む
let elapsed = t0.elapsed();
```

`create_operands` は `StridedArray::col_major(shape)` で415個のテンソルを生成。これはタイム外。
`code.evaluate` 内で414回の中間テンソルの生成・計算・解放が発生。これはタイム内。

### Julia (`strided-rs-benchmark-suite/src/main.jl`)

```julia
# タイム外: 入力テンソルの生成
tensors = create_tensors(shapes, dtype)
# タイム開始
t0 = time_ns()
result = run_fn(tensors)  # 全中間アロケーション含む
elapsed = (time_ns() - t0) / 1e6
```

`create_tensors` は `zeros(Float64, s...)` で415個のテンソルを生成。これはタイム外。
`run_with_path` 内の各 `einsum(code, (ti, tj))` 呼び出しで中間テンソルが生成。これはタイム内。

**結論**: 両者とも計測範囲は同等。入力テンソルのアロケーションはタイム外、中間テンソルのアロケーションはタイム内。

## 3. Rust の中間テンソルアロケーション方法

### BufferPool による再利用 (`strided-opteinsum/src/expr.rs`)

```rust
struct BufferPool {
    f64_pool: HashMap<usize, Vec<Vec<f64>>>,
    c64_pool: HashMap<usize, Vec<Vec<Complex64>>>,
}
```

- `pool_acquire`: HashMap から要素数をキーにバッファを検索。見つかれば再利用、なければ新規アロケート
- `pool_release`: 縮約完了後、入力テンソルのバッファをプールに返却
- プールは `evaluate()` 呼び出しごとに新規作成（呼び出し間の再利用なし）

```rust
// pool_acquire の実装（f64の場合）
fn pool_acquire(pool: &mut BufferPool, dims: &[usize]) -> StridedArray<f64> {
    let total: usize = dims.iter().product();
    match pool.f64_pool.get_mut(&total).and_then(|v| v.pop()) {
        Some(buf) => unsafe { StridedArray::col_major_from_buffer_uninit(buf, dims) },
        None => unsafe { StridedArray::col_major_uninit(dims) },
    }
}
```

### StridedArray のアロケーション (`strided-view/src/view.rs`)

```rust
// 新規アロケーション
pub unsafe fn col_major_uninit(dims: &[usize]) -> Self {
    let total: usize = dims.iter().product();
    let mut data = Vec::with_capacity(total);
    data.set_len(total);
    let strides = col_major_strides(dims);  // Vec<isize> のアロケーション
    Self {
        data,
        dims: Arc::from(dims),  // Arc<[usize]> のアロケーション
        strides: Arc::from(strides.as_slice()),  // Arc<[isize]> のアロケーション
        offset: 0,
    }
}
```

**1回の中間テンソル生成にかかるアロケーション（プールミス時）:**
1. `Vec<f64>` のヒープ確保（データバッファ）
2. `Vec<isize>` のヒープ確保（strides、一時的）
3. `Arc<[usize]>` のヒープ確保（dims）
4. `Arc<[isize]>` のヒープ確保（strides）

→ 最低4回のヒープアロケーション/テンソル

**プールヒット時:**
1. HashMap lookup + Vec::pop（データバッファ再利用）
2. `Vec<isize>` のヒープ確保（strides、一時的）
3. `Arc<[usize]>` のヒープ確保（dims）
4. `Arc<[isize]>` のヒープ確保（strides）

→ データバッファは再利用されるが、メタデータ（dims, strides）は毎回新規アロケーション

### アロケータ

Rust のデフォルトシステムアロケータを使用（macOS では libmalloc）。jemalloc や mimalloc は設定されていない。

## 4. Julia の中間テンソルアロケーション方法

### run_with_path での中間テンソル生成

```julia
# 各縮約ステップで:
code = DynamicEinCode([ii, ij], pair_output)
result = einsum(code, (ti, tj))
```

### einsum 内部のアロケーション

```julia
function einsum(code::AbstractEinsum, xs::Tuple, size_dict)
    y = get_output_array(xs, map(y -> size_dict[y], getiyv(code)), false)
    einsum!(code, xs, y, true, false, size_dict)
end

function get_output_array(xs::NTuple{N, AbstractArray{T,M} where M}, size, fillzero::Bool) where {T,N}
    if fillzero
        zeros(T, size...)
    else
        Array{T}(undef, size...)  # 未初期化配列
    end
end
```

- `Array{T}(undef, size...)` は Julia の GC 管理メモリを使用
- Julia の GC は内部的に **mimalloc** を使用（Julia 1.9以降）
- 小規模アロケーションは mimalloc のスレッドローカルフリーリストから高速に割り当て
- メタデータ（shape, strides）は配列オブジェクト内にインライン格納
- 1回のアロケーションで配列全体（データ+メタデータ）が確保される

### 追加アロケーション（einsum! 内部）

binary contraction の場合（`einsum.jl` 104-124行目）:
```julia
function einsum!(ixs, iy, xs::NTuple{2,Any}, y, sx, sy, size_dict)
    # ... analyze_binary で permutation/reshape を決定
    xs1 = simplifyto(ix1v, c1, x1, size_dict)  # 必要なら permutedims で中間配列生成
    xs2 = simplifyto(ix2v, c2, x2, size_dict)
    x1_ = safe_reshape(xs1, s1)  # reshape（アロケーションなし、ビュー）
    x2_ = safe_reshape(xs2, s2)
    # binary_einsum! で BLAS mul! を使用
end
```

`simplifyto` が入力テンソルの並べ替えが必要な場合、追加の中間配列を生成する。ただし `reshape` はビューなのでアロケーションなし。

## 5. アロケータの差

### Rust: macOS システムアロケータ (libmalloc)

- スレッドセーフだがグローバルロック（small allocation zone）
- 小規模アロケーション（<= 256 bytes）は magazine allocator を使用
- マルチスレッド環境では magazine の競合が発生しうる
- `free()` は即座にメモリを返却

### Julia: mimalloc (GC 管理)

- スレッドローカルのフリーリスト（ロックフリー）
- 極小アロケーション（<= 1024 bytes）は特別に最適化されたセグメントから割り当て
- GC による遅延解放: 使用済みメモリはすぐには返却されず、GC サイクル時にバッチ回収
- 世代別 GC: 短命なオブジェクト（中間テンソル）は young generation で高速回収

## 6. 性能差への影響の推定

### 小規模アロケーションの支配

414回の中間アロケーションのうち305回（73.7%）が64バイト以下。これらのサイズでは:
- 実際の計算（例: 2x2行列の掛け算 = 12 FLOPs）のコストは約 5ns
- Rust のシステムアロケータ呼び出しは 20〜50ns/回
- Julia の mimalloc は 10〜20ns/回
- 4要素のテンソルに対して、dims/strides の Arc アロケーションコストが計算コストを上回る

### BufferPool の効果と限界

Rust の BufferPool は22種類のユニークサイズに対して392回の再利用が理論的に可能。しかし:

1. **データバッファのみ再利用**: dims (`Arc<[usize]>`) と strides (`Arc<[isize]>`) は毎回新規アロケーション
2. **HashMap のオーバーヘッド**: 各 acquire/release で HashMap のハッシュ計算 + lookup が発生
3. **初回のコールドスタート**: evaluate() 呼び出しごとにプール新規作成。最初の414ステップでは多くのアロケーションが必要

### マルチスレッド時の影響

4スレッド環境で Julia が2倍速い観察は、以下の組み合わせで説明可能:
- Rust: システムアロケータのグローバルロック競合 + 各スレッドでの独立アロケーション
- Julia: mimalloc のスレッドローカルフリーリスト + GC の遅延回収による再利用効率

### 推定影響度

| 要因 | 推定影響 |
|------|----------|
| アロケータ性能差（system malloc vs mimalloc） | 中〜高 |
| メタデータの毎回アロケーション（Arc<dims>, Arc<strides>） | 高 |
| BufferPool のオーバーヘッド（HashMap） | 低〜中 |
| マルチスレッド時のアロケータ競合 | 中（4スレッド時） |

## 7. 改善提案

1. **アロケータの変更**: `mimalloc` または `jemalloc` をグローバルアロケータとして設定する
2. **メタデータのインライン化**: 小規模テンソル向けに dims/strides を Arc ではなくスタック上に保持する設計（例: `SmallVec<[isize; 4]>`）
3. **BufferPool の永続化**: evaluate() 呼び出し間でプールを再利用する（ベンチマーク5回の median 計測で効果的）
4. **小テンソルの特殊化**: 要素数 <= 64 のテンソルに対して、アロケーションを回避する固定バッファ方式を検討
5. **スタックアロケーション**: 極小テンソル（<= 32要素）のデータをスタックに配置する `SmallVec` ベースの実装

## 補足: コードの場所

| 項目 | ファイルパス |
|------|------------|
| Rust BufferPool | `strided-opteinsum/src/expr.rs:18-91` |
| Rust StridedArray::col_major_uninit | `strided-view/src/view.rs:876-884` |
| Rust ベンチマーク計測範囲 | `strided-rs-benchmark-suite/src/main.rs:141-148` |
| Julia einsum / get_output_array | `OMEinsum.jl/src/einsum.jl:24-27`, `loop_einsum.jl:48-54` |
| Julia ベンチマーク計測範囲 | `strided-rs-benchmark-suite/src/main.jl:161-166` |
| Julia binary contraction | `OMEinsum.jl/src/binaryrules.jl` |
