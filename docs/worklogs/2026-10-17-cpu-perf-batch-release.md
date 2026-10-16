# 2026-10-17: release after the CPU performance batch

Machine for every number below: AMD EPYC 7713P, shared host, release builds.

## Versions

| Crate | Published | This release | Library change since the published version |
|---|---|---|---|
| `strided-view` | 0.5.0 | **0.5.1** | #308: Phase 3 soundness and boundary fixes (#203, #204, #207, #208) |
| `strided-basic` | 0.5.2 | **0.5.3** | #310: fast `Product`, new `StableProduct`, single-pass pad, feature-independent transpose tile, `run_on` pool handling moved into `threading.rs` |
| `strided-fused` | 0.5.0 | **0.5.1** | #307: static specialization under a serial context (#305); #309: chunked register-machine replay (#304) |
| `strided-traits`, `strided-perm`, `strided-kernel` | 0.5.0 | unchanged | none |

No public item was removed or renamed. The only addition is the
`#[non_exhaustive]` variant `ReduceOp::StableProduct`.

The workspace dependency entries move to `strided-view 0.5.1`,
`strided-basic 0.5.3` and `strided-fused 0.5.1` as caret ranges. The published
manifests of `strided-basic` and `strided-fused` therefore require the
soundness fixes in `strided-view 0.5.1` and the `strided-basic` they were
tested with.

Publish order: `strided-view` → `strided-basic` → `strided-fused`. Publishing
is done by the maintainer.

## Behaviour change to note: `ReduceOp::Product`

`strided-basic 0.5.2` (#302) made the erased product reduction a source-order
left fold. That fold was 3–5.5x slower than a raw loop and never used more than
one thread. The maintainer decided (2026-10-17) that the fastest variant is the
default, matching PyTorch, and that the numerically stable variant is opt-in:

- `ReduceOp::Product` reassociates like `Sum`, using lanes, SIMD and the parallel
  split. As with `torch.prod` (2.14.1 returns `NaN` for f32 and f64), the
  alternating `2.0, 0.5` input of tenferro-rs#2150 can return `NaN`.
- `ReduceOp::StableProduct` tracks a mantissa and an exponent per lane for
  f32/f64, so reassociation causes no overflow or underflow, and it still
  threads. Any zero, subnormal, inf or NaN factor sends that output to the
  source-order fold, so IEEE propagation matches the sequential fold. Complex
  values use the source-order fold; integers use the wrapping product.

kernel_scaling `red_prod_all_8192x4096`: erased goes from 32.9 ms to 11.5 ms at
1T and from 32.9 ms to 6.5 ms at 4T (raw is 11.7 / 6.5 ms). `StableProduct`
costs 1.03–1.6x of `Product` on typical inputs and up to 7.3x in the worst
measured case (axis 1, extreme exponents, 1T).

## Downstream

tenferro-rs pins `strided-basic =0.5.2` and `strided-fused =0.5.0`. When it bumps
to this release:

- `reduce_prod` follows the new `Product` default. The tests
  `test_reduce_prod_survives_lane_reassociation` and
  `test_reduce_prod_follows_source_order_across_routes` assert the 0.5.2 ordered
  fold. They either move to `StableProduct` through an explicit opt-in or assert
  PyTorch semantics.
- tenferro-rs#2161 / #2140 item 3: CPU fusion can be re-enabled on top of the
  #309 chunked replay. That means restoring the pin entry, flipping
  `SELECTS_FUSED_ELEMENTWISE_REGION` and restoring its assertions.
