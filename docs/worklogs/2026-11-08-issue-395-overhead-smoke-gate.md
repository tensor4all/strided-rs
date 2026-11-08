# 2026-11-08: #395 — the overhead smoke gate was decided by noise, and could hang

Scope: `.github/workflows/ci.yml` (`test` job and its `Overhead smoke gates` step)
and the shared gate core `strided-basic/benches/support/mod.rs`. No library API,
no production path; both `overhead_smoke` benches are `harness = false` guards.

Machine for the numbers below: **CI only** (`ubuntu-latest`), taken from
`gh run view --log` on the runs named inline. No local measurement is used as
evidence: the development host was at load 14 on 4 CPUs, which is not a
measurement (see `tensor4all-agent-rules/rules/benchmark/measurement.md`).

## 1. The gate was decided by noise

`strided-rs#391`'s first CI, run
[42451599046](https://github.com/tensor4all/strided-rs/actions/runs/42451599046),
head `064283e`. Attempt 1 and attempt 2 are the **same commit**:

| attempt | S1 A ns | S1 B ns | S1 ratio | verdict |
|---|---|---|---|---|
| 1 | 389.3 | 258.9 | **1.504** | `FAIL (>1.5)` |
| 2 | 297.2 | 297.9 | **0.998** | `<= 1.5` (green) |

A 0.27% exceedance turned the job red, and a re-run of the same tree turned it
green. The statistic was the minimum of 5 rounds of each path **independently**,
which is not a paired statistic and is biased in neither direction reliably:
with `A = [100, 100, 100, 100, 100]` and `B = [50, 100, 100, 100, 100]` the
reported ratio is `min(A)/min(B) = 2.0` while the median paired ratio is `1.0`.
An earlier draft of this worklog said the minimum-of-minimums was "the most
favourable pairing available"; that is wrong and is retracted — a single fast
sample on one side moves the ratio by 2x, which is exactly the kind of swing
that decided #391's CI. The two paths of S1 are *meant* to cost the same and in
fact do.

S1 across the 15 most recent runs on 2026-11-08 (10 PR runs
`42447660181 … 42451061414` plus the 5 consecutive `main` runs
`42452583579, 42452631301, 42452884017, 42453376016, 42453458846`):

```
0.896 0.947 0.896 0.950 0.905 0.958 0.904 0.610 0.907 0.950   (PR runs)
0.994 0.970 0.998 0.960 0.976                                  (main runs)
1.504 0.998                                                    (the flake, attempt 1/2)
```

median 0.950, MAD 0.044. The distribution is tight around 0.95 with one
excursion to 1.504 and one to 0.610 — but S1 is computed as `serial / pooled` and
both sides scale together, so a 0.610 is a fast-serial run, not a defect.

## 2. The rule, and why the width is computed per run

`gate = 1 + 3 * max(1.4826 * MAD, 0.15 * median)`, over that attempt's
`REPS` = 9 paired `A_i / B_i` samples
(`../../strided-basic/benches/support/mod.rs`). `MAD` is used rather than IQR because at n ≈ 5 an IQR is zero or jumps
to a neighbour while a scaled MAD stays defined (the #395 ruling). The 15% floor
keeps a perfectly repeatable run from getting a zero-width gate.

The bound is **anchored at 1.0**, not at the measured median: anchoring at the
median lets a steady regression admit itself, because nine identical ratios give
`MAD = 0`, so a median-anchored bound of `median + 3 * 0.15 * median` would pass a
steady 2x. With the 1.0 anchor that row gets `1 + 3 * 0.3 = 1.9` and fails.

| ratio median | MAD | width | bound `1 + 3w` | verdict |
|---|---|---|---|---|
| 1.000 (parity) | 0 | 0.1500 | 1.450 | PASS |
| 1.400 (steady 1.4x) | 0 | 0.2100 | 1.630 | PASS |
| 1.500 (steady 1.5x) | 0 | 0.2250 | 1.675 | PASS |
| 2.000 (steady 2.0x) | 0 | 0.3000 | 1.900 | **FAIL** |

So a steady regression is caught from about **1.82x** upward (the fixed point of
`m > 1 + 0.45 m`), and a steady 1.5x still passes. Tightening that is one line —
floor the width on the anchor (`0.15 * 1.0`) instead of on the median, which
would put every zero-MAD bound at 1.45 and fail a steady 1.5x. That is the
owner's call, not taken here.

**The width must come from the run being judged, not from a constant baked in
from history.** Calibrating a constant from the same 5 `main` runs gives
median 0.976, MAD 0.016, so the 15% floor dominates and the gate is **1.439**;
including the 1.504 observation instead gives **1.447**; over all 17 samples,
**1.428**. MAD has a 50% breakdown point, so one outlier barely moves it, and
every such constant is *tighter* than the 1.5 that already flaked. A constant
would therefore re-fail the event this issue exists for. Computing the width
from the run's own samples is the only form that adapts to a noisy runner.

Locally (not evidence, just confirming the arithmetic and that the row passes):
S1 ratio 0.998, MAD 0.0218, width 0.1497 (floor), gate 1.449.

## 3. What is *not* claimed

The 1.504 run's within-run dispersion cannot be recovered: the old harness
stored only the minimum of 5 rounds, not the samples. So this change cannot be
shown to have absorbed *that* run — it can only be shown to fit *this* rule and
to keep S1 green from here. From this change on the table prints `reps`, `A med`,
`A p95`, `B med`, `B p95`, `ratio`, `MAD`, `width` and the gate, so the next
occurrence is diagnosable instead of mysterious. If S1 recurs above the gate with
a small MAD, the 15% floor is the parameter to revisit, not the statistic.

**This relaxes the steady band, and that is a decision for the owner, not a
detail.** At parity the bound is ≈1.45–1.46, i.e. tighter than the old fixed 1.5,
and `S3a` (`<= 2.5`) / `S3b` (`<= 4`) are untouched. But a *steady* ratio `m`
passes while `m <= 1 + 0.45 m`, i.e. up to **m = 1.82**, where the fixed 1.5
flagged anything above 1.5. The two cannot both hold: a gate that passes the
1.504 observation that motivated #395 is by definition looser than 1.5. Three
options, in increasing strictness:

| option | bound at MAD 0 | steady band | the 1.504 flake |
|---|---|---|---|
| **A. as implemented** — anchor 1.0, floor `0.15 * median` | 1.45 at parity, rising with the median | passes to 1.82x | passes |
| B. anchor 1.0, floor `0.15 * 1.0` | 1.45 flat | passes to 1.45x | **fails** |
| C. keep the fixed 1.5, change only the statistic | 1.50 | passes to 1.5x | passes if the median-of-9 statistic never reaches 1.504 (unproven) |

A is shipped because it is the form the ruling specifies (`0.15 * median`) with
the anchor the review required; B is a one-line change (`GATE_FLOOR_FRACTION *
PARITY_ANCHOR`) if the owner prefers no relaxation at all; C keeps the old
threshold and bets on the statistic alone.

**Decision (owner, 2026-11-08): A.** A is the ruling's `max(1.4826 * MAD,
0.15 * median)` as written; B is a different rule that pins the median term to the
constant 1.0, and C is the fixed 1.5 the ruling already rejected.

**Follow-up condition for B, recorded so it is not lost.** The only reason A
tolerates a steady band up to 1.82x is that the 1.504 run's within-run MAD is
unknown: the old harness kept only the minimum of five rounds, so nothing says
whether that run was noisy (a wide width, which is the intended behaviour) or
quiet (a genuine 1.5x cost, which A would let through). This change is what makes
the question answerable, because the table now prints `median`, `MAD`, `p95` and
`width` per run. Read the **next 5 runs** after this lands and take B
(`GATE_FLOOR_FRACTION * PARITY_ANCHOR` in `parity_width`, plus this table) if the
floor is what dominates in those runs — i.e. if `1.4826 * MAD < 0.15 * median` in
all five, which every run measured so far satisfies (MAD 0.0009–0.049, so
`1.4826 * MAD` = 0.001–0.073 against a floor of 0.143–0.152). Taking B snaps the
zero-MAD bound back to 1.45 and fails a steady 1.5x, at the cost of re-failing a
repeat of the 1.504 event. If instead one of the five shows `1.4826 * MAD` at or
above the floor, the width is doing real work and A is the right rule to keep.
Do not switch on a single run: the condition is about which term dominates,
observed over five.

## 4. The hang

`overhead smoke` is a **step** inside `cargo test (ubuntu-latest)`, not a job, in
both strided-rs and `tlinalg-rs` (`tlinalg-rs:.github/workflows/ci.yml:100-101`).
A job-level timeout tight enough to catch a hung bench would also kill the
legitimate job, so the budget is on the step, with a job backstop, following
`miri.yml:49,55`.

Measured on run 42451599046 attempt 1:

| step | start | end | duration |
|---|---|---|---|
| `Run tests` | 05:39:11 | 05:40:55 | 1m44s |
| `Overhead smoke gates (ubuntu)` | 05:40:55 | 05:42:29 | **1m34s** |
| whole `cargo test (ubuntu-latest)` job | 05:39:05 | 05:42:30 | **3m25s** |

→ `timeout-minutes: 15` on the step (≈10x) and `30` on the job (≈9x). A timeout
step fails the job while still reaching the `if: always()` artifact-report step.

**The 30+ minute hang could not be reproduced.** Across the 40 most recent
`tlinalg-rs` runs, no job has a missing completion timestamp and the longest
`cargo test (ubuntu-latest)` — the job containing that `overhead smoke` step — is
**23.4 min** (run `42354436364`, head `cf8f59e`). The timeout is therefore a
backstop against a hang we have not measured, not a tuned value.
