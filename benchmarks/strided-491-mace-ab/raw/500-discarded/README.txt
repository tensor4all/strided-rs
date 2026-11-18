Discarded campaign attempts (kept so the accepted pair can be audited against
what was rejected, as #491 did with its contaminated attempt).

* 500-after, 500-before2, 500-before3: failed campaign.sh's 15 % anchor guard.
  The end anchor (fwd f64 1T 128) ran immediately after a 1024-atom VJP cell and
  was 20-26 % slow on BOTH sides: 16.2 ms vs 13.4 ms strided, 57.4-60.2 ms vs
  44.3-44.8 ms llmlip. This is the macOS memory-reclamation effect that
  SETTLE=5 removes; it is not a kernel result.
* 500-after2, 500-after3: passed the guard but ran while the co-tenant load was
  higher; their absolute 4T medians are inflated (e.g. f64 4T 1024 119.70 /
  117.78 ms vs 93.11 ms in the accepted 500-after7) while the ratios agree.
  They are repeats, not the published pair.
* 500-before-early: the first baseline campaign (01:00Z), valid on its own
  (anchor drift 3.7 % strided / 2.4 % llmlip) but its matching after run was one
  of the failed ones; the published before is 500-before4, run back to back with
  500-after4/5/6/7 in the quiet window.
