use crate::split::{chunk_range, range_plan, share};

#[test]
fn shares_sum_to_budget_and_differ_by_at_most_one() {
    for budget in 1..=33 {
        for tasks in 1..=budget {
            let shares: Vec<usize> = (0..tasks).map(|k| share(budget, tasks, k)).collect();
            assert_eq!(shares.iter().sum::<usize>(), budget, "{budget}/{tasks}");
            let (lo, hi) = (*shares.iter().min().unwrap(), *shares.iter().max().unwrap());
            assert!(lo >= 1 && hi - lo <= 1, "{budget}/{tasks}: {shares:?}");
        }
    }
}

#[test]
fn ranges_cover_exactly_once_and_respect_grain() {
    for len in 1..=70 {
        for grain in 1..=9 {
            for budget in 1..=9 {
                let (chunk, tasks) = range_plan(len, grain, budget);
                assert!(tasks >= 1 && tasks <= budget, "{len}/{grain}/{budget}");
                let mut next = 0;
                for k in 0..tasks {
                    let r = chunk_range(len, chunk, k);
                    assert_eq!(r.start, next);
                    assert!(!r.is_empty());
                    if k + 1 < tasks {
                        assert!(r.len() >= grain, "{len}/{grain}/{budget}: {r:?}");
                    }
                    next = r.end;
                }
                assert_eq!(next, len);
            }
        }
    }
}
