//! `Exec::faer` hands out a power-of-two `Par` and runs on the pool.
#![cfg(all(feature = "faer", feature = "parallel"))]

use strided_exec::{Exec, Pool};

fn width(par: faer::Par) -> usize {
    match par {
        faer::Par::Seq => 1,
        faer::Par::Rayon(n) => n.get(),
    }
}

#[test]
fn par_width_is_the_power_of_two_rounding_of_the_budget() {
    let pool = Pool::builder()
        .threads(8)
        .thread_name(|i| format!("faer-{i}"))
        .build()
        .unwrap();
    for (budget, expected) in [(1, 1), (2, 2), (3, 2), (4, 4), (5, 4), (7, 4), (8, 8)] {
        let (w, name) = Exec::new(&pool, budget).unwrap().faer(|par| {
            (
                width(par),
                std::thread::current().name().unwrap_or("").to_string(),
            )
        });
        assert_eq!(w, expected, "budget {budget}");
        if budget > 1 {
            assert!(name.starts_with("faer-"), "{name}");
        }
    }
    assert_eq!(Exec::serial().faer(width), 1);
}

#[test]
fn faer_inside_the_pool_does_not_reinstall() {
    let pool = Pool::builder().threads(4).build().unwrap();
    Exec::new(&pool, 4).unwrap().fork(2, |_, child| {
        let w = child.faer(width);
        assert_eq!(w, 2);
    });
}
