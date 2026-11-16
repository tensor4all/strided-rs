//! One A/B cell: `cargo run --release -- --cell cg-fwd --dtype f64 --threads 1 --atoms 1024`
//!
//! Prints one JSON object (the raw record) and appends it to `raw/<name>.jsonl`
//! when `--raw-dir` is given.

use std::hint::black_box;
use std::time::Duration;

use mace_ab::cg::{Buffers, Cg, Scalar};
use mace_ab::measure::{self, Measured};
use mace_ab::topo;
use strided_basic::Pool;

#[global_allocator]
static ALLOCATOR: mace_ab::measure::Counting = mace_ab::measure::Counting;

struct Args {
    cell: String,
    dtype: String,
    threads: usize,
    atoms: usize,
    samples: usize,
    target_ms: u64,
    raw_dir: Option<String>,
}

fn parse() -> Result<Args, String> {
    let mut args = Args {
        cell: "cg-fwd".to_owned(),
        dtype: "f64".to_owned(),
        threads: 1,
        atoms: 128,
        samples: 15,
        target_ms: 20,
        raw_dir: None,
    };
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = || argv.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--cell" => args.cell = value()?,
            "--dtype" => args.dtype = value()?,
            "--threads" => args.threads = value()?.parse().map_err(|_| "threads")?,
            "--atoms" => args.atoms = value()?.parse().map_err(|_| "atoms")?,
            "--samples" => args.samples = value()?.parse().map_err(|_| "samples")?,
            "--target-ms" => args.target_ms = value()?.parse().map_err(|_| "target-ms")?,
            "--raw-dir" => args.raw_dir = Some(value()?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(args)
}

fn main() {
    let result = parse().and_then(|args| match args.dtype.as_str() {
        "f64" => run::<f64>(&args),
        "f32" => run::<f32>(&args),
        other => Err(format!("unknown dtype {other}")),
    });
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn repeats(atoms: usize) -> Result<[usize; 3], String> {
    match atoms {
        128 => Ok([2, 2, 4]),
        1024 => Ok([4, 4, 8]),
        other => Err(format!("no supercell for {other} atoms")),
    }
}

/// llmlip's parallel feature dispatches through rayon's *global* pool, so the
/// thread count has to be fixed here or "1T" would silently be 32 threads on one
/// core (affinity is not a thread count).
fn fix_llmlip_threads(threads: usize) -> Result<(), String> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|error| format!("rayon pool: {error}"))
}

fn run<T: Scalar>(args: &Args) -> Result<(), String> {
    fix_llmlip_threads(args.threads)?;
    let topology = topo::diamond_si(repeats(args.atoms)?, 6.0)?;
    let mut cg = mace_ab::cg::build::<T>(&topology)?;
    let buffers = mace_ab::buffers(&cg, 11);
    let pool = if args.threads > 1 {
        Some(mace_ab::cg::pool(args.threads)?)
    } else {
        None
    };
    let target = Duration::from_millis(args.target_ms);

    let mut record =
        format!(
        "{{\"cell\":\"{}\",\"dtype\":\"{}\",\"threads\":{},\"atoms\":{},\"nodes\":{},\"edges\":{},\
         \"avg_neighbors\":{:.3},\"feature_dim\":{},\"out_slots\":{},\"weight_numel\":{},\
         \"cg_entries_per_feature\":{},\"vjp_y_paths\":{},\"samples\":{},\"target_ms\":{}",
        args.cell,
        args.dtype,
        args.threads,
        args.atoms,
        cg.num_nodes,
        cg.num_edges,
        topology.avg_neighbors(),
        cg.feature_dim(),
        cg.out_m_total(),
        cg.desc.weight_numel,
        cg.desc.blocks.iter().map(|block| block.cg.len()).sum::<usize>(),
        cg.desc.blocks.iter().map(|block| block.cg.len() * cg.feature_dim()).sum::<usize>(),
        args.samples,
        args.target_ms,
    );

    // Self-checking: the numbers both sides produce for this cell.
    let mut check = String::from(",\"check\":{");
    match args.cell.as_str() {
        "cg-fwd" => {
            let (llmlip, strided) = forward(&mut cg, &buffers, args.threads, pool.as_ref())?;
            let (max_abs, max_rel) = mace_ab::compare_forward(&mut cg, &llmlip, &strided);
            check.push_str(&format!(
                "\"max_abs\":{max_abs:.3e},\"max_rel\":{max_rel:.3e}"
            ));
        }
        "cg-vjp" => {
            let (bar_x, bar_y, bar_w, narrow, wide, grad_w, grad_y) =
                vjp(&mut cg, &buffers, args.threads, pool.as_ref())?;
            let (abs_x, rel_x) = mace_ab::compare_grad_x(&cg, &bar_x, &narrow, &wide);
            let (abs_w, rel_w) = mace_ab::compare_same_order(&bar_w, &grad_w);
            let (abs_y, rel_y) = mace_ab::compare_same_order(&bar_y, &grad_y);
            check.push_str(&format!(
                "\"bar_x_abs\":{abs_x:.3e},\"bar_x_rel\":{rel_x:.3e},\
                 \"bar_w_abs\":{abs_w:.3e},\"bar_w_rel\":{rel_w:.3e},\
                 \"bar_y_abs\":{abs_y:.3e},\"bar_y_rel\":{rel_y:.3e}"
            ));
        }
        other => return Err(format!("unknown cell {other}")),
    }
    check.push('}');
    record.push_str(&check);

    let mut timings = String::from(",\"timings\":{");
    match args.cell.as_str() {
        "cg-fwd" => {
            let mut out =
                vec![T::from_f64(0.0); cg.num_nodes * cg.out_m_total() * cg.feature_dim()];
            let (path, llmlip) = measure_llmlip_fwd(&cg, &buffers, target, args.samples);
            timings.push_str(&format!(
                "\"llmlip\":{},\"llmlip_path\":\"{}\"",
                measured_json(&llmlip),
                path
            ));
            let strided = measure::measure(target, args.samples, || {
                let exec = mace_ab::exec_for(args.threads, pool.as_ref());
                cg.strided_forward(&buffers, black_box(&mut out), exec)
                    .expect("strided forward");
            });
            timings.push_str(&format!(",\"strided\":{}", measured_json(&strided)));
            record.push_str(&timings);
            record.push('}');
            record.push_str(&format!(
                ",\"ratio\":{:.4},\"intermediate_mib_excluded\":{:.1},\"rss_bytes\":{}",
                strided.median_ns / llmlip.median_ns,
                (cg.num_edges as f64 * 5120.0 * std::mem::size_of::<T>() as f64)
                    / (1024.0 * 1024.0),
                measure::peak_rss_bytes()
            ));
        }
        "cg-vjp" => {
            let mut grad_x = vec![T::from_f64(0.0); cg.num_nodes * (128 + 3 * cg.feature_dim())];
            let mut grad_w = vec![T::from_f64(0.0); cg.num_edges * 10 * cg.feature_dim()];
            let mut grad_y = vec![T::from_f64(0.0); cg.num_edges * 16];
            let (path, llmlip) = measure_llmlip_vjp(&mut cg, &buffers, target, args.samples);
            timings.push_str(&format!(
                "\"llmlip\":{},\"llmlip_path\":\"{}\"",
                measured_json(&llmlip),
                path
            ));
            let vjp_x = measure::measure(target, args.samples, || {
                let exec = mace_ab::exec_for(args.threads, pool.as_ref());
                cg.strided_vjp_x(&buffers, black_box(&mut grad_x), exec)
                    .expect("strided vjp-x");
            });
            let vjp_w = measure::measure(target, args.samples, || {
                let exec = mace_ab::exec_for(args.threads, pool.as_ref());
                cg.strided_vjp_w(&buffers, black_box(&mut grad_w), exec)
                    .expect("strided vjp-w");
            });
            let vjp_y = measure::measure(target, args.samples, || {
                let exec = mace_ab::exec_for(args.threads, pool.as_ref());
                cg.strided_vjp_y(&buffers, black_box(&mut grad_y), exec)
                    .expect("strided vjp-y");
            });
            let total = vjp_x.median_ns + vjp_w.median_ns + vjp_y.median_ns;
            timings.push_str(&format!(
                ",\"strided_vjp_x\":{},\"strided_vjp_w\":{},\"strided_vjp_y\":{},\
                 \"strided_total_ns\":{total:.1}",
                measured_json(&vjp_x),
                measured_json(&vjp_w),
                measured_json(&vjp_y)
            ));
            record.push_str(&timings);
            record.push('}');
            record.push_str(&format!(
                ",\"ratio\":{:.4},\"ratio_x\":{:.4},\"ratio_w\":{:.4},\"ratio_y\":{:.4},\
                 \"bar_y_intermediate_mib\":{:.1},\"rss_bytes\":{}",
                total / llmlip.median_ns,
                vjp_x.median_ns / llmlip.median_ns,
                vjp_w.median_ns / llmlip.median_ns,
                vjp_y.median_ns / llmlip.median_ns,
                (cg.num_edges as f64
                    * 16.0
                    * cg.feature_dim() as f64
                    * std::mem::size_of::<T>() as f64)
                    / (1024.0 * 1024.0),
                measure::peak_rss_bytes()
            ));
        }
        _ => unreachable!(),
    }
    record.push('}');
    println!("{record}");
    if let Some(dir) = &args.raw_dir {
        use std::io::Write;
        let path = format!(
            "{dir}/{}-{}-{}T-{}.jsonl",
            args.cell, args.dtype, args.threads, args.atoms
        );
        std::fs::create_dir_all(dir).map_err(|error| format!("{error}"))?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|error| format!("{error}"))?;
        writeln!(file, "{record}").map_err(|error| format!("{error}"))?;
    }
    Ok(())
}

fn measured_json(measured: &Measured) -> String {
    format!(
        "{{\"median_ns\":{:.1},\"min_ns\":{:.1},\"max_ns\":{:.1},\
         \"alloc_bytes_per_call\":{},\"alloc_calls_per_call\":{},\"peak_live_bytes\":{}}}",
        measured.median_ns,
        measured.min_ns,
        measured.max_ns,
        measured.alloc.bytes,
        measured.alloc.calls,
        measured.alloc.peak_live
    )
}

fn measure_llmlip_fwd<T: Scalar>(
    cg: &Cg<T>,
    buffers: &Buffers<T>,
    target: Duration,
    samples: usize,
) -> (&'static str, Measured) {
    let (path, _) = cg.llmlip_forward(buffers).expect("llmlip forward");
    let measured = measure::measure(target, samples, || {
        let (_path, out) = cg.llmlip_forward(buffers).expect("llmlip forward");
        black_box(out);
    });
    (path, measured)
}

fn measure_llmlip_vjp<T: Scalar>(
    cg: &Cg<T>,
    buffers: &Buffers<T>,
    target: Duration,
    samples: usize,
) -> (&'static str, Measured) {
    let (path, _) = cg.llmlip_vjp(buffers).expect("llmlip vjp");
    let measured = measure::measure(target, samples, || {
        let (_path, out) = cg.llmlip_vjp(buffers).expect("llmlip vjp");
        black_box(out);
    });
    (path, measured)
}

fn forward<T: Scalar>(
    cg: &mut Cg<T>,
    buffers: &Buffers<T>,
    threads: usize,
    pool: Option<&Pool>,
) -> Result<(Vec<T>, Vec<T>), String> {
    let (_path, llmlip) = cg.llmlip_forward(buffers)?;
    let mut strided = vec![T::from_f64(0.0); cg.num_nodes * cg.out_m_total() * cg.feature_dim()];
    let exec = mace_ab::exec_for(threads, pool);
    cg.strided_forward(buffers, &mut strided, exec)?;
    Ok((llmlip, strided))
}

fn vjp<T: Scalar>(
    cg: &mut Cg<T>,
    buffers: &Buffers<T>,
    threads: usize,
    pool: Option<&Pool>,
) -> Result<(Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>), String> {
    let (_path, (bar_x, bar_y, bar_w)) = cg.llmlip_vjp(buffers)?;
    let mut grad_x = vec![T::from_f64(0.0); cg.num_nodes * (128 + 3 * cg.feature_dim())];
    let exec = mace_ab::exec_for(threads, pool);
    cg.strided_vjp_x(buffers, &mut grad_x, exec)?;
    let mut grad_w = vec![T::from_f64(0.0); cg.num_edges * 10 * cg.feature_dim()];
    let exec = mace_ab::exec_for(threads, pool);
    cg.strided_vjp_w(buffers, &mut grad_w, exec)?;
    let mut grad_y = vec![T::from_f64(0.0); cg.num_edges * 16];
    let exec = mace_ab::exec_for(threads, pool);
    cg.strided_vjp_y(buffers, &mut grad_y, exec)?;
    let (narrow, wide) = grad_x.split_at(cg.num_nodes * cg.feature_dim());
    Ok((
        bar_x,
        bar_y,
        bar_w,
        narrow.to_vec(),
        wide.to_vec(),
        grad_w,
        grad_y,
    ))
}
