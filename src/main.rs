//! CLI for the 123 Snake engine.
//!
//!   snake eval [--n 200] [--cap 150] [--seed0 1000] [--threads 8] \
//!              [--policies random,greedy:v1,exp:d2:s3:k8:v1]
//!   snake play [--policy greedy:v1] [--seed 42] [--cap 20]

use integer_snake::eval::{evaluate_policy, Summary};
use integer_snake::game::{Board, Mulberry32};
use integer_snake::heuristics::{features, Weights, FEATURE_NAMES, N_FEATURES};
use integer_snake::search::{parse_policy, ExpectimaxPolicy};

fn arg_val(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("eval");
    match cmd {
        "eval" => cmd_eval(&args),
        "play" => cmd_play(&args),
        "tune" => cmd_tune(&args),
        "calibrate" => cmd_calibrate(&args),
        "dump" => cmd_dump(&args),
        "ntuple" => cmd_ntuple(&args),
        other => {
            eprintln!(
                "unknown command '{other}' (expected: eval, play, tune, calibrate, dump, ntuple)"
            );
            std::process::exit(2);
        }
    }
}

/// Afterstate TD(0) + TC self-play training for the n-tuple value network.
///
///   snake ntuple [--games 100000] [--alpha 1.0] [--seed0 0] [--no-2x3]
///                [--eval-every 5000] [--eval-games 100] [--eval-seed0 500000]
///                [--save ml/ntuple-v0.bin] [--load file]
fn cmd_ntuple(args: &[String]) {
    use integer_snake::ntuple::{eval_greedy, train_game, NTupleNet};
    let games: u32 = arg_val(args, "--games")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    let alpha: f32 = arg_val(args, "--alpha")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let seed0: u32 = arg_val(args, "--seed0")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let eval_every: u32 = arg_val(args, "--eval-every")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5000);
    let eval_games: u32 = arg_val(args, "--eval-games")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let eval_seed0: u32 = arg_val(args, "--eval-seed0")
        .and_then(|v| v.parse().ok())
        .unwrap_or(500_000);
    let save = arg_val(args, "--save");
    let with_2x3 = !args.iter().any(|a| a == "--no-2x3");
    let mut net = match arg_val(args, "--load") {
        Some(p) => NTupleNet::load(&p, alpha).expect("load net"),
        None => NTupleNet::new(alpha, with_2x3),
    };
    eprintln!(
        "ntuple: {} games, {} images over {} tables, {} params, alpha {}, 2x3 {}",
        games,
        net.n_images(),
        net.n_tables(),
        net.params(),
        alpha,
        net.with_2x3
    );
    if games == 0 {
        // eval-only: percentiles over the eval block
        let mut sc = integer_snake::ntuple::eval_scores(&net, eval_seed0, eval_games);
        sc.sort_unstable();
        let pct = |p: f64| sc[((sc.len() - 1) as f64 * p) as usize];
        println!(
            "eval n={} seed0={}  mean {:.1}  p10 {}  p50 {}  p90 {}  p99 {}  max {}",
            sc.len(),
            eval_seed0,
            sc.iter().sum::<u64>() as f64 / sc.len() as f64,
            pct(0.10),
            pct(0.50),
            pct(0.90),
            pct(0.99),
            sc[sc.len() - 1]
        );
        return;
    }
    let t0 = std::time::Instant::now();
    let mut window: (u64, u64) = (0, 0); // (games, score) since last report
    for g in 0..games {
        let (score, _) = train_game(&mut net, seed0.wrapping_add(g));
        window.0 += 1;
        window.1 += score;
        if (g + 1) % eval_every == 0 {
            let ev = eval_greedy(&net, eval_seed0, eval_games);
            println!(
                "game {:>7}  train-mean {:>6.1}  eval-greedy {:>6.1}  nonzero {:>9}  {:>5.0}s",
                g + 1,
                window.1 as f64 / window.0 as f64,
                ev,
                net.nonzero(),
                t0.elapsed().as_secs_f64()
            );
            window = (0, 0);
            if let Some(p) = &save {
                net.save(p).expect("save net");
            }
        }
    }
    if let Some(p) = &save {
        net.save(p).expect("save net");
        eprintln!("saved {p}");
    }
}

/// Tournament-style weight tuning: mutate-and-select over the tunable vector.
/// Fitness = mean score over a fixed seed block (common random numbers, so
/// candidate comparisons are paired). w_score stays fixed at 1.0 as the anchor.
fn cmd_tune(args: &[String]) {
    let pop_size: usize = arg_val(args, "--pop")
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let gens: usize = arg_val(args, "--gens")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let n: usize = arg_val(args, "--n")
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    let cap: u32 = arg_val(args, "--cap")
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let seed0: u32 = arg_val(args, "--seed0")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5000);
    let threads: usize = arg_val(args, "--threads")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let base_name = arg_val(args, "--base").unwrap_or_else(|| "t2".to_string());
    let Some(base) = Weights::by_name(&base_name) else {
        eprintln!("unknown base weights '{base_name}'");
        std::process::exit(2);
    };
    // Search budget during fitness evaluation (cheaper than the full report budget).
    let (depth, samples, topk) = (2u32, 8u32, 12usize);

    let mut rng = Mulberry32::new(0x00C0_FFEE);
    let mutate = |t: &[f64; 33], rng: &mut Mulberry32| {
        let mut m = *t;
        for x in m.iter_mut() {
            if rng.next_f64() < 0.6 {
                if *x < 1e-6 {
                    *x = 0.3;
                }
                *x *= (0.35 * (2.0 * rng.next_f64() - 1.0)).exp();
            }
        }
        m
    };

    let fitness = |t: &[f64; 33]| -> f64 {
        let w = base.with_tunable(t);
        evaluate_policy(
            "cand",
            |seed| Box::new(ExpectimaxPolicy::new("cand", w, depth, samples, topk, seed)),
            n,
            cap,
            seed0,
            threads,
        )
        .mean_score
    };

    let mut pop: Vec<[f64; 33]> = vec![base.tunable()];
    while pop.len() < pop_size {
        let t = pop[0];
        pop.push(mutate(&t, &mut rng));
    }

    println!(
        "tuning from '{base_name}': pop {pop_size}, gens {gens}, fitness = mean over {n} episodes (seeds {seed0}+), search d{depth}/s{samples}/k{topk}"
    );
    let mut best: ([f64; 33], f64) = (pop[0], f64::NEG_INFINITY);
    for gen in 0..gens {
        let mut scored: Vec<([f64; 33], f64)> = pop.iter().map(|t| (*t, fitness(t))).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        if scored[0].1 > best.1 {
            best = scored[0];
        }
        println!(
            "gen {gen}: best {:.1}  median {:.1}",
            scored[0].1,
            scored[pop_size / 2].1
        );
        let elites: Vec<[f64; 33]> = scored.iter().take(3).map(|(t, _)| *t).collect();
        pop = elites.clone();
        while pop.len() < pop_size {
            let parent = elites[rng.below(elites.len())];
            pop.push(mutate(&parent, &mut rng));
        }
    }

    println!("\nbest tuned weights (fitness {:.1}):", best.1);
    for (name, v) in Weights::TUNABLE_NAMES.iter().zip(best.0.iter()) {
        println!("  {name}: {v:.3}");
    }
    let tuned = base.with_tunable(&best.0);
    println!(
        "\nfull-budget confirmation (d2/s12/k16, 500 episodes, fresh seeds {}):",
        seed0 + 10_000
    );
    println!("{}", Summary::header());
    for (label, w) in [("base", base), ("tuned", tuned)] {
        let s = evaluate_policy(
            label,
            |seed| Box::new(ExpectimaxPolicy::new(label, w, 2, 12, 16, seed)),
            500,
            cap,
            seed0 + 10_000,
            threads,
        );
        println!("{}", s.row());
    }
}

/// Self-play trajectory dump for neural network training. One JSONL line per
/// visited state: {"g":game,"c":[25 cells],"p":[chosen path cells],"y":remaining}.
/// Terminal states have "p":[]. The path is the teacher policy's actual move.
fn cmd_dump(args: &[String]) {
    let n: usize = arg_val(args, "--n")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    let cap: u32 = arg_val(args, "--cap")
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let seed0: u32 = arg_val(args, "--seed0")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50_000);
    let threads: usize = arg_val(args, "--threads")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let spec = arg_val(args, "--policy").unwrap_or_else(|| "exp:d2:s12:k16:t4".to_string());
    let out_path = arg_val(args, "--out").unwrap_or_else(|| "selfplay.jsonl".to_string());
    if let Err(e) = parse_policy(&spec, 0) {
        eprintln!("{e}");
        std::process::exit(2);
    }
    let spec_ref = &spec;
    let mut lines: Vec<String> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let chunk = n.div_ceil(threads.max(1));
        for t in 0..threads.max(1) {
            let lo = t * chunk;
            let hi = ((t + 1) * chunk).min(n);
            if lo >= hi {
                break;
            }
            handles.push(scope.spawn(move || {
                let mut out: Vec<String> = Vec::new();
                for i in lo..hi {
                    let board_seed = seed0.wrapping_add(i as u32);
                    let mut p = parse_policy(
                        spec_ref,
                        board_seed.wrapping_mul(2_654_435_761).wrapping_add(1),
                    )
                    .expect("validated");
                    let mut b = Board::new_game(board_seed);
                    let mut recs: Vec<(String, u64)> = Vec::new();
                    loop {
                        let cells: Vec<String> = b.cells.iter().map(|v| v.to_string()).collect();
                        if b.moves_made >= cap {
                            recs.push((format!("\"c\":[{}],\"p\":[]", cells.join(",")), b.score));
                            break;
                        }
                        match p.choose(&b) {
                            Some(mv) => {
                                let path: Vec<String> =
                                    mv.path.iter().map(|c| c.to_string()).collect();
                                recs.push((
                                    format!(
                                        "\"c\":[{}],\"p\":[{}]",
                                        cells.join(","),
                                        path.join(",")
                                    ),
                                    b.score,
                                ));
                                b.apply(&mv);
                            }
                            None => {
                                recs.push((
                                    format!("\"c\":[{}],\"p\":[]", cells.join(",")),
                                    b.score,
                                ));
                                break;
                            }
                        }
                    }
                    let fin = b.score;
                    for (body, sc) in recs {
                        out.push(format!("{{\"g\":{i},{body},\"y\":{}}}", fin - sc));
                    }
                }
                out
            }));
        }
        for h in handles {
            lines.extend(h.join().expect("dump worker panicked"));
        }
    });
    std::fs::write(&out_path, lines.join("\n") + "\n").expect("write dump");
    println!("wrote {} states from {n} games to {out_path}", lines.len());
}

/// Value calibration: self-play data generation + ridge regression.
/// Each visited state contributes (features, actual remaining score); the fit
/// is validated on a held-out block of episodes and printed as a Rust literal
/// ready to paste in as a CalVal preset.
#[allow(clippy::needless_range_loop)]
fn cmd_calibrate(args: &[String]) {
    let n: usize = arg_val(args, "--n")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    let cap: u32 = arg_val(args, "--cap")
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let seed0: u32 = arg_val(args, "--seed0")
        .and_then(|v| v.parse().ok())
        .unwrap_or(40_000);
    let threads: usize = arg_val(args, "--threads")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let spec = arg_val(args, "--policy").unwrap_or_else(|| "exp:d2:s8:k12:t4".to_string());
    let lambda: f64 = arg_val(args, "--lambda")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let holdout: f64 = 0.2;

    if let Err(e) = parse_policy(&spec, 0) {
        eprintln!("{e}");
        std::process::exit(2);
    }
    println!("generating {n} self-play episodes with {spec} (seeds {seed0}+)...");
    let spec_ref = &spec;
    // per-episode samples, kept episode-grouped for the holdout split
    let mut episodes: Vec<Vec<([f64; N_FEATURES], f64)>> = Vec::with_capacity(n);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let chunk = n.div_ceil(threads.max(1));
        for t in 0..threads.max(1) {
            let lo = t * chunk;
            let hi = ((t + 1) * chunk).min(n);
            if lo >= hi {
                break;
            }
            handles.push(scope.spawn(move || {
                let mut out = Vec::with_capacity(hi - lo);
                for i in lo..hi {
                    let board_seed = seed0.wrapping_add(i as u32);
                    let mut p = parse_policy(
                        spec_ref,
                        board_seed.wrapping_mul(2_654_435_761).wrapping_add(1),
                    )
                    .expect("validated");
                    let mut b = Board::new_game(board_seed);
                    let mut states: Vec<([f64; N_FEATURES], u64)> = Vec::new();
                    loop {
                        states.push((features(&b), b.score));
                        if b.moves_made >= cap {
                            break;
                        }
                        match p.choose(&b) {
                            Some(mv) => b.apply(&mv),
                            None => break,
                        }
                    }
                    let fin = b.score as f64;
                    out.push(
                        states
                            .into_iter()
                            .map(|(f, sc)| (f, fin - sc as f64))
                            .collect::<Vec<_>>(),
                    );
                }
                out
            }));
        }
        for h in handles {
            episodes.extend(h.join().expect("calibrate worker panicked"));
        }
    });

    let split = ((1.0 - holdout) * episodes.len() as f64) as usize;
    let dim = N_FEATURES + 1; // + bias
    let mut xtx = vec![vec![0.0f64; dim]; dim];
    let mut xty = vec![0.0f64; dim];
    let mut n_train = 0usize;
    for ep in &episodes[..split] {
        for (f, y) in ep {
            let mut row = [0.0f64; 31];
            row[0] = 1.0;
            row[1..].copy_from_slice(f);
            for a in 0..dim {
                for b2 in a..dim {
                    xtx[a][b2] += row[a] * row[b2];
                }
                xty[a] += row[a] * y;
            }
            n_train += 1;
        }
    }
    for a in 0..dim {
        for b2 in 0..a {
            xtx[a][b2] = xtx[b2][a];
        }
        xtx[a][a] += lambda;
    }
    // gaussian elimination with partial pivoting
    let mut m = xtx;
    let mut rhs = xty;
    for col in 0..dim {
        let piv = (col..dim)
            .max_by(|&a, &b2| m[a][col].abs().partial_cmp(&m[b2][col].abs()).unwrap())
            .unwrap();
        m.swap(col, piv);
        rhs.swap(col, piv);
        let d = m[col][col];
        assert!(d.abs() > 1e-12, "singular normal matrix");
        for row in (col + 1)..dim {
            let factor = m[row][col] / d;
            for k in col..dim {
                m[row][k] -= factor * m[col][k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut beta = vec![0.0f64; dim];
    for col in (0..dim).rev() {
        let mut acc = rhs[col];
        for k in (col + 1)..dim {
            acc -= m[col][k] * beta[k];
        }
        beta[col] = acc / m[col][col];
    }

    let metrics = |eps: &[Vec<([f64; N_FEATURES], f64)>]| -> (f64, f64, usize) {
        let (mut sse, mut sst, mut cnt, mut mean) = (0.0, 0.0, 0usize, 0.0);
        for ep in eps {
            for (_, y) in ep {
                mean += y;
                cnt += 1;
            }
        }
        mean /= cnt.max(1) as f64;
        for ep in eps {
            for (f, y) in ep {
                let mut pred = beta[0];
                for i in 0..N_FEATURES {
                    pred += beta[i + 1] * f[i];
                }
                sse += (y - pred) * (y - pred);
                sst += (y - mean) * (y - mean);
            }
        }
        (
            1.0 - sse / sst.max(1e-9),
            (sse / cnt.max(1) as f64).sqrt(),
            cnt,
        )
    };
    let (r2_tr, rmse_tr, n_tr) = metrics(&episodes[..split]);
    let (r2_ho, rmse_ho, n_ho) = metrics(&episodes[split..]);
    println!("train:   n={n_tr} (from {split} episodes)  R2={r2_tr:.4}  RMSE={rmse_tr:.1}");
    println!(
        "holdout: n={n_ho} (from {} episodes)  R2={r2_ho:.4}  RMSE={rmse_ho:.1}",
        episodes.len() - split
    );
    let _ = n_train;

    println!(
        "
coefficients (predicted remaining score):"
    );
    println!("  bias: {:.3}", beta[0]);
    for (i, name) in FEATURE_NAMES.iter().enumerate() {
        println!("  {name}: {:.4}", beta[i + 1]);
    }
    println!(
        "
Rust literal for CalVal:"
    );
    println!("CalVal {{ beta0: {:.6}, betas: [", beta[0]);
    for i in 0..N_FEATURES {
        println!("    {:.6},", beta[i + 1]);
    }
    println!("] }}");
}

fn cmd_eval(args: &[String]) {
    let n: usize = arg_val(args, "--n")
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let cap: u32 = arg_val(args, "--cap")
        .and_then(|v| v.parse().ok())
        .unwrap_or(150);
    let seed0: u32 = arg_val(args, "--seed0")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let threads: usize = arg_val(args, "--threads")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let policies = arg_val(args, "--policies")
        .unwrap_or_else(|| "random,greedy:v1,exp:d2:s3:k8:v1".to_string());

    println!(
        "episodes {n}, move cap {cap}, board seeds {seed0}..{}, {threads} threads",
        seed0 as usize + n - 1
    );
    println!("{}", Summary::header());
    for spec in policies.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if let Err(e) = parse_policy(spec, 0) {
            eprintln!("skipping '{spec}': {e}");
            continue;
        }
        let summary = evaluate_policy(
            spec,
            |seed| parse_policy(spec, seed).expect("validated above"),
            n,
            cap,
            seed0,
            threads,
        );
        println!("{}", summary.row());
    }
}

fn cmd_play(args: &[String]) {
    let seed: u32 = arg_val(args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(42);
    let cap: u32 = arg_val(args, "--cap")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let spec = arg_val(args, "--policy").unwrap_or_else(|| "greedy:v1".to_string());
    let mut policy = match parse_policy(&spec, seed ^ 0xABCD_1234) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    let mut b = Board::new_game(seed);
    println!("policy {} on board seed {seed}\n\n{b}", policy.name());
    while b.moves_made < cap {
        let Some(mv) = policy.choose(&b) else {
            println!("no moves left — game over");
            break;
        };
        let v = b.cells[mv.path[0] as usize];
        let cells: Vec<String> = mv
            .path
            .iter()
            .map(|&i| {
                let (r, c) = integer_snake::game::rc(i as usize);
                format!("({r},{c})")
            })
            .collect();
        b.apply(&mv);
        println!(
            "move {}: chain {} x{} -> {} at {}\n\n{b}",
            b.moves_made,
            v,
            mv.path.len(),
            v * mv.path.len() as u64,
            cells.join("->"),
        );
    }
}
