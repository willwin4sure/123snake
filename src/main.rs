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
        "serve" => cmd_serve(&args),
        other => {
            eprintln!(
                "unknown command '{other}' (expected: eval, play, tune, calibrate, dump, ntuple)"
            );
            std::process::exit(2);
        }
    }
}

/// Local watch server: loads an n-tuple checkpoint and serves a page where
/// the bot plays visibly. Weights never leave the machine.
///
///   snake serve [--model ml/ntuple-v1.bin] [--port 8271]
fn cmd_serve(args: &[String]) {
    use integer_snake::ntuple::NTupleNet;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    let model = arg_val(args, "--model").unwrap_or_else(|| "ml/ntuple-v1.bin".to_string());
    let port: u16 = arg_val(args, "--port")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8271);
    eprintln!("loading {model} ...");
    let net = NTupleNet::load(&model, 1.0).expect("load model");
    eprintln!("{} params; http://127.0.0.1:{port}/", net.params());

    let html = include_str!("../solver/watch.html");
    let mut game: Option<Board> = None;
    let mut history: Vec<Board> = Vec::new();
    let mut srng = Mulberry32::new(0x00C0_FFEE);
    // unbiased action value: V(afterstate) is only trained on-policy and
    // underestimates off-policy moves, so score a chosen move by sampling
    // refills and taking the mean best-reply value
    fn sampled_av_n(
        net: &integer_snake::ntuple::NTupleNet,
        b: &Board,
        mv: &integer_snake::game::Move,
        sum: u64,
        rng: &mut Mulberry32,
        s: u32,
    ) -> f64 {
        let mut acc = 0.0;
        for _ in 0..s {
            let refills: Vec<u64> = (0..mv.path.len() - 1).map(|_| rng.rnd13()).collect();
            let child = b.apply_with_refills(mv, &refills);
            acc += sum as f64
                + net
                    .greedy(&child)
                    .map(|(_, r, a)| r + net.value(&a))
                    .unwrap_or(0.0);
        }
        acc / s as f64
    }
    fn sampled_av(
        net: &integer_snake::ntuple::NTupleNet,
        b: &Board,
        mv: &integer_snake::game::Move,
        sum: u64,
        rng: &mut Mulberry32,
    ) -> f64 {
        sampled_av_n(net, b, mv, sum, rng, 24)
    }
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind");

    fn state_json(b: &Board, v: Option<f64>, hist: usize) -> String {
        let cells: Vec<String> = b.cells.iter().map(|c| c.to_string()).collect();
        format!(
            "{{\"cells\":[{}],\"score\":{},\"moves\":{},\"over\":{},\"hist\":{},\"v\":{}}}",
            cells.join(","),
            b.score,
            b.moves_made,
            u8::from(!b.has_moves()),
            hist,
            v.map_or("null".to_string(), |x| format!("{x:.1}"))
        )
    }

    fn valid_path(b: &Board, path: &[u8]) -> bool {
        use integer_snake::game::{neighbors, CELLS};
        if path.len() < 2 || path.len() > CELLS {
            return false;
        }
        let Some(&c0) = path.first() else {
            return false;
        };
        if c0 as usize >= CELLS {
            return false;
        }
        let v0 = b.cells[c0 as usize];
        let mut mask = 0u32;
        for (k, &c) in path.iter().enumerate() {
            let c = c as usize;
            if c >= CELLS || mask & (1 << c) != 0 || b.cells[c] != v0 {
                return false;
            }
            if k > 0 {
                let prev = path[k - 1] as usize;
                let mut ok = false;
                neighbors(prev, |nb| {
                    if nb == c {
                        ok = true;
                    }
                });
                if !ok {
                    return false;
                }
            }
            mask |= 1 << c;
        }
        true
    }

    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let mut line = String::new();
        if BufReader::new(&stream).read_line(&mut line).is_err() {
            continue;
        }
        let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
        let (route, query) = match path.split_once('?') {
            Some((r, q)) => (r, q),
            None => (path.as_str(), ""),
        };
        let (ctype, body) = match route {
            "/" => ("text/html; charset=utf-8", html.to_string()),
            "/info" => (
                "application/json",
                format!(
                    "{{\"model\":\"{}\",\"params\":{}}}",
                    model.replace('"', ""),
                    net.params()
                ),
            ),
            "/new" => {
                let seed: u32 = query
                    .strip_prefix("seed=")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1);
                let b = Board::new_game(seed);
                history.clear();
                let v = net.greedy(&b).map(|(_, r, a)| r + net.value(&a));
                let json = state_json(&b, v, 0);
                game = Some(b);
                ("application/json", json)
            }
            "/moves" => match &game {
                Some(b) => {
                    let codes = net.encode(&b.cells);
                    let mut ranked: Vec<(Vec<u8>, u64, f64)> = b
                        .legal_moves_capped(integer_snake::game::MOVE_CAP)
                        .into_iter()
                        .map(|mv| {
                            let sum = b.cells[mv.path[0] as usize] * mv.path.len() as u64;
                            let val = sum as f64 + net.value(&net.afterstate(&codes, &mv, sum));
                            (mv.path, sum, val)
                        })
                        .collect();
                    ranked
                        .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                    let total = ranked.len();
                    let rows: Vec<String> = ranked
                        .iter()
                        .take(40)
                        .map(|(p, s, v)| {
                            let pc: Vec<String> = p.iter().map(|c| c.to_string()).collect();
                            format!("{{\"path\":[{}],\"sum\":{s},\"v\":{v:.1}}}", pc.join(","))
                        })
                        .collect();
                    (
                        "application/json",
                        format!("{{\"total\":{total},\"moves\":[{}]}}", rows.join(",")),
                    )
                }
                None => ("application/json", "{\"total\":0,\"moves\":[]}".to_string()),
            },
            "/apply" => {
                let path: Vec<u8> = query
                    .strip_prefix("path=")
                    .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
                    .unwrap_or_default();
                match &mut game {
                    Some(b) if valid_path(b, &path) => {
                        let sum = b.cells[path[0] as usize] * path.len() as u64;
                        history.push(b.clone());
                        let mv = integer_snake::game::Move { path: path.clone() };
                        // fair baseline: the bot's best move, sampled the same way
                        let base = net
                            .greedy(b)
                            .map(|(bm, _, _)| {
                                let bsum = b.cells[bm.path[0] as usize] * bm.path.len() as u64;
                                sampled_av(&net, b, &bm, bsum, &mut srng)
                            })
                            .unwrap_or(0.0);
                        let av = sampled_av(&net, b, &mv, sum, &mut srng);
                        b.apply(&mv);
                        let v = net.greedy(b).map(|(_, r, a)| r + net.value(&a));
                        let st = state_json(b, v, history.len());
                        let pc: Vec<String> = path.iter().map(|c| c.to_string()).collect();
                        (
                            "application/json",
                            format!(
                                "{{\"path\":[{}],\"sum\":{sum},\"av\":{av:.1},\"base\":{base:.1},{}",
                                pc.join(","),
                                st.trim_start_matches('{')
                            ),
                        )
                    }
                    _ => ("application/json", "{\"err\":\"invalid\"}".to_string()),
                }
            }
            "/undo" => match &mut game {
                Some(b) => match history.pop() {
                    Some(prev) => {
                        *b = prev;
                        let v = net.greedy(b).map(|(_, r, a)| r + net.value(&a));
                        ("application/json", state_json(b, v, history.len()))
                    }
                    None => ("application/json", "{\"err\":\"empty\"}".to_string()),
                },
                None => ("application/json", "{\"err\":\"nogame\"}".to_string()),
            },
            "/step" => match &mut game {
                Some(b) if b.has_moves() => {
                    // search mode: rank by the one-ply proxy, then pick the
                    // best of the top 16 by 48-sample expectimax (the
                    // measured test-time-compute sweet spot, +35%)
                    let (mv, av) = if query.contains("search=1") {
                        let codes = net.encode(&b.cells);
                        let mut scored: Vec<(integer_snake::game::Move, u64, f64)> = b
                            .legal_moves_capped(integer_snake::game::MOVE_CAP)
                            .into_iter()
                            .map(|mv| {
                                let sum = b.cells[mv.path[0] as usize] * mv.path.len() as u64;
                                let v = sum as f64 + net.value(&net.afterstate(&codes, &mv, sum));
                                (mv, sum, v)
                            })
                            .collect();
                        scored.sort_by(|x, y| {
                            y.2.partial_cmp(&x.2).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        scored.truncate(16);
                        scored
                            .into_iter()
                            .map(|(mv, sum, _)| {
                                let v = sampled_av_n(&net, b, &mv, sum, &mut srng, 48);
                                (mv, v)
                            })
                            .max_by(|x, y| {
                                x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .expect("moves exist")
                    } else {
                        let (mv, _, _) = net.greedy(b).expect("moves exist");
                        let sum = b.cells[mv.path[0] as usize] * mv.path.len() as u64;
                        let av = sampled_av(&net, b, &mv, sum, &mut srng);
                        (mv, av)
                    };
                    let sum = b.cells[mv.path[0] as usize] * mv.path.len() as u64;
                    let base = av; // the bot plays its own baseline
                    let path_cells: Vec<String> = mv.path.iter().map(|c| c.to_string()).collect();
                    history.push(b.clone());
                    b.apply(&mv);
                    let v = net.greedy(b).map(|(_, r, a)| r + net.value(&a));
                    let st = state_json(b, v, history.len());
                    (
                        "application/json",
                        format!(
                            "{{\"path\":[{}],\"sum\":{sum},\"av\":{av:.1},\"base\":{base:.1},{}",
                            path_cells.join(","),
                            st.trim_start_matches('{')
                        ),
                    )
                }
                _ => ("application/json", "{\"done\":true}".to_string()),
            },
            _ => ("text/plain", "not found".to_string()),
        };
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
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
        .unwrap_or(1000);
    let eval_seed0: u32 = arg_val(args, "--eval-seed0")
        .and_then(|v| v.parse().ok())
        .unwrap_or(500_000);
    let save = arg_val(args, "--save");
    let stage_thresholds: [u32; 2] = arg_val(args, "--stage-thresholds")
        .and_then(|v| {
            let (a, b) = v.split_once(':')?;
            Some([a.parse().ok()?, b.parse().ok()?])
        })
        .unwrap_or([96, 768]);
    let cfg = integer_snake::ntuple::NetConfig {
        alphabet: match arg_val(args, "--alphabet").as_deref() {
            Some("fine") => integer_snake::ntuple::Alphabet::Fine,
            Some("slim") => integer_snake::ntuple::Alphabet::Slim,
            Some("slim89") => integer_snake::ntuple::Alphabet::Slim89,
            _ => {
                if arg_val(args, "--grow").is_some_and(|g| g.contains("alphabet")) {
                    integer_snake::ntuple::Alphabet::Coarse
                } else {
                    integer_snake::ntuple::Alphabet::Base
                }
            }
        },
        with_2x3: !args.iter().any(|a| a == "--no-2x3"),
        pos_2x3: args.iter().any(|a| a == "--pos-2x3"),
        staircase: args.iter().any(|a| a == "--staircase"),
        diagonals: args.iter().any(|a| a == "--diagonals"),
        global: args.iter().any(|a| a == "--global"),
        global2: args.iter().any(|a| a == "--global2"),
        stages: arg_val(args, "--stages")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1),
        stage_thresholds,
        extra: {
            use integer_snake::ntuple::*;
            arg_val(args, "--extra")
                .map(|v| {
                    v.split(',')
                        .map(|f| match f {
                            "bigL" => EX_BIGL,
                            "x" => EX_X,
                            "stair6" => EX_STAIR6,
                            "gated12" => EX_GATED12,
                            "avgdisp" => EX_AVGDISP,
                            "blobtier" => EX_BLOBTIER,
                            "blobalpha" => EX_BLOBALPHA,
                            "blob2" => EX_BLOB2,
                            "freefield" => EX_FREEFIELD,
                            "eqpairs" => EX_EQPAIRS,
                            "pathtier" => EX_PATHTIER,
                            "pathalpha" => EX_PATHALPHA,
                            "path2" => EX_PATH2,
                            other => panic!("unknown --extra flag {other}"),
                        })
                        .fold(0, |a, b| a | b)
                })
                .unwrap_or(0)
        },
    };
    let lambda: f32 = arg_val(args, "--lambda")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let (eps_rank, eps_rand): (f32, f32) = arg_val(args, "--explore")
        .and_then(|v| {
            let (a, b) = v.split_once(':')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((0.0, 0.0));
    let adapt_stages: f64 = arg_val(args, "--adapt-stages")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);

    let grow: Vec<String> = arg_val(args, "--grow")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let start_min = args.iter().any(|a| a == "--start-min");
    let trace_len: usize = arg_val(args, "--trace")
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let mut net = match arg_val(args, "--load") {
        Some(p) => NTupleNet::load(&p, alpha).expect("load net"),
        None => NTupleNet::new(alpha, cfg),
    };
    if start_min {
        use integer_snake::ntuple::{G_ROWS, G_SQ2};
        net.set_active_groups(&[G_ROWS, G_SQ2]);
        eprintln!("start-min: only rows + 2x2 active");
    }
    if adapt_stages > 0.0 {
        net.stage_cap = 0;
        net.promote = true;
        eprintln!("adaptive stages: activate at frac >= {adapt_stages}");
    }
    if let Some(v0) = arg_val(args, "--optimism").and_then(|v| v.parse::<f32>().ok()) {
        net.init_optimistic(v0);
        eprintln!("optimistic init: V0 = {v0}");
    }
    if let Some(bt) = arg_val(args, "--bonus").and_then(|v| v.parse::<f32>().ok()) {
        net.enable_bonus(bt);
        eprintln!("exploration bonus enabled: total {bt}, decays 1/sqrt(1+visits)");
    }
    if args.iter().any(|a| a == "--promote") {
        net.promote = true;
        eprintln!("stage promotion enabled");
    }
    eprintln!(
        "ntuple: {} games, {} images over {} tables, {} params, alpha {}, cfg {:?}",
        games,
        net.n_images(),
        net.n_tables(),
        net.params(),
        alpha,
        net.cfg
    );
    if let Some(spec) = arg_val(args, "--compare-codes") {
        let v: Vec<u8> = spec.split(':').filter_map(|x| x.parse().ok()).collect();
        assert_eq!(v.len(), 3, "--compare-codes a:b:c");
        println!(
            "comparing codes {} vs {} vs {} (per tuple table):",
            v[0], v[1], v[2]
        );
        println!("table  triples   corr(a,b) corr(a,c) corr(b,c)  m|a-b|  m|a-c|  w_std");
        for (t, n, cab, cac, cbc, dab, dac, wstd) in net.compare_codes(v[0], v[1], v[2]) {
            println!(
                "  t{:<3} {:>9}  {:>8.3} {:>9.3} {:>9.3}  {:>6.2}  {:>6.2}  {:>6.2}",
                t, n, cab, cac, cbc, dab, dac, wstd
            );
        }
        return;
    }
    if args.iter().any(|a| a == "--show-global") {
        // decode the learned top-2 position table: which big-tile pair
        // geometries the net values most and least (canonical entries only)
        let rows = net.global_pair_table();
        let mut seen: Vec<(usize, f32, f32)> = rows
            .iter()
            .enumerate()
            .filter(|(_, (w, _))| *w != 0.0)
            .map(|(i, (w, a))| (i, *w, *a))
            .collect();
        seen.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
        let show = |list: &[(usize, f32, f32)]| {
            for (i, w, a) in list {
                let (p1, p2) = (i / 25, i % 25);
                println!(
                    "  top1 at r{}c{}, top2 at r{}c{}: w {:>+9.1}  (|err| mass {:.0})",
                    p1 / 5,
                    p1 % 5,
                    p2 / 5,
                    p2 % 5,
                    w,
                    a
                );
            }
        };
        println!("highest-valued pair geometries:");
        show(&seen[..seen.len().min(10)]);
        println!("lowest-valued pair geometries:");
        let lo = &seen[seen.len().saturating_sub(10)..];
        show(lo);
        return;
    }
    if let Some(spec) = arg_val(args, "--probe") {
        // calibration probe: for states along greedy play, compare each
        // ranked move's claimed value av = r + V(afterstate) against the
        // sampled truth E_refills[r + best reply value]. Positive bias at
        // rank k means V underestimates rank-k moves.
        let (kmax, samples) = spec.split_once(':').expect("--probe ranks:samples");
        let (kmax, samples): (usize, u32) = (
            kmax.parse().expect("ranks"),
            samples.parse().expect("samples"),
        );
        let mut rng = Mulberry32::new(777);
        let mut acc = vec![(0.0f64, 0u32); kmax];
        for g in 0..eval_games {
            let mut b = Board::new_game(eval_seed0 + g);
            let depth = rng.below(40) + 5;
            for _ in 0..depth {
                match net.greedy(&b) {
                    Some((mv, _, _)) => b.apply(&mv),
                    None => break,
                }
            }
            if !b.has_moves() {
                continue;
            }
            let codes = net.encode(&b.cells);
            let mut ranked: Vec<(integer_snake::game::Move, f64, u64)> = b
                .legal_moves_capped(integer_snake::game::MOVE_CAP)
                .into_iter()
                .map(|mv| {
                    let sum = b.cells[mv.path[0] as usize] * mv.path.len() as u64;
                    let av = sum as f64 + net.value(&net.afterstate(&codes, &mv, sum));
                    (mv, av, sum)
                })
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (k, (mv, av, sum)) in ranked.iter().take(kmax).enumerate() {
                let mut m = 0.0;
                for _ in 0..samples {
                    let refills: Vec<u64> = (0..mv.path.len() - 1).map(|_| rng.rnd13()).collect();
                    let child = b.apply_with_refills(mv, &refills);
                    m += *sum as f64
                        + net
                            .greedy(&child)
                            .map(|(_, r, a)| r + net.value(&a))
                            .unwrap_or(0.0);
                }
                acc[k].0 += m / samples as f64 - av;
                acc[k].1 += 1;
            }
        }
        println!("calibration probe: E[realized] - claimed, by move rank");
        for (k, (sum, n)) in acc.iter().enumerate() {
            if *n > 0 {
                println!("  rank {:>2}: {:>+7.1}  (n={})", k + 1, sum / *n as f64, n);
            }
        }
        return;
    }
    if games == 0 {
        // eval-only: percentiles over the eval block; --exp topk:samples
        // switches from one-ply greedy to depth-2 net-leaf expectimax
        let mut sc: Vec<u64> = match arg_val(args, "--exp") {
            Some(spec) => {
                use integer_snake::ntuple::NTupleSearchPolicy;
                use integer_snake::search::Policy;
                let (k, s) = spec.split_once(':').expect("--exp topk:samples");
                let mut pol = NTupleSearchPolicy::new(
                    net,
                    k.parse().expect("topk"),
                    s.parse().expect("samples"),
                    99,
                );
                (0..eval_games)
                    .map(|g| {
                        let mut b = Board::new_game(eval_seed0 + g);
                        while let Some(mv) = pol.choose(&b) {
                            b.apply(&mv);
                        }
                        b.score
                    })
                    .collect()
            }
            None => {
                let pairs = integer_snake::ntuple::eval_scores_tiles(&net, eval_seed0, eval_games);
                if args.iter().any(|a| a == "--tiles") {
                    let mut hist: std::collections::BTreeMap<u64, u32> = Default::default();
                    for (_, t) in &pairs {
                        *hist.entry(*t).or_default() += 1;
                    }
                    println!("max-tile distribution over {} games:", pairs.len());
                    let n = pairs.len() as f64;
                    let mut cum = 0.0;
                    for (t, c) in hist.iter().rev() {
                        cum += *c as f64 / n * 100.0;
                        println!(
                            "  {:>6}: {:>5.1}%  (cum >= {:.1}%)",
                            t,
                            *c as f64 / n * 100.0,
                            cum
                        );
                    }
                }
                pairs.into_iter().map(|p| p.0).collect()
            }
        };
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
                                         // growth controller: on plateau (200k-window mean < 2% over the best so
                                         // far), fire the next pending growth action
    let mut grow_queue: std::collections::VecDeque<String> = grow.iter().cloned().collect();
    let mut gwin: (u64, u64) = (0, 0);
    let mut gbest: f64 = 0.0;
    // adaptive staging: activate stage s when >= adapt_stages of the last
    // 5000 games reached its max-tile threshold
    let mut reach: std::collections::VecDeque<(bool, bool)> = Default::default();
    for g in 0..games {
        let (score, _, maxt) = if lambda > 0.0 {
            integer_snake::ntuple::train_game_lambda_eps(
                &mut net,
                seed0.wrapping_add(g),
                lambda,
                trace_len,
                eps_rank,
                eps_rand,
            )
        } else {
            integer_snake::ntuple::train_game_eps(
                &mut net,
                seed0.wrapping_add(g),
                eps_rank,
                eps_rand,
            )
        };
        if net.cfg.stages > 1 {
            let [t1, t2] = net.cfg.stage_thresholds;
            reach.push_front((maxt >= t1 as u64, maxt >= t2 as u64));
            reach.truncate(5000);
            if adapt_stages > 0.0 && net.stage_cap + 1 < net.cfg.stages && reach.len() == 5000 {
                let first = net.stage_cap == 0;
                let frac = reach
                    .iter()
                    .filter(|r| if first { r.0 } else { r.1 })
                    .count() as f64
                    / 5000.0;
                if frac >= adapt_stages {
                    let s = net.stage_cap + 1;
                    net.activate_stage(s);
                    println!(
                        "game {:>7}  STAGE {} activated (frac {:.2})",
                        g + 1,
                        s,
                        frac
                    );
                }
            }
        }
        if !grow_queue.is_empty() {
            gwin.0 += 1;
            gwin.1 += score;
            if gwin.0 == 200_000 {
                let mean = gwin.1 as f64 / gwin.0 as f64;
                if mean < gbest * 1.02 {
                    let action = grow_queue.pop_front().unwrap();
                    use integer_snake::ntuple::{G_BLK23, G_PLUS, G_STAIR};
                    match action.as_str() {
                        "plus" => net.activate_group(G_PLUS),
                        "stair" => net.activate_group(G_STAIR),
                        "2x3" => net.activate_group(G_BLK23),
                        "alphabet" => net.grow_alphabet(),
                        other => eprintln!("unknown grow action {other}"),
                    }
                    println!(
                        "game {:>7}  GROW: {} (window mean {:.1} vs best {:.1})",
                        g + 1,
                        action,
                        mean,
                        gbest
                    );
                    gbest = 0.0;
                } else {
                    gbest = gbest.max(mean);
                }
                gwin = (0, 0);
            }
        }
        window.0 += 1;
        window.1 += score;
        if (g + 1) % eval_every == 0 {
            if eval_games > 0 {
                let ev = eval_greedy(&net, eval_seed0, eval_games);
                println!(
                    "game {:>7}  train-mean {:>6.1}  eval-greedy {:>6.1}  nonzero {:>9}  {:>5.0}s",
                    g + 1,
                    window.1 as f64 / window.0 as f64,
                    ev,
                    net.nonzero(),
                    t0.elapsed().as_secs_f64()
                );
            } else {
                println!(
                    "game {:>7}  train-mean {:>6.1}  nonzero {:>9}  {:>5.0}s",
                    g + 1,
                    window.1 as f64 / window.0 as f64,
                    net.nonzero(),
                    t0.elapsed().as_secs_f64()
                );
            }
            if net.cfg.stages > 1 && !reach.is_empty() {
                let n = reach.len() as f64;
                let f1 = reach.iter().filter(|r| r.0).count() as f64 / n;
                let f2 = reach.iter().filter(|r| r.1).count() as f64 / n;
                println!("game {:>7}  gatefrac {:.3} {:.3}", g + 1, f1, f2);
            }
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
