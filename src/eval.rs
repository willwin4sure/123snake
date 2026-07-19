//! EV scorer: run a policy over many seeded episodes and summarize.

use crate::game::Board;
use crate::search::Policy;

#[derive(Clone, Copy, Debug)]
pub struct EpisodeStats {
    pub score: u64,
    pub moves: u32,
    pub max_tile: u64,
}

pub fn run_episode(policy: &mut dyn Policy, board_seed: u32, move_cap: u32) -> EpisodeStats {
    let mut b = Board::new_game(board_seed);
    while b.moves_made < move_cap {
        match policy.choose(&b) {
            Some(mv) => b.apply(&mv),
            None => break,
        }
    }
    EpisodeStats {
        score: b.score,
        moves: b.moves_made,
        max_tile: b.max_tile(),
    }
}

#[derive(Clone, Debug)]
pub struct Summary {
    pub name: String,
    pub episodes: usize,
    pub mean_score: f64,
    pub stderr_score: f64,
    pub p10: u64,
    pub p50: u64,
    pub p90: u64,
    pub p99: u64,
    pub mean_moves: f64,
    pub mean_max_tile: f64,
    pub elapsed_s: f64,
}

impl Summary {
    pub fn header() -> String {
        format!(
            "{:<24} {:>5} {:>9} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9} {:>8}",
            "policy", "n", "mean", "±se", "p10", "p50", "p90", "p99", "moves", "maxtile", "time"
        )
    }

    pub fn row(&self) -> String {
        format!(
            "{:<24} {:>5} {:>9.1} {:>6.1} {:>7} {:>7} {:>7} {:>7} {:>7.1} {:>9.1} {:>7.1}s",
            self.name,
            self.episodes,
            self.mean_score,
            self.stderr_score,
            self.p10,
            self.p50,
            self.p90,
            self.p99,
            self.mean_moves,
            self.mean_max_tile,
            self.elapsed_s
        )
    }
}

/// Run `n` episodes (board seeds seed0..seed0+n) across threads.
/// `make_policy` receives a per-episode policy seed so runs are reproducible.
pub fn evaluate_policy<F>(
    name: &str,
    make_policy: F,
    n: usize,
    move_cap: u32,
    seed0: u32,
    threads: usize,
) -> Summary
where
    F: Fn(u32) -> Box<dyn Policy> + Sync,
{
    let start = std::time::Instant::now();
    let mut stats: Vec<EpisodeStats> = Vec::with_capacity(n);
    let make_ref = &make_policy;
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
                    let mut p = make_ref(board_seed.wrapping_mul(2_654_435_761).wrapping_add(1));
                    out.push(run_episode(p.as_mut(), board_seed, move_cap));
                }
                out
            }));
        }
        for h in handles {
            stats.extend(h.join().expect("eval worker panicked"));
        }
    });

    let mut sorted_scores: Vec<u64> = stats.iter().map(|s| s.score).collect();
    sorted_scores.sort_unstable();
    let pct = |q: f64| {
        let n = sorted_scores.len();
        sorted_scores[(((q * n as f64).ceil() as usize).max(1) - 1).min(n - 1)]
    };
    let nf = stats.len() as f64;
    let mean = |f: &dyn Fn(&EpisodeStats) -> f64| stats.iter().map(f).sum::<f64>() / nf;
    let mean_score = mean(&|s| s.score as f64);
    let var = stats
        .iter()
        .map(|s| (s.score as f64 - mean_score).powi(2))
        .sum::<f64>()
        / (nf - 1.0).max(1.0);
    Summary {
        name: name.to_string(),
        episodes: stats.len(),
        mean_score,
        stderr_score: (var / nf).sqrt(),
        p10: pct(0.10),
        p50: pct(0.50),
        p90: pct(0.90),
        p99: pct(0.99),
        mean_moves: mean(&|s| s.moves as f64),
        mean_max_tile: mean(&|s| s.max_tile as f64),
        elapsed_s: start.elapsed().as_secs_f64(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::RandomPolicy;

    #[test]
    fn episodes_are_reproducible() {
        let mut p1 = RandomPolicy::new(3);
        let mut p2 = RandomPolicy::new(3);
        let a = run_episode(&mut p1, 100, 50);
        let b = run_episode(&mut p2, 100, 50);
        assert_eq!(a.score, b.score);
        assert_eq!(a.moves, b.moves);
    }
}
