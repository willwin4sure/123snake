//! Policies: random baseline and depth-limited sampled expectimax.
//!
//! Refills are chance events (uniform 1/2/3 per vacated cell), so the game
//! tree alternates decision nodes and chance nodes. Chance nodes are sampled
//! (`samples` refill vectors per move) rather than enumerated — a k-cell
//! chain has 3^k refill outcomes. Search draws from its own RNG, never the
//! game's: policies cannot peek at the true upcoming refills.

use crate::game::{Board, Move, Mulberry32};
use crate::heuristics::{evaluate, features, Weights, N_FEATURES};

/// A calibrated linear value function: predicted remaining score =
/// beta0 + betas . features(board). Leaf value = banked score + prediction.
#[derive(Clone, Copy)]
pub struct CalVal {
    pub beta0: f64,
    pub betas: [f64; N_FEATURES],
}

pub fn evaluate_calibrated(b: &Board, cal: &CalVal) -> f64 {
    let f = features(b);
    let mut pred = cal.beta0;
    for i in 0..N_FEATURES {
        pred += cal.betas[i] * f[i];
    }
    b.score as f64 + pred
}

pub trait Policy {
    fn name(&self) -> String;
    fn choose(&mut self, b: &Board) -> Option<Move>;
}

/// Uniform random over legal moves.
pub struct RandomPolicy {
    pub rng: Mulberry32,
}

impl RandomPolicy {
    pub fn new(seed: u32) -> Self {
        Self {
            rng: Mulberry32::new(seed),
        }
    }
}

impl Policy for RandomPolicy {
    fn name(&self) -> String {
        "random".into()
    }

    fn choose(&mut self, b: &Board) -> Option<Move> {
        let moves = b.legal_moves();
        if moves.is_empty() {
            return None;
        }
        let i = self.rng.below(moves.len());
        Some(moves[i].clone())
    }
}

/// Depth-limited expectimax with sampled chance nodes.
/// depth 1 == greedy over the value function.
pub struct ExpectimaxPolicy {
    pub w: Weights,
    pub depth: u32,
    pub samples: u32,
    pub topk: usize,
    pub rng: Mulberry32,
    pub label: String,
    /// When set, leaves are evaluated by the calibrated linear model instead
    /// of the hand value function.
    pub cal: Option<CalVal>,
}

impl ExpectimaxPolicy {
    pub fn new(label: &str, w: Weights, depth: u32, samples: u32, topk: usize, seed: u32) -> Self {
        Self {
            w,
            depth,
            samples,
            topk,
            rng: Mulberry32::new(seed),
            label: label.to_string(),
            cal: None,
        }
    }

    fn leaf(&self, b: &Board) -> f64 {
        match &self.cal {
            Some(c) => evaluate_calibrated(b, c),
            None => evaluate(b, &self.w),
        }
    }

    pub fn greedy(label: &str, w: Weights, seed: u32) -> Self {
        Self::new(label, w, 1, 3, usize::MAX, seed)
    }

    /// Cheap deterministic proxy ordering: score each move on a child where
    /// every refill is a 2 (the median spawn), then keep the top k.
    fn prune(&self, b: &Board, mut moves: Vec<Move>) -> Vec<Move> {
        if moves.len() <= self.topk {
            return moves;
        }
        let mut scored: Vec<(f64, Move)> = moves
            .drain(..)
            .map(|mv| {
                let refills = vec![2u64; mv.path.len() - 1];
                let child = b.apply_with_refills(&mv, &refills);
                (self.leaf(&child), mv)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(self.topk);
        scored.into_iter().map(|(_, mv)| mv).collect()
    }

    /// All legal moves with their sampled EVs, best first. `prune_root: false`
    /// ranks the complete action set (used by the analysis/solver UI).
    pub fn ranked(&mut self, b: &Board, prune_root: bool) -> Vec<(Move, f64)> {
        let moves = b.legal_moves();
        let moves = if prune_root {
            self.prune(b, moves)
        } else {
            moves
        };
        let mut out: Vec<(Move, f64)> = moves
            .into_iter()
            .map(|mv| {
                let ev = self.move_ev(b, &mv, self.depth);
                (mv, ev)
            })
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        out
    }

    fn move_ev(&mut self, b: &Board, mv: &Move, depth: u32) -> f64 {
        let n_refill = mv.path.len() - 1;
        let mut acc = 0.0;
        for _ in 0..self.samples {
            let refills: Vec<u64> = (0..n_refill).map(|_| self.rng.rnd13()).collect();
            let child = b.apply_with_refills(mv, &refills);
            acc += if depth <= 1 {
                self.leaf(&child)
            } else {
                self.best_value(&child, depth - 1)
            };
        }
        let mut ev = acc / self.samples as f64;
        if self.w.w_chainlen != 0.0 {
            let excess = (mv.path.len() as f64 - 3.0).max(0.0);
            ev -= self.w.w_chainlen * excess * excess;
        }
        ev
    }

    fn best_value(&mut self, b: &Board, depth: u32) -> f64 {
        let moves = b.legal_moves();
        if moves.is_empty() {
            return self.leaf(b);
        }
        let moves = self.prune(b, moves);
        let mut best = f64::NEG_INFINITY;
        for mv in &moves {
            let ev = self.move_ev(b, mv, depth);
            if ev > best {
                best = ev;
            }
        }
        best
    }
}

impl Policy for ExpectimaxPolicy {
    fn name(&self) -> String {
        self.label.clone()
    }

    fn choose(&mut self, b: &Board) -> Option<Move> {
        self.ranked(b, true).into_iter().next().map(|(mv, _)| mv)
    }
}

/// Single-player MCTS with double progressive widening and value-function
/// leaf evaluation (no rollouts).
///
/// - Selection: UCT over actions, Q normalized to [0,1] by the running
///   min/max of all backed-up values (scores are unbounded).
/// - Priors: the same deterministic proxy the expectimax pruner uses (child
///   with every refill = 2, evaluated by the value function). Actions are
///   considered in prior order, and only the top ceil(k_act * N^a_act) are
///   eligible at a node with N visits (progressive widening over actions).
/// - Chance: each visit samples a refill vector; identical vectors dedupe to
///   the same child. Per action, at most ceil(k_chance * n^a_chance) distinct
///   children may exist; past the cap, an existing child is picked with
///   probability proportional to its sample count (approximating the i.i.d.
///   refill distribution).
/// - Backup: mean of leaf values (the value function at newly created nodes).
/// - Final move: most-visited root action, ties broken by mean value.
pub struct MctsPolicy {
    pub w: Weights,
    pub sims: u32,
    pub c_uct: f64,
    pub k_act: f64,
    pub a_act: f64,
    pub k_chance: f64,
    pub a_chance: f64,
    pub rng: Mulberry32,
    pub label: String,
    bounds: (f64, f64),
}

struct MChild {
    key: u64,
    node: usize,
    count: u32,
}

struct MAction {
    mv: Move,
    prior: f64,
    visits: u32,
    value_sum: f64,
    children: Vec<MChild>,
}

struct MNode {
    board: Board,
    eval: f64,
    visits: u32,
    actions: Vec<MAction>,
}

impl MctsPolicy {
    pub fn new(label: &str, w: Weights, sims: u32, c_uct: f64, seed: u32) -> Self {
        Self {
            w,
            sims,
            c_uct,
            k_act: 2.0,
            a_act: 0.5,
            k_chance: 1.0,
            a_chance: 0.5,
            rng: Mulberry32::new(seed),
            label: label.to_string(),
            bounds: (f64::INFINITY, f64::NEG_INFINITY),
        }
    }

    fn expand_bounds(&mut self, v: f64) {
        self.bounds.0 = self.bounds.0.min(v);
        self.bounds.1 = self.bounds.1.max(v);
    }

    fn norm(&self, v: f64) -> f64 {
        let (lo, hi) = self.bounds;
        if hi - lo < 1e-9 {
            0.5
        } else {
            (v - lo) / (hi - lo)
        }
    }

    fn new_node(&mut self, board: Board) -> MNode {
        let eval = evaluate(&board, &self.w);
        self.expand_bounds(eval);
        let moves = board.legal_moves();
        let mut actions: Vec<MAction> = moves
            .into_iter()
            .map(|mv| {
                let refills = vec![2u64; mv.path.len() - 1];
                let prior = evaluate(&board.apply_with_refills(&mv, &refills), &self.w);
                self.expand_bounds(prior);
                MAction {
                    mv,
                    prior,
                    visits: 0,
                    value_sum: 0.0,
                    children: Vec::new(),
                }
            })
            .collect();
        actions.sort_by(|a, b| b.prior.partial_cmp(&a.prior).unwrap());
        MNode {
            board,
            eval,
            visits: 0,
            actions,
        }
    }

    fn sim(&mut self, tree: &mut Vec<MNode>, idx: usize) -> f64 {
        if tree[idx].actions.is_empty() {
            let v = tree[idx].eval;
            tree[idx].visits += 1;
            self.expand_bounds(v);
            return v;
        }
        let nv = tree[idx].visits;
        let allowed = ((self.k_act * ((nv + 1) as f64).powf(self.a_act)).ceil() as usize)
            .clamp(1, tree[idx].actions.len());
        let ln_n = ((nv + 1) as f64).ln().max(0.0);
        let mut best = 0;
        let mut best_u = f64::NEG_INFINITY;
        for i in 0..allowed {
            let a = &tree[idx].actions[i];
            let q = if a.visits == 0 {
                self.norm(a.prior)
            } else {
                self.norm(a.value_sum / a.visits as f64)
            };
            let u = q + self.c_uct * (ln_n / ((a.visits + 1) as f64)).sqrt();
            if u > best_u {
                best_u = u;
                best = i;
            }
        }

        let n_refill = tree[idx].actions[best].mv.path.len() - 1;
        let refills: Vec<u64> = (0..n_refill).map(|_| self.rng.rnd13()).collect();
        let key = refills.iter().fold(0u64, |k, &v| k * 3 + (v - 1));
        let existing = tree[idx].actions[best]
            .children
            .iter()
            .position(|c| c.key == key);

        let value = if let Some(ci) = existing {
            let child = tree[idx].actions[best].children[ci].node;
            tree[idx].actions[best].children[ci].count += 1;
            self.sim(tree, child)
        } else {
            let a_visits = tree[idx].actions[best].visits;
            let cap = ((self.k_chance * ((a_visits + 1) as f64).powf(self.a_chance)).ceil()
                as usize)
                .max(1);
            if tree[idx].actions[best].children.len() < cap {
                let mv = tree[idx].actions[best].mv.clone();
                let child_board = tree[idx].board.apply_with_refills(&mv, &refills);
                let node = self.new_node(child_board);
                let v = node.eval;
                let child_idx = tree.len();
                tree.push(node);
                tree[child_idx].visits += 1;
                tree[idx].actions[best].children.push(MChild {
                    key,
                    node: child_idx,
                    count: 1,
                });
                v
            } else {
                let total: u32 = tree[idx].actions[best]
                    .children
                    .iter()
                    .map(|c| c.count)
                    .sum();
                let mut r = self.rng.below(total.max(1) as usize) as u32;
                let mut pick = 0;
                for (i, c) in tree[idx].actions[best].children.iter().enumerate() {
                    if r < c.count {
                        pick = i;
                        break;
                    }
                    r -= c.count;
                }
                let child = tree[idx].actions[best].children[pick].node;
                tree[idx].actions[best].children[pick].count += 1;
                self.sim(tree, child)
            }
        };

        let a = &mut tree[idx].actions[best];
        a.visits += 1;
        a.value_sum += value;
        tree[idx].visits += 1;
        self.expand_bounds(value);
        value
    }
}

impl Policy for MctsPolicy {
    fn name(&self) -> String {
        self.label.clone()
    }

    fn choose(&mut self, b: &Board) -> Option<Move> {
        self.bounds = (f64::INFINITY, f64::NEG_INFINITY);
        let root = self.new_node(b.clone());
        if root.actions.is_empty() {
            return None;
        }
        let mut tree = vec![root];
        for _ in 0..self.sims {
            self.sim(&mut tree, 0);
        }
        tree[0]
            .actions
            .iter()
            .max_by(|a, b| {
                let qa = if a.visits > 0 {
                    a.value_sum / a.visits as f64
                } else {
                    f64::NEG_INFINITY
                };
                let qb = if b.visits > 0 {
                    b.value_sum / b.visits as f64
                } else {
                    f64::NEG_INFINITY
                };
                a.visits.cmp(&b.visits).then(qa.partial_cmp(&qb).unwrap())
            })
            .map(|a| a.mv.clone())
    }
}

/// Rollout re-ranker: the inner expectimax proposes its top candidates, then
/// each is re-scored by short greedy playouts with sampled random refills.
/// This looks past the fixed search horizon: the playout actually realizes
/// merge potential (or fails to) under refill luck, instead of the leaf
/// heuristic guessing at it.
pub struct RolloutPolicy {
    pub inner: ExpectimaxPolicy,
    pub rollouts: u32,
    pub len: u32,
    pub cands: usize,
    pub rng: Mulberry32,
}

impl RolloutPolicy {
    fn playout(&self, mut b: Board, rng: &mut Mulberry32) -> f64 {
        for _ in 0..self.len {
            let moves = b.legal_moves();
            if moves.is_empty() {
                break;
            }
            let mut best_i = 0;
            let mut best_v = f64::NEG_INFINITY;
            for (i, mv) in moves.iter().enumerate() {
                let refills = vec![2u64; mv.path.len() - 1];
                let v = evaluate(&b.apply_with_refills(mv, &refills), &self.inner.w);
                if v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            let mv = moves[best_i].clone();
            let refills: Vec<u64> = (0..mv.path.len() - 1).map(|_| rng.rnd13()).collect();
            b = b.apply_with_refills(&mv, &refills);
        }
        evaluate(&b, &self.inner.w)
    }
}

impl Policy for RolloutPolicy {
    fn name(&self) -> String {
        self.inner.label.clone()
    }

    fn choose(&mut self, b: &Board) -> Option<Move> {
        let ranked = self.inner.ranked(b, true);
        if ranked.is_empty() {
            return None;
        }
        let k = self.cands.min(ranked.len());
        // Common random numbers: every candidate replays the same refill
        // streams, so refill luck cancels and only the move choice differs.
        let stream_seeds: Vec<u32> = (0..self.rollouts)
            .map(|_| (self.rng.next_f64() * 4_294_967_296.0) as u32)
            .collect();
        let mut best_mv: Option<Move> = None;
        let mut best_val = f64::NEG_INFINITY;
        for (mv, _) in ranked.into_iter().take(k) {
            let mut acc = 0.0;
            for &seed in &stream_seeds {
                let mut stream = Mulberry32::new(seed);
                let refills: Vec<u64> =
                    (0..mv.path.len() - 1).map(|_| stream.rnd13()).collect();
                acc += self.playout(b.apply_with_refills(&mv, &refills), &mut stream);
            }
            let val = acc / self.rollouts.max(1) as f64;
            if val > best_val {
                best_val = val;
                best_mv = Some(mv);
            }
        }
        best_mv
    }
}

/// First calibration fit (2026-07-19): ridge on 155k states from 2000
/// self-play games of exp:d2:s8:k12:t4. Holdout R2 0.239, RMSE 449.
pub fn c1() -> CalVal {
    CalVal {
        beta0: -488.697,
        betas: [
            7.8070, -2.7676, 53.8095, 0.1974, 0.0000, 24.6987, -6.7524, 217.6770,
            -20.1253, -23.0711, -23.7403, -51.1202, -73.0807, -37.3657, 17.7916,
            12.0566, -17.4748, -1.4949, 24.8876, 39.7930, 14.2424, -13.3476,
            -29.1856, -1.6993, 0.0513, 1.1128, -0.6707, 0.0154, 14.2533, 192.0977,
        ],
    }
}

/// Parse a policy spec string:
///   "random"
///   "greedy:<weights>"            e.g. greedy:v1
///   "exp:d<depth>:s<samples>:k<topk>:<weights>"   e.g. exp:d2:s3:k8:v1
///   "mcts:n<sims>:c<uct>:<weights>"               e.g. mcts:n1000:c1.5:v5t4
///   "roll:d<d>:s<s>:k<k>:r<rollouts>:l<len>:c<cands>:<weights>"
pub fn parse_policy(spec: &str, seed: u32) -> Result<Box<dyn Policy>, String> {
    let parts: Vec<&str> = spec.split(':').collect();
    match parts[0] {
        "random" => Ok(Box::new(RandomPolicy::new(seed))),
        "greedy" => {
            let wname = parts.get(1).copied().unwrap_or("v1");
            let w = Weights::by_name(wname).ok_or(format!("unknown weights '{wname}'"))?;
            Ok(Box::new(ExpectimaxPolicy::greedy(spec, w, seed)))
        }
        "exp" => {
            let mut depth = 2u32;
            let mut samples = 3u32;
            let mut topk = 8usize;
            let mut wname = "v1";
            for p in &parts[1..] {
                if let Some(d) = p.strip_prefix('d') {
                    if let Ok(x) = d.parse() {
                        depth = x;
                        continue;
                    }
                }
                if let Some(s) = p.strip_prefix('s') {
                    if let Ok(x) = s.parse() {
                        samples = x;
                        continue;
                    }
                }
                if let Some(k) = p.strip_prefix('k') {
                    if let Ok(x) = k.parse() {
                        topk = x;
                        continue;
                    }
                }
                wname = p;
            }
            let (w, cal) = if wname == "c1" {
                (Weights::t4(), Some(c1()))
            } else {
                (
                    Weights::by_name(wname).ok_or(format!("unknown weights '{wname}'"))?,
                    None,
                )
            };
            let mut p = ExpectimaxPolicy::new(spec, w, depth, samples, topk, seed);
            p.cal = cal;
            Ok(Box::new(p))
        }
        "roll" => {
            let (mut depth, mut samples, mut topk) = (2u32, 8u32, 12usize);
            let (mut rollouts, mut len, mut cands) = (4u32, 20u32, 6usize);
            let mut wname = "t2";
            for p in &parts[1..] {
                let mut hit = false;
                for (pre, slot) in [("d", 0), ("s", 1), ("k", 2), ("r", 3), ("l", 4), ("c", 5)] {
                    if let Some(x) = p.strip_prefix(pre).and_then(|v| v.parse::<u32>().ok()) {
                        match slot {
                            0 => depth = x,
                            1 => samples = x,
                            2 => topk = x as usize,
                            3 => rollouts = x,
                            4 => len = x,
                            _ => cands = x as usize,
                        }
                        hit = true;
                        break;
                    }
                }
                if !hit {
                    wname = p;
                }
            }
            let w = Weights::by_name(wname).ok_or(format!("unknown weights '{wname}'"))?;
            let inner = ExpectimaxPolicy::new(spec, w, depth, samples, topk, seed);
            Ok(Box::new(RolloutPolicy {
                inner,
                rollouts,
                len,
                cands,
                rng: Mulberry32::new(seed ^ 0x9E37_79B9),
            }))
        }
        "mcts" => {
            let mut sims = 1000u32;
            let mut c_uct = 1.5f64;
            let mut wname = "v5t4";
            for p in &parts[1..] {
                if let Some(n) = p.strip_prefix('n') {
                    if let Ok(x) = n.parse() {
                        sims = x;
                        continue;
                    }
                }
                if let Some(c) = p.strip_prefix('c') {
                    if let Ok(x) = c.parse() {
                        c_uct = x;
                        continue;
                    }
                }
                wname = p;
            }
            let w = Weights::by_name(wname).ok_or(format!("unknown weights '{wname}'"))?;
            Ok(Box::new(MctsPolicy::new(spec, w, sims, c_uct, seed)))
        }
        other => Err(format!("unknown policy '{other}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policies_produce_legal_moves() {
        let b = Board::new_game(5);
        let legal = b.legal_moves();
        let mut r = RandomPolicy::new(9);
        let mv = r.choose(&b).unwrap();
        assert!(legal.contains(&mv));
        let mut g = ExpectimaxPolicy::greedy("greedy:v1", Weights::v1(), 9);
        let mv = g.choose(&b).unwrap();
        assert!(legal.contains(&mv));
        let mut e = ExpectimaxPolicy::new("exp", Weights::v1(), 2, 2, 6, 9);
        let mv = e.choose(&b).unwrap();
        assert!(legal.contains(&mv));
        let mut rp = super::parse_policy("roll:d2:s3:k6:r2:l6:c3:t3", 9).unwrap();
        let mv = rp.choose(&b).unwrap();
        assert!(legal.contains(&mv));
        let mut m = MctsPolicy::new("mcts", Weights::t2(), 200, 1.5, 9);
        let mv = m.choose(&b).unwrap();
        assert!(legal.contains(&mv));
    }

    #[test]
    fn mcts_is_deterministic_per_seed() {
        let b = Board::new_game(21);
        let mut m1 = MctsPolicy::new("mcts", Weights::t2(), 300, 1.5, 7);
        let mut m2 = MctsPolicy::new("mcts", Weights::t2(), 300, 1.5, 7);
        assert_eq!(m1.choose(&b), m2.choose(&b));
    }

    #[test]
    fn search_does_not_touch_game_rng() {
        let b = Board::new_game(11);
        let rng_before = b.rng;
        let mut e = ExpectimaxPolicy::new("exp", Weights::v1(), 2, 3, 8, 1);
        let _ = e.choose(&b);
        assert_eq!(b.rng, rng_before);
    }
}
