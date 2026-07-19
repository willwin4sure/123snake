//! Core game state and rules.

use std::collections::HashSet;
use std::fmt;

pub const N: usize = 5;
pub const CELLS: usize = N * N;

/// Hard cap on distinct moves returned by move generation, and on DFS work,
/// to bound pathological boards (large same-valued components). The DFS
/// budget is sliced per start cell so a huge component near cell 0 cannot
/// starve move enumeration for the rest of the board.
pub const MOVE_CAP: usize = 4096;
const DFS_BUDGET_PER_START: usize = 8_000;

/// mulberry32, bit-compatible with the JS implementation in the web build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mulberry32 {
    pub state: u32,
}

impl Mulberry32 {
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// Uniform f64 in [0, 1), identical to the JS `rand()`.
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6D2B_79F5);
        let mut t = self.state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f64) / 4_294_967_296.0
    }

    /// Uniform tile value in {1, 2, 3}, identical to the JS `rnd13()`.
    pub fn rnd13(&mut self) -> u64 {
        1 + (self.next_f64() * 3.0) as u64
    }

    /// Uniform index in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_f64() * n as f64) as usize).min(n - 1)
    }
}

pub const fn idx(r: usize, c: usize) -> usize {
    r * N + c
}

pub const fn rc(i: usize) -> (usize, usize) {
    (i / N, i % N)
}

/// Calls `f` with each orthogonal neighbor of cell `i`.
#[inline]
pub fn neighbors(i: usize, mut f: impl FnMut(usize)) {
    let r = i / N;
    let c = i % N;
    if r > 0 {
        f(i - N);
    }
    if r + 1 < N {
        f(i + N);
    }
    if c > 0 {
        f(i - 1);
    }
    if c + 1 < N {
        f(i + 1);
    }
}

/// A move: a self-avoiding path of equal-valued cells; the last cell is the
/// head that receives the sum. Cell indices are row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Move {
    pub path: Vec<u8>,
}

impl Move {
    pub fn head(&self) -> usize {
        *self.path.last().expect("move path is never empty") as usize
    }

    pub fn mask(&self) -> u32 {
        self.path.iter().fold(0u32, |m, &i| m | 1 << i)
    }
}

#[derive(Clone, Debug)]
pub struct Board {
    pub cells: [u64; CELLS],
    pub score: u64,
    pub rng: Mulberry32,
    pub moves_made: u32,
}

impl Board {
    /// Fresh 5x5 board of uniform 1/2/3 tiles, redrawn (like the web build)
    /// until at least one move exists.
    pub fn new_game(seed: u32) -> Self {
        let mut rng = Mulberry32::new(seed);
        let mut cells = [0u64; CELLS];
        loop {
            for cell in cells.iter_mut() {
                *cell = rng.rnd13();
            }
            if Self::cells_have_moves(&cells) {
                break;
            }
        }
        Board {
            cells,
            score: 0,
            rng,
            moves_made: 0,
        }
    }

    pub fn from_state(cells: [u64; CELLS], score: u64, rng_state: u32) -> Self {
        Board {
            cells,
            score,
            rng: Mulberry32::new(rng_state),
            moves_made: 0,
        }
    }

    pub fn cells_have_moves(cells: &[u64; CELLS]) -> bool {
        for r in 0..N {
            for c in 0..N {
                let v = cells[idx(r, c)];
                if c + 1 < N && cells[idx(r, c + 1)] == v {
                    return true;
                }
                if r + 1 < N && cells[idx(r + 1, c)] == v {
                    return true;
                }
            }
        }
        false
    }

    pub fn has_moves(&self) -> bool {
        Self::cells_have_moves(&self.cells)
    }

    pub fn is_over(&self) -> bool {
        !self.has_moves()
    }

    /// Number of orthogonally adjacent equal-valued pairs (cheap optionality
    /// proxy used by heuristics).
    pub fn adjacent_pairs(&self) -> u32 {
        let mut n = 0;
        for r in 0..N {
            for c in 0..N {
                let v = self.cells[idx(r, c)];
                if c + 1 < N && self.cells[idx(r, c + 1)] == v {
                    n += 1;
                }
                if r + 1 < N && self.cells[idx(r + 1, c)] == v {
                    n += 1;
                }
            }
        }
        n
    }

    pub fn max_tile(&self) -> u64 {
        *self.cells.iter().max().unwrap()
    }

    /// All legal moves, deduplicated by (cell set, head): two different drag
    /// orders that clear the same cells into the same head are one outcome.
    pub fn legal_moves(&self) -> Vec<Move> {
        self.legal_moves_capped(MOVE_CAP)
    }

    pub fn legal_moves_capped(&self, cap: usize) -> Vec<Move> {
        let mut seen: HashSet<u64> = HashSet::new();
        let mut out: Vec<Move> = Vec::new();
        let mut path: Vec<u8> = Vec::with_capacity(CELLS);
        for start in 0..CELLS {
            if out.len() >= cap {
                break;
            }
            let mut budget = DFS_BUDGET_PER_START;
            path.clear();
            path.push(start as u8);
            self.dfs(
                start,
                self.cells[start],
                1u32 << start,
                &mut path,
                &mut seen,
                &mut out,
                cap,
                &mut budget,
            );
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn dfs(
        &self,
        last: usize,
        v: u64,
        mask: u32,
        path: &mut Vec<u8>,
        seen: &mut HashSet<u64>,
        out: &mut Vec<Move>,
        cap: usize,
        budget: &mut usize,
    ) {
        if out.len() >= cap || *budget == 0 {
            return;
        }
        *budget -= 1;
        if path.len() >= 2 {
            let key = ((mask as u64) << 8) | last as u64;
            if seen.insert(key) {
                out.push(Move { path: path.clone() });
            }
        }
        let mut nbs = [0usize; 4];
        let mut nn = 0;
        neighbors(last, |nb| {
            if self.cells[nb] == v && mask & (1 << nb) == 0 {
                nbs[nn] = nb;
                nn += 1;
            }
        });
        for &nb in &nbs[..nn] {
            path.push(nb as u8);
            self.dfs(nb, v, mask | (1 << nb), path, seen, out, cap, budget);
            path.pop();
        }
    }

    /// Apply a move using the board's own RNG for refills. Refill draws happen
    /// in path order (excluding the head), matching the web build exactly.
    pub fn apply(&mut self, mv: &Move) {
        debug_assert!(mv.path.len() >= 2);
        let head = mv.head();
        let v = self.cells[mv.path[0] as usize];
        let sum = v * mv.path.len() as u64;
        for &ci in &mv.path[..mv.path.len() - 1] {
            self.cells[ci as usize] = self.rng.rnd13();
        }
        self.cells[head] = sum;
        self.score += sum;
        self.moves_made += 1;
    }

    /// Deterministic child for search: refills are supplied (one per non-head
    /// path cell, in path order) instead of drawn from the RNG.
    pub fn apply_with_refills(&self, mv: &Move, refills: &[u64]) -> Board {
        debug_assert_eq!(refills.len(), mv.path.len() - 1);
        let mut b = self.clone();
        let head = mv.head();
        let v = b.cells[mv.path[0] as usize];
        let sum = v * mv.path.len() as u64;
        for (k, &ci) in mv.path[..mv.path.len() - 1].iter().enumerate() {
            b.cells[ci as usize] = refills[k];
        }
        b.cells[head] = sum;
        b.score += sum;
        b.moves_made += 1;
        b
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "score {}  moves {}", self.score, self.moves_made)?;
        for r in 0..N {
            for c in 0..N {
                write!(f, "{:>5}", self.cells[idx(r, c)])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    /// Reference vector generated by the JS mulberry32 in the web build,
    /// seed 123456789.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn mulberry32_matches_js() {
        let mut rng = Mulberry32::new(123_456_789);
        let expected = [
            0.257_790_743_838_995_70,
            0.970_772_111_555_561_42,
            0.785_328_014_288_097_62,
            0.206_164_579_838_514_33,
            0.303_071_887_465_193_87,
            0.747_066_047_042_608_26,
            0.778_733_652_085_065_84,
            0.284_509_629_011_154_17,
        ];
        for &e in &expected {
            assert!((rng.next_f64() - e).abs() < 1e-15);
        }
        let mut rng = Mulberry32::new(123_456_789);
        let expected_13: [u64; 8] = [1, 3, 3, 1, 1, 3, 3, 1];
        for &e in &expected_13 {
            assert_eq!(rng.rnd13(), e);
        }
    }

    #[test]
    fn new_game_is_deterministic_and_playable() {
        let a = Board::new_game(42);
        let b = Board::new_game(42);
        assert_eq!(a.cells, b.cells);
        assert!(a.has_moves());
        assert!(a.cells.iter().all(|&v| (1..=3).contains(&v)));
    }

    #[test]
    fn movegen_single_pair() {
        // 1 2 1 2 1 / 2 1 2 1 2 / ... checkerboard has no moves; then plant one pair.
        let mut cells = [0u64; CELLS];
        for i in 0..CELLS {
            let (r, c) = rc(i);
            cells[i] = if (r + c) % 2 == 0 { 1 } else { 2 };
        }
        let mut b = Board::from_state(cells, 0, 1);
        assert!(b.legal_moves().is_empty());
        assert!(b.is_over());
        // Make cells 0 and 1 both value 7: exactly one pair, two moves (either head).
        b.cells[0] = 7;
        b.cells[1] = 7;
        let mv = b.legal_moves();
        assert_eq!(mv.len(), 2);
        assert!(mv.iter().all(|m| m.path.len() == 2));
        let heads: HashSet<usize> = mv.iter().map(|m| m.head()).collect();
        assert_eq!(heads, HashSet::from([0, 1]));
    }

    #[test]
    fn movegen_dedups_by_mask_and_head() {
        // A 2x2 block of equal tiles: cell sets {a,b} pairs (4 edges x 2 heads = 8),
        // L-triples, and the full square. All distinct (mask, head) outcomes.
        let mut cells = [0u64; CELLS];
        for i in 0..CELLS {
            cells[i] = if i % 2 == 0 {
                90 + i as u64
            } else {
                900 + i as u64
            };
        }
        for &i in &[0usize, 1, 5, 6] {
            cells[i] = 4;
        }
        let b = Board::from_state(cells, 0, 1);
        let moves = b.legal_moves();
        // pairs: edges (0,1),(0,5),(1,6),(5,6) x2 heads = 8
        // triples: 4 L-shapes x 2 heads each (path can start from either end) = 8
        // quads: 4 cells, every cell reachable as a path end = 4 (mask fixed)
        assert_eq!(moves.len(), 20);
        let unique: HashSet<(u32, usize)> = moves.iter().map(|m| (m.mask(), m.head())).collect();
        assert_eq!(unique.len(), moves.len());
    }

    /// "Stop" actions must exist: a chain can end anywhere, so every prefix of
    /// a longer chain (length >= 2) is its own move. An engine that only
    /// enumerates maximal chains is wrong.
    #[test]
    fn stop_actions_are_included() {
        let mut cells = [0u64; CELLS];
        for i in 0..CELLS {
            let (r, c) = rc(i);
            cells[i] = if (r + c) % 2 == 0 { 51 } else { 52 };
        }
        let (a, b_, c_) = (idx(2, 1), idx(2, 2), idx(2, 3));
        cells[a] = 5;
        cells[b_] = 5;
        cells[c_] = 5;
        let b = Board::from_state(cells, 0, 1);
        let moves = b.legal_moves();
        // pairs (a,b) and (b,c) with either head = 4 moves; the full triple can
        // only end at its two ends = 2 moves.
        assert_eq!(moves.len(), 6);
        assert_eq!(moves.iter().filter(|m| m.path.len() == 2).count(), 4);
        assert_eq!(moves.iter().filter(|m| m.path.len() == 3).count(), 2);
        // stopping short of the full chain in both directions is available:
        let has = |cells: &[usize], head: usize| {
            moves
                .iter()
                .any(|m| m.head() == head && m.mask() == cells.iter().fold(0, |k, &i| k | 1 << i))
        };
        assert!(has(&[a, b_], a) && has(&[a, b_], b_));
        assert!(has(&[b_, c_], b_) && has(&[b_, c_], c_));
    }

    /// Coalescing: different drag orders over the same cell set with the same
    /// head are one action. A 2x3 block has many Hamiltonian chains into each
    /// corner; the action list must contain exactly one full-block move per head.
    #[test]
    fn coalescing_merges_equivalent_chains() {
        let mut cells = [0u64; CELLS];
        for i in 0..CELLS {
            let (r, c) = rc(i);
            cells[i] = if (r + c) % 2 == 0 { 51 } else { 52 };
        }
        let block = [
            idx(1, 1),
            idx(1, 2),
            idx(1, 3),
            idx(2, 1),
            idx(2, 2),
            idx(2, 3),
        ];
        for &i in &block {
            cells[i] = 9;
        }
        let b = Board::from_state(cells, 0, 1);
        let moves = b.legal_moves();
        assert!(moves
            .iter()
            .all(|m| m.path.iter().all(|&i| b.cells[i as usize] == 9)));
        let keys: HashSet<(u32, usize)> = moves.iter().map(|m| (m.mask(), m.head())).collect();
        assert_eq!(keys.len(), moves.len(), "every action is unique");
        let full: u32 = block.iter().fold(0, |k, &i| k | 1 << i);
        let corner = idx(1, 1);
        let full_into_corner = moves
            .iter()
            .filter(|m| m.mask() == full && m.head() == corner)
            .count();
        assert_eq!(
            full_into_corner, 1,
            "many chains cover the block into one corner, but it is one action"
        );
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn apply_merges_to_head_and_refills_in_place() {
        let mut cells = [0u64; CELLS];
        for i in 0..CELLS {
            let (r, c) = rc(i);
            cells[i] = if (r + c) % 2 == 0 { 51 } else { 52 };
        }
        cells[idx(2, 1)] = 3;
        cells[idx(2, 2)] = 3;
        cells[idx(2, 3)] = 3;
        let mut b = Board::from_state(cells, 10, 7);
        let mv = Move {
            path: vec![idx(2, 1) as u8, idx(2, 2) as u8, idx(2, 3) as u8],
        };
        let before = b.cells;
        b.apply(&mv);
        assert_eq!(b.cells[idx(2, 3)], 9, "head gets the sum");
        assert!((1..=3).contains(&b.cells[idx(2, 1)]));
        assert!((1..=3).contains(&b.cells[idx(2, 2)]));
        assert_eq!(b.score, 19);
        for i in 0..CELLS {
            if ![idx(2, 1), idx(2, 2), idx(2, 3)].contains(&i) {
                assert_eq!(b.cells[i], before[i], "non-path cells untouched");
            }
        }
    }

    #[test]
    fn apply_with_refills_is_pure() {
        let b = Board::new_game(7);
        let moves = b.legal_moves();
        let mv = &moves[0];
        let refills: Vec<u64> = vec![2; mv.path.len() - 1];
        let c1 = b.apply_with_refills(mv, &refills);
        let c2 = b.apply_with_refills(mv, &refills);
        assert_eq!(c1.cells, c2.cells);
        assert_eq!(b.rng, c1.rng, "search children do not consume the game RNG");
    }
}
