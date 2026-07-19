//! Tile classification and tunable value functions.
//!
//! Domain knowledge encoded here:
//! - Tiles merge only with equal tiles, so a tile's future depends on its
//!   factorization. Fresh spawns are 1/2/3.
//! - The ideal ladder is exactly 2^k·3 (6, 12, 24, 48, ...). Powers of two of
//!   4 and above drift off it (mildly bad); multiples of 9 are worse; 27 and
//!   higher powers of three, plus any prime factor above 3, are dead weight.
//! - Optionality (available merges) keeps the game alive, but pairs differ in
//!   quality: what would the merge produce? (3+3 -> 6 is great, 2+2 -> 4 is not.)
//! - Geometry: big tiles belong on edges/corners, not the center.
//!
//! Every feature is a linear term with its own weight so the whole vector can
//! be fit by regression or tuned by tournament search later.

use crate::game::{idx, Board, CELLS, N};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileClass {
    /// 1, 2, or 3 — spawn material, always fine.
    Small,
    /// On the main ladder: 2^a * 3 exactly (6, 12, 24, 48, 96, ...).
    Smooth,
    /// One extra three: 2^a * 9 (9, 18, 36, 72, 144, ...).
    ThreeHeavy,
    /// 2^a * 27 or worse (27, 54, 81, 108, 216, ...).
    ThreeHi,
    /// Pure power of two >= 4 (4, 8, 16, 32, ...).
    Pow2,
    /// Has a prime factor > 3 (5, 7, 10, 14, 15, 20, ...).
    Off,
}

pub fn classify(v: u64) -> TileClass {
    if v <= 3 {
        return TileClass::Small;
    }
    let mut m = v;
    while m.is_multiple_of(2) {
        m /= 2;
    }
    let mut threes = 0;
    while m.is_multiple_of(3) {
        m /= 3;
        threes += 1;
    }
    if m > 1 {
        TileClass::Off
    } else if threes == 0 {
        TileClass::Pow2
    } else if threes == 1 {
        TileClass::Smooth
    } else if threes == 2 {
        TileClass::ThreeHeavy
    } else {
        TileClass::ThreeHi
    }
}

pub fn is_bad(c: TileClass) -> bool {
    matches!(
        c,
        TileClass::ThreeHeavy | TileClass::ThreeHi | TileClass::Pow2 | TileClass::Off
    )
}

/// Ring distance from the border: 0 for the 16 edge cells (corners included),
/// 1 for the middle ring, 2 for the center cell.
pub fn ring(i: usize) -> u32 {
    let r = i / N;
    let c = i % N;
    r.min(c).min(N - 1 - r).min(N - 1 - c) as u32
}

/// Tiles at or above this value count as "big" for the centrality penalty.
pub const BIG_THRESHOLD: u64 = 12;

/// Symmetry class of a cell on the 5x5 board (up to the 8-fold symmetry):
/// 0 corner, 1 edge next to a corner, 2 edge middle, 3 inner diagonal,
/// 4 inner edge, 5 center.
pub fn sym_class(i: usize) -> usize {
    let r = i / N;
    let c = i % N;
    let a = r.min(N - 1 - r);
    let b = c.min(N - 1 - c);
    match (a.min(b), a.max(b)) {
        (0, 0) => 0,
        (0, 1) => 1,
        (0, 2) => 2,
        (1, 1) => 3,
        (1, 2) => 4,
        _ => 5,
    }
}

/// Size tier for positional terms: 0 below 12, else floor(log2(v/3)) - 1,
/// so 12 -> 1, 24 -> 2, 48 -> 3, 96 -> 4, ..., 768 -> 7.
pub fn size_tier(v: u64) -> f64 {
    if v < BIG_THRESHOLD {
        0.0
    } else {
        ((v / 3) as f64).log2().floor() - 1.0
    }
}

/// Value-function weights. V is the sum of the score term, tile terms,
/// optionality terms, geometry terms, and composition terms, minus death
/// terms. Tile-term magnitudes are log2(v) (or v itself when `linear`).
/// w_score is the unit anchor: keep it at 1.0 and tune the rest in points.
#[derive(Clone, Copy, Debug)]
pub struct Weights {
    pub w_score: f64,
    pub w_smooth: f64,
    /// Penalty for 2^a·9 tiles.
    pub w_three: f64,
    /// Penalty for 2^a·27-and-worse tiles.
    pub w_three_hi: f64,
    pub w_pow2: f64,
    pub w_off: f64,
    pub w_opt: f64,
    pub w_over: f64,
    pub linear: bool,
    /// Apply sqrt to the effective pair count (diminishing returns).
    pub opt_sqrt: bool,
    /// Weight each pair by log2(2v) of its tile value instead of counting 1.
    pub opt_logv: bool,
    /// Credit for pairs whose merge result is a bad class (1.0 = same as good).
    pub w_badpair_frac: f64,
    /// Penalty per 2-tile sitting in a value-2 component of size <= 2.
    pub w_iso2: f64,
    /// Reward for chain potential: sum over components of max(0, size − 2).
    pub w_moves: f64,
    /// Penalty per ring step for big tiles (>= BIG_THRESHOLD) off the border.
    pub w_center_big: f64,
    /// Penalty per ring step for bad-class tiles off the border.
    pub w_center_bad: f64,
    /// Graded low-optionality penalty: w_danger / (effective pairs + 0.5).
    pub w_danger: f64,
    /// Penalty per distinct value on the board (fragmentation).
    pub w_frag: f64,
    /// Reward per spawn-material tile (1/2/3) on the board.
    pub w_small: f64,
    /// Per-value rewards for the individual small tiles (learnable).
    pub w_n1: f64,
    pub w_n2: f64,
    pub w_n3: f64,
    /// Pending-merge potential: for each adjacent equal pair whose merge
    /// result is good (Small/Smooth), reward w_vpair * 2v (score units).
    pub w_vpair: f64,
    /// Coalescing loss: big tiles (>= BIG_THRESHOLD) should share ONE corner.
    /// Loss = min over the 4 corners of sum(log2(v) * manhattan(tile, corner)).
    pub w_coal: f64,
    /// Immediately-achievable value: per same-value component, the max score
    /// extractable if all future refills were unmergeable stones —
    /// max(chain s·v, pair-tree B(s,v) = 2v·⌊s/2⌋ + B(⌊s/2⌋, 2v)).
    pub w_achv: f64,
    /// Positional penalty per size tier for big tiles, indexed by symmetry
    /// class 1..=5 (edge-near-corner, edge-mid, inner-diagonal, inner-edge,
    /// center). Corners are the anchor at zero.
    pub w_pos: [f64; 5],
    /// Quadrant coalescing: reward the best corner 2x2 block's sum of big
    /// tile size tiers — big tiles want to share one quadrant.
    pub w_quad: f64,
    /// Stranded-big penalty: for each big value with an odd count on the
    /// board, penalize by that value's size tier. A lone 48 is frozen real
    /// estate until a twin exists.
    pub w_stranded: f64,
    /// Twin-proximity reward: for each big value with two or more copies,
    /// reward tier * (4 - min pairwise Manhattan distance, floored at 0) —
    /// mergeable pairs should converge.
    pub w_twin: f64,
    /// Move-level penalty on long chains: search docks a move's EV by
    /// w_chainlen * (len - 3)^2. Big tiles have almost no mobility, so
    /// suddenly minting one via a long chain deserves scrutiny.
    pub w_chainlen: f64,
    /// Gather penalty: pairwise tier_i * tier_j * manhattan distance over ALL
    /// big tiles — the big-tile cluster should stay in one region.
    pub w_gather: f64,
    /// Staircase reward: an x tile orthogonally adjacent to a 2x tile, for
    /// x >= 6 (6|12, 12|24, ...), scaled by log2(x). When x doubles, the
    /// result lands next to its next merge partner.
    pub w_stair: f64,
    /// Reward for a value-2 component of size exactly 3: a loaded 6-maker.
    /// Counters the perverse incentive where the iso2 penalty is relieved by
    /// merging a 2-pair into a bad 4 instead of growing it to three.
    pub w_trip2: f64,
}

impl Weights {
    fn base() -> Self {
        Weights {
            w_score: 1.0,
            w_smooth: 0.0,
            w_three: 0.0,
            w_three_hi: 0.0,
            w_pow2: 0.0,
            w_off: 0.0,
            w_opt: 0.0,
            w_over: 0.0,
            linear: false,
            opt_sqrt: false,
            opt_logv: false,
            w_badpair_frac: 1.0,
            w_iso2: 0.0,
            w_moves: 0.0,
            w_center_big: 0.0,
            w_center_bad: 0.0,
            w_danger: 0.0,
            w_frag: 0.0,
            w_small: 0.0,
            w_n1: 0.0,
            w_n2: 0.0,
            w_n3: 0.0,
            w_vpair: 0.0,
            w_coal: 0.0,
            w_achv: 0.0,
            w_pos: [0.0; 5],
            w_quad: 0.0,
            w_stranded: 0.0,
            w_twin: 0.0,
            w_chainlen: 0.0,
            w_gather: 0.0,
            w_stair: 0.0,
            w_trip2: 0.0,
        }
    }

    /// Pure greedy-score: no shaping at all.
    pub fn score_only() -> Self {
        Self::base()
    }

    /// V1: the original user priors, log-scaled tile terms.
    pub fn v1() -> Self {
        Weights {
            w_smooth: 1.0,
            w_pow2: 3.0,
            w_off: 6.0,
            w_opt: 3.0,
            w_over: 300.0,
            ..Self::base()
        }
    }

    /// Pre-feature-expansion champion: 2^k·3 whitelist + sqrt optionality.
    pub fn v5t4() -> Self {
        Weights {
            w_smooth: 1.0,
            w_three: 4.0,
            w_three_hi: 4.0,
            w_pow2: 4.0,
            w_off: 8.0,
            w_opt: 40.0,
            w_over: 1000.0,
            opt_sqrt: true,
            ..Self::base()
        }
    }

    /// T2: tournament-tuned from the combined-feature base (2026-07-18 run
    /// 2, fitness 848.6 reduced budget; 871.7 +/- 22.8 full budget, fresh seeds).
    pub fn t2() -> Self {
        Weights {
            w_smooth: 1.0,
            w_three: 2.945,
            w_three_hi: 3.954,
            w_pow2: 3.894,
            w_off: 8.0,
            w_opt: 27.395,
            w_over: 1000.0,
            opt_sqrt: true,
            w_badpair_frac: 0.774,
            w_iso2: 1.442,
            w_center_big: 1.974,
            w_n1: 1.090,
            w_n2: 0.390,
            w_n3: 0.338,
            ..Self::base()
        }
    }

    /// Parse "base" or "base+field=value+field=value" (e.g. "t1+moves=0.5").
    /// Field names: smooth, three, threehi, pow2, off, opt, over, badfrac,
    /// iso2, moves, cbig, cbad, danger, frag, small.
    /// T3: lean T2 plus quadrant coalescing (big tiles rewarded for sharing
    /// one corner 2x2 block). Champion as of 2026-07-19: 848 +/- 24 at
    /// d2/s8/k12 vs 768 for T2 on the same seeds. The positional class table
    /// (w_pos) helped alone (+60) but overlaps with quad+ring; it stays at
    /// zero pending a joint tune.
    pub fn t3() -> Self {
        Weights {
            w_quad: 2.0,
            ..Self::t2()
        }
    }

    /// T4: T3 plus the staircase reward (x adjacent to 2x for x >= 6).
    /// Champion as of 2026-07-19: 962.7 +/- 25.7 at d2/s12/k16 on fresh
    /// seeds vs 889.9 for T3.
    pub fn t4() -> Self {
        Weights {
            w_stair: 0.5,
            ..Self::t3()
        }
    }

    pub fn by_name(name: &str) -> Option<Self> {
        let mut parts = name.split('+');
        let mut w = Self::named(parts.next()?)?;
        for ov in parts {
            let (k, v) = ov.split_once('=')?;
            let x: f64 = v.parse().ok()?;
            match k {
                "smooth" => w.w_smooth = x,
                "three" => w.w_three = x,
                "threehi" => w.w_three_hi = x,
                "pow2" => w.w_pow2 = x,
                "off" => w.w_off = x,
                "opt" => w.w_opt = x,
                "optlogv" => w.opt_logv = x > 0.5,
                "over" => w.w_over = x,
                "badfrac" => w.w_badpair_frac = x,
                "iso2" => w.w_iso2 = x,
                "moves" => w.w_moves = x,
                "cbig" => w.w_center_big = x,
                "cbad" => w.w_center_bad = x,
                "danger" => w.w_danger = x,
                "frag" => w.w_frag = x,
                "small" => w.w_small = x,
                "n1" => w.w_n1 = x,
                "n2" => w.w_n2 = x,
                "n3" => w.w_n3 = x,
                "vpair" => w.w_vpair = x,
                "coal" => w.w_coal = x,
                "achv" => w.w_achv = x,
                "pos1" => w.w_pos[0] = x,
                "pos2" => w.w_pos[1] = x,
                "pos3" => w.w_pos[2] = x,
                "pos4" => w.w_pos[3] = x,
                "pos5" => w.w_pos[4] = x,
                "quad" => w.w_quad = x,
                "stranded" => w.w_stranded = x,
                "twin" => w.w_twin = x,
                "chainlen" => w.w_chainlen = x,
                "gather" => w.w_gather = x,
                "stair" => w.w_stair = x,
                "trip2" => w.w_trip2 = x,
                _ => return None,
            }
        }
        Some(w)
    }

    fn named(name: &str) -> Option<Self> {
        match name {
            "score" => Some(Self::score_only()),
            "v1" => Some(Self::v1()),
            "v5t4" => Some(Self::v5t4()),
            "t2" => Some(Self::t2()),
            "t3" => Some(Self::t3()),
            "t4" => Some(Self::t4()),
            _ => None,
        }
    }

    /// The tunable parameter vector (w_score stays fixed at 1.0 as the unit
    /// anchor; `linear`/`opt_sqrt` come from the base).
    pub fn tunable(&self) -> [f64; 33] {
        [
            self.w_smooth,
            self.w_three,
            self.w_three_hi,
            self.w_pow2,
            self.w_off,
            self.w_opt,
            self.w_over,
            self.w_badpair_frac,
            self.w_iso2,
            self.w_moves,
            self.w_center_big,
            self.w_center_bad,
            self.w_danger,
            self.w_frag,
            self.w_small,
            self.w_n1,
            self.w_n2,
            self.w_n3,
            self.w_vpair,
            self.w_coal,
            self.w_achv,
            self.w_pos[0],
            self.w_pos[1],
            self.w_pos[2],
            self.w_pos[3],
            self.w_pos[4],
            self.w_quad,
            self.w_stranded,
            self.w_twin,
            self.w_chainlen,
            self.w_gather,
            self.w_stair,
            self.w_trip2,
        ]
    }

    pub const TUNABLE_NAMES: [&'static str; 33] = [
        "w_smooth",
        "w_three",
        "w_three_hi",
        "w_pow2",
        "w_off",
        "w_opt",
        "w_over",
        "w_badpair_frac",
        "w_iso2",
        "w_moves",
        "w_center_big",
        "w_center_bad",
        "w_danger",
        "w_frag",
        "w_small",
        "w_n1",
        "w_n2",
        "w_n3",
        "w_vpair",
        "w_coal",
        "w_achv",
        "w_pos_edge_near",
        "w_pos_edge_mid",
        "w_pos_inner_diag",
        "w_pos_inner_edge",
        "w_pos_center",
        "w_quad",
        "w_stranded",
        "w_twin",
        "w_chainlen",
        "w_gather",
        "w_stair",
        "w_trip2",
    ];

    pub fn with_tunable(&self, t: &[f64; 33]) -> Self {
        let mut w = *self;
        w.w_smooth = t[0].max(0.0);
        w.w_three = t[1].max(0.0);
        w.w_three_hi = t[2].max(0.0);
        w.w_pow2 = t[3].max(0.0);
        w.w_off = t[4].max(0.0);
        w.w_opt = t[5].max(0.0);
        w.w_over = t[6].max(0.0);
        w.w_badpair_frac = t[7].clamp(0.0, 1.0);
        w.w_iso2 = t[8].max(0.0);
        w.w_moves = t[9].max(0.0);
        w.w_center_big = t[10].max(0.0);
        w.w_center_bad = t[11].max(0.0);
        w.w_danger = t[12].max(0.0);
        w.w_frag = t[13].max(0.0);
        w.w_small = t[14].max(0.0);
        w.w_n1 = t[15].max(0.0);
        w.w_n2 = t[16].max(0.0);
        w.w_n3 = t[17].max(0.0);
        w.w_vpair = t[18].max(0.0);
        w.w_coal = t[19].max(0.0);
        w.w_achv = t[20].max(0.0);
        for k in 0..5 {
            w.w_pos[k] = t[21 + k].max(0.0);
        }
        w.w_quad = t[26].max(0.0);
        w.w_stranded = t[27].max(0.0);
        w.w_twin = t[28].max(0.0);
        w.w_chainlen = t[29].max(0.0);
        w.w_gather = t[30].max(0.0);
        w.w_stair = t[31].max(0.0);
        w.w_trip2 = t[32].max(0.0);
        w
    }
}

fn size(v: u64, linear: bool) -> f64 {
    if linear {
        v as f64
    } else {
        (v as f64).log2()
    }
}

fn uf_find(p: &mut [u8; CELLS], mut i: usize) -> usize {
    while p[i] as usize != i {
        p[i] = p[p[i] as usize];
        i = p[i] as usize;
    }
    i
}

#[allow(clippy::needless_range_loop)]
pub fn evaluate(b: &Board, w: &Weights) -> f64 {
    let mut val = w.w_score * b.score as f64;

    let mut small_count = 0u32;
    for i in 0..CELLS {
        let v = b.cells[i];
        let cls = classify(v);
        let sz = size(v, w.linear);
        match cls {
            TileClass::Small => {
                small_count += 1;
                val += match v {
                    1 => w.w_n1,
                    2 => w.w_n2,
                    _ => w.w_n3,
                };
            }
            TileClass::Smooth => val += w.w_smooth * sz,
            TileClass::ThreeHeavy => val -= w.w_three * sz,
            TileClass::ThreeHi => val -= w.w_three_hi * sz,
            TileClass::Pow2 => val -= w.w_pow2 * sz,
            TileClass::Off => val -= w.w_off * sz,
        }
        let rg = ring(i);
        if rg > 0 {
            if v >= BIG_THRESHOLD {
                val -= w.w_center_big * rg as f64 * sz;
            }
            if is_bad(cls) {
                val -= w.w_center_bad * rg as f64 * sz;
            }
        }
        let sc = sym_class(i);
        if sc > 0 {
            let tier = size_tier(v);
            if tier > 0.0 {
                val -= w.w_pos[sc - 1] * tier;
            }
        }
    }
    if w.w_stair != 0.0 {
        for r in 0..N {
            for c in 0..N {
                let a = b.cells[idx(r, c)];
                let mut check = |x: u64, y: u64| {
                    let (lo, hi) = (x.min(y), x.max(y));
                    if lo >= 6 && hi == 2 * lo {
                        val += w.w_stair * (lo as f64).log2();
                    }
                };
                if c + 1 < N {
                    check(a, b.cells[idx(r, c + 1)]);
                }
                if r + 1 < N {
                    check(a, b.cells[idx(r + 1, c)]);
                }
            }
        }
    }
    if w.w_stranded != 0.0 || w.w_twin != 0.0 || w.w_gather != 0.0 {
        let mut bigs: Vec<(u64, usize)> = Vec::new();
        for i in 0..CELLS {
            if b.cells[i] >= BIG_THRESHOLD {
                bigs.push((b.cells[i], i));
            }
        }
        bigs.sort_unstable();
        let mut g = 0;
        while g < bigs.len() {
            let v = bigs[g].0;
            let mut h = g;
            while h < bigs.len() && bigs[h].0 == v {
                h += 1;
            }
            let count = h - g;
            let tier = size_tier(v);
            if w.w_stranded != 0.0 && count % 2 == 1 {
                val -= w.w_stranded * tier;
            }
            if w.w_twin != 0.0 && count >= 2 {
                let mut min_d = usize::MAX;
                for a in g..h {
                    for bb in (a + 1)..h {
                        let (r1, c1) = (bigs[a].1 / N, bigs[a].1 % N);
                        let (r2, c2) = (bigs[bb].1 / N, bigs[bb].1 % N);
                        min_d = min_d.min(r1.abs_diff(r2) + c1.abs_diff(c2));
                    }
                }
                val += w.w_twin * tier * (4.0 - (min_d.min(4)) as f64);
            }
            g = h;
        }
        if w.w_gather != 0.0 && bigs.len() >= 2 {
            for a in 0..bigs.len() {
                for bb in (a + 1)..bigs.len() {
                    let (r1, c1) = (bigs[a].1 / N, bigs[a].1 % N);
                    let (r2, c2) = (bigs[bb].1 / N, bigs[bb].1 % N);
                    let d = (r1.abs_diff(r2) + c1.abs_diff(c2)) as f64;
                    val -= w.w_gather * size_tier(bigs[a].0) * size_tier(bigs[bb].0) * d;
                }
            }
        }
    }
    if w.w_quad != 0.0 {
        let mut best = 0.0f64;
        for &(r0, c0) in &[(0usize, 0usize), (0, 3), (3, 0), (3, 3)] {
            let mut sum = 0.0;
            for dr in 0..2 {
                for dc in 0..2 {
                    sum += size_tier(b.cells[idx(r0 + dr, c0 + dc)]);
                }
            }
            best = best.max(sum);
        }
        val += w.w_quad * best;
    }
    val += w.w_small * small_count as f64;

    // Pairs, split by the quality of what merging the pair would produce.
    // With opt_logv, each pair contributes log2(2v) instead of 1.
    let mut good_pairs = 0u32;
    let mut bad_pairs = 0u32;
    let mut good_mass = 0.0f64;
    let mut bad_mass = 0.0f64;
    let mut vpair_pot = 0.0f64;
    {
        let mut tally = |v: u64| {
            let unit = if w.opt_logv {
                ((2 * v) as f64).log2()
            } else {
                1.0
            };
            match classify(2 * v) {
                TileClass::Small | TileClass::Smooth => {
                    good_pairs += 1;
                    good_mass += unit;
                    vpair_pot += (2 * v) as f64;
                }
                _ => {
                    bad_pairs += 1;
                    bad_mass += unit;
                }
            }
        };
        for r in 0..N {
            for c in 0..N {
                let v = b.cells[idx(r, c)];
                if c + 1 < N && b.cells[idx(r, c + 1)] == v {
                    tally(v);
                }
                if r + 1 < N && b.cells[idx(r + 1, c)] == v {
                    tally(v);
                }
            }
        }
    }
    val += w.w_vpair * vpair_pot;
    let pairs_total = good_pairs + bad_pairs;
    let pairs_eff = good_mass + w.w_badpair_frac * bad_mass;
    let opt_term = if w.opt_sqrt {
        pairs_eff.max(0.0).sqrt()
    } else {
        pairs_eff
    };
    val += w.w_opt * opt_term;
    if pairs_total == 0 {
        val -= w.w_over;
    } else if w.w_danger > 0.0 {
        val -= w.w_danger / (pairs_eff + 0.5);
    }

    // Connected components of equal values (only when a component feature is on).
    if w.w_iso2 != 0.0 || w.w_moves != 0.0 || w.w_achv != 0.0 || w.w_trip2 != 0.0 {
        let mut parent: [u8; CELLS] = [0; CELLS];
        for (i, p) in parent.iter_mut().enumerate() {
            *p = i as u8;
        }
        for r in 0..N {
            for c in 0..N {
                let i = idx(r, c);
                let v = b.cells[i];
                if c + 1 < N && b.cells[i + 1] == v {
                    let (a, bb) = (uf_find(&mut parent, i), uf_find(&mut parent, i + 1));
                    parent[a] = bb as u8;
                }
                if r + 1 < N && b.cells[i + N] == v {
                    let (a, bb) = (uf_find(&mut parent, i), uf_find(&mut parent, i + N));
                    parent[a] = bb as u8;
                }
            }
        }
        let mut comp_size = [0u32; CELLS];
        for i in 0..CELLS {
            let root = uf_find(&mut parent, i);
            comp_size[root] += 1;
        }
        for i in 0..CELLS {
            if uf_find(&mut parent, i) == i {
                let s = comp_size[i];
                if w.w_moves != 0.0 && s >= 3 {
                    val += w.w_moves * (s - 2) as f64;
                }
                if w.w_iso2 != 0.0 && b.cells[i] == 2 && s <= 2 {
                    val -= w.w_iso2 * s as f64;
                }
                if w.w_trip2 != 0.0 && b.cells[i] == 2 && s == 3 {
                    val += w.w_trip2;
                }
                if w.w_achv != 0.0 && s >= 2 {
                    let v = b.cells[i];
                    let chain = s as u64 * v;
                    let mut tot = 0u64;
                    let (mut n, mut cur) = (s as u64, v);
                    while n >= 2 {
                        tot += 2 * cur * (n / 2);
                        n /= 2;
                        cur *= 2;
                    }
                    val += w.w_achv * chain.max(tot) as f64;
                }
            }
        }
    }

    if w.w_coal != 0.0 {
        let mut best = f64::INFINITY;
        let mut any = false;
        for &(cr, cc) in &[(0usize, 0usize), (0, N - 1), (N - 1, 0), (N - 1, N - 1)] {
            let mut cost = 0.0;
            for i in 0..CELLS {
                let v = b.cells[i];
                if v >= BIG_THRESHOLD {
                    any = true;
                    let (r, c) = (i / N, i % N);
                    let dist = r.abs_diff(cr) + c.abs_diff(cc);
                    cost += (v as f64).log2() * dist as f64;
                }
            }
            best = best.min(cost);
        }
        if any {
            val -= w.w_coal * best;
        }
    }

    if w.w_frag != 0.0 {
        let mut vals = b.cells;
        vals.sort_unstable();
        let distinct = 1 + vals.windows(2).filter(|p| p[0] != p[1]).count();
        val -= w.w_frag * distinct as f64;
    }

    val
}

/// Number of raw features emitted by `features()` for value calibration.
pub const N_FEATURES: usize = 30;

/// Human-readable names for the calibration features, index-aligned.
pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "smooth_mass",
    "three_mass",
    "threehi_mass",
    "pow2_mass",
    "off_mass",
    "good_pairs",
    "bad_pairs",
    "sqrt_pairs",
    "pos_corner",
    "pos_edge_near",
    "pos_edge_mid",
    "pos_inner_diag",
    "pos_inner_edge",
    "pos_center",
    "quad_best",
    "stair_sum",
    "iso2_tiles",
    "trip2_comps",
    "n1_count",
    "n2_count",
    "n3_count",
    "distinct_values",
    "chain_potential",
    "stranded_tiers",
    "twin_close",
    "achv_bound",
    "vpair_pot",
    "gather_spread",
    "big_count",
    "dead",
];

/// Raw, unweighted board features for value calibration: remaining-score
/// regression fits one coefficient per entry (plus a bias). Everything the
/// hand value function uses appears here unsigned, so the data decides signs
/// and magnitudes.
#[allow(clippy::needless_range_loop)]
pub fn features(b: &Board) -> [f64; N_FEATURES] {
    let mut f = [0.0f64; N_FEATURES];
    // tile class masses and per-value counts
    for i in 0..CELLS {
        let v = b.cells[i];
        let lg = (v as f64).log2();
        match classify(v) {
            TileClass::Small => match v {
                1 => f[18] += 1.0,
                2 => f[19] += 1.0,
                _ => f[20] += 1.0,
            },
            TileClass::Smooth => f[0] += lg,
            TileClass::ThreeHeavy => f[1] += lg,
            TileClass::ThreeHi => f[2] += lg,
            TileClass::Pow2 => f[3] += lg,
            TileClass::Off => f[4] += lg,
        }
        let tier = size_tier(v);
        if tier > 0.0 {
            f[8 + sym_class(i)] += tier;
            f[28] += 1.0;
        }
    }
    // pair counts and staircase adjacencies
    for r in 0..N {
        for c in 0..N {
            let a = b.cells[idx(r, c)];
            let mut pair = |x: u64, y: u64| {
                if x == y {
                    match classify(2 * x) {
                        TileClass::Small | TileClass::Smooth => {
                            f[5] += 1.0;
                            f[26] += (2 * x) as f64;
                        }
                        _ => f[6] += 1.0,
                    }
                }
                let (lo, hi) = (x.min(y), x.max(y));
                if lo >= 6 && hi == 2 * lo {
                    f[15] += (lo as f64).log2();
                }
            };
            if c + 1 < N {
                pair(a, b.cells[idx(r, c + 1)]);
            }
            if r + 1 < N {
                pair(a, b.cells[idx(r + 1, c)]);
            }
        }
    }
    f[7] = (f[5] + f[6]).sqrt();
    f[29] = if f[5] + f[6] == 0.0 { 1.0 } else { 0.0 };
    // components: iso2 / trip2 / chain potential / achievable bound
    {
        let mut parent: [u8; CELLS] = [0; CELLS];
        for (i, p) in parent.iter_mut().enumerate() {
            *p = i as u8;
        }
        for r in 0..N {
            for c in 0..N {
                let i = idx(r, c);
                let v = b.cells[i];
                if c + 1 < N && b.cells[i + 1] == v {
                    let (a, bb) = (uf_find(&mut parent, i), uf_find(&mut parent, i + 1));
                    parent[a] = bb as u8;
                }
                if r + 1 < N && b.cells[i + N] == v {
                    let (a, bb) = (uf_find(&mut parent, i), uf_find(&mut parent, i + N));
                    parent[a] = bb as u8;
                }
            }
        }
        let mut comp_size = [0u32; CELLS];
        for i in 0..CELLS {
            let root = uf_find(&mut parent, i);
            comp_size[root] += 1;
        }
        for i in 0..CELLS {
            if uf_find(&mut parent, i) == i {
                let sc = comp_size[i] as u64;
                if sc >= 3 {
                    f[22] += (sc - 2) as f64;
                }
                if b.cells[i] == 2 && sc <= 2 {
                    f[16] += sc as f64;
                }
                if b.cells[i] == 2 && sc == 3 {
                    f[17] += 1.0;
                }
                if sc >= 2 {
                    let v = b.cells[i];
                    let chain = sc * v;
                    let mut tot = 0u64;
                    let (mut nn, mut cur) = (sc, v);
                    while nn >= 2 {
                        tot += 2 * cur * (nn / 2);
                        nn /= 2;
                        cur *= 2;
                    }
                    f[25] += chain.max(tot) as f64;
                }
            }
        }
    }
    // distinct values
    {
        let mut vals = b.cells;
        vals.sort_unstable();
        f[21] = (1 + vals.windows(2).filter(|p| p[0] != p[1]).count()) as f64;
    }
    // big-tile geometry: quad, stranded, twin, gather
    {
        let mut bigs: Vec<(u64, usize)> = Vec::new();
        for i in 0..CELLS {
            if b.cells[i] >= BIG_THRESHOLD {
                bigs.push((b.cells[i], i));
            }
        }
        bigs.sort_unstable();
        let mut g = 0;
        while g < bigs.len() {
            let v = bigs[g].0;
            let mut h = g;
            while h < bigs.len() && bigs[h].0 == v {
                h += 1;
            }
            let count = h - g;
            let tier = size_tier(v);
            if count % 2 == 1 {
                f[23] += tier;
            }
            if count >= 2 {
                let mut min_d = usize::MAX;
                for a in g..h {
                    for bb in (a + 1)..h {
                        let (r1, c1) = (bigs[a].1 / N, bigs[a].1 % N);
                        let (r2, c2) = (bigs[bb].1 / N, bigs[bb].1 % N);
                        min_d = min_d.min(r1.abs_diff(r2) + c1.abs_diff(c2));
                    }
                }
                f[24] += tier * (4.0 - (min_d.min(4)) as f64);
            }
            g = h;
        }
        for a in 0..bigs.len() {
            for bb in (a + 1)..bigs.len() {
                let (r1, c1) = (bigs[a].1 / N, bigs[a].1 % N);
                let (r2, c2) = (bigs[bb].1 / N, bigs[bb].1 % N);
                let d = (r1.abs_diff(r2) + c1.abs_diff(c2)) as f64;
                f[27] += size_tier(bigs[a].0) * size_tier(bigs[bb].0) * d;
            }
        }
        let mut best = 0.0f64;
        for &(r0, c0) in &[(0usize, 0usize), (0, 3), (3, 0), (3, 3)] {
            let mut sum = 0.0;
            for dr in 0..2 {
                for dc in 0..2 {
                    sum += size_tier(b.cells[idx(r0 + dr, c0 + dc)]);
                }
            }
            best = best.max(sum);
        }
        f[14] = best;
    }
    f
}

/// Signed contribution of the classic terms (for the policy UI). New feature
/// terms are lumped into `extras` so the total stays exact.
#[derive(Clone, Copy, Debug, Default)]
pub struct EvalBreakdown {
    pub smooth: f64,
    pub three: f64,
    pub pow2: f64,
    pub off: f64,
    pub opt: f64,
    pub over: f64,
    pub pairs: u32,
    pub extras: f64,
    pub total: f64,
}

pub fn evaluate_breakdown(b: &Board, w: &Weights) -> EvalBreakdown {
    let mut bd = EvalBreakdown::default();
    for &v in &b.cells {
        match classify(v) {
            TileClass::Small => {}
            TileClass::Smooth => bd.smooth += w.w_smooth * size(v, w.linear),
            TileClass::ThreeHeavy => bd.three -= w.w_three * size(v, w.linear),
            TileClass::ThreeHi => bd.three -= w.w_three_hi * size(v, w.linear),
            TileClass::Pow2 => bd.pow2 -= w.w_pow2 * size(v, w.linear),
            TileClass::Off => bd.off -= w.w_off * size(v, w.linear),
        }
    }
    bd.pairs = b.adjacent_pairs();
    let opt_term = if w.opt_sqrt {
        (bd.pairs as f64).sqrt()
    } else {
        bd.pairs as f64
    };
    bd.opt = w.w_opt * opt_term;
    if bd.pairs == 0 {
        bd.over = -w.w_over;
    }
    let mut probe = b.clone();
    probe.score = 0;
    bd.total = evaluate(&probe, w);
    bd.extras = bd.total - (bd.smooth + bd.three + bd.pow2 + bd.off + bd.opt + bd.over);
    bd
}

/// Count of tiles in "bad" classes (diagnostic for eval reports).
pub fn bad_tiles(b: &Board) -> u32 {
    b.cells.iter().filter(|&&v| is_bad(classify(v))).count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification() {
        for v in [1, 2, 3] {
            assert_eq!(classify(v), TileClass::Small);
        }
        for v in [6, 12, 24, 48, 96, 192, 384] {
            assert_eq!(classify(v), TileClass::Smooth, "{v}");
        }
        for v in [9, 18, 36, 72, 144, 288] {
            assert_eq!(classify(v), TileClass::ThreeHeavy, "{v}");
        }
        for v in [27, 54, 81, 108, 162, 216, 243] {
            assert_eq!(classify(v), TileClass::ThreeHi, "{v}");
        }
        for v in [4, 8, 16, 32, 64, 128] {
            assert_eq!(classify(v), TileClass::Pow2, "{v}");
        }
        for v in [5, 7, 10, 14, 15, 20, 21, 25, 33, 35, 50] {
            assert_eq!(classify(v), TileClass::Off, "{v}");
        }
    }

    #[test]
    fn ring_geometry() {
        assert_eq!(ring(0), 0); // corner
        assert_eq!(ring(2), 0); // mid-edge
        assert_eq!(ring(6), 1); // middle ring
        assert_eq!(ring(12), 2); // center
    }

    #[test]
    fn v5t4_unchanged_by_new_features() {
        // The pre-expansion champion must score exactly as before the feature
        // expansion: all new weights are zero and w_badpair_frac is 1.0, so
        // pair quality collapses to the raw pair count.
        let b = crate::game::Board::new_game(77);
        let w = Weights::v5t4();
        assert_eq!(w.w_badpair_frac, 1.0);
        let expected = w.w_score * b.score as f64 + w.w_opt * (b.adjacent_pairs() as f64).sqrt();
        assert!((evaluate(&b, &w) - expected).abs() < 1e-9);
    }

    #[test]
    fn breakdown_total_matches_evaluate() {
        let b = crate::game::Board::new_game(99);
        for name in ["v5t4", "t2"] {
            let w = Weights::by_name(name).unwrap();
            let bd = evaluate_breakdown(&b, &w);
            let mut probe = b.clone();
            probe.score = 0;
            assert!((bd.total - evaluate(&probe, &w)).abs() < 1e-9, "{name}");
        }
    }

    #[test]
    fn tunable_roundtrip() {
        let w = Weights::t2();
        let t = w.tunable();
        let w2 = w.with_tunable(&t);
        assert_eq!(w.tunable(), w2.tunable());
    }
}
