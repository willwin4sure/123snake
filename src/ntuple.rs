//! N-tuple network value function with afterstate TD(0) + TC learning,
//! following the 2048 lineage (Szubert & Jaskowski 2014, Jaskowski 2017).
//!
//! The afterstate of a move is the board with the merged sum placed on the
//! head cell and every other path cell marked pending (refill undrawn). The
//! tables learn the expectation over refills implicitly, so training never
//! enumerates the 3^(k-1) refill outcomes.

use crate::game::{idx, rc, Board, Move, Mulberry32, CELLS, MOVE_CAP, N};

/// Base cell alphabet: 0..=3 exact 1/2/3/4, 4..=18 ladder tiers 6*2^0..6*2^14
/// (saturating), 19 pow2 >=8, 20 nine-family 9*2^k, 21 other trash,
/// 22 pending refill.
pub const ALPHABET: usize = 23;
pub const PENDING: u8 = 22;

pub fn encode_cell(v: u64) -> u8 {
    match v {
        1 => 0,
        2 => 1,
        3 => 2,
        4 => 3,
        _ => {
            if v >= 6 && v.is_multiple_of(3) && (v / 3).is_power_of_two() {
                let k = (v / 3).trailing_zeros() - 1;
                return 4 + (k.min(14) as u8);
            }
            if v.is_power_of_two() {
                return 19;
            }
            if v.is_multiple_of(9) && (v / 9).is_power_of_two() {
                return 20;
            }
            21
        }
    }
}

/// Fine alphabet (31 symbols): looser priors — pow2 gets five size tiers
/// (8/16/32/64/128+ -> 19..=23), the nine-family four (9/18/36/72+ ->
/// 24..=27), trash splits by size (<24 -> 28, else 29), pending 30.
pub fn encode_cell_fine(v: u64) -> u8 {
    match v {
        1 => 0,
        2 => 1,
        3 => 2,
        4 => 3,
        _ => {
            if v >= 6 && v.is_multiple_of(3) && (v / 3).is_power_of_two() {
                let k = (v / 3).trailing_zeros() - 1;
                return 4 + (k.min(14) as u8);
            }
            if v.is_power_of_two() {
                return match v {
                    8 => 19,
                    16 => 20,
                    32 => 21,
                    64 => 22,
                    _ => 23,
                };
            }
            if v.is_multiple_of(9) && (v / 9).is_power_of_two() {
                return match v {
                    9 => 24,
                    18 => 25,
                    36 => 26,
                    _ => 27,
                };
            }
            if v < 24 {
                28
            } else {
                29
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Alphabet {
    Base,
    Fine,
}

impl Alphabet {
    pub fn size(self) -> usize {
        match self {
            Alphabet::Base => 23,
            Alphabet::Fine => 31,
        }
    }
    pub fn pending(self) -> u8 {
        (self.size() - 1) as u8
    }
    pub fn encode(self, v: u64) -> u8 {
        match self {
            Alphabet::Base => encode_cell(v),
            Alphabet::Fine => encode_cell_fine(v),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NetConfig {
    pub alphabet: Alphabet,
    pub with_2x3: bool,
    /// Position-dependent 2x3 tables (one per anchor orbit) instead of a
    /// single translation-shared table. ~5.4GB at the base alphabet.
    pub pos_2x3: bool,
    /// 5-cell diagonal staircase tuples (the 2048 papers' snake shapes).
    pub staircase: bool,
    /// The 10 wraparound diagonals (5 per direction, 3 orbit reps): not
    /// chains, but they capture diagonal value arrangement and partition the
    /// board into fixed 5-cell sets.
    pub diagonals: bool,
    /// Global long-range interaction tables that ordinary tuples cannot see:
    /// (a) symmetry-canonical positions of the two largest tiles (625
    /// entries), (b) big-tile dispersion bucket x max-tile tier (160).
    pub global: bool,
}

impl NetConfig {
    pub fn base() -> Self {
        NetConfig {
            alphabet: Alphabet::Base,
            with_2x3: true,
            pos_2x3: false,
            staircase: false,
            diagonals: false,
            global: false,
        }
    }
}

pub fn encode_board(cells: &[u64; CELLS]) -> [u8; CELLS] {
    let mut out = [0u8; CELLS];
    for (o, &v) in out.iter_mut().zip(cells.iter()) {
        *o = encode_cell(v);
    }
    out
}

/// One weight table with temporal-coherence accumulators.
struct Lut {
    w: Vec<f32>,
    e: Vec<f32>,
    a: Vec<f32>,
}

impl Lut {
    fn new(len: usize) -> Self {
        Lut {
            w: vec![0.0; len],
            e: vec![0.0; len],
            a: vec![0.0; len],
        }
    }
}

/// A symmetric image of a base tuple: fixed cell list into a shared table.
struct Image {
    cells: Vec<u8>,
    table: usize,
}

/// The 8 dihedral transforms of the 5x5 board.
fn dihedral(r: usize, c: usize, t: usize) -> (usize, usize) {
    let (mut r, mut c) = (r, c);
    if t & 4 != 0 {
        std::mem::swap(&mut r, &mut c);
    }
    if t & 1 != 0 {
        c = N - 1 - c;
    }
    if t & 2 != 0 {
        r = N - 1 - r;
    }
    (r, c)
}

const SAVE_MAGIC_V2: u32 = 0x4E54_5632; // "NTV2"
const SAVE_MAGIC_V3: u32 = 0x4E54_5633; // "NTV3": adds the diagonals flag
const SAVE_MAGIC: u32 = 0x4E54_5634; // "NTV4": adds the global flag

/// Approximate magnitude of a cell code, for ranking "largest tiles" in the
/// global features. Exact on spawns and the ladder (which dominates big
/// tiles); bucketed families get representative values. Pending refills
/// rank lowest.
fn code_mag(c: u8, a: Alphabet) -> u32 {
    let pow2_rep = |c: u8, base: u8| 8u32 << (c - base); // 8,16,32,64,128
    match a {
        Alphabet::Base => match c {
            0..=3 => c as u32 + 1,
            4..=18 => 6u32 << (c - 4),
            19 => 16,
            20 => 36,
            21 => 30,
            _ => 0,
        },
        Alphabet::Fine => match c {
            0..=3 => c as u32 + 1,
            4..=18 => 6u32 << (c - 4),
            19..=23 => pow2_rep(c, 19),
            24..=27 => 9u32 << (c - 24),
            28 => 10,
            29 => 40,
            _ => 0,
        },
    }
}

pub struct NTupleNet {
    tables: Vec<Lut>,
    images: Vec<Image>,
    pub alpha: f32,
    pub cfg: NetConfig,
    m: usize,
    /// Number of trailing tables that are global features, not tuple images.
    n_globals: usize,
    /// Dihedral index permutations: perms[t][i] = image of cell i.
    perms: Vec<[u8; CELLS]>,
}

impl NTupleNet {
    pub fn new(alpha: f32, cfg: NetConfig) -> Self {
        fn add_base(images: &mut Vec<Image>, cells: &[(usize, usize)], table: usize) {
            for t in 0..8 {
                let img: Vec<u8> = cells
                    .iter()
                    .map(|&(r, c)| {
                        let (rr, cc) = dihedral(r, c, t);
                        idx(rr, cc) as u8
                    })
                    .collect();
                images.push(Image { cells: img, table });
            }
        }
        let m = cfg.alphabet.size();
        let mut tables = Vec::new();
        let mut images = Vec::new();
        let len5 = m.pow(5);
        let len4 = m.pow(4);
        // rows 0..3 (reflections cover rows 3,4 and all columns)
        for r in 0..3 {
            let cells: Vec<_> = (0..N).map(|c| (r, c)).collect();
            tables.push(Lut::new(len5));
            add_base(&mut images, &cells, tables.len() - 1);
        }
        // 2x2 squares, anchor orbit reps
        for &(r, c) in &[(0, 0), (0, 1), (1, 1)] {
            let cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)];
            tables.push(Lut::new(len4));
            add_base(&mut images, &cells, tables.len() - 1);
        }
        // plus shapes, center orbit reps
        for &(r, c) in &[(1, 1), (1, 2), (2, 2)] {
            let cells = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)];
            tables.push(Lut::new(len5));
            add_base(&mut images, &cells, tables.len() - 1);
        }
        // wraparound diagonals D_k = {(r, (r+k) mod 5)}: orbit reps k=0,1,2
        // (reflections/transposes cover the other offsets and the whole
        // anti-diagonal family)
        if cfg.diagonals {
            for k in 0..3 {
                let cells: Vec<_> = (0..N).map(|r| (r, (r + k) % N)).collect();
                tables.push(Lut::new(len5));
                add_base(&mut images, &cells, tables.len() - 1);
            }
        }
        // 5-cell diagonal staircases, anchor orbit reps (positional)
        if cfg.staircase {
            for &(r, c) in &[(0, 0), (0, 1), (0, 2)] {
                let cells = [
                    (r, c),
                    (r + 1, c),
                    (r + 1, c + 1),
                    (r + 2, c + 1),
                    (r + 2, c + 2),
                ];
                tables.push(Lut::new(len5));
                add_base(&mut images, &cells, tables.len() - 1);
            }
        }
        // 2x3 blocks: positional per-orbit tables, or one shared
        // translation-invariant table
        if cfg.with_2x3 {
            let shared = if cfg.pos_2x3 {
                None
            } else {
                tables.push(Lut::new(m.pow(6)));
                Some(tables.len() - 1)
            };
            for &(r, c) in &[(0, 0), (0, 1), (1, 1)] {
                let cells = [
                    (r, c),
                    (r, c + 1),
                    (r, c + 2),
                    (r + 1, c),
                    (r + 1, c + 1),
                    (r + 1, c + 2),
                ];
                let table = match shared {
                    Some(t) => t,
                    None => {
                        tables.push(Lut::new(m.pow(6)));
                        tables.len() - 1
                    }
                };
                add_base(&mut images, &cells, table);
            }
        }
        let mut n_globals = 0;
        if cfg.global {
            tables.push(Lut::new(CELLS * CELLS)); // top-2 positions, canonical
            tables.push(Lut::new(20 * 8)); // dispersion bucket x max tier
            n_globals = 2;
        }
        let perms: Vec<[u8; CELLS]> = (0..8)
            .map(|t| {
                let mut p = [0u8; CELLS];
                for r in 0..N {
                    for c in 0..N {
                        let (rr, cc) = dihedral(r, c, t);
                        p[idx(r, c)] = idx(rr, cc) as u8;
                    }
                }
                p
            })
            .collect();
        NTupleNet {
            tables,
            images,
            alpha,
            cfg,
            m,
            n_globals,
            perms,
        }
    }

    fn index(&self, img: &Image, codes: &[u8; CELLS]) -> usize {
        let mut i = 0usize;
        for &c in &img.cells {
            i = i * self.m + codes[c as usize] as usize;
        }
        i
    }

    /// Indices into the two global tables for this board encoding. Both are
    /// exactly dihedral-invariant: the top-2 pair is selected per transform
    /// (so tie-breaks agree), and dispersion sums over all big tiles.
    fn global_indices(&self, codes: &[u8; CELLS]) -> [usize; 2] {
        let a = self.cfg.alphabet;
        let mags: Vec<u32> = codes.iter().map(|&c| code_mag(c, a)).collect();
        // canonical ordered top-2 pair: run the same index-order tie-break on
        // every transformed board, take the minimal encoding
        let mut g1 = usize::MAX;
        for p in &self.perms {
            let mut inv = [0usize; CELLS];
            for (i, &pi) in p.iter().enumerate() {
                inv[pi as usize] = i;
            }
            let (mut p1, mut p2, mut m1, mut m2) = (0usize, 0usize, 0u32, 0u32);
            for (j, &src) in inv.iter().enumerate() {
                let mg = mags[src];
                if mg > m1 {
                    p2 = p1;
                    m2 = m1;
                    p1 = j;
                    m1 = mg;
                } else if mg > m2 {
                    p2 = j;
                    m2 = mg;
                }
            }
            g1 = g1.min(p1 * CELLS + p2);
        }
        // dispersion: pairwise Manhattan distance over ALL tiles >= 48
        let big: Vec<usize> = (0..CELLS).filter(|&i| mags[i] >= 48).collect();
        let mut disp = 0usize;
        for i in 0..big.len() {
            for j in i + 1..big.len() {
                let (r1, c1) = rc(big[i]);
                let (r2, c2) = rc(big[j]);
                disp += r1.abs_diff(r2) + c1.abs_diff(c2);
            }
        }
        let maxmag = *mags.iter().max().unwrap_or(&1);
        let tier = (32 - (maxmag.max(3) / 3).leading_zeros()).min(7) as usize;
        let g2 = disp.min(19) * 8 + tier;
        [g1, g2]
    }

    pub fn encode(&self, cells: &[u64; CELLS]) -> [u8; CELLS] {
        let mut out = [0u8; CELLS];
        for (o, &v) in out.iter_mut().zip(cells.iter()) {
            *o = self.cfg.alphabet.encode(v);
        }
        out
    }

    pub fn value(&self, codes: &[u8; CELLS]) -> f64 {
        let mut v = 0.0f64;
        for img in &self.images {
            v += self.tables[img.table].w[self.index(img, codes)] as f64;
        }
        if self.n_globals > 0 {
            let g = self.global_indices(codes);
            let base = self.tables.len() - self.n_globals;
            for (k, &gi) in g.iter().enumerate() {
                v += self.tables[base + k].w[gi] as f64;
            }
        }
        v
    }

    /// TC-TD update of V(codes) toward target.
    pub fn update(&mut self, codes: &[u8; CELLS], target: f64) {
        fn bump(t: &mut Lut, i: usize, per: f32, delta: f32) {
            let rate = if t.a[i] > 0.0 {
                (t.e[i].abs() / t.a[i]).min(1.0)
            } else {
                1.0
            };
            t.w[i] += rate * per;
            t.e[i] += delta;
            t.a[i] += delta.abs();
        }
        let delta = (target - self.value(codes)) as f32;
        let per = self.alpha * delta / (self.images.len() + self.n_globals) as f32;
        for k in 0..self.images.len() {
            let i = {
                let img = &self.images[k];
                self.index(img, codes)
            };
            bump(&mut self.tables[self.images[k].table], i, per, delta);
        }
        if self.n_globals > 0 {
            let g = self.global_indices(codes);
            let base = self.tables.len() - self.n_globals;
            for (k, &gi) in g.iter().enumerate() {
                bump(&mut self.tables[base + k], gi, per, delta);
            }
        }
    }

    /// Afterstate codes for a move from `codes` (board encoding): merged sum
    /// on the head cell, pending marks on the vacated cells.
    pub fn afterstate(&self, codes: &[u8; CELLS], mv: &Move, sum: u64) -> [u8; CELLS] {
        let mut a = *codes;
        for &ci in &mv.path[..mv.path.len() - 1] {
            a[ci as usize] = self.cfg.alphabet.pending();
        }
        a[mv.head()] = self.cfg.alphabet.encode(sum);
        a
    }

    pub fn params(&self) -> usize {
        self.tables.iter().map(|t| t.w.len()).sum()
    }

    pub fn n_images(&self) -> usize {
        self.images.len()
    }

    pub fn n_tables(&self) -> usize {
        self.tables.len()
    }

    pub fn nonzero(&self) -> usize {
        self.tables
            .iter()
            .map(|t| t.a.iter().filter(|&&x| x > 0.0).count())
            .sum()
    }

    /// Weights-only snapshot: v2 header (magic, config flags), then a
    /// little-endian f32 stream per table.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let f = std::fs::File::create(path)?;
        let mut wr = std::io::BufWriter::new(f);
        wr.write_all(&SAVE_MAGIC.to_le_bytes())?;
        let fine = u32::from(self.cfg.alphabet == Alphabet::Fine);
        wr.write_all(&fine.to_le_bytes())?;
        wr.write_all(&u32::from(self.cfg.with_2x3).to_le_bytes())?;
        wr.write_all(&u32::from(self.cfg.pos_2x3).to_le_bytes())?;
        wr.write_all(&u32::from(self.cfg.staircase).to_le_bytes())?;
        wr.write_all(&u32::from(self.cfg.diagonals).to_le_bytes())?;
        wr.write_all(&u32::from(self.cfg.global).to_le_bytes())?;
        wr.write_all(&(self.tables.len() as u32).to_le_bytes())?;
        for t in &self.tables {
            wr.write_all(&(t.w.len() as u64).to_le_bytes())?;
            for &x in &t.w {
                wr.write_all(&x.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn load(path: &str, alpha: f32) -> std::io::Result<Self> {
        use std::io::Read;
        let f = std::fs::File::open(path)?;
        let mut rd = std::io::BufReader::new(f);
        let mut b4 = [0u8; 4];
        let mut b8 = [0u8; 8];
        rd.read_exact(&mut b4)?;
        let first = u32::from_le_bytes(b4);
        let (cfg, ntab) = if first == SAVE_MAGIC || first == SAVE_MAGIC_V3 || first == SAVE_MAGIC_V2
        {
            let mut word = || -> std::io::Result<u32> {
                rd.read_exact(&mut b4)?;
                Ok(u32::from_le_bytes(b4))
            };
            let alphabet = if word()? != 0 {
                Alphabet::Fine
            } else {
                Alphabet::Base
            };
            let with_2x3 = word()? != 0;
            let pos_2x3 = word()? != 0;
            let staircase = word()? != 0;
            let diagonals = if first == SAVE_MAGIC_V2 {
                false
            } else {
                word()? != 0
            };
            let global = if first == SAVE_MAGIC {
                word()? != 0
            } else {
                false
            };
            let ntab = word()? as usize;
            (
                NetConfig {
                    alphabet,
                    with_2x3,
                    pos_2x3,
                    staircase,
                    diagonals,
                    global,
                },
                ntab,
            )
        } else {
            // legacy header: [ntab u32][with_2x3 u32]
            rd.read_exact(&mut b4)?;
            let with_2x3 = u32::from_le_bytes(b4) != 0;
            let mut cfg = NetConfig::base();
            cfg.with_2x3 = with_2x3;
            (cfg, first as usize)
        };
        let mut net = NTupleNet::new(alpha, cfg);
        assert_eq!(net.tables.len(), ntab, "table count mismatch");
        for t in &mut net.tables {
            rd.read_exact(&mut b8)?;
            assert_eq!(u64::from_le_bytes(b8) as usize, t.w.len());
            let mut buf = vec![0u8; t.w.len() * 4];
            rd.read_exact(&mut buf)?;
            for (i, ch) in buf.chunks_exact(4).enumerate() {
                t.w[i] = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
            }
        }
        Ok(net)
    }

    /// Greedy move: argmax over legal moves of reward + V(afterstate).
    /// Returns (move, reward, afterstate codes).
    pub fn greedy(&self, b: &Board) -> Option<(Move, f64, [u8; CELLS])> {
        let codes = self.encode(&b.cells);
        let mut best: Option<(Move, f64, [u8; CELLS], f64)> = None;
        for mv in b.legal_moves_capped(MOVE_CAP) {
            let v = b.cells[mv.path[0] as usize];
            let sum = v * mv.path.len() as u64;
            let after = self.afterstate(&codes, &mv, sum);
            let val = sum as f64 + self.value(&after);
            if best.as_ref().is_none_or(|(_, _, _, bv)| val > *bv) {
                best = Some((mv, sum as f64, after, val));
            }
        }
        best.map(|(mv, r, a, _)| (mv, r, a))
    }
}

/// One self-play game with TD(0) updates. Returns (score, moves).
pub fn train_game(net: &mut NTupleNet, seed: u32) -> (u64, u32) {
    let mut b = Board::new_game(seed);
    let mut prev: Option<[u8; CELLS]> = None;
    loop {
        match net.greedy(&b) {
            None => {
                if let Some(pa) = prev {
                    net.update(&pa, 0.0);
                }
                return (b.score, b.moves_made);
            }
            Some((mv, r, after)) => {
                if let Some(pa) = prev {
                    net.update(&pa, r + net.value(&after));
                }
                prev = Some(after);
                b.apply(&mv);
            }
        }
    }
}

/// Greedy-net policy for the eval harness.
pub struct NTuplePolicy {
    pub net: NTupleNet,
}

impl crate::search::Policy for NTuplePolicy {
    fn name(&self) -> String {
        "ntuple-greedy".to_string()
    }
    fn choose(&mut self, b: &Board) -> Option<Move> {
        self.net.greedy(b).map(|(mv, _, _)| mv)
    }
}

/// Depth-2 expectimax over the net: root moves are pruned to the top-k by
/// the one-ply proxy (r + V(afterstate)), then each survivor's chance node
/// is estimated with `samples` sampled refills, valued by the best one-ply
/// reply in the child.
pub struct NTupleSearchPolicy {
    pub net: NTupleNet,
    pub topk: usize,
    pub samples: u32,
    pub rng: Mulberry32,
}

impl NTupleSearchPolicy {
    pub fn new(net: NTupleNet, topk: usize, samples: u32, seed: u32) -> Self {
        NTupleSearchPolicy {
            net,
            topk,
            samples,
            rng: Mulberry32::new(seed),
        }
    }
}

impl crate::search::Policy for NTupleSearchPolicy {
    fn name(&self) -> String {
        format!("ntuple-exp:k{}:s{}", self.topk, self.samples)
    }
    fn choose(&mut self, b: &Board) -> Option<Move> {
        let codes = self.net.encode(&b.cells);
        let mut scored: Vec<(Move, f64, f64)> = b
            .legal_moves_capped(MOVE_CAP)
            .into_iter()
            .map(|mv| {
                let v = b.cells[mv.path[0] as usize];
                let sum = (v * mv.path.len() as u64) as f64;
                let proxy = sum
                    + self
                        .net
                        .value(&self.net.afterstate(&codes, &mv, sum as u64));
                (mv, sum, proxy)
            })
            .collect();
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.topk.max(1));
        let mut best: Option<(Move, f64)> = None;
        for (mv, sum, _) in scored {
            let mut acc = 0.0;
            for _ in 0..self.samples {
                let refills: Vec<u64> = (0..mv.path.len() - 1).map(|_| self.rng.rnd13()).collect();
                let child = b.apply_with_refills(&mv, &refills);
                let reply = self
                    .net
                    .greedy(&child)
                    .map(|(_, r, after)| r + self.net.value(&after))
                    .unwrap_or(0.0);
                acc += reply;
            }
            let val = sum + acc / self.samples as f64;
            if best.as_ref().is_none_or(|(_, bv)| val > *bv) {
                best = Some((mv, val));
            }
        }
        best.map(|(mv, _)| mv)
    }
}

/// Greedy scores over `n` fresh games (no learning).
pub fn eval_scores(net: &NTupleNet, seed0: u32, n: u32) -> Vec<u64> {
    (0..n)
        .map(|s| {
            let mut b = Board::new_game(seed0 + s);
            while let Some((mv, _, _)) = net.greedy(&b) {
                b.apply(&mv);
            }
            b.score
        })
        .collect()
}

/// Mean greedy score over `n` fresh games (no learning).
pub fn eval_greedy(net: &NTupleNet, seed0: u32, n: u32) -> f64 {
    let scores = eval_scores(net, seed0, n);
    scores.iter().sum::<u64>() as f64 / scores.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> NetConfig {
        let mut c = NetConfig::base();
        c.with_2x3 = false;
        c
    }

    #[test]
    fn cell_codes() {
        assert_eq!(encode_cell(1), 0);
        assert_eq!(encode_cell(4), 3);
        assert_eq!(encode_cell(6), 4);
        assert_eq!(encode_cell(12), 5);
        assert_eq!(encode_cell(98304), 18);
        assert_eq!(encode_cell(196_608), 18); // saturates
        assert_eq!(encode_cell(8), 19);
        assert_eq!(encode_cell(16), 19);
        assert_eq!(encode_cell(9), 20);
        assert_eq!(encode_cell(144), 20);
        assert_eq!(encode_cell(27), 21); // 9*3: not 9*2^k
        assert_eq!(encode_cell(5), 21);
    }

    #[test]
    fn fine_cell_codes() {
        assert_eq!(encode_cell_fine(6), 4); // ladder unchanged
        assert_eq!(encode_cell_fine(8), 19);
        assert_eq!(encode_cell_fine(16), 20);
        assert_eq!(encode_cell_fine(128), 23);
        assert_eq!(encode_cell_fine(256), 23);
        assert_eq!(encode_cell_fine(9), 24);
        assert_eq!(encode_cell_fine(72), 27);
        assert_eq!(encode_cell_fine(144), 27);
        assert_eq!(encode_cell_fine(5), 28); // small trash
        assert_eq!(encode_cell_fine(27), 29); // big trash
        assert_eq!(Alphabet::Fine.pending(), 30);
    }

    #[test]
    fn td_learns_toward_target() {
        let mut net = NTupleNet::new(1.0, small_cfg());
        let b = Board::new_game(7);
        let codes = net.encode(&b.cells);
        assert_eq!(net.value(&codes), 0.0);
        // coincident symmetric images can overshoot a single step; repeated
        // updates must still converge on the target
        for _ in 0..50 {
            net.update(&codes, 100.0);
        }
        let v = net.value(&codes);
        assert!((v - 100.0).abs() < 5.0, "v={v}");
    }

    #[test]
    fn symmetry_shared_value() {
        // a board and its transpose must have identical value
        let mut cfg = small_cfg();
        cfg.staircase = true;
        cfg.diagonals = true;
        cfg.global = true;
        let mut net = NTupleNet::new(1.0, cfg);
        let b = Board::new_game(3);
        let codes = net.encode(&b.cells);
        net.update(&codes, 50.0);
        let mut tcells = [0u64; CELLS];
        for r in 0..N {
            for c in 0..N {
                tcells[idx(c, r)] = b.cells[idx(r, c)];
            }
        }
        let tcodes = net.encode(&tcells);
        let (v, tv) = (net.value(&codes), net.value(&tcodes));
        assert!((v - tv).abs() < 1e-9, "{v} vs {tv}");
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = std::env::temp_dir().join("ntuple-test-roundtrip.bin");
        let path = dir.to_str().unwrap();
        let mut cfg = small_cfg();
        cfg.alphabet = Alphabet::Fine;
        cfg.staircase = true;
        let mut net = NTupleNet::new(1.0, cfg);
        let b = Board::new_game(11);
        let codes = net.encode(&b.cells);
        net.update(&codes, 77.0);
        net.save(path).unwrap();
        let loaded = NTupleNet::load(path, 1.0).unwrap();
        assert_eq!(loaded.cfg.alphabet, Alphabet::Fine);
        assert!(loaded.cfg.staircase);
        assert!((loaded.value(&codes) - net.value(&codes)).abs() < 1e-9);
        let _ = std::fs::remove_file(path);
    }
}
