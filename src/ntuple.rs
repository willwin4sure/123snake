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
    /// Gated global tables: top-2 pair and top-3 triple positions, active
    /// only when the tiles are genuinely big (mag >= 24); plus dispersion.
    pub global2: bool,
    /// Weight-set count for multi-stage learning (Jaskowski 2017): 1 or 3.
    /// With 3, the stage is keyed by the board's max tile (<96, <768, >=768)
    /// and every table is replicated per stage.
    pub stages: usize,
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
            global2: false,
            stages: 1,
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
const SAVE_MAGIC_V4: u32 = 0x4E54_5634; // "NTV4": adds the global flag
const SAVE_MAGIC_V5: u32 = 0x4E54_5635; // "NTV5": adds the stage count
const SAVE_MAGIC: u32 = 0x4E54_5636; // "NTV6": adds the global2 flag

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
    /// Number of trailing tables (per stage) that are global features.
    n_globals: usize,
    /// Tables per stage; stage s uses tables[s*ntab_stage..(s+1)*ntab_stage].
    ntab_stage: usize,
    /// Whether promotion enabled and which stages have been initialized.
    pub promote: bool,
    stage_ready: Vec<bool>,
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
        // 5-cell diagonal staircases: the 36 placements decompose into 6
        // orbits under dihedral-8 (computed exhaustively; two orbits are
        // self-symmetric), one positional table per orbit rep
        if cfg.staircase {
            let reps: [[(usize, usize); 5]; 6] = [
                [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
                [(0, 1), (0, 2), (1, 0), (1, 1), (2, 0)],
                [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)],
                [(0, 1), (1, 1), (1, 2), (2, 2), (2, 3)],
                [(0, 2), (1, 1), (1, 2), (2, 0), (2, 1)],
                [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)],
            ];
            for cells in &reps {
                tables.push(Lut::new(len5));
                add_base(&mut images, cells, tables.len() - 1);
            }
        }
        // 2x3 blocks: the 24 placements (both orientations) decompose into 4
        // orbits under dihedral-8 (two of size 8, two of size 4); positional
        // per-orbit tables, or one shared translation-invariant table
        if cfg.with_2x3 {
            let shared = if cfg.pos_2x3 {
                None
            } else {
                tables.push(Lut::new(m.pow(6)));
                Some(tables.len() - 1)
            };
            let reps: [[(usize, usize); 6]; 4] = [
                [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
                [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)],
                [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)],
                [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)],
            ];
            for cells in &reps {
                let table = match shared {
                    Some(t) => t,
                    None => {
                        tables.push(Lut::new(m.pow(6)));
                        tables.len() - 1
                    }
                };
                add_base(&mut images, cells, table);
            }
        }
        let mut n_globals = 0;
        if cfg.global {
            tables.push(Lut::new(CELLS * CELLS)); // top-2 positions, canonical
            tables.push(Lut::new(20 * 8)); // dispersion bucket x max tier
            n_globals += 2;
        }
        if cfg.global2 {
            tables.push(Lut::new(CELLS * CELLS + 1)); // gated top-2 (+inactive)
            tables.push(Lut::new(CELLS * CELLS * CELLS + 1)); // gated top-3
            tables.push(Lut::new(20 * 8)); // dispersion bucket x max tier
            n_globals += 3;
        }
        // multi-stage: replicate the whole per-stage table set
        let ntab_stage = tables.len();
        assert!(cfg.stages == 1 || cfg.stages == 3, "stages must be 1 or 3");
        for _ in 1..cfg.stages {
            for t in 0..ntab_stage {
                let len = tables[t].w.len();
                tables.push(Lut::new(len));
            }
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
        let mut stage_ready = vec![false; cfg.stages];
        stage_ready[0] = true;
        NTupleNet {
            tables,
            images,
            alpha,
            cfg,
            m,
            n_globals,
            ntab_stage,
            promote: false,
            stage_ready,
            perms,
        }
    }

    /// Stage of a board encoding: 0 below max tile 96, 1 below 768, else 2
    /// (always 0 for single-stage nets).
    fn stage_of(&self, codes: &[u8; CELLS]) -> usize {
        if self.cfg.stages == 1 {
            return 0;
        }
        let a = self.cfg.alphabet;
        let mx = codes.iter().map(|&c| code_mag(c, a)).max().unwrap_or(1);
        if mx < 96 {
            0
        } else if mx < 768 {
            1
        } else {
            2
        }
    }

    fn index(&self, img: &Image, codes: &[u8; CELLS]) -> usize {
        let mut i = 0usize;
        for &c in &img.cells {
            i = i * self.m + codes[c as usize] as usize;
        }
        i
    }

    /// Canonical top-k positions: for each dihedral transform, select the
    /// top-k cells by magnitude with index-order tie-breaks on the
    /// transformed board, encode base-25, take the minimum. Exactly
    /// dihedral-invariant.
    fn canonical_topk(&self, mags: &[u32], k: usize) -> usize {
        let mut best = usize::MAX;
        for p in &self.perms {
            let mut inv = [0usize; CELLS];
            for (i, &pi) in p.iter().enumerate() {
                inv[pi as usize] = i;
            }
            let mut top: [(u32, usize); 3] = [(0, 0); 3];
            for (j, &src) in inv.iter().enumerate() {
                let mg = mags[src];
                let mut m = (mg, j);
                for slot in top.iter_mut().take(k) {
                    if m.0 > slot.0 {
                        std::mem::swap(&mut m, slot);
                    }
                }
            }
            let mut enc = 0usize;
            for slot in top.iter().take(k) {
                enc = enc * CELLS + slot.1;
            }
            best = best.min(enc);
        }
        best
    }

    fn dispersion_index(mags: &[u32]) -> usize {
        // pairwise Manhattan distance over ALL tiles >= 48, x max tier
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
        disp.min(19) * 8 + tier
    }

    /// Indices into the active global tables, in table order.
    fn global_indices(&self, codes: &[u8; CELLS]) -> ([usize; 5], usize) {
        let a = self.cfg.alphabet;
        let mags: Vec<u32> = codes.iter().map(|&c| code_mag(c, a)).collect();
        let mut out = [0usize; 5];
        let mut n = 0;
        if self.cfg.global {
            out[n] = self.canonical_topk(&mags, 2);
            out[n + 1] = Self::dispersion_index(&mags);
            n += 2;
        }
        if self.cfg.global2 {
            // sizes of the top three tiles (unsorted scan is fine for gates)
            let mut sorted = mags.clone();
            sorted.sort_unstable_by(|x, y| y.cmp(x));
            out[n] = if sorted[1] >= 24 {
                self.canonical_topk(&mags, 2)
            } else {
                CELLS * CELLS // inactive
            };
            out[n + 1] = if sorted[2] >= 24 {
                self.canonical_topk(&mags, 3)
            } else {
                CELLS * CELLS * CELLS // inactive
            };
            out[n + 2] = Self::dispersion_index(&mags);
            n += 3;
        }
        (out, n)
    }

    pub fn encode(&self, cells: &[u64; CELLS]) -> [u8; CELLS] {
        let mut out = [0u8; CELLS];
        for (o, &v) in out.iter_mut().zip(cells.iter()) {
            *o = self.cfg.alphabet.encode(v);
        }
        out
    }

    pub fn value(&self, codes: &[u8; CELLS]) -> f64 {
        let off = self.stage_of(codes) * self.ntab_stage;
        let mut v = 0.0f64;
        for img in &self.images {
            v += self.tables[off + img.table].w[self.index(img, codes)] as f64;
        }
        if self.n_globals > 0 {
            let (g, n) = self.global_indices(codes);
            let base = off + self.ntab_stage - self.n_globals;
            for (k, &gi) in g.iter().take(n).enumerate() {
                v += self.tables[base + k].w[gi] as f64;
            }
        }
        v
    }

    /// TC-TD update of V(codes) toward target.
    pub fn update(&mut self, codes: &[u8; CELLS], target: f64) {
        let delta = (target - self.value(codes)) as f32;
        self.nudge(codes, delta);
    }

    /// Apply a raw TD error to V(codes) through the TC machinery. `update`
    /// is `nudge(target - V)`; TD(lambda) traces call this with decayed
    /// errors.
    pub fn nudge(&mut self, codes: &[u8; CELLS], delta: f32) {
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
        let stage = self.stage_of(codes);
        if self.promote {
            self.promote_stage(stage);
        }
        let off = stage * self.ntab_stage;
        let per = self.alpha * delta / (self.images.len() + self.n_globals) as f32;
        for k in 0..self.images.len() {
            let i = {
                let img = &self.images[k];
                self.index(img, codes)
            };
            bump(&mut self.tables[off + self.images[k].table], i, per, delta);
        }
        if self.n_globals > 0 {
            let (g, n) = self.global_indices(codes);
            let base = off + self.ntab_stage - self.n_globals;
            for (k, &gi) in g.iter().take(n).enumerate() {
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

    /// Optimistic initialization (Guei & Wu's OTD): fill every weight so an
    /// untouched board evaluates to `v0`. Visited entries correct quickly
    /// (TC starts them at full rate); unvisited ones stay attractive, which
    /// drives systematic exploration of untried moves. Training-time only -
    /// the optimism is baked into the weights, so checkpoints round-trip.
    pub fn init_optimistic(&mut self, v0: f32) {
        let per = v0 / (self.images.len() + self.n_globals) as f32;
        for t in &mut self.tables {
            t.w.fill(per);
        }
    }

    /// Multi-stage weight promotion (Jaskowski 2017): the first time training
    /// touches a stage, copy the previous stage's tables into it so it
    /// refines instead of relearning from zero.
    fn promote_stage(&mut self, stage: usize) {
        if stage == 0 || self.stage_ready[stage] {
            return;
        }
        self.promote_stage(stage - 1);
        for t in 0..self.ntab_stage {
            let (lo, hi) = self.tables.split_at_mut(stage * self.ntab_stage + t);
            let src = &lo[(stage - 1) * self.ntab_stage + t];
            let dst = &mut hi[0];
            dst.w.copy_from_slice(&src.w);
        }
        self.stage_ready[stage] = true;
    }

    /// (weight, |error| mass) rows of the ungated top-2 position table.
    pub fn global_pair_table(&self) -> Vec<(f32, f32)> {
        assert!(self.cfg.global, "net has no --global tables");
        let t = &self.tables[self.ntab_stage - self.n_globals];
        t.w.iter().zip(t.a.iter()).map(|(&w, &a)| (w, a)).collect()
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
        wr.write_all(&u32::from(self.cfg.global2).to_le_bytes())?;
        wr.write_all(&(self.cfg.stages as u32).to_le_bytes())?;
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
        let (cfg, ntab) = if first == SAVE_MAGIC
            || first == SAVE_MAGIC_V5
            || first == SAVE_MAGIC_V4
            || first == SAVE_MAGIC_V3
            || first == SAVE_MAGIC_V2
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
            let global = if first >= SAVE_MAGIC_V4 {
                word()? != 0
            } else {
                false
            };
            let global2 = if first >= SAVE_MAGIC {
                word()? != 0
            } else {
                false
            };
            let stages = if first >= SAVE_MAGIC_V5 {
                word()? as usize
            } else {
                1
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
                    global2,
                    stages,
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

/// Move selection for training: greedy, or with probability `eps_rank` a
/// uniform pick among ranks 2..=8, or with probability `eps_rand` a uniform
/// random legal move. Exploratory afterstates receive TD updates too, which
/// calibrates V on off-policy moves (the probe's rank bias). Returns the
/// chosen (move, reward, afterstate, explored?).
#[allow(clippy::type_complexity)]
fn choose_train(
    net: &NTupleNet,
    b: &Board,
    rng: &mut Mulberry32,
    eps_rank: f32,
    eps_rand: f32,
) -> Option<(Move, f64, [u8; CELLS], bool)> {
    let roll = rng.next_f64() as f32;
    if roll >= eps_rank + eps_rand {
        return net.greedy(b).map(|(mv, r, a)| (mv, r, a, false));
    }
    let codes = net.encode(&b.cells);
    let mut scored: Vec<(Move, u64, f64)> = b
        .legal_moves_capped(MOVE_CAP)
        .into_iter()
        .map(|mv| {
            let sum = b.cells[mv.path[0] as usize] * mv.path.len() as u64;
            let val = sum as f64 + net.value(&net.afterstate(&codes, &mv, sum));
            (mv, sum, val)
        })
        .collect();
    if scored.is_empty() {
        return None;
    }
    let pick = if roll < eps_rand {
        rng.below(scored.len())
    } else {
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        let hi = scored.len().min(8);
        if hi <= 1 {
            0
        } else {
            1 + rng.below(hi - 1)
        }
    };
    let (mv, sum, _) = scored.swap_remove(pick);
    let after = net.afterstate(&codes, &mv, sum);
    Some((mv, sum as f64, after, true))
}

/// One self-play game with TD(0) updates. Returns (score, moves).
pub fn train_game(net: &mut NTupleNet, seed: u32) -> (u64, u32) {
    train_game_eps(net, seed, 0.0, 0.0)
}

/// TD(0) self-play with optional epsilon-exploration.
pub fn train_game_eps(net: &mut NTupleNet, seed: u32, eps_rank: f32, eps_rand: f32) -> (u64, u32) {
    let mut b = Board::new_game(seed);
    let mut xrng = Mulberry32::new(seed ^ 0x9E37_79B9);
    let mut prev: Option<[u8; CELLS]> = None;
    loop {
        match choose_train(net, &b, &mut xrng, eps_rank, eps_rand) {
            None => {
                if let Some(pa) = prev {
                    net.update(&pa, 0.0);
                }
                return (b.score, b.moves_made);
            }
            Some((mv, r, after, _)) => {
                if let Some(pa) = prev {
                    net.update(&pa, r + net.value(&after));
                }
                prev = Some(after);
                b.apply(&mv);
            }
        }
    }
}

/// One self-play game with online TD(lambda): each TD error also nudges the
/// recent afterstates with geometrically decayed weight, propagating credit
/// for long-horizon plays (corner building) much faster than TD(0).
pub fn train_game_lambda(
    net: &mut NTupleNet,
    seed: u32,
    lambda: f32,
    trace_len: usize,
) -> (u64, u32) {
    train_game_lambda_eps(net, seed, lambda, trace_len, 0.0, 0.0)
}

/// TD(lambda) self-play with optional epsilon-exploration; traces are cut at
/// exploratory moves (Watkins), since earlier afterstates must not inherit
/// credit through an action the greedy policy did not choose.
pub fn train_game_lambda_eps(
    net: &mut NTupleNet,
    seed: u32,
    lambda: f32,
    trace_len: usize,
    eps_rank: f32,
    eps_rand: f32,
) -> (u64, u32) {
    fn apply_traces(
        net: &mut NTupleNet,
        traces: &std::collections::VecDeque<[u8; CELLS]>,
        lambda: f32,
        delta: f32,
    ) {
        let mut f = 1.0f32;
        for s in traces {
            net.nudge(s, delta * f);
            f *= lambda;
            if f < 0.05 {
                break;
            }
        }
    }
    let mut b = Board::new_game(seed);
    let mut xrng = Mulberry32::new(seed ^ 0x9E37_79B9);
    let mut traces: std::collections::VecDeque<[u8; CELLS]> = Default::default();
    loop {
        match choose_train(net, &b, &mut xrng, eps_rank, eps_rand) {
            None => {
                if let Some(last) = traces.front() {
                    let delta = -(net.value(last) as f32);
                    apply_traces(net, &traces, lambda, delta);
                }
                return (b.score, b.moves_made);
            }
            Some((mv, r, after, explored)) => {
                if let Some(last) = traces.front() {
                    let delta = (r + net.value(&after) - net.value(last)) as f32;
                    apply_traces(net, &traces, lambda, delta);
                }
                if explored {
                    traces.clear();
                }
                traces.push_front(after);
                traces.truncate(trace_len);
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
        cfg.global2 = true;
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
    fn full_placement_coverage() {
        use std::collections::BTreeSet;
        // every placement of every shape (all dihedral forms x all fitting
        // anchors) must appear among the net's stamped images
        fn placements(shape: &[(usize, usize)]) -> BTreeSet<BTreeSet<usize>> {
            let mut forms: BTreeSet<Vec<(usize, usize)>> = BTreeSet::new();
            for t in 0..8 {
                let f: Vec<(usize, usize)> =
                    shape.iter().map(|&(r, c)| dihedral(r, c, t)).collect();
                let mr = f.iter().map(|p| p.0).min().unwrap();
                let mc = f.iter().map(|p| p.1).min().unwrap();
                let mut norm: Vec<(usize, usize)> =
                    f.iter().map(|&(r, c)| (r - mr, c - mc)).collect();
                norm.sort_unstable();
                forms.insert(norm);
            }
            let mut out = BTreeSet::new();
            for f in forms {
                let h = f.iter().map(|p| p.0).max().unwrap() + 1;
                let w = f.iter().map(|p| p.1).max().unwrap() + 1;
                for ar in 0..=(N - h) {
                    for ac in 0..=(N - w) {
                        out.insert(f.iter().map(|&(r, c)| idx(r + ar, c + ac)).collect());
                    }
                }
            }
            out
        }
        let mut cfg = NetConfig::base();
        cfg.staircase = true;
        cfg.diagonals = true;
        let net = NTupleNet::new(1.0, cfg);
        let stamped: BTreeSet<BTreeSet<usize>> = net
            .images
            .iter()
            .map(|img| img.cells.iter().map(|&c| c as usize).collect())
            .collect();
        for (name, shape) in [
            ("2x2", vec![(0, 0), (0, 1), (1, 0), (1, 1)]),
            ("2x3", vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]),
            ("plus", vec![(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)]),
            ("stair", vec![(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]),
        ] {
            for p in placements(&shape) {
                assert!(stamped.contains(&p), "{name} placement missing: {p:?}");
            }
        }
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = std::env::temp_dir().join("ntuple-test-roundtrip.bin");
        let path = dir.to_str().unwrap();
        let mut cfg = small_cfg();
        cfg.alphabet = Alphabet::Fine;
        cfg.stages = 3;
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
