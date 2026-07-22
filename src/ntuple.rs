//! N-tuple network value function with afterstate TD(0) + TC learning,
//! following the 2048 lineage (Szubert & Jaskowski 2014, Jaskowski 2017).
//!
//! The afterstate of a move is the board with the merged sum placed on the
//! head cell and every other path cell marked pending (refill undrawn). The
//! tables learn the expectation over refills implicitly, so training never
//! enumerates the 3^(k-1) refill outcomes.

use crate::game::{idx, neighbors, rc, Board, Move, Mulberry32, CELLS, MOVE_CAP, N};

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

/// Coarse starter alphabet (12 symbols) for progressive growth: exact
/// 1/2/3/4, ladder tiers folded in pairs (6-12, 24-48, 96-192, 384-768,
/// 1536+), one pow2 bucket, one merged trash bucket, pending.
pub fn encode_cell_coarse(v: u64) -> u8 {
    let f = encode_cell(v);
    COARSE_OF_BASE[f as usize]
}

/// Base(23) code -> Coarse(12) code, also used to seed grown tables.
pub const COARSE_OF_BASE: [u8; 23] = [
    0, 1, 2, 3, // 1,2,3,4
    4, 4, // 6,12
    5, 5, // 24,48
    6, 6, // 96,192
    7, 7, // 384,768
    8, 8, 8, 8, 8, 8, 8,  // 1536..98304
    9,  // pow2
    10, // nine
    10, // trash
    11, // pending
];

/// Slim alphabet (18): tile histogram shows games top out at 384, so all
/// ladder >= 3072 folds into one "large ladder" code.
pub fn encode_cell_slim(v: u64) -> u8 {
    let b = encode_cell(v);
    match b {
        0..=12 => b,   // 1,2,3,4, ladder 6..1536
        13..=18 => 13, // large ladder (3072+)
        19 => 14,      // pow2 >= 8
        20 => 15,      // nine family
        21 => 16,      // trash
        _ => 17,       // pending
    }
}

/// Slim89 (20): slim plus exact 8 and exact 9 split from their buckets.
pub fn encode_cell_slim89(v: u64) -> u8 {
    if v == 8 {
        return 14;
    }
    if v == 9 {
        return 16;
    }
    let b = encode_cell(v);
    match b {
        0..=12 => b,
        13..=18 => 13,
        19 => 15, // pow2 >= 16
        20 => 17, // nine >= 18
        21 => 18, // trash
        _ => 19,  // pending
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Alphabet {
    Base,
    Fine,
    Coarse,
    Slim,
    Slim89,
}

impl Alphabet {
    pub fn size(self) -> usize {
        match self {
            Alphabet::Base => 23,
            Alphabet::Fine => 31,
            Alphabet::Coarse => 12,
            Alphabet::Slim => 18,
            Alphabet::Slim89 => 20,
        }
    }
    pub fn pending(self) -> u8 {
        (self.size() - 1) as u8
    }
    pub fn encode(self, v: u64) -> u8 {
        match self {
            Alphabet::Base => encode_cell(v),
            Alphabet::Fine => encode_cell_fine(v),
            Alphabet::Coarse => encode_cell_coarse(v),
            Alphabet::Slim => encode_cell_slim(v),
            Alphabet::Slim89 => encode_cell_slim89(v),
        }
    }

    /// Whether a code denotes one exact value (needed for same-value
    /// blob/path features; bucket codes cannot prove equality).
    pub fn is_exact(self, c: u8) -> bool {
        match self {
            Alphabet::Base | Alphabet::Fine => c <= 18,
            Alphabet::Coarse => c <= 3,
            Alphabet::Slim => c <= 12,
            Alphabet::Slim89 => c <= 12 || c == 14 || c == 16,
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
    /// Extra-feature bitmask (EX_* constants): new shapes, gated/averaged
    /// global variants, blob / path / census features.
    pub extra: u32,
    /// Gated global tables: top-2 pair and top-3 triple positions, active
    /// only when the tiles are genuinely big (mag >= 24); plus dispersion.
    pub global2: bool,
    /// Weight-set count for multi-stage learning (Jaskowski 2017): 1 or 3.
    /// With 3, the stage is keyed by the board's max tile against
    /// `stage_thresholds` and every table is replicated per stage.
    pub stages: usize,
    /// Max-tile boundaries between stage 0/1 and stage 1/2.
    pub stage_thresholds: [u32; 2],
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
            extra: 0,
            stages: 1,
            stage_thresholds: [96, 768],
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

/// One weight table with temporal-coherence accumulators and (when the
/// exploration bonus is enabled) per-entry visit counts.
struct Lut {
    w: Vec<f32>,
    e: Vec<f32>,
    a: Vec<f32>,
    n: Vec<u32>,
}

impl Lut {
    fn new(len: usize) -> Self {
        Lut {
            w: vec![0.0; len],
            e: vec![0.0; len],
            a: vec![0.0; len],
            n: Vec::new(),
        }
    }
}

/// Shape groups for progressive growth (images activate by group).
pub const G_ROWS: u8 = 0;
pub const G_SQ2: u8 = 1;
pub const G_PLUS: u8 = 2;
pub const G_STAIR: u8 = 3;
pub const G_DIAG: u8 = 4;
pub const G_BLK23: u8 = 5;
pub const G_BIGL: u8 = 6;
pub const G_X: u8 = 7;
pub const G_STAIR6: u8 = 8;
pub const N_GROUPS: usize = 9;

/// Extra-feature bitmask (cfg.extra).
pub const EX_BIGL: u32 = 1;
pub const EX_X: u32 = 2;
pub const EX_STAIR6: u32 = 4;
pub const EX_GATED12: u32 = 8;
pub const EX_AVGDISP: u32 = 16;
pub const EX_BLOBTIER: u32 = 32;
pub const EX_BLOBALPHA: u32 = 64;
pub const EX_BLOB2: u32 = 128;
pub const EX_FREEFIELD: u32 = 256;
pub const EX_EQPAIRS: u32 = 512;
pub const EX_PATHTIER: u32 = 1024;
pub const EX_PATHALPHA: u32 = 2048;
pub const EX_PATH2: u32 = 4096;

/// Global feature kinds, in table order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GKind {
    Pair,
    PairGated(u32),
    PairGated12,
    Disp,
    DispAvg,
    Top3Gated,
    BlobTier,
    BlobAlpha,
    Blob2,
    FreeField,
    EqPairs,
    PathTier,
    PathAlpha,
    Path2,
}

impl GKind {
    fn table_len(self, m: usize) -> usize {
        match self {
            GKind::Pair => CELLS * CELLS,
            GKind::PairGated(_) | GKind::PairGated12 => CELLS * CELLS + 1,
            GKind::Disp => 20 * 8,
            GKind::DispAvg => 10 * 8,
            GKind::Top3Gated => CELLS * CELLS * CELLS + 1,
            GKind::BlobTier | GKind::PathTier => 11 * 8,
            GKind::BlobAlpha | GKind::PathAlpha => 11 * m,
            GKind::Blob2 | GKind::Path2 => 6 * m * 6 * m,
            GKind::FreeField => 26,
            GKind::EqPairs => 25,
        }
    }
}

/// A symmetric image of a base tuple: fixed cell list into a shared table.
struct Image {
    cells: Vec<u8>,
    table: usize,
    group: u8,
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
const SAVE_MAGIC_V6: u32 = 0x4E54_5636; // "NTV6": adds the global2 flag
const SAVE_MAGIC_V7: u32 = 0x4E54_5637; // "NTV7": adds stage thresholds
const SAVE_MAGIC: u32 = 0x4E54_5638; // "NTV8": alphabet id + extras mask

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
        Alphabet::Coarse => match c {
            0..=3 => c as u32 + 1,
            4..=8 => 12u32 << (2 * (c - 4)), // pair reps: 12,48,192,768,3072
            9 => 16,
            10 => 33,
            _ => 0,
        },
        Alphabet::Slim => match c {
            0..=3 => c as u32 + 1,
            4..=12 => 6u32 << (c - 4),
            13 => 3072,
            14 => 16,
            15 => 36,
            16 => 30,
            _ => 0,
        },
        Alphabet::Slim89 => match c {
            0..=3 => c as u32 + 1,
            4..=12 => 6u32 << (c - 4),
            13 => 3072,
            14 => 8,
            15 => 32,
            16 => 9,
            17 => 36,
            18 => 30,
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
    gkinds: Vec<GKind>,
    /// Tables per stage; stage s uses tables[s*ntab_stage..(s+1)*ntab_stage].
    ntab_stage: usize,
    /// Whether promotion enabled and which stages have been initialized.
    pub promote: bool,
    stage_ready: Vec<bool>,
    /// Highest stage currently in use (adaptive staging keeps this at 0
    /// until the trainer activates later stages).
    pub stage_cap: usize,
    /// Active shape groups (progressive growth flips these on).
    active: [bool; N_GROUPS],
    n_active_images: usize,
    /// Cells-per-tuple for each table (0 = global feature table).
    arity: Vec<u8>,
    /// Dihedral index permutations: perms[t][i] = image of cell i.
    perms: Vec<[u8; CELLS]>,
    /// Per-entry exploration bonus scale (OTD-style): each entry adds
    /// bonus/sqrt(1+visits) to TRAINING move selection only. 0 = off.
    pub bonus: f32,
}

impl NTupleNet {
    pub fn new(alpha: f32, cfg: NetConfig) -> Self {
        fn add_base(images: &mut Vec<Image>, cells: &[(usize, usize)], table: usize, group: u8) {
            for t in 0..8 {
                let img: Vec<u8> = cells
                    .iter()
                    .map(|&(r, c)| {
                        let (rr, cc) = dihedral(r, c, t);
                        idx(rr, cc) as u8
                    })
                    .collect();
                images.push(Image {
                    cells: img,
                    table,
                    group,
                });
            }
        }
        let m = cfg.alphabet.size();
        let mut tables = Vec::new();
        let mut images = Vec::new();
        let mut arity: Vec<u8> = Vec::new();
        let len5 = m.pow(5);
        let len4 = m.pow(4);
        // rows 0..3 (reflections cover rows 3,4 and all columns)
        for r in 0..3 {
            let cells: Vec<_> = (0..N).map(|c| (r, c)).collect();
            tables.push(Lut::new(len5));
            arity.push(5);
            add_base(&mut images, &cells, tables.len() - 1, G_ROWS);
        }
        // 2x2 squares, anchor orbit reps
        for &(r, c) in &[(0, 0), (0, 1), (1, 1)] {
            let cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)];
            tables.push(Lut::new(len4));
            arity.push(4);
            add_base(&mut images, &cells, tables.len() - 1, G_SQ2);
        }
        // plus shapes, center orbit reps
        for &(r, c) in &[(1, 1), (1, 2), (2, 2)] {
            let cells = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)];
            tables.push(Lut::new(len5));
            arity.push(5);
            add_base(&mut images, &cells, tables.len() - 1, G_PLUS);
        }
        // wraparound diagonals D_k = {(r, (r+k) mod 5)}: orbit reps k=0,1,2
        // (reflections/transposes cover the other offsets and the whole
        // anti-diagonal family)
        if cfg.diagonals {
            for k in 0..3 {
                let cells: Vec<_> = (0..N).map(|r| (r, (r + k) % N)).collect();
                tables.push(Lut::new(len5));
                arity.push(5);
                add_base(&mut images, &cells, tables.len() - 1, G_DIAG);
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
                arity.push(5);
                add_base(&mut images, cells, tables.len() - 1, G_STAIR);
            }
        }
        // big Ls: 5-cell corner hooks, 6 orbit reps cover all 36 placements
        if cfg.extra & EX_BIGL != 0 {
            let reps: [[(usize, usize); 5]; 6] = [
                [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
                [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
                [(0, 1), (0, 2), (0, 3), (1, 1), (2, 1)],
                [(0, 1), (1, 1), (2, 1), (2, 2), (2, 3)],
                [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)],
                [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)],
            ];
            for cells in &reps {
                tables.push(Lut::new(len5));
                arity.push(5);
                add_base(&mut images, cells, tables.len() - 1, G_BIGL);
            }
        }
        // X shapes: diagonal cross, 3 orbit reps cover all 9 placements
        if cfg.extra & EX_X != 0 {
            let reps: [[(usize, usize); 5]; 3] = [
                [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)],
                [(0, 1), (0, 3), (1, 2), (2, 1), (2, 3)],
                [(1, 1), (1, 3), (2, 2), (3, 1), (3, 3)],
            ];
            for cells in &reps {
                tables.push(Lut::new(len5));
                arity.push(5);
                add_base(&mut images, cells, tables.len() - 1, G_X);
            }
        }
        // 6-cell staircases: 3 orbit reps cover all 24 placements
        if cfg.extra & EX_STAIR6 != 0 {
            let reps: [[(usize, usize); 6]; 3] = [
                [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)],
                [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4)],
                [(0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)],
            ];
            for cells in &reps {
                tables.push(Lut::new(m.pow(6)));
                arity.push(6);
                add_base(&mut images, cells, tables.len() - 1, G_STAIR6);
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
                arity.push(6);
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
                        arity.push(6);
                        tables.len() - 1
                    }
                };
                add_base(&mut images, cells, table, G_BLK23);
            }
        }
        let mut gkinds: Vec<GKind> = Vec::new();
        if cfg.global {
            gkinds.push(if cfg.extra & EX_GATED12 != 0 {
                GKind::PairGated12
            } else {
                GKind::Pair
            });
            gkinds.push(if cfg.extra & EX_AVGDISP != 0 {
                GKind::DispAvg
            } else {
                GKind::Disp
            });
        }
        if cfg.global2 {
            gkinds.push(GKind::PairGated(24));
            gkinds.push(GKind::Top3Gated);
            gkinds.push(GKind::Disp);
        }
        for (bit, k) in [
            (EX_BLOBTIER, GKind::BlobTier),
            (EX_BLOBALPHA, GKind::BlobAlpha),
            (EX_BLOB2, GKind::Blob2),
            (EX_FREEFIELD, GKind::FreeField),
            (EX_EQPAIRS, GKind::EqPairs),
            (EX_PATHTIER, GKind::PathTier),
            (EX_PATHALPHA, GKind::PathAlpha),
            (EX_PATH2, GKind::Path2),
        ] {
            if cfg.extra & bit != 0 {
                gkinds.push(k);
            }
        }
        for k in &gkinds {
            tables.push(Lut::new(k.table_len(m)));
            arity.push(0);
        }
        let n_globals = gkinds.len();
        // multi-stage: replicate the whole per-stage table set
        let ntab_stage = tables.len();
        assert!(cfg.stages == 1 || cfg.stages == 3, "stages must be 1 or 3");
        for _ in 1..cfg.stages {
            for t in 0..ntab_stage {
                let len = tables[t].w.len();
                tables.push(Lut::new(len));
                arity.push(arity[t]);
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
        let n_active_images = images.len();
        NTupleNet {
            tables,
            images,
            alpha,
            cfg,
            m,
            n_globals,
            gkinds,
            ntab_stage,
            promote: false,
            stage_ready,
            stage_cap: cfg.stages - 1,
            active: [true; N_GROUPS],
            n_active_images,
            arity,
            perms,
            bonus: 0.0,
        }
    }

    /// Enable the OTD-style exploration bonus: `total` is the optimism a
    /// fully-novel board carries (split across contributors); it decays per
    /// entry as 1/sqrt(1+visits). Lives outside the learned weights, so TD
    /// targets and eval remain unbiased and TC accumulators stay clean.
    pub fn enable_bonus(&mut self, total: f32) {
        self.bonus = total / (self.n_active_images + self.n_globals) as f32;
        for t in &mut self.tables {
            t.n = vec![0; t.w.len()];
        }
    }

    fn bonus_term(&self, codes: &[u8; CELLS]) -> f64 {
        let off = self.stage_of(codes) * self.ntab_stage;
        let mut b = 0.0f64;
        for img in &self.images {
            if !self.active[img.group as usize] {
                continue;
            }
            let t = &self.tables[off + img.table];
            let n = t.n[self.index(img, codes)];
            b += (self.bonus as f64) / (1.0 + n as f64).sqrt();
        }
        if self.n_globals > 0 {
            let (g, ng) = self.global_indices(codes);
            let base = off + self.ntab_stage - self.n_globals;
            for (k, &gi) in g.iter().take(ng).enumerate() {
                let n = self.tables[base + k].n[gi];
                b += (self.bonus as f64) / (1.0 + n as f64).sqrt();
            }
        }
        b
    }

    /// Training-time greedy: plain V plus the exploration bonus.
    pub fn greedy_bonus(&self, b: &Board) -> Option<(Move, f64, [u8; CELLS])> {
        let codes = self.encode(&b.cells);
        let mut best: Option<(Move, f64, [u8; CELLS], f64)> = None;
        for mv in b.legal_moves_capped(MOVE_CAP) {
            let v = b.cells[mv.path[0] as usize];
            let sum = v * mv.path.len() as u64;
            let after = self.afterstate(&codes, &mv, sum);
            let val = sum as f64 + self.value(&after) + self.bonus_term(&after);
            if best.as_ref().is_none_or(|(_, _, _, bv)| val > *bv) {
                best = Some((mv, sum as f64, after, val));
            }
        }
        best.map(|(mv, r, a, _)| (mv, r, a))
    }

    /// Progressive growth: deactivate every group not in `groups` (their
    /// zero-init tables sit untouched until activated).
    pub fn set_active_groups(&mut self, groups: &[u8]) {
        self.active = [false; N_GROUPS];
        for &g in groups {
            self.active[g as usize] = true;
        }
        self.recount_active();
    }

    pub fn activate_group(&mut self, g: u8) {
        self.active[g as usize] = true;
        self.recount_active();
    }

    pub fn active_groups(&self) -> [bool; N_GROUPS] {
        self.active
    }

    fn recount_active(&mut self) {
        self.n_active_images = self
            .images
            .iter()
            .filter(|i| self.active[i.group as usize])
            .count();
    }

    /// Progressive growth along the alphabet axis: Coarse(12) -> Base(23).
    /// Every tuple table is re-indexed so each fine combination inherits its
    /// coarse parent's weight; TC accumulators start fresh (full rate for
    /// the newly split entries). Global tables are alphabet-free.
    pub fn grow_alphabet(&mut self) {
        assert_eq!(self.cfg.alphabet, Alphabet::Coarse, "can only grow Coarse");
        let fine = Alphabet::Base;
        let mf = fine.size();
        for (t, lut) in self.tables.iter_mut().enumerate() {
            let k = self.arity[t] as usize;
            if k == 0 {
                continue;
            }
            let mut w = vec![0.0f32; mf.pow(k as u32)];
            for (i, slot) in w.iter_mut().enumerate() {
                let mut rest = i;
                let mut ci = 0usize;
                let mut digits = [0usize; 6];
                for d in (0..k).rev() {
                    digits[d] = rest % mf;
                    rest /= mf;
                }
                for &dg in digits.iter().take(k) {
                    ci = ci * 12 + COARSE_OF_BASE[dg] as usize;
                }
                *slot = lut.w[ci];
            }
            let len = w.len();
            lut.w = w;
            lut.e = vec![0.0; len];
            lut.a = vec![0.0; len];
        }
        self.cfg.alphabet = fine;
        self.m = mf;
    }

    /// Adaptive staging: raise the stage cap, promoting weights into the
    /// newly opened stage.
    pub fn activate_stage(&mut self, s: usize) {
        assert!(s < self.cfg.stages);
        self.promote_stage(s);
        self.stage_cap = self.stage_cap.max(s);
    }

    /// Stage of a board encoding: 0 below max tile 96, 1 below 768, else 2
    /// (always 0 for single-stage nets).
    fn stage_of(&self, codes: &[u8; CELLS]) -> usize {
        if self.cfg.stages == 1 || self.stage_cap == 0 {
            return 0;
        }
        let a = self.cfg.alphabet;
        let mx = codes.iter().map(|&c| code_mag(c, a)).max().unwrap_or(1);
        let [t1, t2] = self.cfg.stage_thresholds;
        let raw = if mx < t1 {
            0
        } else if mx < t2 {
            1
        } else {
            2
        };
        raw.min(self.stage_cap)
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

    /// Same-value blobs (connected components of equal exactly-coded cells),
    /// as (code, size, longest simple path within the blob), sorted by size
    /// descending, plus the blob cell masks for adjacency tests.
    fn blobs(&self, codes: &[u8; CELLS]) -> Vec<(u8, u8, u8, u32, u32)> {
        let a = self.cfg.alphabet;
        let mut seen = 0u32;
        let mut out: Vec<(u8, u8, u8, u32, u32)> = Vec::new();
        for start in 0..CELLS {
            if seen & (1 << start) != 0 || !a.is_exact(codes[start]) {
                continue;
            }
            let c = codes[start];
            let mut mask = 0u32;
            let mut stack = vec![start];
            while let Some(i) = stack.pop() {
                if mask & (1 << i) != 0 {
                    continue;
                }
                mask |= 1 << i;
                neighbors(i, |nb| {
                    if codes[nb] == c && mask & (1 << nb) == 0 {
                        stack.push(nb);
                    }
                });
            }
            seen |= mask;
            let size = mask.count_ones() as u8;
            let lp = Self::longest_path(mask);
            // canonical mask key: minimum over dihedral transforms, so blob
            // ordering ties break identically on symmetric boards
            let ckey = self
                .perms
                .iter()
                .map(|p| {
                    let mut m2 = 0u32;
                    for i in 0..CELLS {
                        if mask & (1 << i) != 0 {
                            m2 |= 1 << p[i];
                        }
                    }
                    m2
                })
                .min()
                .unwrap();
            out.push((c, size, lp, mask, ckey));
        }
        out.sort_by(|x, y| {
            y.1.cmp(&x.1)
                .then(x.0.cmp(&y.0))
                .then(y.2.cmp(&x.2))
                .then(x.4.cmp(&y.4))
        });
        out
    }

    /// Longest simple path (in cells) within a cell mask, DFS with a budget.
    fn longest_path(mask: u32) -> u8 {
        fn dfs(cur: usize, visited: u32, mask: u32, budget: &mut u32) -> u8 {
            if *budget == 0 {
                return 1;
            }
            *budget -= 1;
            let mut best = 1u8;
            neighbors(cur, |nb| {
                if mask & (1 << nb) != 0 && visited & (1 << nb) == 0 {
                    let l = 1 + dfs(nb, visited | (1 << nb), mask, budget);
                    if l > best {
                        best = l;
                    }
                }
            });
            best
        }
        let size = mask.count_ones() as u8;
        if size <= 2 {
            return size;
        }
        let mut budget = 20_000u32;
        let mut best = 1u8;
        for i in 0..CELLS {
            if mask & (1 << i) != 0 {
                let l = dfs(i, 1 << i, mask, &mut budget);
                if l > best {
                    best = l;
                }
            }
        }
        best.min(size)
    }

    fn mask_adjacent(m1: u32, m2: u32) -> bool {
        if m1 & m2 != 0 {
            return true;
        }
        for i in 0..CELLS {
            if m1 & (1 << i) != 0 {
                let mut adj = false;
                neighbors(i, |nb| {
                    if m2 & (1 << nb) != 0 {
                        adj = true;
                    }
                });
                if adj {
                    return true;
                }
            }
        }
        false
    }

    /// Indices into the active global tables, in table order.
    fn global_indices(&self, codes: &[u8; CELLS]) -> ([usize; 8], usize) {
        let a = self.cfg.alphabet;
        let m = self.m;
        let mags: Vec<u32> = codes.iter().map(|&c| code_mag(c, a)).collect();
        let maxmag = *mags.iter().max().unwrap_or(&1);
        let tier = (32 - (maxmag.max(3) / 3).leading_zeros()).min(7) as usize;
        let mut sorted = mags.clone();
        sorted.sort_unstable_by(|x, y| y.cmp(x));
        let mut blobs_c: Option<Vec<(u8, u8, u8, u32, u32)>> = None;
        let mut out = [0usize; 8];
        let mut n = 0;
        for k in &self.gkinds {
            out[n] = match *k {
                GKind::Pair => self.canonical_topk(&mags, 2),
                GKind::PairGated12 => {
                    if sorted[1] >= 12 {
                        self.canonical_topk(&mags, 2)
                    } else {
                        CELLS * CELLS
                    }
                }
                GKind::PairGated(t) => {
                    if sorted[1] >= t {
                        self.canonical_topk(&mags, 2)
                    } else {
                        CELLS * CELLS
                    }
                }
                GKind::Top3Gated => {
                    if sorted[2] >= 24 {
                        self.canonical_topk(&mags, 3)
                    } else {
                        CELLS * CELLS * CELLS
                    }
                }
                GKind::Disp => Self::dispersion_index(&mags),
                GKind::DispAvg => {
                    let big: Vec<usize> = (0..CELLS).filter(|&i| mags[i] >= 48).collect();
                    let mut sum = 0usize;
                    let mut np = 0usize;
                    for i in 0..big.len() {
                        for j in i + 1..big.len() {
                            let (r1, c1) = rc(big[i]);
                            let (r2, c2) = rc(big[j]);
                            sum += r1.abs_diff(r2) + c1.abs_diff(c2);
                            np += 1;
                        }
                    }
                    let avg = if np > 0 { (sum + np / 2) / np } else { 0 };
                    avg.min(9) * 8 + tier
                }
                GKind::BlobTier
                | GKind::BlobAlpha
                | GKind::Blob2
                | GKind::PathTier
                | GKind::PathAlpha
                | GKind::Path2 => {
                    let bl = blobs_c.get_or_insert_with(|| self.blobs(codes));
                    match *k {
                        GKind::BlobTier => {
                            let sz = bl.first().map_or(0, |b| b.1) as usize;
                            sz.min(10) * 8 + tier
                        }
                        GKind::BlobAlpha => {
                            let (c, sz) = bl.first().map_or((0u8, 0u8), |b| (b.0, b.1));
                            (sz as usize).min(10) * m + c as usize
                        }
                        GKind::Blob2 => {
                            let (c1, s1) = bl.first().map_or((0u8, 0u8), |b| (b.0, b.1));
                            let (c2, s2) = bl.get(1).map_or((0u8, 0u8), |b| (b.0, b.1));
                            (((s1 as usize).min(5) * m + c1 as usize) * 6 + (s2 as usize).min(5))
                                * m
                                + c2 as usize
                        }
                        GKind::PathTier => {
                            let lp = bl.iter().map(|b| b.2).max().unwrap_or(0) as usize;
                            lp.min(10) * 8 + tier
                        }
                        GKind::PathAlpha => {
                            let best = bl
                                .iter()
                                .max_by_key(|b| {
                                    (b.2, std::cmp::Reverse(b.0), std::cmp::Reverse(b.4))
                                })
                                .map_or((0u8, 0u8), |b| (b.0, b.2));
                            (best.1 as usize).min(10) * m + best.0 as usize
                        }
                        GKind::Path2 => {
                            let b1 = bl
                                .iter()
                                .max_by_key(|b| {
                                    (b.2, std::cmp::Reverse(b.0), std::cmp::Reverse(b.4))
                                })
                                .cloned();
                            let (c1, l1, m1) = b1.map_or((0u8, 0u8, 0u32), |b| (b.0, b.2, b.3));
                            // second: different value OR fully non-adjacent
                            let b2 = bl
                                .iter()
                                .filter(|b| {
                                    !(b.0 == c1 && b.3 == m1)
                                        && (b.0 != c1 || !Self::mask_adjacent(b.3, m1))
                                })
                                .max_by_key(|b| {
                                    (b.2, std::cmp::Reverse(b.0), std::cmp::Reverse(b.4))
                                });
                            let (c2, l2) = b2.map_or((0u8, 0u8), |b| (b.0, b.2));
                            (((l1 as usize).min(5) * m + c1 as usize) * 6 + (l2 as usize).min(5))
                                * m
                                + c2 as usize
                        }
                        _ => unreachable!(),
                    }
                }
                GKind::FreeField => codes
                    .iter()
                    .filter(|&&c| a.is_exact(c) && code_mag(c, a) <= 6)
                    .count(),
                GKind::EqPairs => {
                    let mut pairs = 0usize;
                    for r in 0..N {
                        for c in 0..N {
                            let i = idx(r, c);
                            if !a.is_exact(codes[i]) {
                                continue;
                            }
                            if c + 1 < N && codes[idx(r, c + 1)] == codes[i] {
                                pairs += 1;
                            }
                            if r + 1 < N && codes[idx(r + 1, c)] == codes[i] {
                                pairs += 1;
                            }
                        }
                    }
                    pairs.min(24)
                }
            };
            n += 1;
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
            if !self.active[img.group as usize] {
                continue;
            }
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
        let per = self.alpha * delta / (self.n_active_images + self.n_globals) as f32;
        for k in 0..self.images.len() {
            if !self.active[self.images[k].group as usize] {
                continue;
            }
            let i = {
                let img = &self.images[k];
                self.index(img, codes)
            };
            let t = &mut self.tables[off + self.images[k].table];
            if !t.n.is_empty() {
                t.n[i] = t.n[i].saturating_add(1);
            }
            bump(t, i, per, delta);
        }
        if self.n_globals > 0 {
            let (g, n) = self.global_indices(codes);
            let base = off + self.ntab_stage - self.n_globals;
            for (k, &gi) in g.iter().take(n).enumerate() {
                let t = &mut self.tables[base + k];
                if !t.n.is_empty() {
                    t.n[gi] = t.n[gi].saturating_add(1);
                }
                bump(t, gi, per, delta);
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

    /// Compare learned values across three cell codes: for every visited
    /// context that exists with each code substituted, accumulate pairwise
    /// stats. Returns (table, n_triples, corr_ab, corr_ac, corr_bc,
    /// mean|a-b|, mean|a-c|, w_std) per tuple table.
    #[allow(clippy::type_complexity)]
    pub fn compare_codes(
        &self,
        ca: u8,
        cb: u8,
        cc: u8,
    ) -> Vec<(usize, u64, f64, f64, f64, f64, f64, f64)> {
        let m = self.m;
        let mut out = Vec::new();
        for (t, lut) in self.tables.iter().enumerate().take(self.ntab_stage) {
            let k = self.arity[t] as usize;
            if k == 0 {
                continue;
            }
            let mut n = 0u64;
            let (mut sa, mut sb, mut sc) = (0.0f64, 0.0, 0.0);
            let (mut saa, mut sbb, mut scc) = (0.0f64, 0.0, 0.0);
            let (mut sab, mut sac, mut sbc) = (0.0f64, 0.0, 0.0);
            let (mut dab, mut dac) = (0.0f64, 0.0);
            let mut wsum = 0.0f64;
            let mut wsq = 0.0f64;
            let mut wn = 0u64;
            for i in 0..lut.w.len() {
                let w = lut.w[i] as f64;
                if w != 0.0 {
                    wsum += w;
                    wsq += w * w;
                    wn += 1;
                }
                // find positions holding code ca
                let mut rest = i;
                for d in (0..k).rev() {
                    let digit = (rest % m) as u8;
                    rest /= m;
                    if digit != ca {
                        continue;
                    }
                    let stride = m.pow((k - 1 - d) as u32);
                    let ib = i + (cb as usize - ca as usize) * stride;
                    let ic = i + (cc as usize - ca as usize) * stride;
                    let (wa, wb, wc) = (lut.w[i] as f64, lut.w[ib] as f64, lut.w[ic] as f64);
                    if wa == 0.0 || wb == 0.0 || wc == 0.0 {
                        continue;
                    }
                    n += 1;
                    sa += wa;
                    sb += wb;
                    sc += wc;
                    saa += wa * wa;
                    sbb += wb * wb;
                    scc += wc * wc;
                    sab += wa * wb;
                    sac += wa * wc;
                    sbc += wb * wc;
                    dab += (wa - wb).abs();
                    dac += (wa - wc).abs();
                }
            }
            if n < 100 {
                continue;
            }
            let nf = n as f64;
            let corr = |sxy: f64, sx: f64, sy: f64, sxx: f64, syy: f64| {
                let cov = sxy / nf - (sx / nf) * (sy / nf);
                let vx = sxx / nf - (sx / nf) * (sx / nf);
                let vy = syy / nf - (sy / nf) * (sy / nf);
                if vx <= 0.0 || vy <= 0.0 {
                    return 0.0;
                }
                cov / (vx * vy).sqrt()
            };
            let wstd = if wn > 0 {
                (wsq / wn as f64 - (wsum / wn as f64).powi(2))
                    .max(0.0)
                    .sqrt()
            } else {
                0.0
            };
            out.push((
                t,
                n,
                corr(sab, sa, sb, saa, sbb),
                corr(sac, sa, sc, saa, scc),
                corr(sbc, sb, sc, sbb, scc),
                dab / nf,
                dac / nf,
                wstd,
            ));
        }
        out
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
        let aid: u32 = match self.cfg.alphabet {
            Alphabet::Base => 0,
            Alphabet::Fine => 1,
            Alphabet::Coarse => 2,
            Alphabet::Slim => 3,
            Alphabet::Slim89 => 4,
        };
        wr.write_all(&aid.to_le_bytes())?;
        wr.write_all(&u32::from(self.cfg.with_2x3).to_le_bytes())?;
        wr.write_all(&u32::from(self.cfg.pos_2x3).to_le_bytes())?;
        wr.write_all(&u32::from(self.cfg.staircase).to_le_bytes())?;
        wr.write_all(&u32::from(self.cfg.diagonals).to_le_bytes())?;
        wr.write_all(&u32::from(self.cfg.global).to_le_bytes())?;
        wr.write_all(&u32::from(self.cfg.global2).to_le_bytes())?;
        wr.write_all(&(self.cfg.stages as u32).to_le_bytes())?;
        wr.write_all(&self.cfg.stage_thresholds[0].to_le_bytes())?;
        wr.write_all(&self.cfg.stage_thresholds[1].to_le_bytes())?;
        wr.write_all(&self.cfg.extra.to_le_bytes())?;
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
            || first == SAVE_MAGIC_V7
            || first == SAVE_MAGIC_V6
            || first == SAVE_MAGIC_V5
            || first == SAVE_MAGIC_V4
            || first == SAVE_MAGIC_V3
            || first == SAVE_MAGIC_V2
        {
            let mut word = || -> std::io::Result<u32> {
                rd.read_exact(&mut b4)?;
                Ok(u32::from_le_bytes(b4))
            };
            let aw = word()?;
            let alphabet = if first >= SAVE_MAGIC {
                match aw {
                    1 => Alphabet::Fine,
                    2 => Alphabet::Coarse,
                    3 => Alphabet::Slim,
                    4 => Alphabet::Slim89,
                    _ => Alphabet::Base,
                }
            } else if aw != 0 {
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
            let global2 = if first >= SAVE_MAGIC_V6 {
                word()? != 0
            } else {
                false
            };
            let stages = if first >= SAVE_MAGIC_V5 {
                word()? as usize
            } else {
                1
            };
            let stage_thresholds = if first >= SAVE_MAGIC_V7 {
                [word()?, word()?]
            } else {
                [96, 768]
            };
            let extra = if first >= SAVE_MAGIC { word()? } else { 0 };
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
                    extra,
                    stages,
                    stage_thresholds,
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
        if net.bonus > 0.0 {
            return net.greedy_bonus(b).map(|(mv, r, a)| (mv, r, a, false));
        }
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
    let (s, m, _) = train_game_eps(net, seed, 0.0, 0.0);
    (s, m)
}

/// TD(0) self-play with optional epsilon-exploration.
pub fn train_game_eps(
    net: &mut NTupleNet,
    seed: u32,
    eps_rank: f32,
    eps_rand: f32,
) -> (u64, u32, u64) {
    let mut b = Board::new_game(seed);
    let mut xrng = Mulberry32::new(seed ^ 0x9E37_79B9);
    let mut prev: Option<[u8; CELLS]> = None;
    loop {
        match choose_train(net, &b, &mut xrng, eps_rank, eps_rand) {
            None => {
                if let Some(pa) = prev {
                    net.update(&pa, 0.0);
                }
                return (b.score, b.moves_made, b.max_tile());
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
    let (s, m, _) = train_game_lambda_eps(net, seed, lambda, trace_len, 0.0, 0.0);
    (s, m)
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
) -> (u64, u32, u64) {
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
                return (b.score, b.moves_made, b.max_tile());
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
    eval_scores_tiles(net, seed0, n)
        .into_iter()
        .map(|p| p.0)
        .collect()
}

/// (score, max tile) per game.
pub fn eval_scores_tiles(net: &NTupleNet, seed0: u32, n: u32) -> Vec<(u64, u64)> {
    (0..n)
        .map(|s| {
            let mut b = Board::new_game(seed0 + s);
            while let Some((mv, _, _)) = net.greedy(&b) {
                b.apply(&mv);
            }
            (b.score, b.max_tile())
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
    fn alphabet_growth_preserves_values() {
        let mut cfg = small_cfg();
        cfg.alphabet = Alphabet::Coarse;
        let mut net = NTupleNet::new(1.0, cfg);
        let mut b = Board::new_game(21);
        for _ in 0..40 {
            match net.greedy(&b) {
                Some((mv, r, after)) => {
                    net.update(&after, r + 30.0);
                    b.apply(&mv);
                }
                None => break,
            }
        }
        let before: Vec<f64> = (0..5)
            .map(|s| {
                let bb = Board::new_game(100 + s);
                net.value(&net.encode(&bb.cells))
            })
            .collect();
        net.grow_alphabet();
        assert_eq!(net.cfg.alphabet, Alphabet::Base);
        for (s, bv) in before.iter().enumerate() {
            let bb = Board::new_game(100 + s as u32);
            let av = net.value(&net.encode(&bb.cells));
            assert!((av - bv).abs() < 1e-3, "seed {s}: {av} vs {bv}");
        }
    }

    #[test]
    fn group_activation_changes_capacity() {
        let mut cfg = small_cfg();
        cfg.staircase = true;
        let mut net = NTupleNet::new(1.0, cfg);
        net.set_active_groups(&[G_ROWS, G_SQ2]);
        let b = Board::new_game(9);
        let codes = net.encode(&b.cells);
        net.update(&codes, 80.0);
        let v_small = net.value(&codes);
        net.activate_group(G_PLUS);
        net.activate_group(G_STAIR);
        // newly activated tables are zero-init: value unchanged until trained
        assert!((net.value(&codes) - v_small).abs() < 1e-9);
        let v0 = net.value(&codes);
        net.update(&codes, v0 + 100.0);
        assert!(net.value(&codes) > v0, "activated tables must learn");
    }

    #[test]
    fn slim_extras_symmetric_and_roundtrip() {
        let mut cfg = NetConfig::base();
        cfg.alphabet = Alphabet::Slim;
        cfg.staircase = true;
        cfg.extra = EX_BIGL
            | EX_X
            | EX_STAIR6
            | EX_GATED12
            | EX_AVGDISP
            | EX_BLOBTIER
            | EX_BLOBALPHA
            | EX_BLOB2
            | EX_FREEFIELD
            | EX_EQPAIRS
            | EX_PATHTIER
            | EX_PATHALPHA
            | EX_PATH2;
        let mut net = NTupleNet::new(1.0, cfg);
        let mut b = Board::new_game(5);
        for _ in 0..30 {
            match net.greedy(&b) {
                Some((mv, r, after)) => {
                    net.update(&after, r + 40.0);
                    b.apply(&mv);
                }
                None => break,
            }
        }
        let codes = net.encode(&b.cells);
        let mut tcells = [0u64; CELLS];
        for r in 0..N {
            for c in 0..N {
                tcells[idx(c, r)] = b.cells[idx(r, c)];
            }
        }
        let tcodes = net.encode(&tcells);
        let (v, tv) = (net.value(&codes), net.value(&tcodes));
        assert!((v - tv).abs() < 1e-6, "{v} vs {tv}");
        let path = std::env::temp_dir().join("ntuple-slim-extras.bin");
        let path = path.to_str().unwrap();
        net.save(path).unwrap();
        let loaded = NTupleNet::load(path, 1.0).unwrap();
        assert_eq!(loaded.cfg.alphabet, Alphabet::Slim);
        assert_eq!(loaded.cfg.extra, cfg.extra);
        assert!((loaded.value(&codes) - v).abs() < 1e-6);
        let _ = std::fs::remove_file(path);
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
