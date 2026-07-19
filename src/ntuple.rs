//! N-tuple network value function with afterstate TD(0) + TC learning,
//! following the 2048 lineage (Szubert & Jaskowski 2014, Jaskowski 2017).
//!
//! The afterstate of a move is the board with the merged sum placed on the
//! head cell and every other path cell marked PENDING (refill undrawn). The
//! tables learn the expectation over refills implicitly, so training never
//! enumerates the 3^(k-1) refill outcomes.

use crate::game::{idx, Board, Move, Mulberry32, CELLS, MOVE_CAP, N};

/// Cell alphabet: 0..=3 exact 1/2/3/4, 4..=18 ladder tiers 6*2^0..6*2^14
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

pub struct NTupleNet {
    tables: Vec<Lut>,
    images: Vec<Image>,
    pub alpha: f32,
    pub with_2x3: bool,
}

impl NTupleNet {
    pub fn new(alpha: f32, with_2x3: bool) -> Self {
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
        let mut tables = Vec::new();
        let mut images = Vec::new();
        let len5 = ALPHABET.pow(5);
        let len4 = ALPHABET.pow(4);
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
        // 2x3 blocks: anchor orbit reps share ONE translation-invariant table
        if with_2x3 {
            tables.push(Lut::new(ALPHABET.pow(6)));
            let shared = tables.len() - 1;
            for &(r, c) in &[(0, 0), (0, 1), (1, 1)] {
                let cells = [
                    (r, c),
                    (r, c + 1),
                    (r, c + 2),
                    (r + 1, c),
                    (r + 1, c + 1),
                    (r + 1, c + 2),
                ];
                add_base(&mut images, &cells, shared);
            }
        }
        NTupleNet {
            tables,
            images,
            alpha,
            with_2x3,
        }
    }

    fn index(img: &Image, codes: &[u8; CELLS]) -> usize {
        let mut i = 0usize;
        for &c in &img.cells {
            i = i * ALPHABET + codes[c as usize] as usize;
        }
        i
    }

    pub fn value(&self, codes: &[u8; CELLS]) -> f64 {
        let mut v = 0.0f64;
        for img in &self.images {
            v += self.tables[img.table].w[Self::index(img, codes)] as f64;
        }
        v
    }

    /// TC-TD update of V(codes) toward target.
    pub fn update(&mut self, codes: &[u8; CELLS], target: f64) {
        let delta = (target - self.value(codes)) as f32;
        let per = self.alpha * delta / self.images.len() as f32;
        for img in &self.images {
            let i = Self::index(img, codes);
            let t = &mut self.tables[img.table];
            let rate = if t.a[i] > 0.0 {
                (t.e[i].abs() / t.a[i]).min(1.0)
            } else {
                1.0
            };
            t.w[i] += rate * per;
            t.e[i] += delta;
            t.a[i] += delta.abs();
        }
    }

    /// Afterstate codes for a move from `codes` (board encoding) with tile
    /// value v on the start cell.
    pub fn afterstate(codes: &[u8; CELLS], mv: &Move, sum: u64) -> [u8; CELLS] {
        let mut a = *codes;
        for &ci in &mv.path[..mv.path.len() - 1] {
            a[ci as usize] = PENDING;
        }
        a[mv.head()] = encode_cell(sum);
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

    /// Weights-only snapshot (little-endian f32 stream per table).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let f = std::fs::File::create(path)?;
        let mut wr = std::io::BufWriter::new(f);
        wr.write_all(&(self.tables.len() as u32).to_le_bytes())?;
        wr.write_all(&u32::from(self.with_2x3).to_le_bytes())?;
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
        let ntab = u32::from_le_bytes(b4) as usize;
        rd.read_exact(&mut b4)?;
        let with_2x3 = u32::from_le_bytes(b4) != 0;
        let mut net = NTupleNet::new(alpha, with_2x3);
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
        let codes = encode_board(&b.cells);
        let mut best: Option<(Move, f64, [u8; CELLS], f64)> = None;
        for mv in b.legal_moves_capped(MOVE_CAP) {
            let v = b.cells[mv.path[0] as usize];
            let sum = v * mv.path.len() as u64;
            let after = Self::afterstate(&codes, &mv, sum);
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

#[allow(dead_code)]
fn _rng_unused(_: &mut Mulberry32) {}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn td_learns_toward_target() {
        let mut net = NTupleNet::new(1.0, false);
        let b = Board::new_game(7);
        let codes = encode_board(&b.cells);
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
        let mut net = NTupleNet::new(1.0, false);
        let b = Board::new_game(3);
        let codes = encode_board(&b.cells);
        net.update(&codes, 50.0);
        let mut tcells = [0u64; CELLS];
        for r in 0..N {
            for c in 0..N {
                tcells[idx(c, r)] = b.cells[idx(r, c)];
            }
        }
        let tcodes = encode_board(&tcells);
        let (v, tv) = (net.value(&codes), net.value(&tcodes));
        assert!((v - tv).abs() < 1e-9, "{v} vs {tv}");
    }
}
