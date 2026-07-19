//! C-style wasm exports for the browser solver lab.
//!
//! ABI (no wasm-bindgen, no imports — instantiate the module bare). All JSON
//! results are returned as (pointer from the call, length from `result_len()`).
//!
//! Analysis (stateless, works on whatever board JS supplies):
//! 1. JS writes 25 tile values (u32, row-major) at `input_ptr()`.
//! 2. `analyze(depth, samples, topk, keep, seed)` returns JSON:
//!    {"moves":[{"path":[..],"ev":n,"value":v,"sum":s},...],   best first
//!     "eval":{"smooth":..,"pow2":..,"off":..,"opt":..,"over":..,"pairs":n,"total":..},
//!     "classes":[..25 codes: 0 small, 1 smooth, 2 pow2, 3 off..]}
//!    Analysis samples refills from its own seeded RNG — it never predicts the
//!    game's actual draws.
//!
//! Stateful game (the engine owns the authoritative game + full history):
//! - `game_new(seed)` starts a game (same board generation as the JS build).
//! - `game_apply(len)` applies the move whose path (cell indices) JS wrote at
//!   `input_ptr()`; snapshots history first. Returns 1 on success, 0 if invalid.
//! - `game_undo()` pops one history entry. Returns 1 on success. Unlimited depth.
//! - `game_state()` returns JSON {"cells":[..],"score":n,"over":0|1,"history":n}.

use crate::game::{neighbors, Board, Move, CELLS};
use crate::heuristics::{classify, evaluate_breakdown, TileClass, Weights};
use crate::search::ExpectimaxPolicy;
use std::cell::RefCell;

thread_local! {
    static INPUT: RefCell<[u32; CELLS]> = const { RefCell::new([0; CELLS]) };
    static RESULT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static GAME: RefCell<Option<Board>> = const { RefCell::new(None) };
    static HISTORY: RefCell<Vec<Board>> = const { RefCell::new(Vec::new()) };
}

#[no_mangle]
pub extern "C" fn input_ptr() -> *mut u32 {
    INPUT.with(|i| i.borrow_mut().as_mut_ptr())
}

#[no_mangle]
pub extern "C" fn result_len() -> usize {
    RESULT.with(|r| r.borrow().len())
}

fn set_result(s: String) -> *const u8 {
    RESULT.with(|r| {
        *r.borrow_mut() = s.into_bytes();
        r.borrow().as_ptr()
    })
}

fn weight_rows(w: &Weights) -> String {
    let sz = if w.linear { "v" } else { "log\u{2082}v" };
    let mut rows: Vec<[String; 2]> = vec![["Score".into(), format!("{:.1}", w.w_score)]];
    if w.w_smooth != 0.0 {
        rows.push([
            "Ladder reward (2\u{1d4f}\u{b7}3)".into(),
            format!("{:.1} \u{b7} {sz}", w.w_smooth),
        ]);
    }
    if w.w_three != 0.0 {
        rows.push([
            "9-multiples penalty".into(),
            format!("{:.1} \u{b7} {sz}", w.w_three),
        ]);
    }
    if w.w_three_hi != 0.0 {
        rows.push([
            "27+ penalty".into(),
            format!("{:.1} \u{b7} {sz}", w.w_three_hi),
        ]);
    }
    if w.w_pow2 != 0.0 {
        rows.push([
            "Pow2 penalty".into(),
            format!("{:.1} \u{b7} {sz}", w.w_pow2),
        ]);
    }
    if w.w_off != 0.0 {
        rows.push([
            "Off-family penalty".into(),
            format!("{:.1} \u{b7} {sz}", w.w_off),
        ]);
    }
    if w.w_opt != 0.0 {
        let shape = if w.opt_sqrt {
            "\u{b7} \u{221a}pairs"
        } else {
            "/ pair"
        };
        let bad = if w.w_badpair_frac < 1.0 {
            format!(" (bad pairs {:.2})", w.w_badpair_frac)
        } else {
            String::new()
        };
        rows.push(["Optionality".into(), format!("{:.1} {shape}{bad}", w.w_opt)]);
    }
    if w.w_iso2 != 0.0 {
        rows.push(["Isolated 2s".into(), format!("{:.2} / tile", w.w_iso2)]);
    }
    if w.w_moves != 0.0 {
        rows.push([
            "Chain potential".into(),
            format!("{:.2} / extra cell", w.w_moves),
        ]);
    }
    if w.w_center_big != 0.0 {
        rows.push([
            "Big tile off-border".into(),
            format!("{:.2} \u{b7} ring \u{b7} {sz}", w.w_center_big),
        ]);
    }
    if w.w_center_bad != 0.0 {
        rows.push([
            "Bad tile off-border".into(),
            format!("{:.2} \u{b7} ring \u{b7} {sz}", w.w_center_bad),
        ]);
    }
    if w.w_danger != 0.0 {
        rows.push([
            "Low-option danger".into(),
            format!("{:.2} / (pairs+0.5)", w.w_danger),
        ]);
    }
    if w.w_frag != 0.0 {
        rows.push(["Distinct values".into(), format!("{:.2} / value", w.w_frag)]);
    }
    if w.w_small != 0.0 {
        rows.push(["Spawn material".into(), format!("{:.2} / tile", w.w_small)]);
    }
    if w.w_n1 != 0.0 {
        rows.push(["Reward per 1-tile".into(), format!("{:.2}", w.w_n1)]);
    }
    if w.w_n2 != 0.0 {
        rows.push(["Reward per 2-tile".into(), format!("{:.2}", w.w_n2)]);
    }
    if w.w_n3 != 0.0 {
        rows.push(["Reward per 3-tile".into(), format!("{:.2}", w.w_n3)]);
    }
    if w.w_vpair != 0.0 {
        rows.push([
            "Pending-merge potential".into(),
            format!("{:.2} \u{b7} 2v", w.w_vpair),
        ]);
    }
    if w.w_coal != 0.0 {
        rows.push([
            "Corner coalescing".into(),
            format!("{:.2} \u{b7} dist \u{b7} {sz}", w.w_coal),
        ]);
    }
    if w.w_achv != 0.0 {
        rows.push([
            "Achievable value".into(),
            format!("{:.2} \u{b7} bound", w.w_achv),
        ]);
    }
    if w.w_over != 0.0 {
        rows.push(["Death penalty".into(), format!("{:.0}", w.w_over)]);
    }
    let items: Vec<String> = rows
        .iter()
        .map(|r| format!("[\"{}\",\"{}\"]", r[0], r[1]))
        .collect();
    format!("[{}]", items.join(","))
}

fn class_code(v: u64) -> u8 {
    match classify(v) {
        TileClass::Small => 0,
        TileClass::Smooth => 1,
        TileClass::Pow2 => 2,
        TileClass::ThreeHeavy | TileClass::ThreeHi => 4,
        TileClass::Off => 3,
    }
}

#[no_mangle]
pub extern "C" fn analyze(depth: u32, samples: u32, topk: u32, keep: u32, seed: u32) -> *const u8 {
    let cells_u32 = INPUT.with(|i| *i.borrow());
    let mut cells = [0u64; CELLS];
    for (dst, &src) in cells.iter_mut().zip(cells_u32.iter()) {
        *dst = src as u64;
    }
    let board = Board::from_state(cells, 0, 1);
    let (wname, w) = ("t4", Weights::t4());
    let mut pol = ExpectimaxPolicy::new(
        "analyze",
        w,
        depth.max(1),
        samples.max(1),
        topk.max(1) as usize,
        seed,
    );
    let ranked = pol.ranked(&board, true);

    let mut json = String::from("{\"moves\":[");
    for (i, (mv, ev)) in ranked.iter().take(keep.max(1) as usize).enumerate() {
        if i > 0 {
            json.push(',');
        }
        let v = board.cells[mv.path[0] as usize];
        let path: Vec<String> = mv.path.iter().map(|c| c.to_string()).collect();
        json.push_str(&format!(
            "{{\"path\":[{}],\"ev\":{:.1},\"value\":{},\"sum\":{}}}",
            path.join(","),
            ev,
            v,
            v * mv.path.len() as u64
        ));
    }
    let bd = evaluate_breakdown(&board, &w);
    let classes: Vec<String> = board
        .cells
        .iter()
        .map(|&v| class_code(v).to_string())
        .collect();
    json.push_str(&format!(
        "],\"eval\":{{\"smooth\":{:.1},\"three\":{:.1},\"pow2\":{:.1},\"off\":{:.1},\"opt\":{:.1},\"over\":{:.1},\"pairs\":{},\"total\":{:.1}}},\"classes\":[{}],\"weights\":{{\"name\":\"{}\",\"rows\":{}}}}}",
        bd.smooth, bd.three, bd.pow2, bd.off, bd.opt, bd.over, bd.pairs, bd.total,
        classes.join(","), wname, weight_rows(&w)
    ));
    set_result(json)
}

fn state_json() -> String {
    GAME.with(|g| {
        let g = g.borrow();
        let b = g.as_ref().expect("game_new must be called first");
        let cells: Vec<String> = b.cells.iter().map(|v| v.to_string()).collect();
        let hist = HISTORY.with(|h| h.borrow().len());
        format!(
            "{{\"cells\":[{}],\"score\":{},\"over\":{},\"history\":{}}}",
            cells.join(","),
            b.score,
            u8::from(!b.has_moves()),
            hist
        )
    })
}

#[no_mangle]
pub extern "C" fn game_new(seed: u32) -> *const u8 {
    GAME.with(|g| *g.borrow_mut() = Some(Board::new_game(seed)));
    HISTORY.with(|h| h.borrow_mut().clear());
    set_result(state_json())
}

#[no_mangle]
pub extern "C" fn game_state() -> *const u8 {
    set_result(state_json())
}

#[no_mangle]
pub extern "C" fn game_apply(len: u32) -> u32 {
    let len = len as usize;
    if !(2..=CELLS).contains(&len) {
        return 0;
    }
    let path: Vec<u8> = INPUT.with(|i| i.borrow()[..len].iter().map(|&c| c as u8).collect());
    let ok = GAME.with(|g| {
        let mut g = g.borrow_mut();
        let Some(b) = g.as_mut() else { return false };
        // validate: in range, no repeats, orthogonally chained, equal values
        let mut mask = 0u32;
        let v0 = match path.first().map(|&c| c as usize) {
            Some(c) if c < CELLS => b.cells[c],
            _ => return false,
        };
        for (k, &c) in path.iter().enumerate() {
            let c = c as usize;
            if c >= CELLS || mask & (1 << c) != 0 || b.cells[c] != v0 {
                return false;
            }
            if k > 0 {
                let prev = path[k - 1] as usize;
                let mut adj = false;
                neighbors(prev, |nb| {
                    if nb == c {
                        adj = true;
                    }
                });
                if !adj {
                    return false;
                }
            }
            mask |= 1 << c;
        }
        HISTORY.with(|h| h.borrow_mut().push(b.clone()));
        b.apply(&Move { path });
        true
    });
    u32::from(ok)
}

#[no_mangle]
pub extern "C" fn game_undo() -> u32 {
    let restored = HISTORY.with(|h| h.borrow_mut().pop());
    match restored {
        Some(prev) => {
            GAME.with(|g| *g.borrow_mut() = Some(prev));
            1
        }
        None => 0,
    }
}
