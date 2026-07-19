//! Behavioral probes for specific gameplay scenarios.

use integer_snake::game::{idx, Board, CELLS};
use integer_snake::heuristics::Weights;
use integer_snake::search::ExpectimaxPolicy;

/// A 2x2 block of 3s on an otherwise inert board: pair-then-pair-then-merge
/// scores 24 total vs 12 for the single 4-chain. Probe which route the
/// policy's top move starts.
#[test]
fn two_by_two_square_choice() {
    let mut cells = [0u64; CELLS];
    for (i, c) in cells.iter_mut().enumerate() {
        *c = 1000 + 7 * i as u64; // unique inert junk, no accidental pairs
    }
    for &i in &[idx(1, 1), idx(1, 2), idx(2, 1), idx(2, 2)] {
        cells[i] = 3;
    }
    let b = Board::from_state(cells, 0, 1);
    let mut pol = ExpectimaxPolicy::new("t2", Weights::t2(), 2, 12, 16, 42);
    let ranked = pol.ranked(&b, false);
    assert_eq!(
        ranked[0].0.path.len(),
        2,
        "t2 must start the 2x2 with a pair merge, not a long chain"
    );
}

/// The 1-flood board from a real game (13 ones in one component). The old
/// shared DFS budget was exhausted by the flood component, starving move
/// enumeration for later cells; per-start budget slices must enumerate the
/// far 2-2 pair, and the whole thing must stay fast.
#[test]
fn flood_board_movegen_is_fair_and_bounded() {
    let vals: [u64; 25] = [
        2, 3, 12, 1, 48, //
        3, 1, 1, 1, 1, //
        1, 1, 1, 1, 1, //
        1, 1, 1, 2, 2, //
        24, 1, 3, 2, 12,
    ];
    let b = Board::from_state(vals, 0, 1);
    let start = std::time::Instant::now();
    let moves = b.legal_moves();
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < 500, "movegen too slow: {elapsed:?}");
    assert!(!moves.is_empty());
    // the 2-2 pair at cells 18/19 sits late in scan order — it must be found
    let has_far_pair = moves
        .iter()
        .any(|m| m.path.len() == 2 && m.path.contains(&18) && m.path.contains(&19));
    assert!(
        has_far_pair,
        "far 2-2 pair was starved out of move generation"
    );
}
