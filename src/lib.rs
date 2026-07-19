//! Engine, heuristics, and tree-search solvers for 123 Snake (Integer Snake).
//!
//! The game: a 5x5 grid of numbered tiles. A move traces a self-avoiding
//! orthogonal path through tiles that all share one value; the path collapses
//! into its sum on the final cell, and every vacated cell refills in place
//! with a uniform random 1, 2, or 3. Score is the cumulative sum of merged
//! tiles. The game ends when no two adjacent tiles are equal.
//!
//! The RNG is mulberry32, bit-compatible with the web build, so a (seed, move
//! sequence) pair replays identically in both implementations.

pub mod eval;
pub mod game;
pub mod heuristics;
pub mod search;
#[cfg(target_arch = "wasm32")]
pub mod wasm_api;
