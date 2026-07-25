# Ideas log — n-tuple program

Living log of open ideas, running experiments, and tested negatives.
(Heuristic-era results live in README.md and the wave summaries in
results/; this file tracks the research frontier.)

## Open (untested)

- **CNN + n-tuple hybrid** (2026-07-23): V = n-tuple tables + small CNN
  residual, trained jointly by TD. Tables give cheap bulk value; the CNN
  learns global geometry the tuples can't see (our hand-built globals —
  blob/path/eqpairs — are special cases of what a conv stack could learn).
  Cost concern: even a tiny CNN is ~25-100x a table eval on CPU, so
  usable as (a) training-time head distilled back into tables/globals,
  (b) leaf evaluator for search only, or (c) inference if quantized and
  genuinely tiny. Literature: n-tuples (Szubert/Jaskowski/Matsuzaki) and
  CNN agents (incl. stochastic MuZero on 2048) exist separately; a
  joint tables+conv-residual TD net appears rare/unexplored.
- **run42 shape family** (after run42's +263): variants of the
  line-plus-support motif — 3+2, 5+2 (if it fits), 4+3, double-flanked
  4-run, offset variants; positional (per-orbit) run42 tables.
- **Memory-layout/throughput levers** (2026-07-24, MEASURED via
  `snake membench`): realistic 30:1 read:update mix is layout-invariant
  (3.9 ns/op SoA = AoS = prefetch; OoO already hides latency) — AoS
  refactor and prefetch NOT worth it (AoS only wins 33% on pure
  updates). DONE: --threads N parallel eval (5.4x); --train-threads N
  hogwild trainer VALIDATED: 4.2x at 4 threads (perfect), 6.8x at 8
  (85% eff), zero quality loss (1822/1837/1820 at 200k games). Not viable: hugepages on
  macOS/ARM; f16 for TD.
- **TDLeaf-style fine-tune**: search-in-training; update at PV leaves /
  realized-path leaves toward search-backed targets. Cheap version:
  fine-tune a champion a few 100k games with 4x27-class search.
- **Exact-chance-node TD targets**: bootstrap V(afterstate) against the
  enumerated (or without-replacement) expectation over refills of the
  best-reply backup instead of the single realized refill. Reachable
  states only; unbiased; large target-variance reduction. The surviving
  descendant of the refill-consistency thread.
- **Coherent-head training**: use the leave-one-out lift LV' as the value
  head during training (coherence as structure, not post-hoc surgery);
  or the least-squares projection of V onto range(L) as distillation.
  Deprioritized after post-hoc variants failed (see negatives).
- **Symmetry/commitment features**: QuadSpread (top-4 tiles: # distinct
  corner quadrants x tier), QuadProfile (sorted per-quadrant max tiers),
  AnchorGradient (# adjacent pairs downhill-from-anchor x tier),
  LadderChain (longest adjacent doubling chain x tier — the endgame
  merge mechanism made visible). Motivated by DispAvg decode: endgame
  wants concentration, midgame tolerates spread; policy still plays
  four corners (see AS1 running).
- **eqpairs x tier**: phase-cross the strongest global curve (+601 span,
  currently phase-blind) to see if liquidity preference flips late.
- **d3 probe at the d2 optimum**: 8:96:8:96 (and asymmetric root-heavy
  variants) now that small-k/deep-s is established.
- **Compressed blob-path cartesians**: tier-compressed BP22 (~1.7M),
  sizes-only B3P3 (46k), asymmetric full-top + size-only-rest (~15M).
  Full B3xP3 infeasible (108^6). NOTE: A19 showed blob3 marginal over
  path3 is +5 — paths subsume blobs, so these are low priority.
- **Fold sub-tuples into covering tables at deployment**: exact rewrite;
  only 2x2 -> pos-2x3 qualifies in current zoo (~10% of lookups). Parked.
- **MCTS revisit** once V is search-calibrated (old finding: loses to
  expectimax under uncalibrated V).
- **Rerun TTC cartesian on each new champion** (AC1 sweep running; A3
  optimum was 8x96; KE10 gained far more from search than A3 — search
  responsiveness varies per net and must be re-measured).

## Running

- AC-10M ladder: AC1 (pos23+bigL+stair6), AC2 (+eqpairs), AC3
  (+eqpairs+blob2+path2), AC4 (blob3+path3+tiny, no eqpairs). Marginal
  reads: AC2-AC1 = eqpairs; AC3-AC2 = blob2+path2; AC4 vs AC3.
- A20-blob4 / A21-path4 / A22-bp22 (136M-entry cartesian globals),
  A23-tinytuples (pair2/line3/smallL3/line4, 352 images),
  A24-run42 (4-in-line + adjacent 2-run, 2 shared m^6 tables).

## Negative results (tested)

- **Commitment exploration (AS1, 2026-07-23)**: -25 x avg-dist selection
  bonus annealed over 2.5M/5M on the AC1 recipe -> 2707 @30k vs AC1
  2693: PARITY. Exploration tax fully recovered, +900k extra nonzero
  entries explored, but committed play showed no exploitable value at
  greedy horizon. Four-corners stands (for now); suspicion shifts to
  horizon (search-in-training). Steering is free at this dose — c=50+
  or ladder-adjacency steering remain cheap follow-ups.

- **Refill-consistency surgery** (2026-07-23): hard projection (pending
  entry := mean of refinements) -128 @5.6sigma greedy, neutral under
  search; symmetric Kaczmarz smoothing worse (~-570 at n=100). Cause
  (user diagnosis): PENDING is an observable — "the chain passed HERE" —
  so pending vs exact entries condition on different events; equating
  them pools across contexts. The correctly-conditioned constraint is
  TD itself. V* leave-one-out lift (--star): -55 (ns) at n=3000; parity
  at best.
- **Value-blind globals are nulls**: blobtier 2534, pathtier 2531,
  freefield 2544, avgdisp 2521, gated12 2523 (all ~baseline 2540 @30k).
  Value-aware versions of the same ideas win (blobalpha +46, blob2 +130,
  path2 +145, path3 +183). Lesson: cross size/shape with VALUE, not tier.
- **X-shape tuple**: 2532 @30k — null (n=1000's "harmful" was noise).
- **slim89 (exact 8/9 codes)**: 2541 — null; buckets already carry it.
- **blob3 on top of path3**: +5 marginal (2728 vs 2723) — paths subsume
  blobs; don't spend params on blob variants when paths present.
- **10M vs 5M at pos23 level**: 2650 vs 2641 — plateau by 5M for slim
  single-feature nets (combos may differ; AC-10M ladder testing).
- **OTD decaying bonus (O10)**: 2166 — far below baseline. **Baked
  optimism (L10)**: 2541 @30k = exact parity (n=1000's 2613 was noise);
  also poisons TC accumulators mechanically. **Staging (M10 promote /
  gated)**: 2516/2556 — at or below plain within noise.
- **Wide/starved search**: top-k > 8-12 loses (winner's curse); s=9
  column worst everywhere; 32x96 dominated at 4x cost of 16x48.
- **Sampled (with-replacement) chance nodes**: exact enum / without-
  replacement worth +21% strength AND 5.5x speed at 16:48 — noise was
  costing ~20% of playing strength.
- **Wave-1**: diagonals, TD(lambda=.8), 3-stage zero-init all finish
  below base at plateau (image dilution / trace smear) despite early
  acceleration.
- n=1000 benchmarks unreliable below ~80-point effects (three verdicts
  flipped at 30k). Judge at 30k (SEM ~16).
