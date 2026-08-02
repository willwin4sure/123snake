# Wave 7: trimmed baseline + seeded feature arms

Recipe changes vs wave 6: eqpairs REMOVED (tiny tuples subsume);
carousel 0.25 in every run; evals 10k games every 200k (SEM ~30).
Shared-table audit: only tiny families share tables (by design);
run42pos and all other tuple families are positional.

## Phase 1 — tiny race (2.5M fresh each, sequential 8T)
- W7-base: trimmed recipe (bigL,stair6,blob3,path3,tiny,run42pos,t321,fish)
- W7-tiny2: + diag2/T4/S4/L4 (shared per family, +232 images)
- Verdict: last-3-eval means; tiny2 wins ties (and any gap <= 25).
- Winner snapshot -> ml/ntuple-W7-seed.bin

## Phase 2 — machinery (validated)
- --seed-from: signature-matched table transplant (exact-eval validated)
- chain-12 alphabet: slim->12 digit remap preserving merge equality
  through 48 (CH12_OF_SLIM)
- hash8: FNV-1a into 2^26 slots, corner-anchored

## Phase 3 — seeded arms (+2.5M each from the seed, sequential 8T)
| arm | adds | new params |
|---|---|---|
| W7-cont | nothing (control) | 0 |
| W7-hex6 | hook6 (6 orbits) + zig6 (2) | +272M |
| W7-l44 | corner L44, 3 orbits @ 12^7 | +107M |
| W7-stair7 | stair7, 3 orbits @ 12^7 | +107M |
| W7-hash8 | hashed corner 8-tuple | +67M |

Verdict per arm: final-1M mean eval vs W7-cont; 30k bench for winners.
Dash: port 8279 (fill %, delta vs matched control).
