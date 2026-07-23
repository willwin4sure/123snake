# AC combo runs (wave 5)

All runs share the KE recipe: slim alphabet, rows + 2x2 + plus + 5-cell
staircase tuples, global top-2 pair, TC alpha 2.0, eps-greedy 3% rank +
1% random, eval every 100k on 3000 seeds at 500000+, training seeds
1e8+, 10M games, NTV9 checkpoints.

Common flags: `--games 10000000 --alphabet slim --staircase --global
--pos-2x3 --alpha 2.0 --explore 0.03:0.01 --eval-every 100000
--eval-games 3000`

| run | --extra | extra bits | features beyond base+pos23 |
|---|---|---|---|
| AC1-10M | bigL,stair6 | 5 | big L (6x 5-tuples), 6-cell staircases (3x 6-tuples) |
| AC2-10M | bigL,stair6,eqpairs | 517 | AC1 + adjacent-equal-pairs count (25 entries) |
| AC3-10M | bigL,stair6,eqpairs,blob2,path2 | 4741 | AC2 + top-2 blobs (size x value)^2 + top-2 disjoint paths (len x value)^2 |
| AC4-10M | bigL,stair6,blob3,path3,tiny | 286725 | AC1 + top-3 blobs + top-3 paths (stacked, (6m)^3 each) + tiny tuples (pair2/line3/smallL3/line4 shared tables), NO eqpairs |

Marginal reads: AC2-AC1 = eqpairs; AC3-AC2 = blob2+path2; AC4 vs AC3 =
{blob3+path3+tiny} vs {eqpairs+blob2+path2}.

Reference points (5M, 30k-game benchmarks, seeds 15000+): slim baseline
A1 2540; single features path2 2685, blob3 2702, blob2 2670, pos23 2641,
eqpairs 2639, bigL 2607, stair6 2578; AC1-triple @5M = 2693.
Superseded 5M runs: AC1-triple (kept, benchmarked), AC2/AC3/AC4 5M
killed mid-run 2026-07-23 when the line moved to 10M.
