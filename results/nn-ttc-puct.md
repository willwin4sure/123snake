# NN test-time compute: root-PUCT sweep (2026-07-29)

Checkpoint q128x8-trieval-dh (policy 755 mean/100 games). 100 games per
arm, shared seeds 12000+. PUCT: root visits over the full legal move
set, priors from the factored policy (one batched forward over the move
trie), Q from refill-sampled value-head child evals.

| arm | mean | ms/move |
|---|---|---|
| policy only        | 755 | -    |
| value-greedy k=5   | 230 | -    |
| puct 32 sims       | 869 | 12.7 |
| puct 128 sims      | 392 | 24.9 |
| puct 512 sims (93g)| ~440| 71   |

Verdict: the sims-scaling curve INVERTS. 32 sims (+15%) keeps the
prior in charge; 128+ lets Q estimates dominate and PUCT hunts down
the value head's off-manifold optimism (same failure as value-greedy's
collapse). Search amplifies value error. Fix is value robustness
(DAgger), not more compute.
