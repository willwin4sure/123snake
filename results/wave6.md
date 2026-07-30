# Wave 6: 2048-paper ideas on the AC7 base (10M, hogwild 8T, sequential)

All runs: exact AC7 config (616 images, 1.09B params) + one intervention.
Benchmarks @30k seeds 15000+ appended to wave6-bench.txt on completion.
Champion reference: AC7-10M @30k = 3644.

| run | intervention | rationale (Jaskowski 2018 / Szubert-Jaskowski) |
|---|---|---|
| AC7-base2 | none | reproducibility baseline for the wave |
| AC7-lam05 | --lambda 0.5 (Watkins-cut traces, len 16) | TC(lambda) was the single biggest training-speed win in the 2048 paper |
| AC7-carousel | --carousel 0.25 (restart from snapshots at max-tile 192/768, ring 2048/worker) | late-game states are undertrained; restart shaping oversamples them |

Queued candidates (need decisions/work): 2-stage weight promotion (doubles
table memory to 26GB), optimistic init replacing eps-greedy, exact
chance-node TD targets (equal-time comparison), delayed/batched TC updates.
