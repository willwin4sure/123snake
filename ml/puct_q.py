#!/usr/bin/env python3
"""Root-PUCT test-time compute for the distilled net.

At each decision state, enumerate all legal moves (paths, len 2..6).
Priors come from the factored policy: P(start) * prod P(dir | prefix)
* P(finish | path), with every prefix evaluated in one batched forward.
Each simulation samples a refill of the vacated cells and scores the
child decision board with the value head; Q(m) = sum(m) + E[2^v - 1].
Visits follow PUCT with min-max-normalised Q. Plays argmax visits.

Usage: puct_q.py [games] [ckpt] [sims] [cpuct] [ch] [blocks] [kmode]
                 [hmode] [seed0]
"""

import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "ml")
from train_q2 import (CELLS, DELTAS, FINISH, N, Net, dir_mask, embed_cells,
                      start_mask)

DIR_OF = {}
for d, (dr, dc) in DELTAS.items():
    DIR_OF[dr * N + dc] = d


def enumerate_moves(cells):
    cand = []
    for i in range(CELLS):
        r, c = divmod(i, N)
        ok = False
        for dr, dc in DELTAS.values():
            rr, cc = r + dr, c + dc
            if 0 <= rr < N and 0 <= cc < N and cells[rr * N + cc] == cells[i]:
                ok = True
                break
        if not ok:
            continue
        stack = [[i]]
        while stack:
            path = stack.pop()
            if len(path) >= 2:
                cand.append(list(path))
            if len(path) >= 6:
                continue
            h = path[-1]
            hr, hc = divmod(h, N)
            for dr, dc in DELTAS.values():
                rr, cc = hr + dr, hc + dc
                nb = rr * N + cc
                if (0 <= rr < N and 0 <= cc < N and nb not in path
                        and cells[nb] == cells[i]):
                    stack.append(path + [nb])
    return cand


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def priors(net, dev, cells, cand):
    prefixes = set()
    for path in cand:
        for ln in range(1, len(path) + 1):
            prefixes.add(tuple(path[:ln]))
    prefixes = sorted(prefixes, key=lambda p: (len(p), p))
    emb = embed_cells(cells)
    planes = np.zeros((len(prefixes) + 1, 14, N, N), dtype=np.float32)
    planes[:, :9] = emb
    for bi, pref in enumerate(prefixes, start=1):
        for p in pref:
            planes[bi, 9, p // N, p % N] = 1.0
        h = pref[-1]
        planes[bi, 10, h // N, h % N] = 1.0
        planes[bi, 11] = 1.0
    with torch.no_grad():
        start, dirs, _ = net(torch.from_numpy(planes).to(dev))
    sl = start[0].cpu().numpy()
    sm = start_mask(cells)
    sl[~sm] = -1e9
    sp = softmax(sl)
    dirs = dirs.cpu().numpy()
    dird = {}
    for bi, pref in enumerate(prefixes, start=1):
        h = pref[-1]
        dl = dirs[bi, h].copy()
        dm = dir_mask(cells, list(pref))
        dl[~dm] = -1e9
        dird[pref] = softmax(dl)
    ps = []
    for path in cand:
        p = sp[path[0]]
        for t in range(1, len(path)):
            pref = tuple(path[:t])
            d = DIR_OF[path[t] - path[t - 1]]
            p *= dird[pref][d]
        p *= dird[tuple(path)][FINISH]
        ps.append(p)
    ps = np.array(ps, dtype=np.float64)
    s = ps.sum()
    if s < 1e-12:
        return np.full(len(cand), 1.0 / len(cand))
    return ps / s


def value_children(net, dev, boards):
    planes = np.zeros((len(boards), 14, N, N), dtype=np.float32)
    for bi, bc in enumerate(boards):
        planes[bi, :9] = embed_cells(bc)
    with torch.no_grad():
        _, _, vals = net(torch.from_numpy(planes).to(dev))
    return vals.cpu().numpy().reshape(-1)


def puct_move(net, dev, cells, cand, pri, rng, sims, cpuct, batch=48):
    nc = len(cand)
    sums = [cells[path[0]] * len(path) for path in cand]
    Nv = np.zeros(nc)
    W = np.zeros(nc)
    done = 0
    while done < sims:
        b = min(batch, sims - done)
        ntmp = Nv.copy()
        picks = []
        for _ in range(b):
            if (Nv > 0).any():
                qbar = np.where(Nv > 0, W / np.maximum(Nv, 1e-9), 0.0)
                lo, hi = qbar[Nv > 0].min(), qbar[Nv > 0].max()
                qn = np.where(Nv > 0, (qbar - lo) / max(hi - lo, 1e-9), 0.5)
            else:
                qn = np.full(nc, 0.5)
            u = qn + cpuct * pri * np.sqrt(ntmp.sum() + 1) / (1 + ntmp)
            i = int(np.argmax(u))
            picks.append(i)
            ntmp[i] += 1
        boards = []
        for i in picks:
            path = cand[i]
            nb = list(cells)
            for p in path[:-1]:
                nb[p] = rng.randint(1, 3)
            nb[path[-1]] = sums[i]
            boards.append(nb)
        vals = value_children(net, dev, boards)
        for i, v in zip(picks, vals):
            Nv[i] += 1
            W[i] += sums[i] + (2.0 ** float(v) - 1.0)
        done += b
    best = int(np.argmax(Nv))
    ties = np.flatnonzero(Nv == Nv[best])
    if len(ties) > 1:
        qbar = np.where(Nv > 0, W / np.maximum(Nv, 1e-9), -1e18)
        best = int(ties[np.argmax(qbar[ties])])
    return best


def puct_game(net, dev, seed, sims, cpuct):
    rng = random.Random(seed)
    cells = [rng.randint(1, 3) for _ in range(CELLS)]
    score = 0
    moves = 0
    while True:
        cand = enumerate_moves(cells)
        if not cand:
            break
        pri = priors(net, dev, cells, cand)
        best = puct_move(net, dev, cells, cand, pri, rng, sims, cpuct)
        path = cand[best]
        sm_sum = cells[path[0]] * len(path)
        for p in path[:-1]:
            cells[p] = rng.randint(1, 3)
        cells[path[-1]] = sm_sum
        score += sm_sum
        moves += 1
        if moves > 3000:
            break
    return score, moves


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "ml/qnet-q128x8-trieval-dh.pt"
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    cpuct = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
    ch = int(sys.argv[5]) if len(sys.argv) > 5 else 128
    blocks = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    kmode = sys.argv[7] if len(sys.argv) > 7 else "k3"
    hmode = sys.argv[8] if len(sys.argv) > 8 else "d"
    seed0 = int(sys.argv[9]) if len(sys.argv) > 9 else 12000
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = Net(ch=ch, blocks=blocks, kmode=kmode, hmode=hmode).to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev))
    net.eval()
    t0 = time.time()
    scores = []
    tot_moves = 0
    for g in range(games):
        sc, mv = puct_game(net, dev, seed0 + g, sims, cpuct)
        scores.append(sc)
        tot_moves += mv
    el = time.time() - t0
    scores.sort()
    n = len(scores)
    print(f"puct sims={sims} c={cpuct} n={n}  mean {sum(scores)/n:.1f}  "
          f"p50 {scores[n//2]}  p90 {scores[9*n//10]}  max {scores[-1]}  "
          f"| moves={tot_moves} ms/mv={el*1000/max(tot_moves,1):.1f}")


if __name__ == "__main__":
    main()
