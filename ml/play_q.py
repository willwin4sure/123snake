#!/usr/bin/env python3
"""Play 123 Snake with the distilled policy net only (no search): greedy
factored policy — argmax masked start cell, then argmax dir/finish.

Usage: .venv/bin/python ml/play_q.py [games] [ml/qnet.pt]
"""

import random
import sys

import numpy as np
import torch

sys.path.insert(0, "ml")
from train_q2 import (CELLS, DELTAS, FINISH, N, Net, dir_mask, embed_cells,
                      start_mask)


def legal_start(cells, i):
    r, c = divmod(i, N)
    for dr, dc in DELTAS.values():
        rr, cc = r + dr, c + dc
        if 0 <= rr < N and 0 <= cc < N and cells[rr * N + cc] == cells[i]:
            return True
    return False


def play_game(net, dev, seed, states=None):
    rng = random.Random(seed)
    cells = [rng.randint(1, 3) for _ in range(CELLS)]
    score = 0
    moves = 0
    while True:
        sm = start_mask(cells)
        if not sm.any():
            break
        if states is not None:
            states.append(list(cells))
        planes = np.zeros((1, 14, N, N), dtype=np.float32)
        planes[0, :9] = embed_cells(cells)
        with torch.no_grad():
            start, dirs, _ = net(torch.from_numpy(planes).to(dev))
        sl = start[0].cpu().numpy()
        sl[~sm] = -1e9
        path = [int(np.argmax(sl))]
        for _ in range(24):
            planes2 = np.zeros((1, 14, N, N), dtype=np.float32)
            planes2[0, :9] = embed_cells(cells)
            for p in path:
                planes2[0, 9, p // N, p % N] = 1.0
            h = path[-1]
            planes2[0, 10, h // N, h % N] = 1.0
            planes2[0, 11] = 1.0
            with torch.no_grad():
                _, dirs2, _ = net(torch.from_numpy(planes2).to(dev))
            dl = dirs2[0, h].cpu().numpy()
            dm = dir_mask(cells, path)
            dl[~dm] = -1e9
            if not dm.any():
                break
            d = int(np.argmax(dl))
            if d == FINISH:
                break
            dr, dc = DELTAS[d]
            path.append((h // N + dr) * N + (h % N + dc))
        if len(path) < 2:
            # net picked a dead start; fall back to any legal pair
            done = False
            for i in range(CELLS):
                if sm[i]:
                    r, c = divmod(i, N)
                    for dr, dc in DELTAS.values():
                        rr, cc = r + dr, c + dc
                        if (0 <= rr < N and 0 <= cc < N
                                and cells[rr * N + cc] == cells[i]):
                            path = [i, rr * N + cc]
                            done = True
                            break
                if done:
                    break
            if not done:
                break
        v = cells[path[0]]
        sm_sum = v * len(path)
        for p in path[:-1]:
            cells[p] = rng.randint(1, 3)
        cells[path[-1]] = sm_sum
        score += sm_sum
        moves += 1
        if moves > 3000:
            break
    return score, moves


def value_greedy_game(net, dev, seed, k=5):
    rng = random.Random(seed)
    cells = [rng.randint(1, 3) for _ in range(CELLS)]
    score = 0
    moves = 0
    while True:
        cand = []
        for i in range(CELLS):
            if not legal_start(cells, i):
                continue
            stack = [[i]]
            while stack:
                path = stack.pop()
                if len(path) >= 2:
                    cand.append(list(path))
                if len(path) >= 6:
                    continue
                h = path[-1]
                r, c = divmod(h, N)
                for dr, dc in DELTAS.values():
                    rr, cc = r + dr, c + dc
                    nb = rr * N + cc
                    if (0 <= rr < N and 0 <= cc < N and nb not in path
                            and cells[nb] == cells[i]):
                        stack.append(path + [nb])
        if not cand:
            break
        boards = []
        meta = []
        for path in cand:
            v = cells[path[0]]
            sm_sum = v * len(path)
            for _ in range(k):
                nc = cells[:]
                for p in path[:-1]:
                    nc[p] = rng.randint(1, 3)
                nc[path[-1]] = sm_sum
                boards.append(nc)
            meta.append((path, sm_sum))
        planes = np.zeros((len(boards), 14, N, N), dtype=np.float32)
        for bi, bc in enumerate(boards):
            planes[bi, :9] = embed_cells(bc)
        with torch.no_grad():
            _, _, vals = net(torch.from_numpy(planes).to(dev))
        vals = vals.cpu().numpy().reshape(len(cand), k).mean(axis=1)
        qs = [m[1] + (2.0 ** vv - 1.0) for m, vv in zip(meta, vals)]
        best = int(np.argmax(qs))
        path, sm_sum = meta[best]
        for p in path[:-1]:
            cells[p] = rng.randint(1, 3)
        cells[path[-1]] = sm_sum
        score += sm_sum
        moves += 1
        if moves > 3000:
            break
    return score, moves


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "ml/qnet.pt"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ch = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    blocks = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    kmode = sys.argv[6] if len(sys.argv) > 6 else "k3"
    hmode = sys.argv[7] if len(sys.argv) > 7 else "s"
    net = Net(ch=ch, blocks=blocks, kmode=kmode, hmode=hmode).to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev))
    net.eval()
    mode = sys.argv[3] if len(sys.argv) > 3 else "policy"
    states_out = sys.argv[8] if len(sys.argv) > 8 else None
    seed0 = int(sys.argv[9]) if len(sys.argv) > 9 else 12000
    if states_out:
        import json
        import os
        import time
        lf = (open(os.environ["PROG_LOG"], "a", buffering=1)
              if os.environ.get("PROG_LOG") else None)
        t0 = time.time()
        states = []
        scores = []
        for g in range(games):
            sc, mv = play_game(net, dev, seed0 + g, states)
            scores.append(sc)
            if lf:
                lf.write(f"PROG {g + 1} {sc} {mv} {time.time() - t0:.1f}\n")
        with open(states_out, "w") as f:
            for c in states:
                f.write(json.dumps({"cells": c}) + "\n")
        print(f"dumped {len(states)} decision states to {states_out}")
    else:
        fn = value_greedy_game if mode == "value" else play_game
        import os
        import time
        lf = (open(os.environ["PROG_LOG"], "a", buffering=1)
              if os.environ.get("PROG_LOG") else None)
        t0 = time.time()
        scores = []
        for g in range(games):
            sc, mv = fn(net, dev, 12000 + g)
            scores.append(sc)
            if lf:
                lf.write(f"PROG {g + 1} {sc} {mv} {time.time() - t0:.1f}\n")
    scores.sort()
    n = len(scores)
    print(f"net-{mode} n={n}  mean {sum(scores)/n:.1f}  "
          f"p50 {scores[n//2]}  p90 {scores[9*n//10]}  max {scores[-1]}")


if __name__ == "__main__":
    main()
