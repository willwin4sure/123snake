#!/usr/bin/env python3
"""Let the distilled network play 123 Snake greedily (no search) and report
its score distribution. Reference points from the Rust harness (500 games):
random ~92, greedy-t4 (1-ply hand eval) ~427, teacher exp:d2:s12:k16:t4 ~963.

Usage: .venv/bin/python ml/play_net.py [n_games] [checkpoint]
"""

import math
import random
import sys

import numpy as np
import torch

from train_distill import (
    CELLS,
    DELTAS,
    FINISH,
    N,
    Net,
    dir_mask,
    embed_cells,
    start_mask,
)


def new_board(rng):
    while True:
        cells = [rng.randint(1, 3) for _ in range(CELLS)]
        if has_moves(cells):
            return cells


def has_moves(cells):
    for r in range(N):
        for c in range(N):
            i = r * N + c
            if c + 1 < N and cells[i] == cells[i + 1]:
                return True
            if r + 1 < N and cells[i] == cells[i + N]:
                return True
    return False


@torch.no_grad()
def pick_move(net, cells, device):
    planes = np.zeros((1, 14, N, N), dtype=np.float32)
    planes[0, :9] = embed_cells(cells)
    sl, _, _ = net(torch.from_numpy(planes).to(device))
    sm = torch.from_numpy(start_mask(cells)).to(device)
    if not bool(sm.any()):
        return None
    first = int(sl[0].masked_fill(~sm, -1e9).argmax())
    prefix = [first]
    for _ in range(CELLS):
        planes[0, 9:12] = 0.0
        for pcell in prefix:
            planes[0, 9, pcell // N, pcell % N] = 1.0
        h = prefix[-1]
        planes[0, 10, h // N, h % N] = 1.0
        planes[0, 11] = 1.0
        _, dl, _ = net(torch.from_numpy(planes).to(device))
        dm = torch.from_numpy(dir_mask(cells, prefix)).to(device)
        if not bool(dm.any()):
            return None
        d = int(dl[0, :, h].masked_fill(~dm, -1e9).argmax())
        if d == FINISH:
            return prefix
        dr, dc = DELTAS[d]
        prefix.append((h // N + dr) * N + (h % N + dc))
    return prefix if len(prefix) >= 2 else None


def play(net, seed, device, cap=300):
    rng = random.Random(seed)
    cells = new_board(rng)
    score = 0
    for _ in range(cap):
        if not has_moves(cells):
            break
        mv = pick_move(net, cells, device)
        if mv is None or len(mv) < 2:
            break
        v = cells[mv[0]]
        total = v * len(mv)
        for ci in mv[:-1]:
            cells[ci] = rng.randint(1, 3)
        cells[mv[-1]] = total
        score += total
    return score


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "ml/net-v0.pt"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = Net().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    scores = sorted(play(net, 1000 + i, device) for i in range(n))
    arr = np.array(scores, dtype=np.float64)
    print(
        f"net-only greedy play, {n} games: mean {arr.mean():.1f}  "
        f"p50 {int(np.percentile(arr, 50))}  p90 {int(np.percentile(arr, 90))}  "
        f"max {scores[-1]}"
    )


if __name__ == "__main__":
    main()
