#!/usr/bin/env python3
"""Supervised distillation of the hand policy (t4 expectimax) into a
policy+value network, proof-of-concept scale.

Action space is click-level submoves (matching the game's tap mode, no
take-backs):
  - first click: one of 25 cells, masked to cells with an equal neighbor
  - then repeatedly: one of 5 actions (up/down/left/right/finish), masked to
    legal extensions; finish only once the path has >= 2 cells

Network: shared conv trunk on 14 input planes (value embedding: log2, prime
exponents of 2/3/5/7, off-residual flag, one-hots for 1/2/3; plus path planes:
visited, head; plus an in-path flag), with three heads:
  - start head: per-cell logit (used when the path is empty)
  - direction head: per-cell 5 logits, gathered at the current head cell
  - value head: predicts log2(1 + remaining score) on clean states

Symmetrization: a random dihedral transform is applied per batch.

Usage: .venv/bin/python ml/train_distill.py [data.jsonl] [epochs]
"""

import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 5
CELLS = 25
UP, DOWN, LEFT, RIGHT, FINISH = 0, 1, 2, 3, 4
DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# ---------------------------------------------------------------- embeddings

def cell_channels(v: int) -> list:
    if v <= 0:
        return [0.0] * 9
    e2 = e3 = e5 = e7 = 0
    m = v
    while m % 2 == 0:
        m //= 2
        e2 += 1
    while m % 3 == 0:
        m //= 3
        e3 += 1
    while m % 5 == 0:
        m //= 5
        e5 += 1
    while m % 7 == 0:
        m //= 7
        e7 += 1
    return [
        math.log2(v) / 10.0,
        e2 / 8.0,
        e3 / 5.0,
        e5 / 3.0,
        e7 / 3.0,
        1.0 if m > 1 else 0.0,
        1.0 if v == 1 else 0.0,
        1.0 if v == 2 else 0.0,
        1.0 if v == 3 else 0.0,
    ]


EMB = {}


def embed_cells(cells) -> np.ndarray:
    """9 x 5 x 5 float32 planes for the board values."""
    out = np.zeros((9, N, N), dtype=np.float32)
    for i, v in enumerate(cells):
        ch = EMB.get(v)
        if ch is None:
            ch = cell_channels(v)
            EMB[v] = ch
        out[:, i // N, i % N] = ch
    return out


def start_mask(cells) -> np.ndarray:
    m = np.zeros(CELLS, dtype=bool)
    for i in range(CELLS):
        r, c = divmod(i, N)
        for dr, dc in DELTAS.values():
            rr, cc = r + dr, c + dc
            if 0 <= rr < N and 0 <= cc < N and cells[rr * N + cc] == cells[i]:
                m[i] = True
                break
    return m


def dir_mask(cells, prefix) -> np.ndarray:
    """Legal 5-way mask given a nonempty path prefix."""
    m = np.zeros(5, dtype=bool)
    head = prefix[-1]
    r, c = divmod(head, N)
    v = cells[prefix[0]]
    for d, (dr, dc) in DELTAS.items():
        rr, cc = r + dr, c + dc
        nb = rr * N + cc
        if 0 <= rr < N and 0 <= cc < N and cells[nb] == v and nb not in prefix:
            m[d] = True
    if len(prefix) >= 2:
        m[FINISH] = True
    return m


# ------------------------------------------------------------------ symmetry

def build_transforms():
    """8 dihedral transforms as cell-index permutations: perm[t][old] = new."""
    perms = []
    base = np.arange(CELLS).reshape(N, N)
    for t in range(8):
        g = np.rot90(base, t % 4)
        if t >= 4:
            g = np.flip(g, axis=1)
        perm = np.zeros(CELLS, dtype=np.int64)
        for r in range(N):
            for c in range(N):
                perm[g[r, c]] = r * N + c
        perms.append(perm)
    return perms


PERMS = build_transforms()


def transform_planes(x: torch.Tensor, t: int) -> torch.Tensor:
    x = torch.rot90(x, t % 4, dims=(-2, -1))
    if t >= 4:
        x = torch.flip(x, dims=(-1,))
    return x


def transform_dir(d: int, t: int) -> int:
    if d == FINISH:
        return FINISH
    dr, dc = DELTAS[d]
    # transform two points and rederive the delta
    a, b = 2 * N + 2, (2 + dr) * N + (2 + dc)  # around center, always in range
    pa, pb = PERMS[t][a], PERMS[t][b]
    ndr, ndc = pb // N - pa // N, pb % N - pa % N
    for dd, (xr, xc) in DELTAS.items():
        if (xr, xc) == (ndr, ndc):
            return dd
    raise AssertionError


DIR_MAP = [[transform_dir(d, t) for d in range(5)] for t in range(8)]

# ------------------------------------------------------------------- dataset

def load(path):
    states = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            states.append((rec["g"], rec["c"], rec["p"], rec["y"]))
    return states


def expand(states):
    """Each state becomes submove rows: (state_idx, step) with
    step 0 = first click, step k>=1 = k-th direction decision (k==len path
    means FINISH). Value target attaches to step-0 rows."""
    rows = []
    for si, (_, _, path, _) in enumerate(states):
        if len(path) >= 2:
            for step in range(len(path) + 1):
                rows.append((si, step))
        else:
            rows.append((si, 0))  # terminal state: value-only row
    return rows


class Batch:
    pass


def make_batch(states, rows, idxs, t):
    """Build tensors for rows[idxs] under dihedral transform t."""
    B = len(idxs)
    planes = np.zeros((B, 14, N, N), dtype=np.float32)
    is_start = np.zeros(B, dtype=bool)
    start_tgt = np.zeros(B, dtype=np.int64)
    start_msk = np.zeros((B, CELLS), dtype=bool)
    head_idx = np.zeros(B, dtype=np.int64)
    dir_tgt = np.zeros(B, dtype=np.int64)
    dir_msk = np.zeros((B, 5), dtype=bool)
    has_val = np.zeros(B, dtype=bool)
    val_tgt = np.zeros(B, dtype=np.float32)
    has_pol = np.zeros(B, dtype=bool)

    perm = PERMS[t]
    for bi, ri in enumerate(idxs):
        si, step = rows[ri]
        _, cells, path, y = states[si]
        planes[bi, :9] = embed_cells(cells)
        if step == 0:
            is_start[bi] = True
            has_val[bi] = True
            val_tgt[bi] = math.log2(1.0 + y)
            if len(path) >= 2:
                has_pol[bi] = True
                start_tgt[bi] = perm[path[0]]
                sm = start_mask(cells)
                start_msk[bi, perm] = sm  # start_msk[new] = sm[old]
        else:
            prefix = path[:step]
            has_pol[bi] = True
            for pcell in prefix:
                planes[bi, 9, pcell // N, pcell % N] = 1.0
            h = prefix[-1]
            planes[bi, 10, h // N, h % N] = 1.0
            planes[bi, 11] = 1.0  # in-path flag
            head_idx[bi] = perm[h]
            dm = dir_mask(cells, prefix)
            if step < len(path):
                d = next(
                    dd
                    for dd, (dr, dc) in DELTAS.items()
                    if (path[step] // N - h // N, path[step] % N - h % N) == (dr, dc)
                )
            else:
                d = FINISH
            dir_tgt[bi] = DIR_MAP[t][d]
            dir_msk[bi, [DIR_MAP[t][dd] for dd in range(5)]] = dm

    b = Batch()
    b.planes = transform_planes(torch.from_numpy(planes), t)
    b.is_start = torch.from_numpy(is_start)
    b.start_tgt = torch.from_numpy(start_tgt)
    b.start_msk = torch.from_numpy(start_msk)
    b.head_idx = torch.from_numpy(head_idx)
    b.dir_tgt = torch.from_numpy(dir_tgt)
    b.dir_msk = torch.from_numpy(dir_msk)
    b.has_val = torch.from_numpy(has_val)
    b.val_tgt = torch.from_numpy(val_tgt)
    b.has_pol = torch.from_numpy(has_pol)
    return b


# ------------------------------------------------------------------- network

class Net(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.stem = nn.Conv2d(14, ch, 3, padding=1)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(ch, ch, 3, padding=1) for _ in range(4)]
        )
        self.start_head = nn.Conv2d(ch, 1, 1)
        self.dir_head = nn.Conv2d(ch, 5, 1)
        self.val_conv = nn.Conv2d(ch, 8, 1)
        self.val_fc = nn.Sequential(
            nn.Linear(8 * CELLS, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        h = F.relu(self.stem(x))
        for blk in self.blocks:
            h = F.relu(blk(h) + h)
        start_logits = self.start_head(h).flatten(1)          # B x 25
        dir_logits = self.dir_head(h).flatten(2)              # B x 5 x 25
        v = self.val_fc(F.relu(self.val_conv(h)).flatten(1)).squeeze(-1)
        return start_logits, dir_logits, v


def masked_ce(logits, mask, target):
    logits = logits.masked_fill(~mask, -1e9)
    return F.cross_entropy(logits, target, reduction="sum")


def run_epoch(net, opt, states, rows, device, batch_size=512, train=True, row_cap=150_000):
    order = np.random.permutation(len(rows))[:row_cap]
    tot_pol = tot_pol_n = tot_val = tot_val_n = 0.0
    acc_start = acc_start_n = acc_dir = acc_dir_n = 0
    net.train(train)
    for lo in range(0, len(order), batch_size):
        idxs = order[lo : lo + batch_size]
        t = np.random.randint(8)
        b = make_batch(states, rows, idxs, t)
        planes = b.planes.to(device)
        start_logits, dir_logits, v = net(planes)

        loss = torch.zeros((), device=device)
        sel = b.is_start & b.has_pol
        if sel.any():
            sl = start_logits[sel.to(device)]
            msk = b.start_msk[sel].to(device)
            tgt = b.start_tgt[sel].to(device)
            loss = loss + masked_ce(sl, msk, tgt)
            acc_start += (sl.masked_fill(~msk, -1e9).argmax(1) == tgt).sum().item()
            acc_start_n += int(sel.sum())
            tot_pol_n += int(sel.sum())
        sel = (~b.is_start) & b.has_pol
        if sel.any():
            dl = dir_logits[sel.to(device)]
            dl = dl.gather(
                2, b.head_idx[sel].to(device).view(-1, 1, 1).expand(-1, 5, 1)
            ).squeeze(-1)
            msk = b.dir_msk[sel].to(device)
            tgt = b.dir_tgt[sel].to(device)
            loss = loss + masked_ce(dl, msk, tgt)
            acc_dir += (dl.masked_fill(~msk, -1e9).argmax(1) == tgt).sum().item()
            acc_dir_n += int(sel.sum())
            tot_pol_n += int(sel.sum())
        tot_pol += float(loss.detach())

        if b.has_val.any():
            sel = b.has_val
            vl = F.mse_loss(
                v[sel.to(device)], b.val_tgt[sel].to(device), reduction="sum"
            )
            loss = loss + vl
            tot_val += float(vl)
            tot_val_n += int(sel.sum())

        if train:
            opt.zero_grad()
            (loss / max(len(idxs), 1)).backward()
            opt.step()

    return {
        "pol_loss": tot_pol / max(tot_pol_n, 1),
        "val_rmse_log": math.sqrt(tot_val / max(tot_val_n, 1)),
        "start_acc": acc_start / max(acc_start_n, 1),
        "dir_acc": acc_dir / max(acc_dir_n, 1),
    }


@torch.no_grad()
def exact_match(net, states, device, limit=4000):
    """Greedy-decode a full move per state and compare with the teacher path."""
    net.eval()
    picks = [s for s in states if len(s[2]) >= 2][:limit]
    hits = 0
    for _, cells, path, _ in picks:
        planes = np.zeros((1, 14, N, N), dtype=np.float32)
        planes[0, :9] = embed_cells(cells)
        x = torch.from_numpy(planes).to(device)
        sl, _, _ = net(x)
        sm = torch.from_numpy(start_mask(cells)).to(device)
        first = int(sl[0].masked_fill(~sm, -1e9).argmax())
        prefix = [first]
        ok = first == path[0]
        for _ in range(CELLS):
            planes[0, 9:12] = 0.0
            for pcell in prefix:
                planes[0, 9, pcell // N, pcell % N] = 1.0
            h = prefix[-1]
            planes[0, 10, h // N, h % N] = 1.0
            planes[0, 11] = 1.0
            x = torch.from_numpy(planes).to(device)
            _, dl, _ = net(x)
            logits = dl[0, :, h]
            dm = torch.from_numpy(dir_mask(cells, prefix)).to(device)
            if not bool(dm.any()):
                break
            d = int(logits.masked_fill(~dm, -1e9).argmax())
            if d == FINISH:
                ok = ok and prefix == path
                break
            dr, dc = DELTAS[d]
            nxt = (h // N + dr) * N + (h % N + dc)
            prefix.append(nxt)
            if len(prefix) > len(path) or prefix != path[: len(prefix)]:
                ok = False
        hits += int(ok)
    return hits / max(len(picks), 1)


@torch.no_grad()
def value_corr(net, states, device):
    net.eval()
    preds, ys = [], []
    for lo in range(0, len(states), 512):
        chunk = states[lo : lo + 512]
        planes = np.stack([np.pad(embed_cells(c), ((0, 5), (0, 0), (0, 0))) for _, c, _, _ in chunk])
        _, _, v = net(torch.from_numpy(planes.astype(np.float32)).to(device))
        preds.extend(v.cpu().numpy().tolist())
        ys.extend(math.log2(1.0 + y) for _, _, _, y in chunk)
    p, y = np.array(preds), np.array(ys)
    return float(np.corrcoef(p, y)[0, 1])


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "ml/selfplay-t4.jsonl"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    states = load(data)
    games = sorted({g for g, _, _, _ in states})
    holdout_games = set(games[int(0.9 * len(games)) :])
    train_states = [s for s in states if s[0] not in holdout_games]
    ho_states = [s for s in states if s[0] in holdout_games]
    train_rows = expand(train_states)
    ho_rows = expand(ho_states)
    print(
        f"device {device} | {len(train_states)} train states -> {len(train_rows)} rows | "
        f"{len(ho_states)} holdout states"
    )
    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(epochs):
        t0 = time.time()
        tr = run_epoch(net, opt, train_states, train_rows, device, train=True)
        ho = run_epoch(net, opt, ho_states, ho_rows, device, train=False, row_cap=30_000)
        print(
            f"epoch {ep}: train start_acc {tr['start_acc']:.3f} dir_acc {tr['dir_acc']:.3f} "
            f"val_rmse(log2) {tr['val_rmse_log']:.3f} | holdout start_acc {ho['start_acc']:.3f} "
            f"dir_acc {ho['dir_acc']:.3f} val_rmse {ho['val_rmse_log']:.3f} "
            f"({time.time()-t0:.0f}s)"
        )
    em = exact_match(net, ho_states, device)
    vc = value_corr(net, ho_states, device)
    print(f"holdout exact full-move match: {em:.3f}")
    print(f"holdout value corr (log2 space): {vc:.3f}")
    torch.save(net.state_dict(), "ml/net-v0.pt")
    print("saved ml/net-v0.pt")


if __name__ == "__main__":
    main()
