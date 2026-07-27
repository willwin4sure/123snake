#!/usr/bin/env python3
"""Distill AC7's full-width depth-2 search into a policy+value CNN.

Data: ml/data/qdump-*.jsonl — every decision state of teacher self-play
with Q for EVERY legal move (chain), Q = reward + E_refill[best-reply V].

Targets:
  - value: log2(1 + max_j Q_j) on clean states (the search root value)
  - policy: p(move_j) = softmax(Q_j / tau) over all legal chains, factored
    into the click-level action space (start cell, then dir/finish masked),
    trained with soft targets:
      start row:  P(start=s) = sum of p_j over moves starting at s
      dir rows along the argmax-Q move's prefixes:
                  P(d | prefix) = mass(prefix + d) / mass(prefix)
                  P(finish | prefix) = p(move == prefix) / mass(prefix)
    (chain prefixes of length >= 2 are themselves legal moves, so the
    finish mass is always well-defined.)

Usage: .venv/bin/python ml/train_q.py [data.jsonl] [epochs] [tau]
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


def cell_channels(v):
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


def embed_cells(cells):
    out = np.zeros((9, N, N), dtype=np.float32)
    for i, v in enumerate(cells):
        ch = EMB.get(v)
        if ch is None:
            ch = cell_channels(v)
            EMB[v] = ch
        out[:, i // N, i % N] = ch
    return out


def start_mask(cells):
    m = np.zeros(CELLS, dtype=bool)
    for i in range(CELLS):
        r, c = divmod(i, N)
        for dr, dc in DELTAS.values():
            rr, cc = r + dr, c + dc
            if 0 <= rr < N and 0 <= cc < N and cells[rr * N + cc] == cells[i]:
                m[i] = True
                break
    return m


def dir_mask(cells, prefix):
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


def build_transforms():
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


def transform_planes(x, t):
    x = torch.rot90(x, t % 4, dims=(-2, -1))
    if t >= 4:
        x = torch.flip(x, dims=(-1,))
    return x


def transform_dir(d, t):
    if d == FINISH:
        return FINISH
    dr, dc = DELTAS[d]
    a, b = 2 * N + 2, (2 + dr) * N + (2 + dc)
    pa, pb = PERMS[t][a], PERMS[t][b]
    ndr, ndc = pb // N - pa // N, pb % N - pa % N
    for dd, (xr, xc) in DELTAS.items():
        if (xr, xc) == (ndr, ndc):
            return dd
    raise AssertionError


DIR_MAP = [[transform_dir(d, t) for d in range(5)] for t in range(8)]


def load(path, tau):
    """Per state: cells, argmax path, soft start dist, per-prefix dir dists,
    value target log2(1+maxQ)."""
    states = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            cells = rec["cells"]
            paths = [tuple(m["path"]) for m in rec["moves"]]
            qs = np.array([m["q"] for m in rec["moves"]], dtype=np.float64)
            p = np.exp((qs - qs.max()) / tau)
            p /= p.sum()
            best = paths[int(np.argmax(qs))]
            sd = np.zeros(CELLS, dtype=np.float32)
            for pa, pj in zip(paths, p):
                sd[pa[0]] += pj
            dir_rows = []
            for step in range(1, len(best) + 1):
                prefix = best[:step]
                mass = 0.0
                dd = np.zeros(5, dtype=np.float64)
                for pa, pj in zip(paths, p):
                    if pa[: len(prefix)] != prefix:
                        continue
                    mass += pj
                    if len(pa) == len(prefix):
                        dd[FINISH] += pj
                    else:
                        nxt = pa[len(prefix)]
                        h = prefix[-1]
                        ndr, ndc = nxt // N - h // N, nxt % N - h % N
                        for d, (xr, xc) in DELTAS.items():
                            if (xr, xc) == (ndr, ndc):
                                dd[d] += pj
                                break
                if mass <= 0:
                    break
                dir_rows.append((prefix, (dd / mass).astype(np.float32)))
            y = math.log2(1.0 + max(qs.max(), 0.0))
            states.append((cells, best, sd, dir_rows, y))
    return states


def expand(states):
    rows = []
    for si, (_, best, _, dir_rows, _) in enumerate(states):
        rows.append((si, 0))
        for k in range(len(dir_rows)):
            rows.append((si, k + 1))
    return rows


class Batch:
    pass


def make_batch(states, rows, idxs, t):
    B = len(idxs)
    planes = np.zeros((B, 14, N, N), dtype=np.float32)
    is_start = np.zeros(B, dtype=bool)
    start_dist = np.zeros((B, CELLS), dtype=np.float32)
    start_msk = np.zeros((B, CELLS), dtype=bool)
    head_idx = np.zeros(B, dtype=np.int64)
    dir_dist = np.zeros((B, 5), dtype=np.float32)
    dir_msk = np.zeros((B, 5), dtype=bool)
    has_val = np.zeros(B, dtype=bool)
    val_tgt = np.zeros(B, dtype=np.float32)

    perm = PERMS[t]
    for bi, ri in enumerate(idxs):
        si, step = rows[ri]
        cells, best, sd, dir_rows, y = states[si]
        planes[bi, :9] = embed_cells(cells)
        if step == 0:
            is_start[bi] = True
            has_val[bi] = True
            val_tgt[bi] = y
            start_dist[bi, perm] = sd
            start_msk[bi, perm] = start_mask(cells)
        else:
            prefix, dd = dir_rows[step - 1]
            for pcell in prefix:
                planes[bi, 9, pcell // N, pcell % N] = 1.0
            h = prefix[-1]
            planes[bi, 10, h // N, h % N] = 1.0
            planes[bi, 11] = 1.0
            head_idx[bi] = perm[h]
            dm = dir_mask(cells, list(prefix))
            for d in range(5):
                dir_dist[bi, DIR_MAP[t][d]] = dd[d]
                dir_msk[bi, DIR_MAP[t][d]] = dm[d]

    b = Batch()
    b.planes = transform_planes(torch.from_numpy(planes), t)
    b.is_start = torch.from_numpy(is_start)
    b.start_dist = torch.from_numpy(start_dist)
    b.start_msk = torch.from_numpy(start_msk)
    b.head_idx = torch.from_numpy(head_idx)
    b.dir_dist = torch.from_numpy(dir_dist)
    b.dir_msk = torch.from_numpy(dir_msk)
    b.has_val = torch.from_numpy(has_val)
    b.val_tgt = torch.from_numpy(val_tgt)
    return b


class Net(nn.Module):
    def __init__(self, ch=64, blocks=4):
        super().__init__()
        self.stem = nn.Conv2d(14, ch, 3, padding=1)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(ch, ch, 3, padding=1) for _ in range(blocks)]
        )
        self.start_head = nn.Conv2d(ch, 1, 1)
        self.dir_head = nn.Conv2d(ch, 5, 1)
        self.val_conv = nn.Conv2d(ch, 8, 1)
        self.val_fc = nn.Linear(8 * CELLS, 1)

    def forward(self, x):
        h = F.relu(self.stem(x))
        for blk in self.blocks:
            h = h + F.relu(blk(h))
        start = self.start_head(h).flatten(1)
        dirs = self.dir_head(h).flatten(2).transpose(1, 2)
        val = self.val_fc(F.relu(self.val_conv(h)).flatten(1)).squeeze(-1)
        return start, dirs, val


def masked_kl(logits, mask, target):
    logits = logits.masked_fill(~mask, -1e9)
    logp = F.log_softmax(logits, dim=-1)
    contrib = torch.where(
        target > 0,
        target * (target.clamp_min(1e-9).log() - logp),
        torch.zeros_like(target),
    )
    return contrib.sum(-1)


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "ml/data/qdump-ac7-400g.jsonl"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tau = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
    ch = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    blocks = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    run = sys.argv[6] if len(sys.argv) > 6 else f"q{ch}x{blocks}"
    ckpt_path = f"ml/qnet-{run}.pt"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {data_path} (tau {tau}) ...", flush=True)
    states = load(data_path, tau)
    rng = np.random.default_rng(7)
    order = rng.permutation(len(states))
    n_val = max(1, len(states) // 20)
    val_states = [states[i] for i in order[:n_val]]
    tr_states = [states[i] for i in order[n_val:]]
    tr_rows = expand(tr_states)
    va_rows = expand(val_states)
    print(f"{len(tr_states)} train states / {len(val_states)} val; "
          f"{len(tr_rows)} rows", flush=True)

    net = Net(ch=ch, blocks=blocks).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    BS = 512

    def run_batch(states_, rows_, idxs, train):
        t = int(rng.integers(8)) if train else 0
        b = make_batch(states_, rows_, idxs, t)
        planes = b.planes.to(dev)
        start, dirs, val = net(planes)
        loss = torch.zeros((), device=dev)
        logs = {}
        st = b.is_start.to(dev)
        if st.any():
            kl_s = masked_kl(start[st], b.start_msk.to(dev)[st],
                             b.start_dist.to(dev)[st])
            pol_rows = b.start_msk.any(-1).to(dev)[st]
            if pol_rows.any():
                loss = loss + kl_s[pol_rows].mean()
                logs["kl_start"] = kl_s[pol_rows].mean().item()
            v_err = F.mse_loss(val[st], b.val_tgt.to(dev)[st])
            loss = loss + v_err
            logs["v_mse"] = v_err.item()
        nd = ~b.is_start.to(dev)
        if nd.any():
            hd = b.head_idx.to(dev)[nd]
            dl = dirs[nd].gather(1, hd[:, None, None].expand(-1, 1, 5)).squeeze(1)
            kl_d = masked_kl(dl, b.dir_msk.to(dev)[nd], b.dir_dist.to(dev)[nd])
            loss = loss + kl_d.mean()
            logs["kl_dir"] = kl_d.mean().item()
        return loss, logs, (start, dirs, val, b)

    for ep in range(epochs):
        net.train()
        perm_rows = rng.permutation(len(tr_rows))
        t0 = time.time()
        agg = {}
        nb = 0
        for lo in range(0, len(perm_rows), BS):
            idxs = perm_rows[lo:lo + BS]
            loss, logs, _ = run_batch(tr_states, tr_rows, idxs, True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        line = " ".join(f"{k} {v / nb:.4f}" for k, v in sorted(agg.items()))
        print(f"epoch {ep + 1}/{epochs}  {line}  ({time.time() - t0:.0f}s)",
              flush=True)

        net.eval()
        with torch.no_grad():
            vp, vt, s_hit, s_n = [], [], 0, 0
            for lo in range(0, len(va_rows), BS):
                idxs = np.arange(lo, min(lo + BS, len(va_rows)))
                _, _, (start, dirs, val, b) = run_batch(val_states, va_rows, idxs, False)
                st = b.is_start
                if st.any():
                    vp.append(val.cpu()[st])
                    vt.append(b.val_tgt[st])
                    lg = start.cpu()[st].masked_fill(~b.start_msk[st], -1e9)
                    pol = b.start_msk[st].any(-1)
                    pred = lg.argmax(-1)
                    tgt = b.start_dist[st].argmax(-1)
                    s_hit += (pred[pol] == tgt[pol]).sum().item()
                    s_n += pol.sum().item()
            vp = torch.cat(vp).numpy()
            vt = torch.cat(vt).numpy()
            corr = float(np.corrcoef(vp, vt)[0, 1])
            top1 = s_hit / max(1, s_n)
            print(f"  val: value corr {corr:.3f}  start top1 {top1:.3f}",
                  flush=True)
        torch.save(net.state_dict(), ckpt_path)
        play_mean = None
        try:
            import subprocess
            out = subprocess.run(
                [sys.executable, "ml/play_q.py", "20", ckpt_path, "policy",
                 str(ch), str(blocks)],
                capture_output=True, text=True, timeout=600,
            ).stdout
            for tok in out.split():
                if tok == "mean":
                    continue
            import re as _re
            m = _re.search(r"mean ([\d.]+)", out)
            if m:
                play_mean = float(m.group(1))
        except Exception:
            pass
        with open("ml/data/nn-metrics.jsonl", "a") as mf:
            mf.write(json.dumps({
                "run": run, "epoch": ep + 1,
                "kl_start": round(agg.get("kl_start", 0) / nb, 4),
                "kl_dir": round(agg.get("kl_dir", 0) / nb, 4),
                "v_mse": round(agg.get("v_mse", 0) / nb, 4),
                "corr": round(corr, 4), "top1": round(top1, 4),
                "play20": play_mean,
            }) + "\n")
        print(f"  play20 {play_mean}", flush=True)
    print(f"saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
