#!/usr/bin/env python3
"""Heuristic v3 for Colony: One, built from the Level 4 tournament loss analysis.

v2 (build_model.py) = material + centre control, capture-bias policy.
The recorded tournament losses show three blind spots of v2:
  1. Kernel left on its home corner and raided in two actions.
  2. Kernel burnt by the firewall (rounds 8/16/24/26) -- v2 ignores plane 14.
  3. Kernel walking into Trace / Daemon firing lines.

v3 adds, all as fixed ONNX ops (no training):
  value  += KCEN  * (ownK - enemyK) . centre_map
         -= KFW   * urgency * (ownK - enemyK) . next_ring_to_lock
         -= PFW   * urgency * sum_t w_t (own_t - enemy_t) . next_ring_to_lock
         -= KTH   * tanh(ownK . enemy_reach) * (1.5 - actions_left/2)
         += KTH   * tanh(enemyK . own_reach) * (0.5 + actions_left/2)
  where `reach` = squares a side can capture on within one 2-action turn
  (piece footprints composed move+capture, blockers ignored), via a 13x13 Conv.
  policy += KPOL  * K(from) * centre(to)   - KPOLT * K(from) * enemy_reach(to)
         +  CPOL  * own(from) * centre(to) + KCAP  * enemyK(to)

  python build_model_v3.py out.onnx          # env vars override weights
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

import build_model as bm


def _f(name, default):
    return float(os.environ.get(name, default))


KCEN = _f("KCEN", 0.15)
KFW = _f("KFW", 2.0)
PFW = _f("PFW", 0.5)
KTH = _f("KTH", 1.0)
KPOL = _f("KPOL", 0.5)
KPOLT = _f("KPOLT", 1.5)
CPOL = _f("CPOL", 0.15)
KCAP = _f("KCAP", 3.0)
FW_POW = int(_f("FW_POW", 1))     # urgency exponent for firewall terms
GUARD = _f("GUARD", 0.0)          # >0: unguarded kernels are threatened more
FW_SYM = int(_f("FW_SYM", 1))     # 1: symmetric firewall term, 0: own side only
R = 6  # conv radius -> 13x13 footprints


# ------------------------------------------------------------- footprints
def _orth(n):
    return {(d * k, 0) for d in (1, -1) for k in range(1, n + 1)} | \
           {(0, d * k) for d in (1, -1) for k in range(1, n + 1)}


def _diag(lo, hi):
    return {(a * k, b * k) for a in (1, -1) for b in (1, -1) for k in range(lo, hi + 1)}


KNIGHT = {(a, b) for a in (-2, -1, 1, 2) for b in (-2, -1, 1, 2) if abs(a) != abs(b)}
KING = _orth(1) | _diag(1, 1)

# (moves, captures) per plane order Ping, Daemon, Exploit, Trace, Ice, Kernel
FOOT = [
    (_orth(2), _diag(1, 2)),            # Ping
    (_orth(3), _orth(3)),               # Daemon (shoot adj, move-capture 2-3)
    (KNIGHT, KNIGHT),                   # Exploit
    (_orth(1), _diag(2, 3)),            # Trace
    (_orth(2) | _diag(1, 2), _orth(2) | _diag(1, 2)),  # Ice
    (KING, KING),                       # Kernel
]


def reach2(moves, caps):
    """Offsets at which a piece can capture within its 2-action turn."""
    first = moves | caps | {(0, 0)}
    out = set()
    for m in first:
        for c in caps:
            off = (m[0] + c[0], m[1] + c[1])
            if off != (0, 0) and max(abs(off[0]), abs(off[1])) <= R:
                out.add(off)
    return out


def reach_conv_weight():
    """(2,16,13,13): channel 0 = own reach map, channel 1 = enemy reach map."""
    w = np.zeros((2, 16, 2 * R + 1, 2 * R + 1), dtype=np.float32)
    for t, (m, c) in enumerate(FOOT):
        for (dr, dc) in reach2(m, c):
            w[0, t, R + dr, R + dc] = 1.0        # own pieces  (planes 0-5)
            w[1, t + 6, R + dr, R + dc] = 1.0    # enemy pieces (planes 6-11)
    return torch.from_numpy(w)


def ring_masks():
    """ring r (0 = centre 2x2 .. 3 = edge) as 4 maps (4,1,8,8)."""
    m = np.zeros((4, 1, 8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            ring = int(max(abs(r - 3.5), abs(c - 3.5)) - 0.5)
            m[ring, 0, r, c] = 1.0
    return torch.from_numpy(m)


def onehot(r, c):
    m = torch.zeros(1, 1, 8, 8)
    m[0, 0, r, c] = 1.0
    return m


class HeuristicV3(nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.zeros(16); cen = torch.zeros(16); en = torch.zeros(16); own = torch.zeros(16)
        for t, pw in enumerate(bm.PIECE_W):
            w[t], w[t + 6] = pw, -pw
            cen[t], cen[t + 6] = bm.CENTER_W, -bm.CENTER_W
            own[t], en[t + 6] = 1.0, 1.0
        self.register_buffer("wval", w.view(1, 16, 1, 1))
        self.register_buffer("wcen", cen.view(1, 16, 1, 1))
        self.register_buffer("wown", own.view(1, 16, 1, 1))
        self.register_buffer("wenemy", en.view(1, 16, 1, 1))
        self.register_buffer("center_map", torch.from_numpy(bm.center_map()))
        self.register_buffer("reach_w", reach_conv_weight())
        self.register_buffer("rings", ring_masks())
        # diagonal probe squares telling whether ring r is locked
        self.register_buffer("probe", torch.cat(
            [onehot(3, 3), onehot(2, 2), onehot(1, 1), onehot(0, 0)], 1))  # (1,4,8,8) r=0..3
        self.register_buffer("ones_col", torch.ones(64, 1))
        # 3x3 neighbourhood (minus centre) over non-kernel planes -> bodyguard counts
        g = torch.zeros(2, 16, 3, 3)
        for t in range(5):
            g[0, t] = 1.0; g[1, t + 6] = 1.0
        g[:, :, 1, 1] = 0.0
        self.register_buffer("guard_w", g)

    def forward(self, obs):
        B = obs.shape[0]
        K = obs[:, 5:6]            # own kernel      (B,1,8,8)
        EK = obs[:, 11:12]         # enemy kernel
        L = obs[:, 12:13]          # locked squares
        A = obs[:, 13:14, 0:1, 0:1].reshape(-1)   # actions_left/2  (B,)
        U = obs[:, 14:15, 0:1, 0:1].reshape(-1)   # firewall urgency (B,)

        # ---- v2 terms
        mat = (obs * self.wval).sum(dim=1, keepdim=True)
        cen = (obs * self.wcen).sum(dim=1, keepdim=True) * self.center_map
        v = (mat + cen).sum(dim=(1, 2, 3))

        # ---- kernel centralisation
        v = v + KCEN * ((K - EK) * self.center_map).sum(dim=(1, 2, 3))

        # ---- firewall: next ring to lock = outermost not-yet-locked ring
        lock = (L * self.probe).sum(dim=(2, 3))            # (B,4) locked flag ring 0..3
        one = lock[:, :1] * 0.0 + 1.0                      # (B,1) ones without Shape op
        lock_next = torch.cat([lock[:, 1:], one], dim=1)   # ring r+1 locked (ring 4 := locked)
        ind = lock_next - lock                             # (B,4) 1 exactly on next ring
        next_ring = (ind.reshape(-1, 4, 1, 1) * self.rings.reshape(1, 4, 8, 8)).sum(dim=1, keepdim=True)
        if FW_SYM:
            kfw = ((K - EK) * next_ring).sum(dim=(1, 2, 3))
            pfw = (mat * next_ring).sum(dim=(1, 2, 3))
        else:  # own-side only: the enemy can simply step inward, we may not see it
            kfw = (K * next_ring).sum(dim=(1, 2, 3))
            own_mat = (obs * torch.relu(self.wval)).sum(dim=1, keepdim=True)
            pfw = (own_mat * next_ring).sum(dim=(1, 2, 3))
        Up = U
        for _ in range(FW_POW - 1):
            Up = Up * U
        v = v - Up * (KFW * kfw + PFW * pfw)

        # ---- kernel threats via footprint convolution
        reach = torch.nn.functional.conv2d(obs, self.reach_w, padding=R)  # (B,2,8,8)
        own_reach, enemy_reach = reach[:, 0:1], reach[:, 1:2]
        if GUARD > 0:
            guards = torch.nn.functional.conv2d(obs, self.guard_w, padding=1)  # (B,2,8,8)
            own_g = torch.tanh(GUARD * (K * guards[:, 0:1]).sum(dim=(1, 2, 3)))
            en_g = torch.tanh(GUARD * (EK * guards[:, 1:2]).sum(dim=(1, 2, 3)))
        else:
            own_g = en_g = 0.0
        my_threat = torch.tanh((EK * own_reach).sum(dim=(1, 2, 3))) * (1.0 - 0.7 * en_g)
        their_threat = torch.tanh((K * enemy_reach).sum(dim=(1, 2, 3))) * (1.0 - 0.7 * own_g)
        v = v + KTH * (my_threat * (0.5 + A) - their_threat * (1.5 - A))

        value = torch.tanh(bm.VALUE_SCALE * v)

        # ---- policy priors over (from, to)
        enemy_to = (obs * self.wenemy).sum(dim=1).flatten(1)          # (B,64)
        own_from = (obs * self.wown).sum(dim=1).flatten(1)
        k_from = K.flatten(1)
        ek_to = EK.flatten(1)
        cen_to = self.center_map.flatten(1)                            # (1,64)
        thr_to = enemy_reach.flatten(1)                                # (B,64)
        row = lambda x: x.unsqueeze(1)      # (B,1,64)  depends on 'to'
        col = lambda x: x.unsqueeze(2)      # (B,64,1)  depends on 'from'
        grid = bm.POLICY_SCALE * row(enemy_to) * self.ones_col
        grid = grid + KCAP * row(ek_to) * self.ones_col
        grid = grid + col(k_from) * (KPOL * cen_to.unsqueeze(0) - KPOLT * row(thr_to))
        grid = grid + CPOL * col(own_from) * cen_to.unsqueeze(0)
        policy = grid.flatten(1)
        return policy, value


def main():
    from train_az import export_onnx, validate
    out = sys.argv[1] if len(sys.argv) > 1 else "model_v3.onnx"
    m = HeuristicV3().eval()
    export_onnx(m, out)
    ok, txt = validate(out)
    print(f"{out}: {'ACCEPTED' if ok else 'REJECTED'}  ({os.path.getsize(out)} B)")
    if not ok:
        print(txt)
        sys.exit(1)


if __name__ == "__main__":
    main()
