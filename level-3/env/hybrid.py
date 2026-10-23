#!/usr/bin/env python3
"""Hybrid ONNX models: hand-crafted heuristic value (build_model.py) combined
with the policy (and optionally value) of a self-play-trained net checkpoint.

  python hybrid.py CKPT.pt OUT_PREFIX [--channels 48 --blocks 4]

Writes OUT_PREFIX_A.onnx  heuristic value          + net policy
       OUT_PREFIX_B.onnx  blended value (heur/net) + net policy
       OUT_PREFIX_C.onnx  heuristic value          + net policy + capture bias
and validates each with tools/validate_submission.py.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn

import build_model as bm
from train_az import export_onnx, make_net, validate


class Hybrid(nn.Module):
    def __init__(self, net, value_mode="heur", blend=0.3, capture_bias=0.0,
                 policy_scale=1.0):
        super().__init__()
        self.net = net
        self.value_mode, self.blend = value_mode, blend
        self.capture_bias, self.policy_scale = capture_bias, policy_scale
        w = torch.zeros(16)
        c = torch.zeros(16)
        e = torch.zeros(16)
        for t, pw in enumerate(bm.PIECE_W):
            w[t], w[t + 6] = pw, -pw
            c[t], c[t + 6] = bm.CENTER_W, -bm.CENTER_W
            e[t + 6] = 1.0
        self.register_buffer("wval", w.view(1, 16, 1, 1))
        self.register_buffer("wcen", c.view(1, 16, 1, 1))
        self.register_buffer("wenemy", e.view(1, 16, 1, 1))
        self.register_buffer("center_map", torch.from_numpy(bm.center_map()))
        self.register_buffer("ones_col", torch.ones(64, 1))

    def forward(self, obs):
        logits, v_net = self.net(obs)
        mat = (obs * self.wval).sum(dim=1, keepdim=True)
        cen = (obs * self.wcen).sum(dim=1, keepdim=True) * self.center_map
        heur = torch.tanh(bm.VALUE_SCALE * (mat + cen).sum(dim=(1, 2, 3)))
        if self.value_mode == "heur":
            value = heur
        else:
            value = (1.0 - self.blend) * heur + self.blend * v_net
        policy = logits * self.policy_scale
        if self.capture_bias:
            enemy_row = (obs * self.wenemy).sum(dim=1).flatten(1).unsqueeze(1)
            grid = enemy_row * self.ones_col            # (B,64,64), depends on 'to'
            policy = policy + self.capture_bias * grid.flatten(1)
        return policy, value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("prefix")
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--blend", type=float, default=0.3)
    ap.add_argument("--capture-bias", type=float, default=1.0)
    args = ap.parse_args()

    net = make_net(args.channels, args.blocks)
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    net.eval()

    variants = {
        "A": Hybrid(net, "heur"),
        "B": Hybrid(net, "blend", blend=args.blend),
        "C": Hybrid(net, "heur", capture_bias=args.capture_bias),
        "D": Hybrid(net, "blend", blend=args.blend, capture_bias=2.0,
                    policy_scale=1.5),
    }
    for tag, m in variants.items():
        out = f"{args.prefix}_{tag}.onnx"
        m.eval()
        export_onnx(m, out)
        ok, txt = validate(out)
        # sanity: compare heuristic value with the reference ONNX
        import onnxruntime as ort
        x = np.random.default_rng(0).random((3, 16, 8, 8), dtype=np.float32)
        x = (x > 0.9).astype(np.float32)
        ref = ort.InferenceSession("model.onnx").run(None, {"observation": x})[1]
        got = ort.InferenceSession(out).run(None, {"observation": x})[1]
        dev = float(np.abs(ref - got).max()) if tag != "B" else float("nan")
        print(f"{out}: {'ACCEPTED' if ok else 'REJECTED'}  "
              f"max|value-heur|={dev:.4f}")
        if not ok:
            print("   ", txt.strip().splitlines()[-1])


if __name__ == "__main__":
    main()
