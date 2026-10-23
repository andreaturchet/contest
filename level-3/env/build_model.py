#!/usr/bin/env python3
"""
Build a hand-crafted policy/value ONNX model for Colony: One (Level 3).

No training: the referee's MCTS supplies the lookahead, so the model only has to
evaluate positions sensibly. We encode two classic heuristics directly as an
ONNX graph:

  value  = tanh(VALUE_SCALE * material_diff)
           material_diff = sum over board of (own piece weights - enemy weights),
           computed with a 1x1 convolution over the 16 input planes.

  policy = POLICY_SCALE * enemy_occupancy[to_square]
           i.e. every (from, to) action whose target square holds an enemy piece
           gets a higher prior -> a capture bias, like the greedy bot, but the
           MCTS then adds the lookahead greedy lacks.

Planes (from docs/observation.md): 0-5 own [Ping,Daemon,Exploit,Trace,Ice,Kernel],
6-11 enemy (same order), 12 locked, 13 actions_left, 14 urgency, 15 side-to-move.
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 18

# Per-piece material weights, indexed like planes 0-5
# [Ping, Daemon, Exploit, Trace, Ice, Kernel], scaled from GreedyAgent's values
# (Ping 15, Daemon 45, Exploit 25, Trace 30, Ice 50, Kernel 100) / 50.
PIECE_W = [float(x) for x in os.environ.get(
    "PIECE_W", "0.30,0.90,0.50,0.60,1.00,2.00").split(",")]
VALUE_SCALE = float(os.environ.get("VALUE_SCALE", "0.55"))
POLICY_SCALE = float(os.environ.get("POLICY_SCALE", "2.0"))
# Center-control weight: bonus for own pieces (penalty for enemy) by how central
# their square is. Aligns with the Kernel-endgame tiebreak (central Kernel wins).
CENTER_W = float(os.environ.get("CENTER_W", "0.08"))


def center_map():
    """8x8 map, higher toward the centre (max(|r-3.5|,|c-3.5|) inverted)."""
    m = np.zeros((1, 1, 8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            cheb = max(abs(r - 3.5), abs(c - 3.5))  # 0.5 .. 3.5
            m[0, 0, r, c] = (3.5 - cheb)            # 0 edge .. 3 centre
    return m


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build():
    # ---- value head weights: 1x1 conv, +w on own planes, -w on enemy planes
    wval = np.zeros((1, 16, 1, 1), dtype=np.float32)
    for t, w in enumerate(PIECE_W):
        wval[0, t, 0, 0] = w          # own planes 0-5
        wval[0, t + 6, 0, 0] = -w     # enemy planes 6-11
    # ---- own/enemy occupancy sums (for centre control and capture policy)
    wown = np.zeros((1, 16, 1, 1), dtype=np.float32)
    wenemy = np.zeros((1, 16, 1, 1), dtype=np.float32)
    for t in range(6):
        wown[0, t, 0, 0] = 1.0
        wenemy[0, t + 6, 0, 0] = 1.0
    # centre-control signed weight = CENTER_W on own, -CENTER_W on enemy
    wcen = np.zeros((1, 16, 1, 1), dtype=np.float32)
    for t in range(6):
        wcen[0, t, 0, 0] = CENTER_W
        wcen[0, t + 6, 0, 0] = -CENTER_W

    inits = [
        const("Wval", wval),
        const("Wcen", wcen),
        const("Wenemy", wenemy),
        const("center_map", center_map()),
        const("val_axes", np.array([1, 2, 3], dtype=np.int64)),
        const("val_scale", np.array(VALUE_SCALE, dtype=np.float32)),
        const("enemy_shape", np.array([0, 1, 64], dtype=np.int64)),
        const("ones_col", np.ones((64, 1), dtype=np.float32)),
        const("pol_scale", np.array(POLICY_SCALE, dtype=np.float32)),
    ]

    nodes = [
        # ----- value: material + centre control -----
        helper.make_node("Conv", ["observation", "Wval"], ["mat_map"]),
        helper.make_node("Conv", ["observation", "Wcen"], ["cen_sig"]),
        helper.make_node("Mul", ["cen_sig", "center_map"], ["cen_map"]),
        helper.make_node("Add", ["mat_map", "cen_map"], ["val_map"]),
        helper.make_node("ReduceSum", ["val_map", "val_axes"], ["val_sum"],
                         keepdims=0),
        helper.make_node("Mul", ["val_sum", "val_scale"], ["val_scaled"]),
        helper.make_node("Tanh", ["val_scaled"], ["value"]),
        # ----- policy: capture bias (target square holds an enemy) -----
        helper.make_node("Conv", ["observation", "Wenemy"], ["enemy_map"]),
        helper.make_node("Reshape", ["enemy_map", "enemy_shape"], ["enemy_row"]),
        # outer broadcast: (B,1,64) * (64,1) -> (B,64,64), value depends only on 'to'
        helper.make_node("Mul", ["enemy_row", "ones_col"], ["pol_grid"]),
        helper.make_node("Flatten", ["pol_grid"], ["pol_flat"], axis=1),
        helper.make_node("Mul", ["pol_flat", "pol_scale"], ["policy"]),
    ]

    obs = helper.make_tensor_value_info(
        "observation", TensorProto.FLOAT, ["batch", 16, 8, 8])
    policy = helper.make_tensor_value_info(
        "policy", TensorProto.FLOAT, ["batch", 4096])
    value = helper.make_tensor_value_info(
        "value", TensorProto.FLOAT, ["batch"])

    graph = helper.make_graph(nodes, "colony_one_heuristic",
                              [obs], [policy, value], initializer=inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=9)
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "model.onnx"
    onnx.save(build(), out)
    print("wrote", out)
