#!/usr/bin/env python3
"""
Interactive Level 2 assistant.

Each round, type the current positions and it prints the CYAN turn (two actions)
to play. Powered by the engine in level2.py.

Run:
    python3 play.py                # depth 3 (fast)
    python3 play.py --depth 4      # stronger, slower

Each prompt accepts a chess-like square (e.g. f4, a8). You can answer the first
prompt with ALL positions on one line to go fast, e.g.:
    f4 b6 a8            ->  my Kernel=f4, my Ping=b6, enemy Kernel=a8
    f4 b6 a8 h1         ->  ...and enemy Ping=h1
    f4 - a8             ->  '-' means that piece is gone

Commands at any prompt:  q = quit,  r N = set round number to N (default auto +1)
"""

import sys

from level2 import (
    State, best_turn, parse_square, render,
    CYAN, MAGENTA, KERNEL, PING, WIN,
)


def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        print()
        sys.exit(0)


def parse_piece(tok):
    """'-'/'none'/'' -> None, else (x,y)."""
    tok = tok.strip().lower()
    if tok in ("", "-", "none", "x"):
        return None
    return parse_square(tok)


def build_state(ck, cp, ek, ep, rnd):
    pieces = {}
    if ck:
        pieces[ck] = (CYAN, KERNEL)
    if cp:
        pieces[cp] = (CYAN, PING)
    if ek:
        pieces[ek] = (MAGENTA, KERNEL)
    if ep:
        pieces[ep] = (MAGENTA, PING)
    return State(pieces, CYAN, rnd)


def main():
    args = sys.argv[1:]
    depth = 3
    if "--depth" in args:
        i = args.index("--depth")
        depth = int(args[i + 1])

    print("COLONY: ONE - Level 2 assistant (CYAN).  depth=%d" % depth)
    print("Type 'q' to quit. First prompt accepts all squares on one line.\n")

    rnd = 1
    while True:
        print(f"--- Round {rnd} ---")
        first = ask("Your Kernel (or 'Kernel Ping Enemy [EnemyPing]'): ")
        if first.lower() in ("q", "quit", "exit"):
            break
        if first.lower().startswith("r "):
            rnd = int(first.split()[1])
            continue

        toks = first.split()
        if len(toks) >= 3:                       # all-on-one-line fast path
            ck = parse_piece(toks[0])
            cp = parse_piece(toks[1])
            ek = parse_piece(toks[2])
            ep = parse_piece(toks[3]) if len(toks) >= 4 else None
        else:
            ck = parse_piece(first)
            cp = parse_piece(ask("Your Ping (or '-'): "))
            ek = parse_piece(ask("Enemy Kernel: "))
            ep = parse_piece(ask("Enemy Ping (blank if none): "))

        if ck is None:
            print("  ! You have no Kernel - that's a loss. Check input.\n")
            continue
        if ek is None:
            print("  ! No enemy Kernel entered - nothing to beat.\n")
            continue

        state = build_state(ck, cp, ek, ep, rnd)
        print(state_board(state))

        labels, val = best_turn(state, depth)
        if labels is None:
            print("  No legal turn.\n")
        else:
            verdict = ("FORCED WIN" if val >= WIN - 100
                       else "losing - best try" if val <= -WIN + 100
                       else f"eval {val:.0f}")
            print(f"  >> PLAY: {'   then   '.join(labels)}    [{verdict}]\n")

        rnd += 1


def state_board(state):
    return render(state)


if __name__ == "__main__":
    main()
