#!/usr/bin/env python3
"""
COLONY: ONE - Level 1 exact solver.

Given a board, find two CYAN (you) actions that either capture MAGENTA's Kernel
(instant win) or leave your Kernel on a tile that MAGENTA cannot capture with its
own two actions ("safe tile").

The search is exhaustive: the state space of one turn (two actions) per side is
tiny, so the answer is provably correct, not a heuristic.

--------------------------------------------------------------------------------
RULES IMPLEMENTED (from the level sheet)
  Board : 8x8. Columns A-H (x=1..8), rows 1-8 (y=1..8).
  Teams : CYAN = you, MAGENTA = enemy. A turn = exactly 2 actions, no skipping;
          the same piece may act twice.
  Kernel: per action moves 1 step to any adjacent tile (king move). Moving onto an
          enemy captures it. Lose your Kernel => you lose; capture theirs => you win.
  Ping  : move = up to 2 tiles orthogonally onto empty tiles only, no jumping.
          capture = the FIRST piece on a diagonal within 2 tiles, if it is an enemy
          (the Ping moves onto it). Cannot capture orthogonally; cannot move
          diagonally onto an empty tile.
--------------------------------------------------------------------------------
INPUT FORMAT (two accepted styles, auto-detected)

  Style A - grid, 8 rows top (row 8) to bottom (row 1), 8 chars each:
      . = empty
      k = your (CYAN) Kernel     p = your (CYAN) Ping
      K = enemy (MAGENTA) Kernel P = enemy (MAGENTA) Ping
    Example (scene 1):
      ........
      ........
      ........
      ....k...
      P......K      <-- wait: see note; use style B if unsure
      ........
      ........
      ........

  Style B - one piece per line: "<team> <type> <square>", blank lines/# ignored:
      cyan   K e4
      magenta K h4
      magenta P a4

Run:
  python3 solver.py scene.txt
  cat scene.txt | python3 solver.py
"""

import sys
from itertools import product

N = 8
CYAN, MAGENTA = "C", "M"
KERNEL, PING = "K", "P"


def in_board(x, y):
    return 1 <= x <= N and 1 <= y <= N


def sq_name(x, y):
    return "ABCDEFGH"[x - 1] + str(y)


def parse_square(s):
    s = s.strip().lower()
    x = "abcdefgh".index(s[0]) + 1
    y = int(s[1:])
    return (x, y)


# ------------------------------------------------------------------ parsing
def parse(text):
    """Return dict {(x,y): (team, type)}."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    grid_rows = [ln for ln in lines if set(ln) <= set(".kpKP") and len(ln) == N]
    board = {}
    if len(grid_rows) == N:  # Style A
        for r, row in enumerate(grid_rows):
            y = N - r  # first row is row 8
            for c, ch in enumerate(row):
                if ch == ".":
                    continue
                x = c + 1
                team = CYAN if ch.islower() else MAGENTA
                typ = KERNEL if ch.upper() == "K" else PING
                board[(x, y)] = (team, typ)
        return board
    # Style B
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        team = CYAN if parts[0].lower().startswith("c") else MAGENTA
        typ = KERNEL if parts[1].upper() == "K" else PING
        board[parse_square(parts[2])] = (team, typ)
    return board


# ------------------------------------------------------------------ moves
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]
KING = DIAG + ORTHO


def gen_actions(board, team):
    """Yield (label, new_board) for every legal action of `team`."""
    enemy = MAGENTA if team == CYAN else CYAN
    for (x, y), (t, typ) in list(board.items()):
        if t != team:
            continue
        if typ == KERNEL:
            for dx, dy in KING:
                nx, ny = x + dx, y + dy
                if not in_board(nx, ny):
                    continue
                occ = board.get((nx, ny))
                if occ and occ[0] == team:
                    continue  # friendly block
                nb = dict(board)
                del nb[(x, y)]
                nb[(nx, ny)] = (team, KERNEL)  # capture if enemy was there
                cap = occ[1] if occ else None
                yield (f"{sq_name(x,y)}->{sq_name(nx,ny)}", nb, cap)
        else:  # PING
            # move: up to 2 orthogonal onto empty, no jumping
            for dx, dy in ORTHO:
                for dist in (1, 2):
                    nx, ny = x + dx * dist, y + dy * dist
                    if not in_board(nx, ny):
                        break
                    # path must be empty
                    blocked = any(
                        (x + dx * s, y + dy * s) in board for s in range(1, dist + 1)
                    )
                    if blocked:
                        break
                    nb = dict(board)
                    del nb[(x, y)]
                    nb[(nx, ny)] = (team, PING)
                    yield (f"{sq_name(x,y)}->{sq_name(nx,ny)}", nb, None)
            # capture: first piece on a diagonal within 2, if enemy
            for dx, dy in DIAG:
                for dist in (1, 2):
                    tx, ty = x + dx * dist, y + dy * dist
                    if not in_board(tx, ty):
                        break
                    occ = board.get((tx, ty))
                    if occ is None:
                        continue
                    # first piece on this diagonal
                    if occ[0] == enemy:
                        nb = dict(board)
                        del nb[(x, y)]
                        nb[(tx, ty)] = (team, PING)
                        yield (f"{sq_name(x,y)}x{sq_name(tx,ty)}", nb, occ[1])
                    break  # blocked by first piece either way


def kernel_pos(board, team):
    for pos, (t, typ) in board.items():
        if t == team and typ == KERNEL:
            return pos
    return None


# ------------------------------------------------------------------ safety
def magenta_can_capture_cyan_kernel(board):
    """True if MAGENTA has a <=2-action sequence that captures the CYAN Kernel."""
    for _l1, b1, cap1 in gen_actions(board, MAGENTA):
        if cap1 == KERNEL and kernel_pos(b1, CYAN) is None:
            return True
        # MAGENTA's second action
        if kernel_pos(b1, CYAN) is None:
            return True  # already gone
        for _l2, b2, cap2 in gen_actions(b1, MAGENTA):
            if kernel_pos(b2, CYAN) is None:
                return True
    return False


def solve(board):
    """Return (list_of_2_action_labels, final_square, kind) or None."""
    # pre-pass: a single action that captures the enemy Kernel wins outright
    for l1, b1, cap1 in gen_actions(board, CYAN):
        if cap1 == KERNEL:
            return ([l1], sq_name(*kernel_pos(b1, CYAN)), "capture-win")
    best = None
    for l1, b1, cap1 in gen_actions(board, CYAN):
        # we must play a second action (cannot skip)
        for l2, b2, cap2 in gen_actions(b1, CYAN):
            if cap2 == KERNEL:
                return ([l1, l2], sq_name(*kernel_pos(b2, CYAN)), "capture-win")
            kp = kernel_pos(b2, CYAN)
            if kp is None:
                continue  # shouldn't happen on our own turn
            if not magenta_can_capture_cyan_kernel(b2):
                sol = ([l1, l2], sq_name(*kp), "safe-tile")
                # prefer the fewest "wasted"/back-and-forth actions; first found is fine
                if best is None:
                    best = sol
    return best


def render(board):
    rows = []
    for y in range(N, 0, -1):
        row = []
        for x in range(1, N + 1):
            occ = board.get((x, y))
            if not occ:
                row.append(".")
            else:
                ch = occ[1]
                row.append(ch.lower() if occ[0] == CYAN else ch.upper())
        rows.append(f"{y} " + " ".join(row))
    rows.append("  " + " ".join("ABCDEFGH"))
    return "\n".join(rows)


def main():
    text = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    board = parse(text)
    if kernel_pos(board, CYAN) is None:
        print("No CYAN Kernel found in input.", file=sys.stderr)
        sys.exit(1)
    print("Board:")
    print(render(board))
    print()
    sol = solve(board)
    if sol is None:
        print("No safe two-action solution exists for this scene.")
        sys.exit(2)
    actions, final, kind = sol
    print(f"Solution ({kind}): " + "  then  ".join(actions))
    print(f"CYAN Kernel ends on {final}.")


if __name__ == "__main__":
    main()
