#!/usr/bin/env python3
"""
COLONY: ONE - Level 2 game engine + solver.

Level 2 is a full two-player game (you = CYAN, enemy = MAGENTA). Unlike level 1
(one-shot "reach a safe tile"), here you must WIN. So this is an adversarial
search: iterative-deepening alpha-beta over whole turns (a turn = 2 actions),
returning the CYAN turn that forces the best outcome.

You consult it each CYAN turn: give it the current board + round number, it
prints the two actions to play.

================================================================================
RULES IMPLEMENTED
  Board/teams/actions/Kernel/Ping : as level 1 (king-step Kernel; Ping moves <=2
      orthogonally onto empty, no jumping; Ping captures the first enemy on a
      diagonal within 2). A turn = exactly 2 actions, no skipping, a piece may
      act twice.

  BODYGUARD: a Kernel with any friendly piece in its 8 neighbours is "protected".
      An attack (move OR shot) on a protected Kernel destroys one guard instead:
      the Kernel survives, the attacker STAYS on its tile, one guard removed per
      attack (a 2nd attack then hits the Kernel). The firewall ignores guards.
      (In this game each side has at most one Ping, so the guard is unique.)

  FIREWALL: after MAGENTA's turn in rounds 8, 16, 24, 26 the outer ring vanishes
      (8x8 -> 6x6 -> 4x4 -> 2x2 -> none). Pieces on the vanishing ring die.
      Kernel killed by firewall => other team wins; both at once => MAGENTA.
      Max 26 rounds; if nobody won, MAGENTA wins. No draws.

  KERNEL ENDGAME: if only the two Kernels remain at the end of a turn:
      - within 2 tiles (Chebyshev, diagonals count) => the team that JUST moved loses;
      - else the Kernel on the more central ring wins (centre 2x2 > 4x4 > 6x6 > edge);
        same ring => MAGENTA wins.

ASSUMPTIONS TO VERIFY against the live engine (flagged, rarely matter in short
scenes): exact round index the scene starts on; which guard dies if a Kernel ever
has 2+ guards; whether "attacker stays" also applies to a Ping shot (assumed yes).
================================================================================
INPUT (same two styles as level 1's solver.py, auto-detected)
  Grid, 8 rows top->bottom:  . empty | k/p = your CYAN Kernel/Ping | K/P = MAGENTA
  or lines:  cyan K e4 / cyan P d4 / magenta K h8 ...
  Optional directives:  #round N   (default 1)   #side cyan|magenta (default cyan)

Run:  python3 level2.py scene.txt [--depth 6]
"""

import sys
from functools import lru_cache

N0 = 8
CYAN, MAGENTA = "C", "M"
KERNEL, PING = "K", "P"
WIN = 10**9

DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]
KING = DIAG + ORTHO
FIREWALL_ROUNDS = {8: (2, 7), 16: (3, 6), 24: (4, 5), 26: None}


def other(team):
    return MAGENTA if team == CYAN else CYAN


def sq_name(x, y):
    return "ABCDEFGH"[x - 1] + str(y)


def parse_square(s):
    s = s.strip().lower()
    return ("abcdefgh".index(s[0]) + 1, int(s[1:]))


# ------------------------------------------------------------------ state
class State:
    __slots__ = ("pieces", "side", "rnd", "lo", "hi")

    def __init__(self, pieces, side=CYAN, rnd=1, lo=1, hi=N0):
        self.pieces = pieces          # {(x,y): (team, type)}
        self.side = side              # team to move
        self.rnd = rnd                # current round number
        self.lo, self.hi = lo, hi     # live board bounds (firewall)

    def copy(self, pieces=None, side=None, rnd=None, lo=None, hi=None):
        return State(
            pieces if pieces is not None else dict(self.pieces),
            side if side is not None else self.side,
            rnd if rnd is not None else self.rnd,
            lo if lo is not None else self.lo,
            hi if hi is not None else self.hi,
        )

    def key(self):
        return (
            frozenset(self.pieces.items()),
            self.side, self.rnd, self.lo, self.hi,
        )

    def in_board(self, x, y):
        return self.lo <= x <= self.hi and self.lo <= y <= self.hi

    def kernel(self, team):
        for pos, (t, typ) in self.pieces.items():
            if t == team and typ == KERNEL:
                return pos
        return None


def is_protected(pieces, pos, owner):
    x, y = pos
    for dx, dy in KING:
        occ = pieces.get((x + dx, y + dy))
        if occ and occ[0] == owner:
            return True
    return False


def find_guard(pieces, pos, owner):
    x, y = pos
    for dx, dy in KING:
        p = (x + dx, y + dy)
        occ = pieces.get(p)
        if occ and occ[0] == owner:
            return p
    return None


# ----------------------------------------------------- single-action generator
def gen_actions(state, team):
    """Yield (label, new_pieces, captured_type). captured_type==KERNEL => win."""
    enemy = other(team)
    pieces = state.pieces
    for (x, y), (t, typ) in list(pieces.items()):
        if t != team:
            continue
        if typ == KERNEL:
            for dx, dy in KING:
                nx, ny = x + dx, y + dy
                if not state.in_board(nx, ny):
                    continue
                occ = pieces.get((nx, ny))
                if occ and occ[0] == team:
                    continue
                if occ and occ[0] == enemy:
                    if occ[1] == KERNEL and is_protected(pieces, (nx, ny), enemy):
                        g = find_guard(pieces, (nx, ny), enemy)
                        nb = dict(pieces); del nb[g]          # attacker stays
                        yield (f"{sq_name(x,y)}x{sq_name(nx,ny)}(guard)", nb, None)
                    else:
                        nb = dict(pieces); del nb[(x, y)]
                        nb[(nx, ny)] = (team, KERNEL)
                        yield (f"{sq_name(x,y)}x{sq_name(nx,ny)}", nb, occ[1])
                else:
                    nb = dict(pieces); del nb[(x, y)]
                    nb[(nx, ny)] = (team, KERNEL)
                    yield (f"{sq_name(x,y)}->{sq_name(nx,ny)}", nb, None)
        else:  # PING
            for dx, dy in ORTHO:                               # move
                for dist in (1, 2):
                    nx, ny = x + dx * dist, y + dy * dist
                    if not state.in_board(nx, ny):
                        break
                    if any((x + dx * s, y + dy * s) in pieces for s in range(1, dist + 1)):
                        break
                    nb = dict(pieces); del nb[(x, y)]
                    nb[(nx, ny)] = (team, PING)
                    yield (f"{sq_name(x,y)}->{sq_name(nx,ny)}", nb, None)
            for dx, dy in DIAG:                                # capture
                for dist in (1, 2):
                    tx, ty = x + dx * dist, y + dy * dist
                    if not state.in_board(tx, ty):
                        break
                    occ = pieces.get((tx, ty))
                    if occ is None:
                        continue
                    if occ[0] == enemy:
                        if occ[1] == KERNEL and is_protected(pieces, (tx, ty), enemy):
                            g = find_guard(pieces, (tx, ty), enemy)
                            nb = dict(pieces); del nb[g]       # ping stays
                            yield (f"{sq_name(x,y)}x{sq_name(tx,ty)}(guard)", nb, None)
                        else:
                            nb = dict(pieces); del nb[(x, y)]
                            nb[(tx, ty)] = (team, PING)
                            yield (f"{sq_name(x,y)}x{sq_name(tx,ty)}", nb, occ[1])
                    break   # blocked by first piece on the diagonal


# ----------------------------------------------------- post-turn resolution
def apply_firewall(pieces, lo, hi):
    """Remove outer ring. Return (new_pieces, new_lo, new_hi, cyan_k_dead, mag_k_dead)."""
    ck_dead = mk_dead = False
    nb = {}
    for (x, y), v in pieces.items():
        if x == lo or x == hi or y == lo or y == hi:
            if v[1] == KERNEL:
                if v[0] == CYAN:
                    ck_dead = True
                else:
                    mk_dead = True
            continue
        nb[(x, y)] = v
    return nb, lo + 1, hi - 1, ck_dead, mk_dead


def centrality(pos, lo, hi):
    c = (lo + hi) / 2
    return max(abs(pos[0] - c), abs(pos[1] - c))  # smaller = more central


def resolve_after_turn(state, mover):
    """
    Apply firewall (only after MAGENTA) + endgame checks.
    Return (new_state, winner) where winner is CYAN/MAGENTA/None.
    """
    pieces, lo, hi, rnd = dict(state.pieces), state.lo, state.hi, state.rnd

    if mover == MAGENTA:
        if rnd in FIREWALL_ROUNDS:
            if FIREWALL_ROUNDS[rnd] is None:      # round 26: board gone
                return state, MAGENTA
            pieces, lo, hi, ckd, mkd = apply_firewall(pieces, lo, hi)
            if ckd and mkd:
                return state, MAGENTA
            if ckd:
                return state, MAGENTA
            if mkd:
                return state, CYAN
        rnd += 1
        if rnd > 26:
            return state, MAGENTA

    ck, mk = None, None
    for pos, (t, typ) in pieces.items():
        if typ == KERNEL:
            if t == CYAN: ck = pos
            else: mk = pos
    if ck is None:
        return state, MAGENTA
    if mk is None:
        return state, CYAN

    # kernel endgame: only two kernels remain
    if len(pieces) == 2:
        if max(abs(ck[0] - mk[0]), abs(ck[1] - mk[1])) <= 2:
            winner = other(mover)             # team that just moved loses
            return state, winner
        cc, cm = centrality(ck, lo, hi), centrality(mk, lo, hi)
        if cc < cm:
            return state, CYAN
        elif cm < cc:
            return state, MAGENTA
        else:
            return state, MAGENTA             # same ring => MAGENTA

    ns = State(pieces, other(mover), rnd, lo, hi)
    return ns, None


# ----------------------------------------------------- whole-turn generator
def legal_turns(state):
    """Yield (labels, result_state, winner_or_None) for every legal full turn."""
    team = state.side
    out = []
    for l1, b1, cap1 in gen_actions(state, team):
        if cap1 == KERNEL:
            out.append(([l1], None, team))            # captured enemy kernel: win now
            continue
        mid = state.copy(pieces=b1)
        found = False
        for l2, b2, cap2 in gen_actions(mid, team):
            found = True
            if cap2 == KERNEL:
                out.append(([l1, l2], None, team))
                continue
            after = state.copy(pieces=b2)
            ns, winner = resolve_after_turn(after, team)
            out.append(([l1, l2], ns, winner))
        if not found:
            after = state.copy(pieces=b1)             # no 2nd action available
            ns, winner = resolve_after_turn(after, team)
            out.append(([l1], ns, winner))
    return out


# ----------------------------------------------------- evaluation + search
def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _king_mobility(state, pos):
    """Empty in-board squares the piece at pos could step to (raw escape room)."""
    n = 0
    for dx, dy in KING:
        nx, ny = pos[0] + dx, pos[1] + dy
        if state.in_board(nx, ny) and (nx, ny) not in state.pieces:
            n += 1
    return n


def evaluate(state):
    """Heuristic, CYAN perspective. Rewards hunting the enemy Kernel."""
    ck = state.kernel(CYAN)
    mk = state.kernel(MAGENTA)
    if ck is None:
        return -WIN
    if mk is None:
        return WIN
    s = 0
    # material: our Ping is a big asset, an enemy Ping a liability
    for pos, (t, typ) in state.pieces.items():
        if typ == PING:
            s += 120 if t == CYAN else -120
    # push the enemy Kernel to the edge; keep ours central
    s += centrality(mk, state.lo, state.hi) * 10
    s -= centrality(ck, state.lo, state.hi) * 6
    # choke the enemy Kernel's escape room
    s -= _king_mobility(state, mk) * 9
    # close in with both attackers (Chebyshev distance to the enemy Kernel)
    s -= _cheb(ck, mk) * 3
    pp = next((p for p, v in state.pieces.items() if v == (CYAN, PING)), None)
    if pp is not None:
        s -= _cheb(pp, mk) * 3
    # guard bookkeeping
    if is_protected(state.pieces, ck, CYAN):
        s += 8
    return s


def search(state, depth, alpha, beta):
    """Negamax-ish alpha-beta from `state.side`'s perspective -> CYAN-signed score."""
    turns = legal_turns(state)
    if state.side == CYAN:
        best = -WIN - 1
        for labels, ns, winner in turns:
            if winner is not None:
                val = WIN - (10 - depth) if winner == CYAN else -WIN + (10 - depth)
            elif depth <= 1:
                val = evaluate(ns)
            else:
                val = search(ns, depth - 1, alpha, beta)
            if val > best:
                best = val
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best
    else:
        best = WIN + 1
        for labels, ns, winner in turns:
            if winner is not None:
                val = WIN - (10 - depth) if winner == CYAN else -WIN + (10 - depth)
            elif depth <= 1:
                val = evaluate(ns)
            else:
                val = search(ns, depth - 1, alpha, beta)
            if val < best:
                best = val
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best


def best_turn(state, max_depth):
    """Iterative deepening; return (labels, score). Stops early on a forced win."""
    best = None
    for depth in range(1, max_depth + 1):
        cur_best, cur_val = None, -WIN - 1
        for labels, ns, winner in legal_turns(state):
            if winner is not None:
                val = WIN if winner == CYAN else -WIN
            else:
                val = search(ns, depth - 1, -WIN - 1, WIN + 1)
            if val > cur_val:
                cur_val, cur_best = val, labels
        best = (cur_best, cur_val)
        if cur_val >= WIN - 100:          # forced win found; no need to search deeper
            break
    return best


# ------------------------------------------------------------------ parsing / io
def parse(text):
    pieces, side, rnd = {}, CYAN, 1
    lines = [ln.rstrip() for ln in text.splitlines()]
    grid_rows = [ln for ln in lines if set(ln) <= set(".kpKP") and len(ln) == N0]
    if len(grid_rows) == N0:
        for r, row in enumerate(grid_rows):
            y = N0 - r
            for c, ch in enumerate(row):
                if ch == ".":
                    continue
                team = CYAN if ch.islower() else MAGENTA
                typ = KERNEL if ch.upper() == "K" else PING
                pieces[(c + 1, y)] = (team, typ)
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#round"):
            rnd = int(s.split()[1]); continue
        if s.startswith("#side"):
            side = CYAN if s.split()[1].lower().startswith("c") else MAGENTA; continue
        if s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) >= 3 and parts[0].lower()[0] in "cm" and parts[1].upper() in "KP":
            team = CYAN if parts[0].lower().startswith("c") else MAGENTA
            typ = KERNEL if parts[1].upper() == "K" else PING
            pieces[parse_square(parts[2])] = (team, typ)
    return State(pieces, side, rnd)


def render(state):
    rows = []
    for y in range(state.hi, state.lo - 1, -1):
        row = []
        for x in range(state.lo, state.hi + 1):
            occ = state.pieces.get((x, y))
            if not occ:
                row.append(".")
            else:
                ch = occ[1]
                row.append(ch.lower() if occ[0] == CYAN else ch.upper())
        rows.append(f"{y} " + " ".join(row))
    rows.append("  " + " ".join("ABCDEFGH"[state.lo - 1:state.hi]))
    return "\n".join(rows)


def main():
    args = [a for a in sys.argv[1:]]
    depth = 6
    if "--depth" in args:
        i = args.index("--depth"); depth = int(args[i + 1]); del args[i:i + 2]
    text = open(args[0]).read() if args else sys.stdin.read()
    state = parse(text)
    print("Board (round %d, %s to move):" % (state.rnd, "CYAN" if state.side == CYAN else "MAGENTA"))
    print(render(state))
    print()
    labels, val = best_turn(state, depth)
    if labels is None:
        print("No legal turn."); return
    verdict = ("FORCED WIN" if val >= WIN - 100
               else "FORCED LOSS" if val <= -WIN + 100
               else f"best effort (eval {val})")
    print(f"Play: {'  then  '.join(labels)}    [{verdict}]")


if __name__ == "__main__":
    main()
