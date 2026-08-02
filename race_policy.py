"""Greedy move policy for the timed round.

This game is a race, not a puzzle: a countdown runs (~3 minutes observed) and
the round ends when it expires, scoring however many cards reached the
foundations. That makes deep search the wrong tool twice over:

- Time spent thinking is time not playing. The search solver was given 5s per
  cycle to return, in practice, a single move (hidden cards mean it can't plan
  past the first face-down card anyway, so it reports "exhausted" constantly).
  At ~3s/cycle for ~1 useful move, a 3-minute round allows ~55 moves - fewer
  than a full solve needs, so the round can expire before a win is reachable
  even with perfect play.
- Search optimizes for "is this deal solvable", not "what earns the most in
  the next 180 seconds". Observed live: it spent the back half of a round
  sliding one exposed 8S run between two columns, which reveals nothing and
  founds nothing - each shuffle burning a move the clock was charging for.

So this module ranks the legal moves generate_moves() already produces by how
much progress each makes, picks the best instantly, and - critically - refuses
the null shuffle outright. It reuses the real move generator and rules rather
than reimplementing them, so anything illegal here was already illegal there.
"""

from freecell_solver import (
    UNKNOWN, rank_val, generate_moves, apply_move, can_found,
)

# Score weights. Ordering matters far more than exact magnitudes: the intent
# is a strict preference ladder, with ties broken by cheap positional sense.
SCORE_FOUNDATION = 1000      # a card banked is the only thing that scores
SCORE_REVEAL = 500           # uncovering a face-down card is the engine of progress
SCORE_EMPTY_COLUMN = 120     # emptying a column entirely (King parking space)
SCORE_WASTE_TO_COL = 60      # frees the waste and puts a known card in play
SCORE_FROM_WASTE_DEPTH = 8   # prefer digging the waste over idle tableau moves
SCORE_DRAW = 10              # better than nothing, but never over a real move
SCORE_NULL_SHUFFLE = -10_000 # no reveal/found/empty on its own - see lookahead

# Bounded lookahead, used ONLY when every legal move is a bare relocation (the
# stuck endgame that froze a live round at 12 cards). A relocation that reveals
# or founds nothing by itself can still be the setup that unblocks a foundation
# a move or two later ("move this run aside to expose the 2 spades needs").
# We look a few plies ahead over KNOWN cards - drawing/redealing reveals
# physically unknown cards, so it isn't simulated - and, if a setup shuffle
# demonstrably leads to progress, take it instead of drawing past a winnable
# position. Depth stays small: it runs per cycle and must return instantly.
LOOKAHEAD_DEPTH = 3
GAIN_FOUNDATION = 100
GAIN_REVEAL = 40
GAIN_EMPTY = 20


def _col_reveals(col, run_length):
    """Does lifting run_length cards off this column expose a face-down card?"""
    if run_length >= len(col):
        return False
    return col[-run_length - 1] == UNKNOWN


def _empties_column(col, run_length):
    return run_length >= len(col)


def _empty_count(state):
    return sum(1 for c in state.cols if not c)


def _gains_empty_column(state, move):
    """Does this move strictly increase the number of empty columns?

    Asked by simulation rather than by inspecting the source column alone:
    moving a run out of a column it fully occupies empties that column, but
    if it lands on an empty column the net change is zero. That distinction
    is exactly what lets a run slide back and forth between two positions
    while each individual move looks like progress.
    """
    try:
        after = apply_move(state, move)
    except Exception:
        return False
    return _empty_count(after) > _empty_count(state)


def score_move(state, move):
    """Rank one legal move by progress-per-move. Higher is better.

    A move that neither banks a card, reveals a face-down card, nor empties a
    column is a null shuffle - the exposed-run relocation that burned half a
    round live. It scores below every alternative including a draw, so it is
    only ever chosen when literally nothing else is legal.
    """
    kind = move[0]

    if kind in ("col_to_found", "waste_to_found"):
        # Banking is the only scoring action; prefer low ranks first so the
        # foundations stay level and more cards stay eligible.
        card = move[2] if kind == "col_to_found" else move[1]
        score = SCORE_FOUNDATION - rank_val(card[0])
        if kind == "col_to_found":
            col = state.cols[move[1]]
            if _col_reveals(col, 1):
                score += SCORE_REVEAL
            elif _empties_column(col, 1):
                score += SCORE_EMPTY_COLUMN
        return score
    # (col_to_col handled below via simulation)

    if kind == "waste_to_col":
        # Always at least mildly useful: the waste only moves forward.
        return SCORE_WASTE_TO_COL + SCORE_FROM_WASTE_DEPTH

    if kind == "col_to_col":
        _, ci, cj, card, run_length = move
        col = state.cols[ci]
        if _col_reveals(col, run_length):
            # Deeper stacks are worth more: they unlock more of the board.
            return SCORE_REVEAL + min(len(col), 12)
        if _gains_empty_column(state, move):
            return SCORE_EMPTY_COLUMN
        # Neither reveals a card nor gains an empty column: pure relocation
        # of already-exposed cards. This is the move that burned half a live
        # round sliding one run between two columns.
        return SCORE_NULL_SHUFFLE

    if kind in ("draw", "redeal"):
        return SCORE_DRAW

    return 0


def _move_gain(state, move):
    """Progress a single move makes, for lookahead accounting (0 = none)."""
    kind = move[0]
    if kind in ("col_to_found", "waste_to_found"):
        g = GAIN_FOUNDATION
        if kind == "col_to_found" and _col_reveals(state.cols[move[1]], 1):
            g += GAIN_REVEAL
        return g
    if kind == "col_to_col":
        _, ci, cj, card, run_length = move
        if _col_reveals(state.cols[ci], run_length):
            return GAIN_REVEAL
        if _gains_empty_column(state, move):
            return GAIN_EMPTY
    return 0


def _reachable_gain(state, depth, seen):
    """Best total progress (founds + reveals + empties) reachable within `depth`
    plies over known cards. Draw/redeal are not simulated (they turn over cards
    we can't see). A visited-set prevents the relocation shuffles from looping
    inside the search."""
    if depth == 0:
        return 0
    best = 0
    for m in generate_moves(state):
        if m[0] in ("draw", "redeal"):
            continue
        try:
            ns = apply_move(state, m)
        except Exception:
            continue
        k = ns.key()
        if k in seen:
            continue
        seen.add(k)
        val = _move_gain(state, m) + _reachable_gain(ns, depth - 1, seen)
        seen.discard(k)
        if val > best:
            best = val
    return best


def choose_move(state, exclude=None, allow_null_shuffle=False,
                lookahead_depth=LOOKAHEAD_DEPTH):
    """Best single move for the clock, or None if nothing is worth doing.

    Direct progress (a foundation play, a reveal, an empty-column gain, a waste
    unload) is always taken immediately. Only when every legal move is a bare
    relocation does the bounded lookahead run, to distinguish a setup shuffle
    that unblocks a foundation from a dead one. Returns None rather than a dead
    shuffle unless allow_null_shuffle is set, so the caller can draw instead.
    """
    moves = generate_moves(state, exclude=exclude)
    if not moves:
        return None
    scored = [(score_move(state, m), m) for m in moves]
    scored.sort(key=lambda sm: sm[0], reverse=True)
    best_score, best_move = scored[0]
    if best_score > SCORE_NULL_SHUFFLE / 2 or allow_null_shuffle:
        return best_move

    # Everything legal is a bare relocation. Before drawing past the position,
    # check whether any of these relocations is actually a setup that leads to
    # a foundation or reveal within a few plies. If so, take the one that
    # unblocks the most - this is what lets the endgame progress instead of
    # freezing (previously it drew here and spun until the clock ran out).
    if lookahead_depth > 0:
        best_setup, best_gain = None, 0
        for _score, m in scored:
            try:
                ns = apply_move(state, m)
            except Exception:
                continue
            gain = _reachable_gain(ns, lookahead_depth - 1, {state.key(), ns.key()})
            if gain > best_gain:
                best_gain, best_setup = gain, m
        if best_setup is not None:
            return best_setup

    # No relocation reaches any progress: draw to turn over new cards (same
    # clock cost, but the shuffle provably changes nothing).
    if state.stock_remaining > 0:
        return ("draw",)
    if state.stock_total > 0:
        return ("redeal",)
    # No stock left and only dead shuffles remain: genuinely stuck.
    return None


def plan_batch(state, max_moves=8, exclude=None):
    """Greedy sequence of moves from one screen read.

    Screen capture plus board read is the dominant per-cycle cost (~1s), so
    playing several moves per read is the main lever on moves-per-minute. The
    simulation uses the real apply_move, and stops at the first move whose
    outcome we can't predict:

    - draw/redeal reveal physically unknown cards, so anything planned after
      one would be built on invented state
    - a move that exposes a face-down card makes the rest of that column
      UNKNOWN to us; it is safe to execute (it's the move we want most) but we
      stop after it and re-read rather than plan blind.
    """
    batch = []
    sim = state
    seen = {sim.key()}
    for _ in range(max_moves):
        move = choose_move(sim, exclude=exclude if not batch else None)
        if move is None:
            break
        batch.append(move)
        if move[0] in ("draw", "redeal"):
            break
        reveals = (
            move[0] == "col_to_col" and _col_reveals(sim.cols[move[1]], move[4])
            or move[0] == "col_to_found" and _col_reveals(sim.cols[move[1]], 1)
        )
        sim = apply_move(sim, move)
        if reveals:
            break
        # Defensive: never let the local simulation loop even if scoring
        # somehow permits a cycle.
        k = sim.key()
        if k in seen:
            break
        seen.add(k)
    return batch
