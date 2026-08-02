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
SCORE_NULL_SHUFFLE = -10_000 # never: no reveal, no found, no empty column


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


def choose_move(state, exclude=None, allow_null_shuffle=False):
    """Best single move for the clock, or None if nothing is worth doing.

    Returns None rather than a null shuffle unless allow_null_shuffle is set,
    so the caller can draw instead of burning the move (and the round's time)
    on a board position that doesn't change.
    """
    moves = generate_moves(state, exclude=exclude)
    if not moves:
        return None
    scored = [(score_move(state, m), m) for m in moves]
    scored.sort(key=lambda sm: sm[0], reverse=True)
    best_score, best_move = scored[0]
    if best_score > SCORE_NULL_SHUFFLE / 2 or allow_null_shuffle:
        return best_move

    # Everything legal is a null shuffle. generate_moves() only offers a draw
    # when nothing else is legal, so it won't rescue us here - a shuffle is
    # "a move" as far as it's concerned. Drawing is strictly better: it costs
    # the same clock and at least turns over new cards, whereas the shuffle
    # provably changes nothing. This is the branch that ends the oscillation.
    if state.stock_remaining > 0:
        return ("draw",)
    if state.stock_total > 0:
        return ("redeal",)
    # No stock left and only shuffles remain: the position is genuinely stuck.
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
