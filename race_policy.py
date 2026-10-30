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
    UNKNOWN, RANK_ORDER, rank_val, generate_moves, apply_move, can_found,
    can_stack, is_safe_autoplay,
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
# Small per-ply cost so a gain reached sooner beats the same gain reached
# later. Without it, a lateral shuffle that merely *preserves* a reachable
# gain ties the move that actually advances toward it, and the tie breaks
# on move-generation order - observed as the bot ping-ponging one card
# between two symmetric landing columns forever while the payoff line
# (e.g. "unstack the 10, then the Queen reveals") never got played.
GAIN_STEP = 1


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
            # Prefer the column with the most face-down cards left: digging
            # there unlocks the most of the board. (Ranking by total column
            # length, face-up cards included, won 8-14 fewer games per 2000
            # simulated deals.)
            return SCORE_REVEAL + 10 * sum(1 for c in col if c == UNKNOWN)
        if _gains_empty_column(state, move):
            return SCORE_EMPTY_COLUMN
        # Neither reveals a card nor gains an empty column: pure relocation
        # of already-exposed cards. This is the move that burned half a live
        # round sliding one run between two columns.
        return SCORE_NULL_SHUFFLE

    if kind in ("draw", "redeal"):
        return SCORE_DRAW

    return 0


def _move_gain(state, move, after, defer=False):
    """Progress a single move makes, for lookahead accounting (0 = none).

    `after` is the already-simulated result of applying `move` to `state`:
    the caller has it in hand anyway, so empty-column gain is a direct
    count comparison instead of a second apply_move per search node.
    Mirrors score_move's ladder (found > reveal > empty), including the
    empty-column credit when founding a column's last card.
    """
    kind = move[0]
    if defer and _plain_docking(state, move):
        return 0  # held back (deferred docking): no reason to set it up
    if kind in ("col_to_found", "waste_to_found"):
        g = GAIN_FOUNDATION
        if kind == "col_to_found":
            col = state.cols[move[1]]
            if _col_reveals(col, 1):
                g += GAIN_REVEAL
            elif _empties_column(col, 1):
                g += GAIN_EMPTY
        return g
    if kind == "col_to_col":
        _, ci, cj, card, run_length = move
        if _col_reveals(state.cols[ci], run_length):
            return GAIN_REVEAL
        if _empty_count(after) > _empty_count(state):
            return GAIN_EMPTY
    return 0


def _reachable_gain(state, depth, seen, defer=False):
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
        # GAIN_STEP charges each ply, so equal raw gains rank by how soon
        # they arrive (see the constant's comment). The floor of 0 ("stop
        # here") keeps gainless tails from dragging a real gain negative.
        val = (_move_gain(state, m, ns, defer) - GAIN_STEP
               + _reachable_gain(ns, depth - 1, seen, defer))
        seen.discard(k)
        if val > best:
            best = val
    return best


# ------------------------------------------------------------------------------
# Foundation pull-down ("found_to_col")
# ------------------------------------------------------------------------------
# Klondike lets the top card of a foundation come back down onto the tableau.
# Observed live: the bot stopped as "stuck" with 5S banked, a 6D open on the
# tableau and a red 4 coming up in the draw cycle. Pulling the 5S back down
# onto the 6D gave the red 4 a home and the round carried on - but no move
# generator here offered that move, so the bot drew through the stock until
# the no-progress stop fired.
#
# A pull-down undoes a foundation play, so taken carelessly it can loop
# forever (down, back up, down...). It is therefore boxed in on every side:
#   1. Only the race policy proposes it; generate_moves() never does, so the
#      search and Monte Carlo solvers are unaffected.
#   2. Last resort: considered only when the policy would otherwise draw,
#      redeal, or give up - never when any direct progress move exists.
#   3. Immediate payoff: the pulled card must at once accept the waste top
#      or a column run whose move reveals a face-down card. (Emptying a
#      column was tried as a payoff too; over 2000 simulated deals it lost
#      cards on net, so it doesn't count.) That follow-up is played in the same batch, which covers the
#      pulled card so it cannot simply go straight back up.
#   4. Never a card nobody could need: if the card would be a safe autoplay
#      straight back to its foundation (so no tableau card can ever want to
#      sit on it - this also rules out every Ace and 2), it stays put.
#   5. Once per card: the caller passes every card already pulled down this
#      run in `pulled_down`, and a card in that set is never pulled again.
#      With 52 cards that bounds the total number of pull-downs per run, so
#      even a pathological board cannot cycle on this move.
#   6. At most one pull-down chain per batch (plan_batch), of at most
#      MAX_PULLDOWN_CHAIN pulls; the payoff must land on the last card
#      pulled. A King may come down into an empty column.
#   7. The bot's no-progress stop still applies on top of all of this.


def _pulldown_followup(after, card, cj):
    """The move that uses a just-pulled-down card, or None.

    `after` is the state with `card` already sitting on top of column cj.
    Preference: a column run that reveals a face-down card, then the waste
    top. A follow-up that only relocates exposed cards (or merely empties a
    column) is not a payoff and is never accepted.
    """
    reveal_move = None
    for ci, col in enumerate(after.cols):
        if ci == cj or not col or col[-1] == UNKNOWN:
            continue
        max_run = 1
        while (max_run < len(col) and col[-max_run - 1] != UNKNOWN
               and can_stack(col[-max_run], col[-max_run - 1])):
            max_run += 1
        for run_length in range(1, max_run + 1):
            start = col[-run_length]
            if not can_stack(start, card):
                continue
            if _col_reveals(col, run_length):
                reveal_move = reveal_move or ("col_to_col", ci, cj, start, run_length)
    if reveal_move:
        return reveal_move
    waste_top = after.waste[-1] if after.waste and after.waste[-1] != UNKNOWN else None
    if waste_top is not None and can_stack(waste_top, card):
        return ("waste_to_col", cj, waste_top)
    return None


# Longest chain of pull-downs considered before the payoff move. Live
# 2026-10-10 the bot stopped at 44 cards: J-D on the waste needed a black
# queen, and the only way was K-S off its foundation into an empty column,
# then Q-S onto the K-D. Chains are rare, so keep them short.
MAX_PULLDOWN_CHAIN = 3


def _pulldown_moves(state, pulled_down, used):
    """Single pull-down candidates: a foundation top onto a column card that
    takes it, or a King into the first empty column."""
    found = state.found_dict()
    for suit, top in sorted(found.items()):
        if top < 0:
            continue
        card = (RANK_ORDER[top], suit)
        if card in pulled_down or card in used:
            continue
        lowered = dict(found)
        lowered[suit] = top - 1
        if is_safe_autoplay(card, lowered):
            continue
        empty_offered = False
        for cj, col in enumerate(state.cols):
            if not col:
                if card[0] == "K" and not empty_offered:
                    empty_offered = True
                    yield ("found_to_col", card, cj)
                continue
            if col[-1] != UNKNOWN and can_stack(card, col[-1]):
                yield ("found_to_col", card, cj)


def choose_pulldown(state, pulled_down=frozenset()):
    """Best (pulldown_moves, followup_move) for this state, or None.

    pulldown_moves is a chain of 1..MAX_PULLDOWN_CHAIN pull-downs; the
    follow-up must land on the LAST card pulled, so every pull in the chain
    is needed for the payoff. Applies every guard in the section comment
    above except "last resort", which choose_move() enforces.
    """
    best, best_key = None, None

    def search(sim, chain, used):
        nonlocal best, best_key
        if len(chain) >= MAX_PULLDOWN_CHAIN:
            return
        for move in _pulldown_moves(sim, pulled_down, used):
            after = apply_move(sim, move)
            card, cj = move[1], move[2]
            follow = _pulldown_followup(after, card, cj)
            new_chain = chain + [move]
            if follow is not None:
                # shorter chains first; reveal (0) beats waste (1); then the
                # lowest cards, keeping foundations as high as possible.
                kind_rank = 0 if follow[0] == "col_to_col" else 1
                key = (len(new_chain), kind_rank, sum(rank_val(m[1][0]) for m in new_chain))
                if best_key is None or key < best_key:
                    best_key, best = key, (tuple(new_chain), follow)
            else:
                search(after, new_chain, used | {card})

    search(state, [], frozenset())
    return best


# ------------------------------------------------------------------------------
# Auto-complete endgame
# ------------------------------------------------------------------------------
# Observed live (user report 2026-10-10): once every tableau card is face up
# AND the stock and waste are both empty, the game finishes the round by
# itself. Docking cards one at a time before that point only burns clock, so
# with nothing face down the policy's job becomes getting the remaining
# stock/waste cards into play, then stopping.


def no_hidden_cards(state):
    return all(card != UNKNOWN for col in state.cols for card in col)


def auto_complete_ready(state):
    """Every card face up and the draw pile fully played: the game takes over."""
    return state.stock_remaining == 0 and not state.waste and no_hidden_cards(state)


def _waste_playable(state):
    top = state.waste[-1] if state.waste else None
    if top is None or top == UNKNOWN:
        return False
    if can_found(top, state.found_dict()):
        return True
    return any((not col and top[0] == "K") or (col and can_stack(top, col[-1]))
               for col in state.cols)


ENDGAME_DOCK_DEPTH = 8


def _legal_dockings(state):
    found = state.found_dict()
    return [("col_to_found", ci, col[-1]) for ci, col in enumerate(state.cols)
            if col and can_found(col[-1], found)]


def _first_enabling_move(state, candidates, goal=_waste_playable):
    """First move of the shortest line that reaches `goal` (by default: the
    waste card becomes playable).

    The line starts with one of `candidates` and continues with dockings
    only. Dockings only ever raise the foundations, so the search cannot
    cycle; depth and a visited set keep it small.
    """
    frontier = []
    seen = set()
    for m in candidates:
        try:
            nxt = apply_move(state, m)
        except Exception:
            continue
        if goal(nxt):
            return m
        if nxt.key() not in seen:
            seen.add(nxt.key())
            frontier.append((nxt, m))
    for _ in range(ENDGAME_DOCK_DEPTH - 1):
        nxt_frontier = []
        for sim, first in frontier:
            for d in _legal_dockings(sim):
                after = apply_move(sim, d)
                if goal(after):
                    return first
                if after.key() not in seen:
                    seen.add(after.key())
                    nxt_frontier.append((after, first))
        frontier = nxt_frontier
        if not frontier:
            break
    return None


def _choose_endgame(state, exclude, pulldown_or):
    """Move choice once no face-down cards remain (see section above)."""
    exclude = set(exclude or ())
    # Tableau-to-foundation plays are never forced here: they only help when
    # they open a landing spot for the waste card.
    found = state.found_dict()
    dockings = {("col_to_found", ci, col[-1]) for ci, col in enumerate(state.cols)
                if col and can_found(col[-1], found)}
    moves = generate_moves(state, exclude=exclude | dockings)
    waste_moves = [m for m in moves if m[0] in ("waste_to_found", "waste_to_col")]
    if waste_moves:
        waste_moves.sort(key=lambda m: 0 if m[0] == "waste_to_found" else 1)
        return waste_moves[0]
    if not _waste_playable(state):
        first = _first_enabling_move(state, [m for m in dockings if m not in exclude]
                                     + [m for m in moves if m[0] == "col_to_col"])
        if first is not None:
            return first
    stock = [m for m in moves if m[0] in ("draw", "redeal")]
    if not stock:
        if state.stock_remaining > 0:
            stock = [("draw",)]
        elif state.stock_total > 0 and state.waste:
            stock = [("redeal",)]
    stock = [m for m in stock if m not in exclude]
    return pulldown_or(stock[0] if stock else None)


# ------------------------------------------------------------------------------
# Deferred docking (user report 2026-10-16): with face-down cards still on the
# board the policy used to dock every eligible tableau card in rank order -
# one live batch spent 12 taps banking 3S..9D while 3 cards sat face down and
# the stock still held cards. The game docks everything itself once the board
# is face up and the draw pile is empty, so those taps were clock spent on
# work auto-complete does for free, and docking can take away a landing spot
# a waste card needed. So a "plain" docking - one that reveals nothing and
# empties no column - is held back. It is played only when a short docking
# chain opens the waste card or a reveal/empty-column move, or when nothing
# but dead shuffles remains with the draw pile exhausted. The caller's stall
# fallback (endgame=False) restores eager docking so an unwinnable round
# still banks its cards before the clock runs out.


DEFER_DOCKING = False  # off: cost wins in simulation (2026-10-16), see section comment
# Hold dockings back only this close to the end (face-down cards left).
DEFER_DOCKING_MAX_HIDDEN = 4


def _defer_docking(state, endgame):
    return (endgame and DEFER_DOCKING
            and sum(c == UNKNOWN for col in state.cols for c in col) <= DEFER_DOCKING_MAX_HIDDEN)


def _plain_docking(state, move):
    """A docking only auto-complete needs: it reveals nothing, empties no
    column, and the next card of its suit already sits face up on the
    tableau - so banking it now cannot let a draw-pile or face-down card go
    up. (Holding back every non-revealing docking, low cards included, cost
    ~12% of wins in simulation: waste cards lost their foundation home.)"""
    if move[0] != "col_to_found":
        return False
    col = state.cols[move[1]]
    if _col_reveals(col, 1) or _empties_column(col, 1):
        return False
    rank, suit = move[2]
    i = RANK_ORDER.index(rank)
    if i + 1 >= len(RANK_ORDER):
        return True  # a King: nothing builds on it
    nxt = (RANK_ORDER[i + 1], suit)
    return any(nxt in c for c in state.cols)


def _opens_progress(state):
    """Waste card playable, or a tableau move that reveals or empties a column."""
    if _waste_playable(state):
        return True
    for m in generate_moves(state):
        if m[0] == "col_to_found":
            col = state.cols[m[1]]
            if _col_reveals(col, 1) or _empties_column(col, 1):
                return True
        elif m[0] == "col_to_col":
            if _col_reveals(state.cols[m[1]], m[4]) or _gains_empty_column(state, m):
                return True
    return False


def choose_move(state, exclude=None, allow_null_shuffle=False,
                lookahead_depth=LOOKAHEAD_DEPTH, pulled_down=None, endgame=True):
    """Best single move for the clock, or None if nothing is worth doing.

    Direct progress (a foundation play, a reveal, an empty-column gain, a waste
    unload) is always taken immediately. Only when every legal move is a bare
    relocation does the bounded lookahead run, to distinguish a setup shuffle
    that unblocks a foundation from a dead one. Returns None rather than a dead
    shuffle unless allow_null_shuffle is set, so the caller can draw instead.
    """
    def pulldown_or(fallback):
        # Last resort only (see the pull-down section above): reached solely
        # when the alternative is a draw, a redeal, or giving up. Passing
        # pulled_down=None disables pull-downs entirely.
        if pulled_down is not None:
            pick = choose_pulldown(state, pulled_down)
            if pick is not None and (not exclude or pick[0][0] not in exclude):
                return pick[0][0]
        return fallback

    if endgame and auto_complete_ready(state):
        return None  # the game finishes the round itself from here
    if endgame and no_hidden_cards(state):
        return _choose_endgame(state, exclude, pulldown_or)

    moves = generate_moves(state, exclude=exclude)
    if not moves:
        return pulldown_or(None)
    # With confirmed stock history generate_moves() also offers the draw
    # next to tableau moves. The ladder below is unchanged by that: real
    # progress first, then setup relocations found by lookahead, and only
    # then the draw (simulated over 2000 deals, letting a known draw outrank
    # setup relocations cost ~1.6 banked cards per game).
    stock_moves = [m for m in moves if m[0] in ("draw", "redeal")]
    moves = [m for m in moves if m[0] not in ("draw", "redeal")]
    # Deferred docking (see section above): only while endgame is on, i.e.
    # until the caller's stall fallback asks for cards to be banked.
    held = [m for m in moves if _plain_docking(state, m)] if _defer_docking(state, endgame) else []
    if held:
        moves = [m for m in moves if m not in held]
        held.sort(key=lambda m: rank_val(m[2][0]))
    if not moves and not held:
        return pulldown_or(stock_moves[0])
    scored = sorted(((score_move(state, m), m) for m in moves),
                    key=lambda sm: sm[0], reverse=True)
    if scored and (scored[0][0] > SCORE_NULL_SHUFFLE / 2 or allow_null_shuffle):
        return scored[0][1]
    if held:
        first = _first_enabling_move(state, held, goal=_opens_progress)
        if first is not None:
            return first

    # Everything legal is a bare relocation. Before drawing past the position,
    # check whether any of these relocations is actually a setup that leads to
    # a foundation or reveal within a few plies. If so, take the one that
    # unblocks the most - this is what lets the endgame progress instead of
    # freezing (previously it drew here and spun until the clock ran out).
    if lookahead_depth > 0 and scored:
        best_setup, best_gain = None, 0
        for _score, m in scored:
            try:
                ns = apply_move(state, m)
            except Exception:
                continue
            gain = _reachable_gain(ns, lookahead_depth - 1, {state.key(), ns.key()},
                                   defer=_defer_docking(state, endgame))
            if gain > best_gain:
                best_gain, best_setup = gain, m
        if best_setup is not None:
            return best_setup

    # No relocation reaches any progress: draw to turn over new cards (same
    # clock cost, but the shuffle provably changes nothing).
    if stock_moves:
        return pulldown_or(stock_moves[0])
    if state.stock_remaining > 0:
        return pulldown_or(("draw",))
    if state.stock_total > 0 and state.waste:
        return pulldown_or(("redeal",))
    # Draw pile exhausted: bank a held card rather than stall.
    if held:
        return pulldown_or(held[0])
    # No stock left and only dead shuffles remain: genuinely stuck.
    return pulldown_or(None)


def _waste_top_known(state):
    return bool(state.waste) and state.waste[-1] != UNKNOWN


def plan_batch(state, max_moves=8, exclude=None, pulled_down=None, endgame=True):
    """Greedy sequence of moves from one screen read.

    Screen capture plus board read is the dominant per-cycle cost (~1s), so
    playing several moves per read is the main lever on moves-per-minute. The
    simulation uses the real apply_move, and stops at the first move whose
    outcome we can't predict:

    - draw/redeal reveal physically unknown cards, so anything planned after
      one would be built on invented state. Exception: when confirmed stock
      history (stock_history.py) supplies the drawn identities, planning
      continues through the draw and through waste plays.
    - a newly exposed card remains UNKNOWN, so its column is unavailable
      until the next read. Independent moves on known cards can continue,
      but drawing waits until the newly exposed card has been read.

    endgame: with every tableau card face up, play the stock/waste out and
    leave docking to the game's auto-complete (see that section). Callers
    turn it off when that endgame stalls, so cards still get banked before
    the clock runs out.

    pulled_down: cards already pulled off a foundation this run (see the
    pull-down section). None disables pull-downs. A pull-down is always
    followed in the same batch by the move it was taken for, and at most
    one pull-down happens per batch. A batch never ends on a pull-down: if
    there is no room for its follow-up, the pull-down is not played.
    """
    batch = []
    sim = state
    seen = {sim.key()}
    needs_reveal_read = False
    forced = []
    pulldown_done = False
    for _ in range(max_moves):
        if forced:
            move = forced.pop(0)
        else:
            move = choose_move(
                sim, exclude=exclude if not batch else None,
                pulled_down=None if (pulled_down is None or pulldown_done) else pulled_down,
                endgame=endgame,
            )
        if move is None:
            break
        if move[0] == "found_to_col" and not pulldown_done:
            pick = choose_pulldown(sim, pulled_down or frozenset())
            plan = list(pick[0]) + [pick[1]] if pick and pick[0][0] == move else None
            if plan is None or len(batch) + len(plan) > max_moves:
                # No room for the pull-down and its payoff in this batch (or
                # no payoff after all): play what we would have played
                # without pull-downs instead - usually the draw.
                pulldown_done = True
                move = choose_move(sim, exclude=exclude if not batch else None, endgame=endgame)
                if move is None:
                    break
            else:
                pulldown_done = True
                forced = plan[1:]
        if move[0] in ("draw", "redeal"):
            if needs_reveal_read:
                break
            nxt = apply_move(sim, move)
            if nxt.key() in seen:
                break
            seen.add(nxt.key())
            batch.append(move)
            sim = nxt
            # Keep planning only when stock history has confirmed the card
            # this draw turns up (a redeal turns up nothing; the draw that
            # follows it is judged on its own). Without confirmed history the
            # new waste top is UNKNOWN and the batch ends here, as before.
            if move[0] == "draw" and not _waste_top_known(sim):
                break
            continue
        reveals = (
            move[0] == "col_to_col" and _col_reveals(sim.cols[move[1]], move[4])
            or move[0] == "col_to_found" and _col_reveals(sim.cols[move[1]], 1)
        )
        nxt = apply_move(sim, move)
        # Defensive: never let the local simulation loop even if scoring
        # somehow permits a cycle. Checked BEFORE committing the move to the
        # batch - a move that walks the board back to a state this batch
        # already visited provably undoes progress, so it must not be
        # physically executed as the batch's parting shot.
        k = nxt.key()
        if k in seen:
            break
        seen.add(k)
        batch.append(move)
        sim = nxt
        needs_reveal_read = needs_reveal_read or reveals
        if move[0] in ("waste_to_found", "waste_to_col"):
            # The next waste card was partly covered; read its full face
            # before deciding to play it or draw past it - unless stock
            # history already confirmed it. An emptied waste also ends the
            # batch: the read may not show older waste behind the fan.
            if not _waste_top_known(sim):
                break
    # Belt and braces: a pull-down must never be the last physical move of a
    # batch (its follow-up is what makes it worth doing).
    while batch and batch[-1][0] == "found_to_col":
        batch.pop()
    return batch
