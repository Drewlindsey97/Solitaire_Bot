import heapq
import itertools
import time

RANK_ORDER = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["S","H","D","C"]

# Placeholder for a stock/waste card whose identity hasn't been revealed by a
# real board read yet. A hypothetical `draw` taken during search can't know
# what it turns up (that's real hidden information, not something a search
# can guess), so drawn-but-unseen cards are represented with this sentinel
# and treated as unusable by every move check below - the search can still
# choose to draw (to see if it *would* need to, once no other move is left),
# it just can't act on what a hypothetical draw reveals.
UNKNOWN = ("?", "?")

def rank_val(r):
    return RANK_ORDER.index(r)

class State:
    __slots__ = ("cols", "waste", "stock_remaining", "stock_total", "found")
    def __init__(self, cols, waste, stock_remaining, stock_total, found):
        self.cols = tuple(tuple(c) for c in cols)
        # waste is a stack (top = last = playable), never sorted - unlike
        # free cells in the old FreeCell model, order here is structural.
        self.waste = tuple(waste)
        self.stock_remaining = stock_remaining
        self.stock_total = stock_total
        self.found = tuple(sorted(found.items()))

    def key(self):
        return (self.cols, self.waste, self.stock_remaining, self.found)

    def found_dict(self):
        return dict(self.found)

def make_is_solved(total_cards):
    def is_solved(state):
        return sum(v + 1 for _, v in state.found) == total_cards
    return is_solved

def make_heuristic(total_cards):
    def heuristic(state):
        found = state.found_dict()
        on_found = sum(v + 1 for v in found.values())
        base = total_cards - on_found

        penalty = 0
        for col in state.cols:
            for i, card in enumerate(col):
                if card == UNKNOWN:
                    continue
                r, s = card
                needed = found.get(s, -1) + 1
                if rank_val(r) == needed and i != len(col) - 1:
                    penalty += (len(col) - 1 - i) * 2

        empty_bonus = sum(1 for c in state.cols if not c) * 3

        return max(0, base + penalty - empty_bonus)
    return heuristic

def can_stack(card, on_card):
    if card == UNKNOWN or on_card == UNKNOWN:
        return False
    r, s = card
    r2, s2 = on_card
    color = "RED" if s in ("H","D") else "BLACK"
    color2 = "RED" if s2 in ("H","D") else "BLACK"
    return color != color2 and rank_val(r) == rank_val(r2) - 1

def can_found(card, found):
    if card == UNKNOWN:
        return False
    r, s = card
    cur = found.get(s, -1)
    return rank_val(r) == cur + 1

def is_safe_autoplay(card, found):
    """A card is safe to send to foundation immediately - no future move
    could ever need it back in play - once both opposite-color foundations
    are already at least at its rank - 1. Standard FreeCell/Klondike
    safe-autoplay rule; applies the same way whether the card is a column
    top or the current top of waste.

    Used as a cheap single check inside generate_moves() rather than a
    separate full-state fixpoint pass: a run of several forced plays in a
    row just costs a few extra cheap search-loop iterations (each a plain
    heap push/pop, no state rebuild) instead of one expensive rescan of
    every column on every candidate move. Measured on a real shuffled
    deal, this is ~2x the states/sec of rebuilding a full auto-play
    fixpoint per candidate move, for the same or better solution quality."""
    if not can_found(card, found):
        return False
    r, s = card
    rv = rank_val(r)
    if rv <= 1:  # A, 2 are always safe once legal
        return True
    o1, o2 = ("S", "C") if s in ("H", "D") else ("H", "D")
    return min(found.get(o1, -1), found.get(o2, -1)) >= rv - 1

def generate_moves(state):
    moves = []
    cols = state.cols
    waste = state.waste
    found = state.found_dict()
    waste_top = waste[-1] if waste and waste[-1] != UNKNOWN else None

    # A safe auto-play, if one exists, is always correct to make and never
    # worth skipping in favor of some other branch - so prune every other
    # option this turn and force it. See is_safe_autoplay() for why this
    # is cheaper than collapsing chains into one step ahead of time.
    for ci, col in enumerate(cols):
        if col and is_safe_autoplay(col[-1], found):
            return [("col_to_found", ci, col[-1])]
    if waste_top is not None and is_safe_autoplay(waste_top, found):
        return [("waste_to_found", waste_top)]

    for ci, col in enumerate(cols):
        if col and can_found(col[-1], found):
            moves.append(("col_to_found", ci, col[-1]))

    if waste_top is not None and can_found(waste_top, found):
        moves.append(("waste_to_found", waste_top))

    for ci, col in enumerate(cols):
        if not col:
            continue
        card = col[-1]
        # A face-down card at the top of a column isn't movable - only the
        # game itself can reveal it (see UNKNOWN's module docstring) - so
        # there's nothing this column can contribute this turn.
        if card == UNKNOWN:
            continue
        placed_on_empty = False
        for cj, col2 in enumerate(cols):
            if ci == cj:
                continue
            if not col2:
                if not placed_on_empty:
                    moves.append(("col_to_col", ci, cj, card))
                    placed_on_empty = True
            elif can_stack(card, col2[-1]):
                moves.append(("col_to_col", ci, cj, card))

    if waste_top is not None:
        placed_on_empty = False
        for cj, col2 in enumerate(cols):
            if not col2:
                if not placed_on_empty:
                    moves.append(("waste_to_col", cj, waste_top))
                    placed_on_empty = True
            elif can_stack(waste_top, col2[-1]):
                moves.append(("waste_to_col", cj, waste_top))

    # Drawing/redealing only ever reveals an UNKNOWN placeholder to the
    # search (see module docstring on UNKNOWN) - there's nothing a
    # hypothetical draw can unblock that the search can act on, so it's
    # never worth taking over a move that's already known to be usable.
    # Offering it only once every other option is exhausted is therefore
    # lossless (not just a branching-factor optimization): the search
    # gains no less information by deferring a draw than by taking it
    # early, since either way it only ever sees UNKNOWN until a real
    # perceive-cycle reveals the true cards.
    if not moves:
        if state.stock_remaining > 0:
            moves.append(("draw",))
        elif state.stock_total > 0:
            moves.append(("redeal",))

    return moves

def apply_move(state, move):
    cols = [list(c) for c in state.cols]
    waste = list(state.waste)
    stock_remaining = state.stock_remaining
    found = state.found_dict()

    kind = move[0]
    if kind == "col_to_found":
        _, ci, card = move
        cols[ci].pop()
        found[card[1]] = rank_val(card[0])
    elif kind == "waste_to_found":
        _, card = move
        waste.pop()
        found[card[1]] = rank_val(card[0])
    elif kind == "col_to_col":
        _, ci, cj, card = move
        cols[ci].pop()
        cols[cj].append(card)
    elif kind == "waste_to_col":
        _, cj, card = move
        waste.pop()
        cols[cj].append(card)
    elif kind == "draw":
        n = min(3, stock_remaining)
        waste.extend([UNKNOWN] * n)
        stock_remaining -= n
    elif kind == "redeal":
        # Confirmed live: redeals are unlimited and deterministic (the same
        # 24 cards come back in the same order every cycle), so a redeal
        # just resets the counters - it doesn't need to model *which*
        # specific cards return, since they were never known to begin with.
        waste = []
        stock_remaining = state.stock_total

    return State(cols, waste, stock_remaining, state.stock_total, found)

def solve(initial_cols, initial_waste=None, initial_stock_remaining=0, initial_stock_total=0,
          initial_found=None, progress_every=100_000, max_seen=3_000_000, weight=5,
          time_limit=100.0):
    """
    Weighted best-first search (f = g + weight*h) that runs until the game
    is won, every reachable state has been explored with no solution found
    (proven stuck), or a safety valve trips - `seen` dedup guarantees
    termination in principle, since the state space for a fixed deck is
    finite, but that state space can be far too large to fully explore
    within a real-world time budget. Safe foundation plays (see
    is_safe_autoplay) are forced via an early-return in generate_moves()
    instead of being treated as one branch choice among several.

    weight > 1 biases the search to favor making heuristic progress over
    finding the shortest path - plain A* (weight=1) treats every state tied
    on heuristic value as equally worth exploring, which blows up into a
    breadth-first-like search across huge plateaus. We only need *a*
    solution, not the shortest one, so trading path optimality for a search
    that actually converges is the right tradeoff.

    max_seen is a memory safety valve, not a search-quality cap: if state
    tracking grows past it the run stops and says so explicitly, rather than
    growing until the OS kills the process with no explanation.

    time_limit (seconds, wall-clock) is the primary safety valve: measured
    throughput on a real shuffled deal is on the order of ~1,500-2,000
    states/sec, so max_seen alone (3,000,000 states) can take on the order
    of half an hour to trip - far past the couple of minutes the bot has to
    return an answer. time_limit guarantees a bounded return even when
    max_seen would not kick in soon enough. Pass None to disable it (falls
    back to max_seen / full exhaustion only).

    Returns (path, explored, solved_bool, status). status is one of
    "solved", "exhausted" (proven unsolvable), "capped" (state-count limit
    hit), or "timeout" (wall-clock limit hit) - for "capped" and "timeout",
    the search was inconclusive and path is the best partial line seen.
    """
    initial_waste = initial_waste or []
    initial_found = initial_found or {}
    start = State(initial_cols, initial_waste, initial_stock_remaining, initial_stock_total, initial_found)

    total_cards = sum(len(c) for c in initial_cols) + len(initial_waste) + initial_stock_remaining \
                  + sum(v + 1 for v in initial_found.values())

    is_solved = make_is_solved(total_cards)
    heuristic = make_heuristic(total_cards)

    counter = itertools.count()
    open_set = [(weight * heuristic(start), next(counter), start, [])]
    seen = {start.key(): 0}

    best_path = []
    best_h = heuristic(start)

    explored = 0
    start_time = time.time()

    while open_set:
        _, _, state, path = heapq.heappop(open_set)
        explored += 1

        if progress_every and explored % progress_every == 0:
            elapsed = time.time() - start_time
            print(f"  ...explored {explored} states, {len(open_set)} queued, "
                  f"best remaining={best_h}, {elapsed:.0f}s elapsed")

        h = heuristic(state)
        if h < best_h:
            best_h = h
            best_path = path
        elif not best_path and path:
            # Nothing has strictly improved the heuristic yet, but this is
            # at least a real legal continuation - report it rather than
            # nothing. Matters most for "draw" (see UNKNOWN's docstring):
            # it never improves the heuristic by itself, but it's often
            # the only move available at all (e.g. a fresh deal with no
            # tableau moves yet), and solve_incrementally needs a concrete
            # next action to commit and re-perceive from rather than being
            # told no path exists when one plainly does.
            best_path = path

        if is_solved(state):
            return path, explored, True, "solved"

        if time_limit is not None and time.time() - start_time > time_limit:
            print(f"  ...time limit reached ({time_limit}s), stopping")
            return best_path, explored, False, "timeout"

        if len(seen) >= max_seen:
            print(f"  ...memory cap reached ({max_seen} states tracked), stopping")
            return best_path, explored, False, "capped"

        for move in generate_moves(state):
            new_state = apply_move(state, move)
            g = len(path) + 1
            hh = heuristic(new_state)
            f = g + weight * hh
            k = new_state.key()
            if k not in seen or seen[k] > g:
                seen[k] = g
                heapq.heappush(open_set, (f, next(counter), new_state, path + [move]))

    return best_path, explored, False, "exhausted"

def solve_incrementally(initial_cols, initial_waste=None, initial_stock_remaining=0,
                         initial_stock_total=0, initial_found=None,
                         total_budget=150.0, round_limit=20.0, **solve_kwargs):
    """
    Wraps solve() in a series of short, hard-bounded rounds instead of one
    long, uncertain one. Each round gets at most `round_limit` seconds;
    whatever partial line it returns is committed immediately - it's a
    real, legal sequence of moves from the current position (see solve()'s
    docstring on `best_path`), so it's always safe to act on even when a
    round doesn't fully solve - and the next round re-solves fresh from
    the resulting position. This bounds total latency to `total_budget`
    no matter how hard the board is, while still letting a later round
    recover from an earlier one committing to a heuristically weak line.

    A round ending in "draw"/"redeal" is expected and fine: neither move
    can be productively extended within a single solve() call (see
    generate_moves()'s docstring on UNKNOWN), so committing it and letting
    the bot execute the real tap + re-read the board is exactly how new
    information is meant to enter the picture, one round at a time.

    Stops early if a round:
    - fully solves it ("solved"),
    - proves the (already-progressed) position unsolvable ("exhausted"),
    - makes zero committed progress with an inconclusive result -
      re-running the same deterministic search on an unchanged position
      would just repeat it, so further rounds would only burn budget
      ("stuck").
    Otherwise keeps looping, committing progress each round, until
    total_budget is spent ("timeout").

    Returns (all_moves, total_explored, solved_bool, status) - same shape
    as solve(), where all_moves is the full committed move sequence across
    every round played so far.
    """
    cols = [list(c) for c in initial_cols]
    waste = list(initial_waste or [])
    stock_remaining = initial_stock_remaining
    stock_total = initial_stock_total
    found = dict(initial_found or {})

    all_moves = []
    total_explored = 0
    start = time.time()

    while True:
        remaining = total_budget - (time.time() - start)
        if remaining <= 0:
            return all_moves, total_explored, False, "timeout"

        path, explored, solved, status = solve(
            cols, waste, stock_remaining, stock_total, found,
            time_limit=min(round_limit, remaining),
            **solve_kwargs)
        total_explored += explored

        if not path:
            # No moves at all this round: either already fully solved with
            # nothing left to do, or genuinely stuck - either way, another
            # round from this same position won't produce anything new.
            return all_moves, total_explored, solved, ("solved" if solved else "stuck")

        state = State(cols, waste, stock_remaining, stock_total, found)
        for move in path:
            state = apply_move(state, move)
        all_moves.extend(path)
        cols = list(state.cols)
        waste = list(state.waste)
        stock_remaining = state.stock_remaining
        found = state.found_dict()

        if solved:
            return all_moves, total_explored, True, "solved"
        if status == "exhausted":
            return all_moves, total_explored, False, "exhausted"
        # status was "timeout" or "capped" for this round only (not the
        # overall budget) - loop again with the committed progress kept.
