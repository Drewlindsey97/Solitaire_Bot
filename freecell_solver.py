import heapq
import itertools
import time

RANK_ORDER = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["S","H","D","C"]

def rank_val(r):
    return RANK_ORDER.index(r)

class State:
    __slots__ = ("cols", "free", "found")
    def __init__(self, cols, free, found):
        self.cols = tuple(tuple(c) for c in cols)
        self.free = tuple(sorted(free))
        self.found = tuple(sorted(found.items()))

    def key(self):
        return (self.cols, self.free, self.found)

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
            for i, (r, s) in enumerate(col):
                needed = found.get(s, -1) + 1
                if rank_val(r) == needed and i != len(col) - 1:
                    penalty += (len(col) - 1 - i) * 2

        empty_bonus = sum(1 for c in state.cols if not c) * 3
        free_bonus = (4 - len(state.free))

        return max(0, base + penalty - empty_bonus - free_bonus)
    return heuristic

def can_stack(card, on_card):
    r, s = card
    r2, s2 = on_card
    color = "RED" if s in ("H","D") else "BLACK"
    color2 = "RED" if s2 in ("H","D") else "BLACK"
    return color != color2 and rank_val(r) == rank_val(r2) - 1

def can_found(card, found):
    r, s = card
    cur = found.get(s, -1)
    return rank_val(r) == cur + 1

def is_safe_autoplay(card, found):
    """A card is safe to send to foundation immediately - no future move
    could ever need it back in play - once both opposite-color foundations
    are already at least at its rank - 1. Standard FreeCell auto-play rule.

    Used as a cheap single check inside generate_moves() rather than a
    separate full-state fixpoint pass: a run of several forced plays in a
    row just costs a few extra cheap search-loop iterations (each a plain
    heap push/pop, no state rebuild) instead of one expensive rescan of
    every column on every candidate move. Measured on a real shuffled
    deal, this is ~2x the states/sec of rebuilding a full auto-play
    fixpoint per candidate move, for the same or better solution quality."""
    r, s = card
    if not can_found(card, found):
        return False
    rv = rank_val(r)
    if rv <= 1:  # A, 2 are always safe once legal
        return True
    o1, o2 = ("S", "C") if s in ("H", "D") else ("H", "D")
    return min(found.get(o1, -1), found.get(o2, -1)) >= rv - 1

def generate_moves(state, last_move=None):
    moves = []
    cols = state.cols
    free = state.free
    found = state.found_dict()

    # A safe auto-play, if one exists, is always correct to make and never
    # worth skipping in favor of some other branch - so prune every other
    # option this turn and force it. See is_safe_autoplay() for why this
    # is cheaper than collapsing chains into one step ahead of time.
    for ci, col in enumerate(cols):
        if col and is_safe_autoplay(col[-1], found):
            return [("col_to_found", ci, col[-1])]
    for card in free:
        if is_safe_autoplay(card, found):
            return [("free_to_found", card)]

    for ci, col in enumerate(cols):
        if col and can_found(col[-1], found):
            moves.append(("col_to_found", ci, col[-1]))

    for card in free:
        if can_found(card, found):
            moves.append(("free_to_found", card))

    for ci, col in enumerate(cols):
        if not col:
            continue
        card = col[-1]
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

    for card in free:
        placed_on_empty = False
        for cj, col2 in enumerate(cols):
            if not col2:
                if not placed_on_empty:
                    moves.append(("free_to_col", cj, card))
                    placed_on_empty = True
            elif can_stack(card, col2[-1]):
                moves.append(("free_to_col", cj, card))

    if len(free) < 4:
        for ci, col in enumerate(cols):
            if col:
                move = ("col_to_free", ci, col[-1])
                if last_move and last_move[0] == "free_to_col" and last_move[2] == col[-1]:
                    continue
                moves.append(move)

    return moves

def apply_move(state, move):
    cols = [list(c) for c in state.cols]
    free = list(state.free)
    found = state.found_dict()

    kind = move[0]
    if kind == "col_to_found":
        _, ci, card = move
        cols[ci].pop()
        found[card[1]] = rank_val(card[0])
    elif kind == "free_to_found":
        _, card = move
        free.remove(card)
        found[card[1]] = rank_val(card[0])
    elif kind == "col_to_col":
        _, ci, cj, card = move
        cols[ci].pop()
        cols[cj].append(card)
    elif kind == "col_to_free":
        _, ci, card = move
        cols[ci].pop()
        free.append(card)
    elif kind == "free_to_col":
        _, cj, card = move
        free.remove(card)
        cols[cj].append(card)

    return State(cols, free, found)

def solve(initial_cols, initial_free=None, initial_found=None,
          progress_every=100_000, max_seen=3_000_000, weight=5,
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
    initial_free = initial_free or []
    initial_found = initial_found or {}
    start = State(initial_cols, initial_free, initial_found)

    total_cards = sum(len(c) for c in initial_cols) + len(initial_free) \
                  + sum(v + 1 for v in initial_found.values())

    is_solved = make_is_solved(total_cards)
    heuristic = make_heuristic(total_cards)

    counter = itertools.count()
    open_set = [(weight * heuristic(start), next(counter), start, [], None)]
    seen = {start.key(): 0}

    best_path = []
    best_h = heuristic(start)

    explored = 0
    start_time = time.time()

    while open_set:
        _, _, state, path, last_move = heapq.heappop(open_set)
        explored += 1

        if progress_every and explored % progress_every == 0:
            elapsed = time.time() - start_time
            print(f"  ...explored {explored} states, {len(open_set)} queued, "
                  f"best remaining={best_h}, {elapsed:.0f}s elapsed")

        h = heuristic(state)
        if h < best_h:
            best_h = h
            best_path = path

        if is_solved(state):
            return path, explored, True, "solved"

        if time_limit is not None and time.time() - start_time > time_limit:
            print(f"  ...time limit reached ({time_limit}s), stopping")
            return best_path, explored, False, "timeout"

        if len(seen) >= max_seen:
            print(f"  ...memory cap reached ({max_seen} states tracked), stopping")
            return best_path, explored, False, "capped"

        for move in generate_moves(state, last_move):
            new_state = apply_move(state, move)
            g = len(path) + 1
            hh = heuristic(new_state)
            f = g + weight * hh
            k = new_state.key()
            if k not in seen or seen[k] > g:
                seen[k] = g
                heapq.heappush(open_set, (f, next(counter), new_state, path + [move], move))

    return best_path, explored, False, "exhausted"

def solve_incrementally(initial_cols, initial_free=None, initial_found=None,
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
    free = list(initial_free or [])
    found = dict(initial_found or {})

    all_moves = []
    total_explored = 0
    start = time.time()

    while True:
        remaining = total_budget - (time.time() - start)
        if remaining <= 0:
            return all_moves, total_explored, False, "timeout"

        path, explored, solved, status = solve(
            cols, free, found,
            time_limit=min(round_limit, remaining),
            **solve_kwargs)
        total_explored += explored

        if not path:
            # No moves at all this round: either already fully solved with
            # nothing left to do, or genuinely stuck - either way, another
            # round from this same position won't produce anything new.
            return all_moves, total_explored, solved, ("solved" if solved else "stuck")

        state = State(cols, free, found)
        for move in path:
            state = apply_move(state, move)
        all_moves.extend(path)
        cols, free, found = list(state.cols), list(state.free), state.found_dict()

        if solved:
            return all_moves, total_explored, True, "solved"
        if status == "exhausted":
            return all_moves, total_explored, False, "exhausted"
        # status was "timeout" or "capped" for this round only (not the
        # overall budget) - loop again with the committed progress kept.
