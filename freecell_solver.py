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

def generate_complete_moves(state):
    moves = []
    cols = state.cols
    free = state.free
    found = state.found_dict()

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
                moves.append(("col_to_free", ci, col[-1]))

    return moves


def generate_moves(state, last_move=None):
    cols = state.cols
    free = state.free
    found = state.found_dict()

    # Check for safe auto-plays to foundation. If found, prune all other
    # branches and play immediately. Complete legality is exposed separately
    # by generate_complete_moves().
    for ci, col in enumerate(cols):
        if col and can_found(col[-1], found):
            card = col[-1]
            r, s = card
            rv = rank_val(r)
            if rv <= 1:  # Ace or 2 is always safe
                return [("col_to_found", ci, card)]
            opposite_suits = ("S", "C") if s in ("H", "D") else ("H", "D")
            if all(found.get(osut, -1) >= rv - 1 for osut in opposite_suits):
                return [("col_to_found", ci, card)]

    for card in free:
        if can_found(card, found):
            r, s = card
            rv = rank_val(r)
            if rv <= 1:  # Ace or 2 is always safe
                return [("free_to_found", card)]
            opposite_suits = ("S", "C") if s in ("H", "D") else ("H", "D")
            if all(found.get(osut, -1) >= rv - 1 for osut in opposite_suits):
                return [("free_to_found", card)]

    moves = generate_complete_moves(state)
    if last_move:
        moves = [
            move for move in moves
            if not (
                move[0] == "col_to_free"
                and last_move[0] == "free_to_col"
                and last_move[2] == move[2]
            )
        ]
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
          max_states=500000, time_limit=5.0):
    """
    Time-boxed best-effort search.
    Returns (path, explored, solved_bool).
    If not fully solved within the time/state budget, returns the best
    partial plan found so far (closest to solved by heuristic).
    """
    initial_free = initial_free or []
    initial_found = initial_found or {}
    start = State(initial_cols, initial_free, initial_found)

    total_cards = sum(len(c) for c in initial_cols) + len(initial_free) \
                  + sum(v + 1 for v in initial_found.values())

    is_solved = make_is_solved(total_cards)
    heuristic = make_heuristic(total_cards)

    counter = itertools.count()
    open_set = [(heuristic(start), next(counter), start, [], None)]
    seen = {start.key(): 0}

    best_state = start
    best_path = []
    best_h = heuristic(start)

    explored = 0
    start_time = time.time()

    while open_set and explored < max_states:
        if time.time() - start_time > time_limit:
            break

        _, _, state, path, last_move = heapq.heappop(open_set)
        explored += 1

        h = heuristic(state)
        if h < best_h:
            best_h = h
            best_state = state
            best_path = path

        if is_solved(state):
            return path, explored, True

        for move in generate_moves(state, last_move):
            new_state = apply_move(state, move)
            g = len(path) + 1
            hh = heuristic(new_state)
            f = g + hh
            k = new_state.key()
            if k not in seen or seen[k] > g:
                seen[k] = g
                heapq.heappush(open_set, (f, next(counter), new_state, path + [move], move))

    return best_path, explored, False
