import heapq
import time
from collections import deque

RANK_ORDER = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
SUITS = ['C','D','H','S']

def rank_val(r):
    return RANK_ORDER.index(r)


def card_color(card):
    _, s = card
    return 'RED' if s in ('H','D') else 'BLACK'

class KlondikeState:
    """State for Klondike solver.
    tableau: list of (faceDown list, faceUp list) ; faceDown bottom->top, faceUp bottom->top
    foundations: dict suit->top_rank_index (-1 if empty)
    stock: list face-down (top at end)
    waste: list face-up (top at end)
    redeals: int remaining
    """
    __slots__ = ('tableau','foundations','stock','waste','redeals')

    def __init__(self, tableau, foundations=None, stock=None, waste=None, redeals=1):
        # normalize deep copy-ish (assumes caller gives lists)
        self.tableau = [ (list(fd), list(fu)) for fd, fu in tableau ]
        self.foundations = {s: -1 for s in SUITS}
        if foundations:
            self.foundations.update({s: r for s,r in foundations.items()})
        self.stock = list(stock or [])
        self.waste = list(waste or [])
        self.redeals = int(redeals)

    def key(self):
        # produce hashable key capturing full state
        tkey = tuple((tuple(fd), tuple(fu)) for fd,fu in self.tableau)
        fkey = tuple((s,self.foundations[s]) for s in SUITS)
        return (tkey, fkey, tuple(self.stock), tuple(self.waste), self.redeals)

    def is_won(self):
        return sum(v+1 for v in self.foundations.values()) == 52

# Move formats
# ('col_to_col', src, dst, run_cards_tuple)
# ('col_to_found', src, card)
# ('waste_to_col', dst, card)
# ('waste_to_found', card)
# ('draw', n)
# ('recycle',)


def valid_sequence(seq):
    # seq is list of cards bottom->top
    if not seq:
        return False
    for i in range(len(seq)-1):
        lower = seq[i+1]
        upper = seq[i]
        r_lower = rank_val(lower[0])
        r_upper = rank_val(upper[0])
        if r_upper != r_lower + 1:
            return False
        if card_color(lower) == card_color(upper):
            return False
    return True


def generate_moves(state, draw_n=3):
    moves = []
    # tableau->foundation and tableau->tableau
    for i,(fd,fu) in enumerate(state.tableau):
        if fu:
            top = fu[-1]
            # to foundation
            cur = state.foundations[top[1]]
            if rank_val(top[0]) == cur + 1:
                moves.append(('col_to_found', i, top))
            # consider runs
            # build bottom->top sequence of faceUp cards
            seq = list(fu)
            # consider moving suffixes of seq
            for start in range(len(seq)):
                run = seq[start:]
                if not valid_sequence(run):
                    continue
                # onto non-empty columns
                for j,(fd2,fu2) in enumerate(state.tableau):
                    if i==j:
                        continue
                    if fu2:
                        dest_top = fu2[-1]
                        if rank_val(run[0][0]) == rank_val(dest_top[0]) - 1 and card_color(run[0]) != card_color(dest_top):
                            moves.append(('col_to_col', i, j, tuple(run)))
                    else:
                        # empty destination: only allow if run starts with King
                        if run[0][0] == 'K':
                            moves.append(('col_to_col', i, j, tuple(run)))
    # waste moves
    if state.waste:
        top = state.waste[-1]
        cur = state.foundations[top[1]]
        if rank_val(top[0]) == cur + 1:
            moves.append(('waste_to_found', top))
        for j,(fd2,fu2) in enumerate(state.tableau):
            if fu2:
                dest_top = fu2[-1]
                if rank_val(top[0]) == rank_val(dest_top[0]) - 1 and card_color(top) != card_color(dest_top):
                    moves.append(('waste_to_col', j, top))
            else:
                if top[0] == 'K':
                    moves.append(('waste_to_col', j, top))
    # draw
    if state.stock:
        moves.append(('draw', min(draw_n, len(state.stock))))
    else:
        # recycle if allowed
        if state.waste and state.redeals > 0:
            moves.append(('recycle',))
    return moves


def apply_move(state, move, draw_n=3):
    # returns new KlondikeState
    tableau = [(list(fd), list(fu)) for fd,fu in state.tableau]
    foundations = dict(state.foundations)
    stock = list(state.stock)
    waste = list(state.waste)
    redeals = state.redeals

    kind = move[0]
    if kind == 'col_to_found':
        _, src, card = move
        # pop from faceUp
        tableau[src][1].pop()
        foundations[card[1]] += 1
        # reveal underlying faceDown
        if tableau[src][1]==[] and tableau[src][0]:
            # flip one
            tableau[src][1].append(tableau[src][0].pop())
    elif kind == 'waste_to_found':
        _, card = move
        waste.pop()
        foundations[card[1]] += 1
    elif kind == 'col_to_col':
        _, src, dst, run = move
        run_cards = list(run)
        # remove run from source
        for _ in range(len(run_cards)):
            tableau[src][1].pop()
        # append to dest preserving order
        tableau[dst][1].extend(run_cards)
        # reveal underlying card if any
        if tableau[src][1]==[] and tableau[src][0]:
            tableau[src][1].append(tableau[src][0].pop())
    elif kind == 'waste_to_col':
        _, dst, card = move
        waste.pop()
        tableau[dst][1].append(card)
    elif kind == 'draw':
        _, n = move
        # draw n cards from stock to waste (top at end)
        for _ in range(n):
            waste.append(stock.pop())
    elif kind == 'recycle':
        stock = list(reversed(waste))
        waste = []
        redeals -= 1
    else:
        raise ValueError('unknown move')

    return KlondikeState(tableau, foundations, stock, waste, redeals)


def heuristic(state):
    # score higher is better
    score = 0
    # foundation progress
    score += 100 * sum(v+1 for v in state.foundations.values())
    # reveal face-down
    hidden_remaining = 0
    faceup_count = 0
    movable_sequences = 0
    for fd,fu in state.tableau:
        hidden_remaining += len(fd)
        faceup_count += len(fu)
        # count movable sequences roughly
        if len(fu) > 1:
            # compute longest suffix that is valid
            for start in range(len(fu)):
                run = fu[start:]
                if valid_sequence(run):
                    movable_sequences += 1
                    break
    score += 1000 * (0)  # immediate reveal handled in move scoring
    score += 300 * sum(1 for fd,fu in state.tableau if not fu and fd==[])
    score += 50 * faceup_count
    score += 25 * len(generate_moves(state))
    score += 15 * movable_sequences
    score -= 20 * hidden_remaining
    score -= 50 * (len(state.stock)//3)
    # dead position penalty handled elsewhere
    return score


def solve(initial_state, time_limit=5.0):
    start_time = time.time()
    counter = 0
    pq = []
    seen = {}
    start_key = initial_state.key()
    start_score = heuristic(initial_state)
    heapq.heappush(pq, (-start_score, counter, initial_state, []))
    seen[initial_state.key()] = 0

    while pq:
        if time.time() - start_time > time_limit:
            break
        _, _, state, path = heapq.heappop(pq)
        if state.is_won():
            return path, True
        moves = generate_moves(state)
        # order moves by immediate priority: reveal, foundation safe, empty creation
        scored = []
        for m in moves:
            new = apply_move(state, m)
            sc = heuristic(new)
            # immediate bonuses
            bonus = 0
            if m[0] in ('col_to_found','waste_to_found'):
                bonus += 500
            if m[0] == 'col_to_col':
                # if the move uncovers a faceDown at the source, big bonus
                src = m[1]
                run_len = len(m[3])
                if len(state.tableau[src][0]) > 0 and len(state.tableau[src][1]) == run_len:
                    bonus += 1000
            scored.append((sc+bonus, m, new))
        # sort descending
        scored.sort(reverse=True, key=lambda t:t[0])
        for sc, m, new in scored:
            k = new.key()
            if k in seen:
                continue
            seen[k]=1
            counter += 1
            heapq.heappush(pq, (-sc, counter, new, path+[m]))
    # no solution found in time
    return None, False
