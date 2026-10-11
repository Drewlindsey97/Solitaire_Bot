"""Foundation pull-down (found_to_col) for the race policy.

Covers the live stall of 2026-10-10 (5S banked, 6D open, red 4 in the draw
cycle) and, with a full Klondike simulator over random deals, the guarantee
that matters most: pull-downs can never make the bot loop.
"""
import random

import pytest

from freecell_solver import UNKNOWN, RANK_ORDER, State, apply_move, can_stack, can_found
from race_policy import choose_pulldown, plan_batch

U = UNKNOWN


def stuck_state(waste_top=("4", "H"), found=None):
    """Board from live_session3 cycle 40, where the bot drew until it stopped."""
    cols = [
        [("K", "D"), ("Q", "S"), ("J", "D"), ("10", "C"), ("9", "D"), ("8", "S"),
         ("7", "D"), ("6", "C"), ("5", "D"), ("4", "C")],
        [],
        [],
        [U, U, ("J", "S")],
        [("K", "C"), ("Q", "H"), ("J", "C"), ("10", "H"), ("9", "C"), ("8", "D"),
         ("7", "S"), ("6", "D")],
        [],
        [U, ("8", "C"), ("7", "H")],
    ]
    found = found if found is not None else {"C": 1, "D": 2, "H": 1, "S": 4}
    return State(cols, [U, U, waste_top], 13, 24, found)


def test_without_pulldowns_the_live_stall_only_draws():
    assert plan_batch(stuck_state(), max_moves=10) == [("draw",)]


def test_pulldown_frees_the_live_stall():
    batch = plan_batch(stuck_state(), max_moves=10, pulled_down=set())
    assert batch == [
        ("found_to_col", ("5", "S"), 4),
        ("waste_to_col", 4, ("4", "H")),
    ]


def test_apply_move_found_to_col():
    after = apply_move(stuck_state(), ("found_to_col", ("5", "S"), 4))
    assert after.found_dict()["S"] == 3
    assert after.cols[4][-1] == ("5", "S")


def test_card_is_never_pulled_twice():
    assert plan_batch(stuck_state(), max_moves=10, pulled_down={("5", "S")}) == [("draw",)]


def test_no_payoff_means_no_pulldown():
    # Q of clubs on the waste can't use a 5 of spades.
    assert plan_batch(stuck_state(waste_top=("Q", "C")), max_moves=10,
                      pulled_down=set()) == [("draw",)]


def test_excluded_pulldown_is_respected():
    exclude = {("found_to_col", ("5", "S"), 4)}
    assert plan_batch(stuck_state(), max_moves=10, exclude=exclude,
                      pulled_down=set()) == [("draw",)]


def test_one_move_batches_never_strand_a_pulldown():
    # A pull-down without its payoff is pure loss, so with no room for both
    # the batch falls back to the normal move instead of coming back empty.
    assert plan_batch(stuck_state(), max_moves=1, pulled_down=set()) == [("draw",)]


def test_safe_cards_are_never_pulled():
    # With both red piles at 4, no tableau card can ever need the 5S, so
    # pulling it down could only waste a move.
    state = stuck_state(found={"C": 1, "D": 4, "H": 4, "S": 4})
    assert choose_pulldown(state, set()) is None


def test_pulldown_only_when_otherwise_drawing():
    # A real move is available (4H could go on... nothing, but 7H onto 8S? no -
    # give it a direct foundation play instead): 3H on the waste banks, so
    # no pull-down may be chosen.
    state = stuck_state(waste_top=("3", "H"))
    batch = plan_batch(state, max_moves=10, pulled_down=set())
    assert batch[0] == ("waste_to_found", ("3", "H"))
    assert all(m[0] != "found_to_col" for m in batch)


# ---------------------------------------------------------------------------
# Full-game simulation: real deck, real hidden information, the bot's own
# stuck stop. Proves pull-downs stay bounded and every game terminates.
# ---------------------------------------------------------------------------
SUITS = ["S", "H", "D", "C"]
STUCK_LIMIT = 15
CYCLE_CAP = 3000


class Game:
    def __init__(self, seed, memory=False):
        # memory=True models stock_history.py: a card's identity is known
        # once it has been the exposed waste top, and redeals keep order.
        self.memory = memory
        self.known = set()
        deck = [(r, s) for s in SUITS for r in RANK_ORDER]
        random.Random(seed).shuffle(deck)
        self.cols = []  # list of [card, face_up]
        for i in range(7):
            col = [[deck.pop(), False] for _ in range(i + 1)]
            col[-1][1] = True
            self.cols.append(col)
        self.stock = deck  # draw from the end
        self.waste = []
        self.found = {}

    def visible(self):
        cols = [[c if up else U for c, up in col] for col in self.cols]
        if self.memory:
            waste = [c if c in self.known else U for c in self.waste]
            stock = [c if c in self.known else U for c in reversed(self.stock)]
            waste[-1:] = self.waste[-1:]
            return State(cols, waste, len(self.stock), 24, self.found, stock=stock)
        waste = [U] * max(0, len(self.waste) - 1) + self.waste[-1:]
        return State(cols, waste, len(self.stock), 24, self.found)

    def _see_waste_top(self):
        if self.waste:
            self.known.add(self.waste[-1])

    def _flip(self, ci):
        if self.cols[ci] and not self.cols[ci][-1][1]:
            self.cols[ci][-1][1] = True

    def play(self, move):
        """Apply a move to the TRUE board, asserting it is legal there."""
        kind = move[0]
        if kind == "col_to_found":
            _, ci, card = move
            assert self.cols[ci][-1] == [card, True] and can_found(card, self.found)
            self.cols[ci].pop()
            self.found[card[1]] = RANK_ORDER.index(card[0])
            self._flip(ci)
        elif kind == "waste_to_found":
            _, card = move
            assert self.waste[-1] == card and can_found(card, self.found)
            self.waste.pop()
            self.found[card[1]] = RANK_ORDER.index(card[0])
            self._see_waste_top()
        elif kind == "col_to_col":
            _, ci, cj, card, n = move
            run = self.cols[ci][-n:]
            assert run[0] == [card, True] and all(up for _, up in run)
            dest = self.cols[cj]
            assert (card[0] == "K" and not dest) or (dest and can_stack(card, dest[-1][0]))
            del self.cols[ci][-n:]
            dest.extend(run)
            self._flip(ci)
        elif kind == "waste_to_col":
            _, cj, card = move
            assert self.waste[-1] == card
            dest = self.cols[cj]
            assert (card[0] == "K" and not dest) or (dest and can_stack(card, dest[-1][0]))
            self.waste.pop()
            dest.append([card, True])
            self._see_waste_top()
        elif kind == "found_to_col":
            _, card, cj = move
            assert self.found.get(card[1]) == RANK_ORDER.index(card[0])
            dest = self.cols[cj]
            assert ((card[0] == "K" and not dest)
                    or (dest and dest[-1][1] and can_stack(card, dest[-1][0])))
            self.found[card[1]] -= 1
            if self.found[card[1]] < 0:
                del self.found[card[1]]
            dest.append([card, True])
        elif kind == "draw":
            assert self.stock
            for _ in range(min(3, len(self.stock))):
                self.waste.append(self.stock.pop())
            self._see_waste_top()
        elif kind == "redeal":
            # With stock and waste both empty the real game ignores the tap.
            assert not self.stock
            self.stock = list(reversed(self.waste))
            self.waste = []
        else:
            raise AssertionError(move)

    def founded(self):
        return sum(v + 1 for v in self.found.values())


def run_game(seed, pulldowns=True, memory=False, endgame=True):
    game = Game(seed, memory=memory)
    pulled = set() if pulldowns else None
    best, idle, stats = float("-inf"), 0, {"pulldowns": 0, "cycles": 0, "moves": 0}
    for cycle in range(CYCLE_CAP):
        stats["cycles"] = cycle
        state = game.visible()
        face_down = sum(1 for c in state.cols for x in c if x == U)
        tableau = sum(len(c) for c in state.cols)
        progress = (game.founded() + len(pulled or ())) * 100 + tableau - face_down
        if progress > best:
            best, idle = progress, 0
        else:
            idle += 1
        if not game.stock and not game.waste and all(up for col in game.cols for _, up in col):
            # The real game auto-completes from here (user report 2026-10-10).
            stats["auto_completed"] = True
            game.found = {s_: 12 for s_ in SUITS}
            break
        if game.founded() == 52 or idle >= STUCK_LIMIT:
            break
        batch = plan_batch(state, max_moves=10, pulled_down=pulled,
                           endgame=endgame and idle < 8)
        if not batch:
            break
        assert batch[-1][0] != "found_to_col"
        for i, move in enumerate(batch):
            if move[0] == "found_to_col":
                assert move[1] not in pulled, "card pulled down twice"
                pulled.add(move[1])
                stats["pulldowns"] += 1
                nxt = batch[i + 1]
                if nxt[0] != "found_to_col":
                    # end of a chain: the payoff lands on the last card pulled
                    assert nxt[0] in ("waste_to_col", "col_to_col")
                    assert nxt[2 if nxt[0] == "col_to_col" else 1] == move[2]
                chain = 1
                while i - chain >= 0 and batch[i - chain][0] == "found_to_col":
                    chain += 1
                assert chain <= 3, "pull-down chain longer than the cap"
            game.play(move)
            stats["moves"] += 1
    else:
        pytest.fail(f"seed {seed}: no stop after {CYCLE_CAP} cycles")
    stats["founded"] = game.founded()
    return stats


@pytest.mark.parametrize("memory", [False, True], ids=["no-history", "stock-history"])
@pytest.mark.parametrize("seed", range(150))
def test_simulated_games_terminate_and_pulldowns_stay_bounded(seed, memory):
    stats = run_game(seed, memory=memory)
    assert stats["pulldowns"] <= 52
    assert stats["cycles"] < CYCLE_CAP


# ---------------------------------------------------------------------------
# Bot side: gesture, safety checks, board model, foundation history.
# ---------------------------------------------------------------------------
def _card(rank, suit, **extra):
    color = "RED" if suit in ("H", "D") else "BLACK"
    return dict({"rank": rank, "suit": suit, "color": color, "score": 1.0}, **extra)


def _bot_board():
    board = {f"col{i}": [] for i in range(7)}
    board["col4"] = [_card("K", "C"), _card("Q", "H"), _card("J", "C"), _card("10", "H"),
                     _card("9", "C"), _card("8", "D"), _card("7", "S"), _card("6", "D")]
    board["waste"] = [_card("4", "H", x=514)]
    board["foundation"] = [_card("3", "D", reliable=True), _card("2", "C", reliable=True),
                           _card("5", "S", reliable=True), _card("2", "H", reliable=True)]
    return board


def _events():
    log = []
    return log, (lambda name, **data: log.append((name, data)))


def test_execute_pulldown_swipes_from_the_spade_pile_to_the_column():
    import solitaire_auto_bot as bot
    log, emit = _events()
    board = _bot_board()
    assert bot.execute_move(board, ("found_to_col", ("5", "S"), 4), sim_mode=True, event_logger=emit)
    planned = [d for n, d in log if n == "gesture_planned"][0]
    assert planned["gesture"] == "swipe"
    sx = bot.FOUNDATION_X[2] + bot.SLOT_W / 2
    assert planned["start"]["x"] == int(sx)
    assert planned["end"]["x"] == int(bot.TABLEAU_X[4] + bot.COL_WIDTH / 2)


def test_execute_pulldown_refuses_when_the_pile_shows_another_card():
    import solitaire_auto_bot as bot
    log, emit = _events()
    board = _bot_board()
    board["foundation"][2] = _card("4", "S", reliable=True)
    assert not bot.execute_move(board, ("found_to_col", ("5", "S"), 4), sim_mode=True, event_logger=emit)
    assert any(d.get("reason") == "foundation_top_mismatch" for n, d in log if n == "move_rejected")


def test_execute_pulldown_refuses_an_illegal_destination():
    import solitaire_auto_bot as bot
    log, emit = _events()
    board = _bot_board()
    board["col4"][-1] = _card("6", "C")  # black 6 can't take the black 5
    assert not bot.execute_move(board, ("found_to_col", ("5", "S"), 4), sim_mode=True, event_logger=emit)
    assert any(d.get("reason") == "tableau_destination_mismatch" for n, d in log if n == "move_rejected")


def test_board_model_and_history_follow_a_pulldown():
    import solitaire_auto_bot as bot
    board = _bot_board()
    accepted = {i: dict(c) for i, c in enumerate(board["foundation"])}
    pending = {2: [("S", 4), 0.0]}
    slot = bot.record_pulldown_in_history(board, accepted, pending, ("5", "S"))
    bot.apply_move_to_board(board, ("found_to_col", ("5", "S"), 4))
    assert slot == 2 and 2 not in pending
    assert board["foundation"][2]["rank"] == "4"
    assert board["col4"][-1]["rank"] == "5" and board["col4"][-1]["suit"] == "S"

    # The next real frame shows 4S on the pile. Without the history update
    # this read was rejected for 5 seconds ("contradicts history").
    frame = _bot_board()
    frame["foundation"][2] = _card("4", "S", reliable=True)
    log, emit = _events()
    notes = bot.reconcile_foundation_reads(frame, accepted, pending, 2, emit, 100.0)
    assert notes == []
    assert frame["foundation"][2]["rank"] == "4"
    assert not any(n == "foundation_read_rejected" for n, _ in log)



def test_live_44_card_stall_is_freed_by_a_pulldown_chain():
    # live_session8 cycle 47: spades and hearts complete, D at 8, C at 10;
    # K-D alone, K-C Q-D J-C 10-D, J-D on the waste and nothing else legal.
    cols = [[("K", "D")], [], [("K", "C"), ("Q", "D"), ("J", "C"), ("10", "D")], [], [], [], []]
    state = State(cols, [U, U, ("J", "D")], 0, 24, {"S": 12, "H": 12, "D": 7, "C": 9})
    assert plan_batch(state, max_moves=10) == [("redeal",)]
    assert plan_batch(state, max_moves=10, pulled_down=set()) == [
        ("found_to_col", ("K", "S"), 1),
        ("found_to_col", ("Q", "S"), 0),
        ("waste_to_col", 0, ("J", "D")),
    ]


def test_chain_needs_room_for_every_step():
    cols = [[("K", "D")], [], [("K", "C"), ("Q", "D"), ("J", "C"), ("10", "D")], [], [], [], []]
    state = State(cols, [U, U, ("J", "D")], 0, 24, {"S": 12, "H": 12, "D": 7, "C": 9})
    assert plan_batch(state, max_moves=2, pulled_down=set()) == [("redeal",)]


def test_chain_respects_cards_already_pulled():
    cols = [[("K", "D")], [], [("K", "C"), ("Q", "D"), ("J", "C"), ("10", "D")], [], [], [], []]
    state = State(cols, [U, U, ("J", "D")], 0, 24, {"S": 12, "H": 12, "D": 7, "C": 9})
    assert plan_batch(state, max_moves=10, pulled_down={("K", "S")}) == [("redeal",)]


def test_execute_king_pulldown_into_an_empty_column():
    import solitaire_auto_bot as bot
    log, emit = _events()
    board = {f"col{i}": [] for i in range(7)}
    board["col0"] = [_card("K", "D")]
    board["waste"] = [_card("J", "D", x=514)]
    board["foundation"] = [_card("K", "S", reliable=True), _card("K", "H", reliable=True),
                           _card("8", "D", reliable=True), _card("10", "C", reliable=True)]
    assert bot.execute_move(board, ("found_to_col", ("K", "S"), 1), sim_mode=True, event_logger=emit)
    bot.apply_move_to_board(board, ("found_to_col", ("K", "S"), 1))
    assert board["foundation"][0]["rank"] == "Q"
    # second pull of the chain comes from the same pile, now showing QS
    assert bot.execute_move(board, ("found_to_col", ("Q", "S"), 0), sim_mode=True, event_logger=emit)
    assert not any(n == "move_rejected" for n, _ in log)
