"""Learn deterministic draw order from verified captures, never gesture dispatch.

The screenshot only shows the last three waste slots, not the whole waste.
Synchronize at an empty waste; afterwards retain the covered history and
verify each predicted action against the next clean capture. A mismatch
drops history instead of feeding guessed cards to the solver.
"""

from freecell_solver import State, UNKNOWN, apply_move, rank_val


class StockHistory:
    def __init__(self, stock_total=24):
        self.stock_total = stock_total
        self.state = None
        self.pending = []

    def record_move(self, move):
        """Queue a dispatched move; its result is not trusted yet."""
        self.pending.append(move)

    @staticmethod
    def _layout_matches(expected, observed):
        if expected.found != observed.found:
            return False
        for before, after in zip(expected.cols, observed.cols):
            if len(before) != len(after):
                return False
            if any(a != UNKNOWN and a != b for a, b in zip(before, after)):
                return False
        return True

    @staticmethod
    def _waste_matches(expected, visible):
        if not expected:
            return not visible
        # Some layouts fan only the unplayed part of the latest draw, not
        # three cards from the entire waste. Missing covered slots do not
        # erase the older confirmed history; an empty fan cannot confirm it.
        if not 1 <= len(visible) <= min(3, len(expected)):
            return False
        tail = expected[-len(visible):] if visible else ()
        # Covered OCR slots are deliberately unknown. Only exposed identities
        # can confirm or contradict a remembered slot.
        return all(a == UNKNOWN or b == UNKNOWN or a == b
                   for a, b in zip(tail, visible))

    @staticmethod
    def _identities_consistent(state):
        seen = set()
        found = state.found_dict()
        for pile in list(state.cols) + [state.waste, state.stock]:
            for card in pile:
                if card == UNKNOWN:
                    continue
                if card in seen or rank_val(card[0]) <= found.get(card[1], -1):
                    return False
                seen.add(card)
        return True

    def observe(self, reading):
        """Enrich a build_solver_state result; return diagnostic notes.

        Empty-waste captures establish exact remaining cycle size. Nonempty
        mid-game attachment stays unsynchronized until that boundary is seen.
        """
        reading["stock_history_synced"] = False
        reading["stock_cards"] = None
        reading["previous_moves_confirmed"] = None
        notes = []
        observed = State(reading["cols"], reading["waste"],
                         max(0, reading["stock_remaining"]), self.stock_total,
                         reading["found"])
        previous, pending = self.state, self.pending
        reading["previous_moves_dispatched"] = len(pending)
        self.pending = []
        if reading["issues"]:
            self.state = None
            if previous is not None:
                notes.append("stock history discarded after an untrusted capture")
            return notes

        expected = previous
        if expected is not None:
            try:
                for move in pending:
                    expected = apply_move(expected, move)
            except (IndexError, KeyError, ValueError):
                expected = None

        # A failed stock tap can leave the same exposed card. Never count it
        # as an unknown draw merely because the predicted slot was unknown.
        failed_draw = (previous is not None and pending and pending[-1][0] == "draw"
                and previous.waste and observed.waste
                and previous.waste[-1] != UNKNOWN
                and previous.waste[-1] == observed.waste[-1])
        if failed_draw:
            expected = previous

        if (expected is not None and self._layout_matches(expected, observed)
                and self._waste_matches(expected.waste, observed.waste)):
            waste = list(expected.waste)
            if waste and observed.waste[-1] != UNKNOWN:
                waste[-1] = observed.waste[-1]
            verified = State(observed.cols, waste, expected.stock_remaining,
                             self.stock_total, observed.found_dict(), stock=expected.stock)
            if self._identities_consistent(verified):
                self.state = verified
                if pending:
                    reading["previous_moves_confirmed"] = 0 if failed_draw else len(pending)
            else:
                self.state = None
                notes.append("stock history contradicted by a duplicate or founded identity")
        else:
            self.state = None
            if (pending and previous is not None
                    and self._layout_matches(previous, observed)
                    and self._waste_matches(previous.waste, observed.waste)):
                reading["previous_moves_confirmed"] = 0
            if previous is not None:
                notes.append("stock history discarded: capture did not confirm the dispatched moves")

        if self.state is None and not observed.waste:
            remaining = reading["stock_remaining"]
            if 0 <= remaining <= self.stock_total:
                self.state = observed
                notes.append("stock history synchronized at an empty waste")

        if self.state is not None:
            reading["waste"] = list(self.state.waste)
            reading["stock_remaining"] = self.state.stock_remaining
            reading["stock_cards"] = self.state.stock
            reading["stock_history_synced"] = True
        return notes
