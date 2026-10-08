"""Shared board-dict -> solver-state conversion.

Every consumer of read_board() (the live bot, hybrid_runner, parallel_search,
benchmarks) needs the same conversion: columns to (rank, suit) tuples,
waste to tuples, foundation slots to a {suit: rank_val} map, and the
card-conservation stock count. Before this module existed the conversion was
copy-pasted in four places and reader-trust fixes had to be remembered in
each; now the trust rules (score gating, the reader's `reliable` flag,
foundation suit-collision handling) live here once.

VALID_SUITS check matters: match_suit() fails closed with suit "?" when the
color evidence is mixed, and those cards must become truncation points, not
solver cards.
"""

from freecell_solver import rank_val, UNKNOWN, RANK_ORDER

VALID_SUITS = ("S", "H", "D", "C")
RED_SUITS = ("H", "D")


def _continues_tableau_run(upper, lower):
    """True if `lower` can legally sit directly beneath `upper` in a
    tableau column: alternating color, exactly one rank down. A column is
    always a valid alternating-descending run in real Klondike, so a card
    that fails this is reader hallucination (e.g. the row-count estimate
    in detect_column_height overshooting into noise below the real last
    card) - direct evidence the read is garbage, not a legal position.
    """
    upper_rank, upper_suit = upper
    lower_rank, lower_suit = lower
    upper_red = upper_suit in RED_SUITS
    lower_red = lower_suit in RED_SUITS
    return upper_red != lower_red and rank_val(lower_rank) == rank_val(upper_rank) - 1


def card_is_resolved(card):
    """A card read trustworthy enough to hand the solver as (rank, suit).

    Zero score is the reader's own unreliability flag (mid-animation column,
    mixed-ink suit color); a missing/invalid suit means suit matching failed
    closed. Score is read fail-closed: a card dict without a score key is
    untrusted, not implicitly perfect.
    """
    return (
        card.get("rank") in RANK_ORDER
        and card.get("suit") in VALID_SUITS
        and card.get("score", 0.0) > 0.0
    )


def build_solver_state(board, stock_total=24):
    """Convert a read_board() dict into solver inputs plus a trust report.

    Returns a dict with:
      cols:              list of 7 lists of (rank, suit) / UNKNOWN
      waste:             list of (rank, suit)
      found:             {suit: rank_val} from trusted foundation reads only
      foundation_reads:  per-slot list: None (empty), dict (trusted read),
                         or "unreliable" (occupied but untrusted this frame)
      stock_remaining:   card-conservation remainder
      truncated_columns: column indices cut short at an unresolved card
      issues:            human-readable strings, one per trust problem -
                         empty means the whole frame was read cleanly
    """
    issues = []

    cols = []
    truncated_columns = []
    for idx in range(7):
        col = []
        cards = board.get(f"col{idx}", [])
        prev_card = None  # (rank, suit) directly above the next slot, or
                           # None when it's unknown/hidden and can't be
                           # checked against
        for pos, card in enumerate(cards):
            if card and card.get("rank") == "?" and card.get("color") == "?":
                # face-down card: identity unknown but it occupies a real
                # slot, so it must stay in the column
                col.append(UNKNOWN)
                prev_card = None
                continue
            if card and card_is_resolved(card):
                this_card = (card["rank"], card["suit"])
                if prev_card is not None and not _continues_tableau_run(prev_card, this_card):
                    # A real column is always alternating-descending; a card
                    # that breaks that chain is a phantom read (e.g. the
                    # reader overshooting past the true bottom card), not a
                    # legal position. Same treatment as any other unresolved
                    # read: truncate here rather than hand the solver a
                    # card that cannot really be there.
                    col.extend([UNKNOWN] * (len(cards) - pos))
                    truncated_columns.append(idx)
                    issues.append(
                        f"col{idx}: {card['rank']}{card['suit']} at row {pos} "
                        f"does not continue the legal alternating-descending "
                        f"run below {prev_card[0]}{prev_card[1]}; treating it "
                        f"and the rest of the column as unknown"
                    )
                    break
                col.append(this_card)
                prev_card = this_card
            else:
                # Unresolved read: this card and everything under it become
                # UNKNOWN placeholders rather than being dropped - they are
                # real cards occupying real slots, and dropping them would
                # both misattribute them to stock (breaking the
                # conservation count below) and misplace every later
                # per-row coordinate. The solver already treats UNKNOWN as
                # a hidden card pending reveal.
                col.extend([UNKNOWN] * (len(cards) - pos))
                truncated_columns.append(idx)
                issues.append(
                    f"col{idx}: unresolved card {card!r} at row {pos}; "
                    f"rest of column read as unknown"
                )
                break
        cols.append(col)

    waste = []
    for card in board.get("waste", []):
        if card and card.get("covered"):
            waste.append(UNKNOWN)
            continue
        if card and card_is_resolved(card):
            waste.append((card["rank"], card["suit"]))
        else:
            # same conservation argument as the column case above
            waste.append(UNKNOWN)
            issues.append(f"waste: unresolved card {card!r}; read as unknown")

    # Foundation slots. read_slot marks each occupied slot reliable/not;
    # additionally two slots can never hold the same suit, so a duplicate
    # means at least one read is garbage - keep the higher-scoring one.
    foundation_reads = []
    by_suit = {}
    for slot_idx, card in enumerate(board.get("foundation", [])):
        if not card:
            foundation_reads.append(None)
            continue
        if not card_is_resolved(card) or not card.get("reliable", True):
            foundation_reads.append("unreliable")
            issues.append(f"foundation slot {slot_idx}: untrusted read {card!r}")
            continue
        foundation_reads.append(card)
        suit = card["suit"]
        if suit in by_suit:
            prev_idx, prev = by_suit[suit]
            issues.append(
                f"foundation: slots {prev_idx} and {slot_idx} both read as "
                f"{suit}; keeping the stronger read (earlier slot on a tie)"
            )
            if card.get("score", 0.0) <= prev.get("score", 0.0):
                foundation_reads[slot_idx] = "unreliable"
                continue
            foundation_reads[prev_idx] = "unreliable"
        by_suit[suit] = (slot_idx, card)

    found = {suit: rank_val(card["rank"]) for suit, (_, card) in by_suit.items()}

    # Card-conservation cross-check: a foundation pile at rank v holds EVERY
    # card of that suit up to v, so any such card still visible in a column
    # or the waste is a physical contradiction - one of the two reads is
    # garbage (e.g. an animation graphic template-matching as a high
    # foundation card while the real card sits mid-flight in a column).
    # The foundation side loses the tie: foundation-slot misreads are the
    # documented failure class here, a visible tableau/waste card is direct
    # evidence, and the failure directions are asymmetric - wrongly dropping
    # a foundation pile inflates stock (caught by the bounds check / retried
    # by the live loop), while wrongly trusting one force-feeds the solver
    # an illegal autoplay it will physically execute.
    contradicted = set()
    for pile in list(cols) + [waste]:
        for card in pile:
            if card == UNKNOWN:
                continue
            rank, suit = card
            if suit in found and rank_val(rank) <= found[suit]:
                issues.append(
                    f"({rank},{suit}) visible on the board but the {suit} "
                    f"foundation already reads {found[suit]}; dropping the "
                    f"foundation read as the likelier garbage"
                )
                contradicted.add(suit)
    for suit in contradicted:
        slot_idx, _card = by_suit.pop(suit)
        foundation_reads[slot_idx] = "unreliable"
        del found[suit]

    # A plausible count is not enough: the deck contains each identity once.
    # Neither duplicate read wins on score alone. Preserve their slots, but
    # block both identities (and any tableau run depending on them).
    locations = {}
    for ci, pile in enumerate(cols + [waste]):
        for row, identity in enumerate(pile):
            if identity != UNKNOWN:
                locations.setdefault(identity, []).append((ci, row))
    for identity, duplicates in locations.items():
        if len(duplicates) < 2:
            continue
        labels = [f"col{ci} row {row}" if ci < 7 else f"waste row {row}"
                  for ci, row in duplicates]
        issues.append(f"duplicate card {identity[0]}{identity[1]} at "
                      f"{', '.join(labels)}; treating conflicting reads as unknown")
        for ci, row in duplicates:
            if ci < 7:
                cols[ci][row:] = [UNKNOWN] * (len(cols[ci]) - row)
                if ci not in truncated_columns:
                    truncated_columns.append(ci)
            else:
                waste[row] = UNKNOWN
    truncated_columns.sort()

    # Every real card is in exactly one of: a column (revealed or face-down),
    # the waste, a foundation, or undrawn stock - so stock is the remainder.
    # Self-correcting each cycle, but only as good as the reads above:
    # anything wrongly dropped from cols/waste/found is misattributed to
    # stock, hence the bounds check.
    stock_remaining = 52 - sum(len(c) for c in cols) - len(waste) \
        - sum(v + 1 for v in found.values())
    if stock_remaining < 0 or stock_remaining > stock_total:
        issues.append(
            f"impossible stock_remaining={stock_remaining} "
            f"(valid range 0-{stock_total}); some read above is wrong"
        )

    return {
        "cols": cols,
        "waste": waste,
        "found": found,
        "foundation_reads": foundation_reads,
        "stock_remaining": stock_remaining,
        "truncated_columns": truncated_columns,
        "issues": issues,
    }
