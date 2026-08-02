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

from freecell_solver import rank_val, UNKNOWN

VALID_SUITS = ("S", "H", "D", "C")


def card_is_resolved(card):
    """A card read trustworthy enough to hand the solver as (rank, suit).

    Zero score is the reader's own unreliability flag (mid-animation column,
    mixed-ink suit color); a missing/invalid suit means suit matching failed
    closed. Score is read fail-closed: a card dict without a score key is
    untrusted, not implicitly perfect.
    """
    return (
        card.get("rank") not in (None, "?")
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
        for pos, card in enumerate(cards):
            if card and card.get("rank") == "?" and card.get("color") == "?":
                # face-down card: identity unknown but it occupies a real
                # slot, so it must stay in the column
                col.append(UNKNOWN)
                continue
            if card and card_is_resolved(card):
                col.append((card["rank"], card["suit"]))
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
        if not card or "suit" not in card:
            foundation_reads.append(None)
            continue
        if not card.get("reliable", card_is_resolved(card)):
            foundation_reads.append("unreliable")
            issues.append(f"foundation slot {slot_idx}: untrusted read {card!r}")
            continue
        foundation_reads.append(card)
        suit = card["suit"]
        if suit in by_suit:
            prev_idx, prev = by_suit[suit]
            issues.append(
                f"foundation: slots {prev_idx} and {slot_idx} both read as "
                f"{suit}; keeping the higher-scoring one"
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
    for pile in list(cols) + [waste]:
        for card in pile:
            if card is UNKNOWN:
                continue
            rank, suit = card
            if suit in found and rank_val(rank) <= found[suit]:
                issues.append(
                    f"({rank},{suit}) visible on the board but the {suit} "
                    f"foundation already reads {found[suit]}; contradictory frame"
                )

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
