#!/usr/bin/env python3

from board_reader_lib import read_board, STOCK_TOTAL
from freecell_solver import State, rank_val, UNKNOWN
from monte_carlo_solver import (
    choose_move_monte_carlo,
    print_statistics,
)
from solitaire_auto_bot import assign_pseudo_suits


def main():
    board = read_board("Gameplay/frame_0108.png")

    assign_pseudo_suits(board)

    columns = []

    for column_index in range(7):
        column = []

        for card in board[f"col{column_index}"]:
            if card.get("rank") == "?" and card.get("color") == "?":
                # face-down card - still occupies a real slot in this
                # column, so it can't just be dropped (see solitaire_auto_bot's
                # main() for the same fix and why it matters).
                column.append(UNKNOWN)
                continue

            if "suit" not in card:
                print(
                    f"Skipping unresolved card in col{column_index}: "
                    f"{card}"
                )
                break

            column.append((card["rank"], card["suit"]))

        columns.append(column)

    waste = []

    for card in board["waste"]:
        if card and "suit" in card:
            waste.append((card["rank"], card["suit"]))

    foundations = {}

    for card in board["foundation"]:
        if card and "suit" in card:
            foundations[card["suit"]] = rank_val(card["rank"])

    stock_remaining = 52 - sum(len(c) for c in columns) - len(waste) \
        - sum(v + 1 for v in foundations.values())

    state = State(
        cols=columns,
        waste=waste,
        stock_remaining=stock_remaining,
        stock_total=STOCK_TOTAL,
        found=foundations,
    )

    print("Solver state:")
    print("Columns:", columns)
    print("Waste:", waste)
    print("Foundations:", foundations)
    print("Stock remaining:", stock_remaining)

    best_move, statistics = choose_move_monte_carlo(
        state=state,
        simulations=5000,
        time_limit=10.0,
        max_depth=150,
        seed=42,
    )

    print(f"\nSelected move: {best_move}")
    print_statistics(statistics)


if __name__ == "__main__":
    main()
