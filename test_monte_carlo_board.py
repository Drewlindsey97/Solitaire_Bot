#!/usr/bin/env python3

from board_reader_lib import read_board
from freecell_solver import State, rank_val
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
                continue

            if "suit" not in card:
                print(
                    f"Skipping unresolved card in col{column_index}: "
                    f"{card}"
                )
                break

            column.append((card["rank"], card["suit"]))

        columns.append(column)

    free_cells = []

    for card in board["free_cells"]:
        if card and "suit" in card:
            free_cells.append((card["rank"], card["suit"]))

    foundations = {}

    for card in board["foundation"]:
        if card and "suit" in card:
            foundations[card["suit"]] = rank_val(card["rank"])

    state = State(
        cols=columns,
        free=free_cells,
        found=foundations,
    )

    print("Solver state:")
    print("Columns:", columns)
    print("Free cells:", free_cells)
    print("Foundations:", foundations)

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
