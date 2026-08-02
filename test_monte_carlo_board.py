#!/usr/bin/env python3

from board_reader_lib import read_board, STOCK_TOTAL
from freecell_solver import State
from monte_carlo_solver import (
    choose_move_monte_carlo,
    print_statistics,
)
from solitaire_auto_bot import assign_pseudo_suits
from solver_state import build_solver_state


def main():
    board = read_board("Gameplay/frame_0108.png")

    assign_pseudo_suits(board)

    solver_state = build_solver_state(board, stock_total=STOCK_TOTAL)
    for msg in solver_state["issues"]:
        print(f"[Warn] {msg}")

    state = State(
        cols=solver_state["cols"],
        waste=solver_state["waste"],
        stock_remaining=solver_state["stock_remaining"],
        stock_total=STOCK_TOTAL,
        found=solver_state["found"],
    )

    print("Solver state:")
    print("Columns:", solver_state["cols"])
    print("Waste:", solver_state["waste"])
    print("Foundations:", solver_state["found"])
    print("Stock remaining:", solver_state["stock_remaining"])

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
