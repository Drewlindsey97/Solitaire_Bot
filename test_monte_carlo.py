#!/usr/bin/env python3

from freecell_solver import State
from monte_carlo_solver import (
    choose_move_monte_carlo,
    print_statistics,
)


def main():
    columns = [
        [("K", "S"), ("Q", "H"), ("J", "C")],
        [("K", "D"), ("Q", "C")],
        [("10", "D")],
        [],
        [],
        [],
        [],
    ]

    free_cells = [
        ("A", "S"),
    ]

    foundations = {}

    state = State(
        cols=columns,
        free=free_cells,
        found=foundations,
    )

    best_move, statistics = choose_move_monte_carlo(
        state=state,
        simulations=1_000,
        time_limit=3.0,
        max_depth=100,
        seed=42,
    )

    print(f"\nSelected move: {best_move}")
    print_statistics(statistics)


if __name__ == "__main__":
    main()
