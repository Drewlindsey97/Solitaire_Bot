#!/usr/bin/env python3

import random
import time
from dataclasses import dataclass
from typing import Optional

from freecell_solver import (
    State,
    apply_move,
    generate_complete_moves,
    generate_moves,
    rank_val,
)


WIN_SCORE = 1_000_000.0


@dataclass
class MoveStatistics:
    move: tuple
    visits: int = 0
    wins: int = 0
    total_score: float = 0.0

    @property
    def average_score(self) -> float:
        if self.visits == 0:
            return float("-inf")

        return self.total_score / self.visits

    @property
    def win_rate(self) -> float:
        if self.visits == 0:
            return 0.0

        return self.wins / self.visits


def total_card_count(state: State) -> int:
    """
    Count all cards represented by the solver state.

    Foundation values are stored as zero-based rank values:
    Ace = 0, Two = 1, ..., King = 12.
    Therefore, a foundation value of 3 represents four cards.
    """
    tableau_cards = sum(len(column) for column in state.cols)
    free_cards = len(state.free)
    foundation_cards = sum(value + 1 for _, value in state.found)

    return tableau_cards + free_cards + foundation_cards


def foundation_card_count(state: State) -> int:
    return sum(value + 1 for _, value in state.found)


def is_solved(state: State, expected_total: int) -> bool:
    return foundation_card_count(state) == expected_total


def count_empty_columns(state: State) -> int:
    return sum(1 for column in state.cols if not column)


def count_open_free_cells(state: State) -> int:
    return 4 - len(state.free)


def mobility(state: State) -> int:
    """
    Number of legal moves available from this state.

    More legal moves generally means the position is less constrained.
    """
    return len(generate_moves(state))


def evaluate_state(state: State, expected_total: int) -> float:
    """
    Score an unfinished rollout.

    Foundation progress matters most. Open free cells, empty columns,
    and available moves are smaller bonuses.
    """
    foundation_cards = foundation_card_count(state)

    if foundation_cards == expected_total:
        return WIN_SCORE

    score = 0.0

    score += foundation_cards * 1_000.0
    score += count_empty_columns(state) * 40.0
    score += count_open_free_cells(state) * 20.0
    score += mobility(state) * 2.0

    # Keeping cards trapped in free cells reduces flexibility.
    score -= len(state.free) * 8.0

    return score


def move_priority(state: State, move: tuple) -> float:
    """
    Give the rollout policy a mild preference for productive moves.

    This does not choose the final move. It only makes random simulations
    less likely to spend all their time shuffling cards pointlessly.
    """
    kind = move[0]

    priority = 1.0

    if kind in ("col_to_found", "free_to_found"):
        priority += 12.0

    elif kind == "free_to_col":
        priority += 5.0

    elif kind == "col_to_col":
        priority += 3.0

    elif kind == "col_to_free":
        priority += 0.5

    try:
        next_state = apply_move(state, move)

        if count_empty_columns(next_state) > count_empty_columns(state):
            priority += 4.0

        if count_open_free_cells(next_state) > count_open_free_cells(state):
            priority += 3.0

    except (IndexError, KeyError, ValueError):
        return 0.0

    return max(priority, 0.01)


def choose_weighted_move(
    state: State,
    moves: list[tuple],
    rng: random.Random,
) -> tuple:
    """
    Randomly choose a move while favoring moves with higher priority.
    """
    weights = [move_priority(state, move) for move in moves]

    return rng.choices(
        population=moves,
        weights=weights,
        k=1,
    )[0]


def rollout(
    starting_state: State,
    expected_total: int,
    rng: random.Random,
    max_depth: int,
) -> tuple[float, bool, int]:
    """
    Play a randomized simulated continuation.

    Returns:
        score:
            Evaluation of the final simulated state.

        won:
            True when all represented cards reached foundations.

        depth:
            Number of simulated moves performed.
    """
    state = starting_state
    previous_move: Optional[tuple] = None
    visited = {state.key()}

    for depth in range(max_depth):
        if is_solved(state, expected_total):
            return WIN_SCORE, True, depth

        moves = generate_moves(state, previous_move)

        if not moves:
            return evaluate_state(state, expected_total), False, depth

        # Remove moves that immediately revisit a state already encountered
        # during this rollout.
        candidates = []

        for move in moves:
            try:
                next_state = apply_move(state, move)
            except (IndexError, KeyError, ValueError):
                continue

            if next_state.key() not in visited:
                candidates.append((move, next_state))

        if not candidates:
            return evaluate_state(state, expected_total), False, depth

        candidate_moves = [move for move, _ in candidates]

        selected_move = choose_weighted_move(
            state,
            candidate_moves,
            rng,
        )

        next_state = next(
            candidate_state
            for move, candidate_state in candidates
            if move == selected_move
        )

        previous_move = selected_move
        state = next_state
        visited.add(state.key())

    return evaluate_state(state, expected_total), False, max_depth


def choose_move_monte_carlo(
    state: State,
    simulations: int = 1_000,
    time_limit: float = 3.0,
    max_depth: int = 100,
    seed: Optional[int] = None,
    legal_moves: Optional[list[tuple]] = None,
) -> tuple[Optional[tuple], list[MoveStatistics]]:
    """
    Evaluate every legal opening move using randomized rollouts.

    The simulation budget is distributed round-robin so every legal move
    receives trials instead of one move consuming the whole time limit.
    """
    if legal_moves is None:
        legal_moves = generate_moves(state)
    else:
        complete_moves = set(generate_complete_moves(state))
        illegal_moves = [
            move for move in legal_moves
            if move not in complete_moves
        ]
        if illegal_moves:
            raise ValueError(f"Illegal Monte Carlo candidate move(s): {illegal_moves!r}")

    if not legal_moves:
        return None, []

    if len(legal_moves) == 1:
        only_move = legal_moves[0]

        return only_move, [
            MoveStatistics(
                move=only_move,
                visits=1,
                wins=0,
                total_score=0.0,
            )
        ]

    rng = random.Random(seed)
    expected_total = total_card_count(state)

    statistics = [
        MoveStatistics(move=move)
        for move in legal_moves
    ]

    started_at = time.monotonic()
    completed_simulations = 0
    move_index = 0

    while completed_simulations < simulations:
        elapsed = time.monotonic() - started_at

        if elapsed >= time_limit:
            break

        stats = statistics[move_index]
        move_index = (move_index + 1) % len(statistics)

        try:
            first_state = apply_move(state, stats.move)
        except (IndexError, KeyError, ValueError):
            stats.visits += 1
            stats.total_score += float("-inf")
            completed_simulations += 1
            continue

        score, won, _ = rollout(
            starting_state=first_state,
            expected_total=expected_total,
            rng=rng,
            max_depth=max_depth,
        )

        stats.visits += 1
        stats.total_score += score

        if won:
            stats.wins += 1

        completed_simulations += 1

    statistics.sort(
        key=lambda item: (
            item.win_rate,
            item.average_score,
            item.visits,
        ),
        reverse=True,
    )

    return statistics[0].move, statistics


def print_statistics(statistics: list[MoveStatistics]) -> None:
    print("\nMonte Carlo results:")

    for index, stats in enumerate(statistics, start=1):
        print(
            f"{index:>2}. {stats.move!s:<45} "
            f"trials={stats.visits:<5} "
            f"wins={stats.wins:<5} "
            f"win_rate={stats.win_rate:>7.2%} "
            f"average={stats.average_score:>10.2f}"
        )
