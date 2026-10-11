from unittest.mock import patch

import pytest

from freecell_solver import State, UNKNOWN
from monte_carlo_solver import choose_move_monte_carlo, rollout
from solitaire_auto_bot import parse_args
import solitaire_auto_bot as bot


def test_default_is_race_with_short_explicit_solver_budget():
    args = parse_args([])
    assert args.solver == "race"
    assert args.solver_time_limit == .5
    assert not args.save_screenshots
    assert parse_args(["--solver", "search", "--solver-time-limit", "2"]).solver_time_limit == 2


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_decision_budgets_are_rejected(value):
    with pytest.raises(SystemExit):
        parse_args(["--solver-time-limit", value])


def test_single_legal_move_does_not_claim_a_simulation():
    state = State([[UNKNOWN]] + [[]] * 6, [], 3, 24, {})
    move, stats = choose_move_monte_carlo(state)
    assert move == ("draw",)
    assert sum(s.visits for s in stats) == 0


def test_expired_rollout_returns_before_selecting_another_move():
    state = State([[UNKNOWN]] + [[]] * 6, [], 3, 24, {})
    with patch("monte_carlo_solver.time.monotonic", return_value=5), \
            patch("monte_carlo_solver.choose_weighted_move") as choose:
        _, won, depth = rollout(state, 4, None, deadline=4)
    assert not won and depth == 0
    choose.assert_not_called()


def test_zero_simulations_selects_productive_move_without_fabricating_trials():
    state = State([[("A", "H")], [("5", "S")]] + [[]] * 5, [("4", "H")], 3, 24, {})
    move, stats = choose_move_monte_carlo(state, simulations=0)
    assert move == ("col_to_found", 0, ("A", "H"))
    assert sum(s.visits for s in stats) == 0


@pytest.mark.parametrize("extra, swipe", [([], None), (["--swipe-ms", "750"], 750)])
def test_fast_cli_preserves_drag_timing_unless_explicitly_overridden(tmp_path, extra, swipe):
    image = tmp_path / "frame.png"
    image.touch()
    board = {f"col{i}": [dict(rank="?", suit="?", color="?", score=0)] * 4
             for i in range(7)}
    board.update(foundation=[None] * 4,
                 waste=[dict(rank="A", suit="H", color="RED", score=.9)])
    argv = ["solitaire_auto_bot.py", "--sim", str(image), "--fast"] + extra
    with patch("sys.argv", argv), patch.object(bot, "read_board", return_value=board), \
            patch.object(bot.bridge, "configure_timing") as configure, \
            patch.object(bot, "execute_move", return_value=True) as execute:
        bot.main()
    assert execute.call_args.args[1] == ("waste_to_found", ("A", "H"))
    if swipe is None:
        configure.assert_not_called()
    else:
        configure.assert_called_once_with(swipe_ms=swipe)
