"""Hybrid Monte Carlo + focused search runner.

1) Run Monte Carlo rollouts for `mc_time` seconds to rank legal opening moves.
2) For the top `k` moves, run the deterministic search `solve()` with `per_move_time`
   seconds each from the position reached after applying that move.
3) Report if any found a full solution within the total budget.

Usage: python3 hybrid_runner.py <board_image_path>
"""
import sys
import time
from pathlib import Path

from board_reader_lib import read_board, STOCK_TOTAL
from solitaire_auto_bot import assign_pseudo_suits
from solver_state import build_solver_state
from freecell_solver import State, apply_move, solve
from monte_carlo_solver import choose_move_monte_carlo, print_statistics


def build_state_from_image(img_path):
    board = read_board(str(img_path))
    assign_pseudo_suits(board)
    state = build_solver_state(board, stock_total=STOCK_TOTAL)
    for msg in state['issues']:
        print(f'[Warn] {msg}')
    return state['cols'], state['waste'], state['stock_remaining'], state['found']


def run_hybrid(img_path, mc_time=30.0, top_k=3, per_move_time=50.0, total_search_budget=180.0):
    print('Building state from', img_path)
    cols, waste, stock_remaining, found = build_state_from_image(img_path)
    state = State(cols, waste, stock_remaining, STOCK_TOTAL, found)
    print('State: cols=', [len(c) for c in cols], 'waste=', waste, 'found=', found)

    # 1) Monte Carlo ranking
    print(f'Running Monte Carlo for {mc_time}s')
    move, stats = choose_move_monte_carlo(state, simulations=1000000, time_limit=mc_time, max_depth=200)
    print('\nMonte Carlo selected:', move)
    print_statistics(stats[:min(10, len(stats))])

    # pick top_k moves (from stats sorted by win_rate/avg)
    top_moves = [s.move for s in stats[:top_k]]
    print('\nTop moves to search:', top_moves)

    total_start = time.time()
    for idx, m in enumerate(top_moves):
        elapsed = time.time() - total_start
        remaining = max(0.0, total_search_budget - elapsed)
        allowed = min(per_move_time, remaining)
        if allowed <= 0:
            break
        print(f"\nSearching after applying move {idx+1}/{len(top_moves)}: {m} (allowed {allowed:.1f}s)")
        try:
            first_state = apply_move(state, m)
        except Exception as e:
            print('Failed to apply move to state:', e)
            continue
        path, explored, solved, status = solve(
            list(first_state.cols), list(first_state.waste), first_state.stock_remaining,
            first_state.stock_total, first_state.found_dict(), time_limit=allowed
        )
        if path and solved:
            print('Found full solution after move', m)
            return True, m, path
        else:
            print('Search result:', status, 'solved=', solved, 'explored=', explored, 'path_len=', len(path) if path else 0)
    return False, None, None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 hybrid_runner.py <board_image_path>')
        raise SystemExit(2)
    img = Path(sys.argv[1])
    ok, move, path = run_hybrid(img, mc_time=30.0, top_k=3, per_move_time=50.0)
    if ok:
        print('SOLUTION FOUND via move', move)
    else:
        print('No full solution found by hybrid run')
