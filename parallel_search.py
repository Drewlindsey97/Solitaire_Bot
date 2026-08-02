#!/usr/bin/env python3
"""Run focused deterministic searches in parallel for each legal opening move.

Usage: python3 parallel_search.py <image-path> --workers N --time 180
"""
import argparse
import math
import multiprocessing as mp
import time
from pathlib import Path

from board_reader_lib import read_board, STOCK_TOTAL
from solitaire_auto_bot import assign_pseudo_suits
from solver_state import build_solver_state
from freecell_solver import State, apply_move, generate_moves
from freecell_solver import solve as solve_fn


def build_state(img_path):
    board = read_board(str(img_path))
    assign_pseudo_suits(board)
    state = build_solver_state(board, stock_total=STOCK_TOTAL)
    for msg in state['issues']:
        print(f'[Warn] {msg}')
    return State(state['cols'], state['waste'], state['stock_remaining'],
                 STOCK_TOTAL, state['found'])


def worker_task(args):
    move, state_serialized, time_limit = args
    # Recreate state from serialized data to avoid pickling-freecell objects
    cols, waste, stock_remaining, found = state_serialized
    first_state = apply_move(State(cols, waste, stock_remaining, STOCK_TOTAL, found), move)
    # convert first_state to raw lists for solve_fn
    cols_list = list(first_state.cols)
    waste_list = list(first_state.waste)
    stock_rem = first_state.stock_remaining
    stock_total = first_state.stock_total
    found_dict = first_state.found_dict()
    path, explored, solved, status = solve_fn(cols_list, waste_list, stock_rem, stock_total, found_dict, time_limit=time_limit)
    return (move, solved, status, len(path) if path else 0, explored, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--time', type=int, default=180)
    parser.add_argument('--top', type=int, default=0, help='If >0, only test top K moves (by generate_moves order)')
    args = parser.parse_args()

    img = Path(args.img)
    if not img.exists():
        print('Image not found', img)
        return

    state = build_state(img)
    legal = generate_moves(state)
    if not legal:
        print('No legal moves found; nothing to search')
        return

    if args.top > 0:
        legal = legal[:args.top]

    print(f'Found {len(legal)} candidate moves; launching up to {args.workers} parallel searches (time per move {args.time}s)')

    # serialize state as raw lists for workers
    state_serialized = (list(state.cols), list(state.waste), state.stock_remaining, state.found_dict())

    pool = mp.Pool(processes=args.workers)
    tasks = [(m, state_serialized, args.time) for m in legal]
    results_async = pool.map_async(worker_task, tasks)

    start = time.time()
    try:
        # Pool runs tasks in batches of args.workers, so the global wait
        # must cover every batch, not just one.
        batches = math.ceil(len(legal) / args.workers)
        results = results_async.get(timeout=batches * args.time + 5)
    except mp.TimeoutError:
        print('Global timeout reached; terminating workers')
        pool.terminate()
        pool.join()
        return
    finally:
        pool.close()
        pool.join()

    # Check results for a solved path
    for move, solved, status, path_len, explored, path in results:
        print('Move:', move, 'solved:', solved, 'status:', status, 'path_len:', path_len, 'explored:', explored)
        if solved:
            print('Solution found! First move:', move)
            print('Full path length:', path_len)
            return
    print('No full solution found among tested moves')

if __name__ == '__main__':
    main()
