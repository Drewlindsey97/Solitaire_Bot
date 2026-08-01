#!/usr/bin/env python3
"""Run focused deterministic searches in parallel for each legal opening move.

Usage: python3 parallel_search.py <image-path> --workers N --time 180
"""
import argparse
import math
import multiprocessing as mp
import time
from pathlib import Path

from board_reader_lib import read_board
from solitaire_auto_bot import assign_pseudo_suits, UNKNOWN, rank_val
from freecell_solver import State, apply_move, generate_moves
from freecell_solver import solve as solve_fn


def build_state(img_path):
    board = read_board(str(img_path))
    assign_pseudo_suits(board)
    cols = []
    for idx in range(7):
        col = []
        for card in board.get(f'col{idx}', []):
            if card and card.get('rank') == '?' and card.get('color') == '?':
                col.append(UNKNOWN)
                continue
            if card and 'suit' in card:
                col.append((card['rank'], card['suit']))
            else:
                break
        cols.append(col)
    waste = []
    for card in board.get('waste', []):
        if card and 'suit' in card:
            waste.append((card['rank'], card['suit']))
    found = {}
    for card in board.get('foundation', []):
        if card and 'suit' in card:
            found[card['suit']] = rank_val(card['rank'])
    stock_remaining = 52 - sum(len(c) for c in cols) - len(waste) - sum(v+1 for v in found.values())
    return State(cols, waste, stock_remaining, 24, found)


def worker_task(args):
    move, state_serialized, time_limit = args
    # Recreate state from serialized data to avoid pickling-freecell objects
    cols, waste, stock_remaining, found = state_serialized
    first_state = apply_move(State(cols, waste, stock_remaining, 24, found), move)
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
