#!/usr/bin/env python3
"""
Run hybrid_runner.run_hybrid across the first 20 sample Gameplay images and record results.
Quick baseline: mc_time=10s, top_k=3, per_move_time=20s (short to keep total runtime reasonable).
Outputs CSV to logs/benchmark_20_results.csv
"""
import time
from pathlib import Path
import csv

from hybrid_runner import run_hybrid

GAMEPLAY_DIR = Path(__file__).parent / 'Gameplay'
OUT_DIR = Path(__file__).parent / 'logs'
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / 'benchmark_20_results.csv'

# collect first 20 frame images sorted
frames = sorted([p for p in GAMEPLAY_DIR.iterdir() if p.suffix.lower()=='.png'])[:20]
if not frames:
    print('No frames found in', GAMEPLAY_DIR)
    raise SystemExit(1)

rows = []
for img in frames:
    t0 = time.time()
    try:
        ok, move, path = run_hybrid(img, mc_time=10.0, top_k=3, per_move_time=20.0)
        duration = time.time() - t0
        rows.append({'image': img.name, 'solved': int(ok), 'first_move': str(move), 'path_len': len(path) if path else 0, 'duration_s': round(duration,2)})
        print(f'Image {img.name}: solved={ok} move={move} time={duration:.1f}s')
    except Exception as e:
        duration = time.time() - t0
        rows.append({'image': img.name, 'solved': 0, 'first_move': 'ERROR', 'path_len': 0, 'duration_s': round(duration,2)})
        print(f'Image {img.name}: ERROR {e}')

# write CSV
with open(OUT_FILE, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image','solved','first_move','path_len','duration_s'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print('Wrote results to', OUT_FILE)
