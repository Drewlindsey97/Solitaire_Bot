#!/usr/bin/env python3
"""
Collect failing images from benchmark CSV and copy them to tests/failing/ with metadata.
"""
import csv
from pathlib import Path
import shutil

ROOT = Path(__file__).parent
LOG = ROOT / 'logs' / 'benchmark_20_results.csv'
OUT = ROOT / 'tests' / 'failing'
OUT.mkdir(parents=True, exist_ok=True)

if not LOG.exists():
    print('Benchmark CSV not found at', LOG)
    raise SystemExit(1)

rows = []
with LOG.open() as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

failures = [row for row in rows if row.get('solved') in ('0','False','')]
if not failures:
    print('No failures found in', LOG)
    raise SystemExit(0)

meta = []
for row in failures:
    img = ROOT / 'Gameplay' / row['image']
    if not img.exists():
        print('Image missing:', img)
        continue
    dest = OUT / img.name
    shutil.copy2(img, dest)
    meta.append({'image': img.name, 'duration_s': row.get('duration_s',''), 'path_len': row.get('path_len','')})

import json
with (OUT / 'metadata.json').open('w') as f:
    json.dump(meta, f, indent=2)

print('Copied', len(meta), 'failing images to', OUT)
