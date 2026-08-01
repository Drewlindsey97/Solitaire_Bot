import json
from pathlib import Path
from hybrid_runner import run_hybrid

ROOT = Path(__file__).parent
FAIL_DIR = ROOT / 'failing'
META = FAIL_DIR / 'metadata.json'


def test_failing_deals_runs():
    assert META.exists(), 'metadata.json missing in tests/failing'
    meta = json.loads(META.read_text())
    for entry in meta:
        img = FAIL_DIR / entry['image']
        assert img.exists(), f'Missing image {img}'
        ok, move, path = run_hybrid(img, mc_time=1.0, top_k=1, per_move_time=1.0)
        # Test simply ensures the pipeline executes without raising and returns expected tuple
        assert isinstance(ok, bool)
        assert isinstance(path, (list, type(None)))
