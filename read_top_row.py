import cv2
import glob
import os

img = cv2.imread("Gameplay/frame_0108.png")

# top row: 4 free cells + foundation area
# each slot ~95px wide, same rank crop style as "last card" (big centered digit)
SLOT_Y = 303
SLOT_W, SLOT_H = 95, 90

FREE_CELL_X = [10, 110, 210, 310]
FOUNDATION_X = [472]  # only one foundation pile visible in this frame; may need more slots

def load_templates(folder):
    t = {}
    for path in glob.glob(f"{folder}/*.png"):
        name = os.path.splitext(os.path.basename(path))[0]
        t[name] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return t

TEMPLATES_LAST = load_templates("templates_last")

_CANONICAL_CLASSIFY = None

def classify_suit_color(patch):
    # Stale local copy replaced by a delegate to the canonical classifier
    # in the top-level board_reader_lib.py (this legacy script had drifted
    # behind its fixes - it still used the old mean-BGR test). The
    # canonical version returns (color, confident); legacy callers here
    # only use the color string. Imported by file path because pipeline/
    # has its own module named board_reader_lib.
    global _CANONICAL_CLASSIFY
    if _CANONICAL_CLASSIFY is None:
        import importlib.util
        from pathlib import Path
        here = Path(__file__).resolve()
        for cand in (here.parent / "board_reader_lib.py",
                     here.parent.parent / "board_reader_lib.py"):
            if cand.exists() and cand != here:
                spec = importlib.util.spec_from_file_location("_canonical_brl", str(cand))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _CANONICAL_CLASSIFY = mod.classify_suit_color
                break
    res = _CANONICAL_CLASSIFY(patch)
    # normalize: the canonical classifier returns (color, confident), but a
    # delegate chained through another legacy delegate gets a plain string
    return res[0] if isinstance(res, tuple) else res

def match_rank(patch, template_set):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    best_name, best_score = "?", -1
    for name, tmpl in template_set.items():
        resized = cv2.resize(gray, (tmpl.shape[1], tmpl.shape[0]))
        result = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score

def read_slot(x):
    patch = img[SLOT_Y:SLOT_Y+SLOT_H, x:x+SLOT_W]

    # check if slot is empty (mostly uniform background, low variance)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if gray.std() < 15:
        return None  # empty slot

    name, score = match_rank(patch, TEMPLATES_LAST)
    color = classify_suit_color(img[SLOT_Y+20:SLOT_Y+40, x+55:x+90])
    return {"rank": name, "color": color, "score": round(float(score), 2)}

print("--- Free Cells ---")
free_cells = []
for i, x in enumerate(FREE_CELL_X):
    result = read_slot(x)
    free_cells.append(result)
    print(f"  slot {i}: {result}")

print("--- Foundation ---")
foundation = []
for i, x in enumerate(FOUNDATION_X):
    result = read_slot(x)
    foundation.append(result)
    print(f"  pile {i}: {result}")
