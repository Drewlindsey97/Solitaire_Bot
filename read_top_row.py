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

def classify_suit_color(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mask = gray < 200
    if mask.sum() < 5:
        return "?"
    b, g, r = patch[mask].mean(axis=0)
    if r > g + 15 and r > b:
        return "RED"
    return "BLACK"

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
