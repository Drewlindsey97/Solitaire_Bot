import cv2
import numpy as np
import glob
import os

img = cv2.imread("Gameplay/frame_0108.png")

tableau_boxes = [
    (0, 10, 507, 471),
    (1, 111, 507, 423),
    (2, 212, 507, 519),
    (3, 313, 507, 135),
    (4, 414, 507, 327),
]

STEP = 50
RANK_W, RANK_H = 45, 45

# load templates
templates = {}
for path in glob.glob("templates/*.png"):
    name = os.path.splitext(os.path.basename(path))[0]
    t_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    templates[name] = t_img

def classify_suit_color(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mask = gray < 200
    if mask.sum() < 5:
        return "?"
    b, g, r = patch[mask].mean(axis=0)
    if r > g + 15 and r > b:
        return "RED"
    return "BLACK"

def match_rank(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    best_name, best_score = "?", -1
    for name, tmpl in templates.items():
        # resize patch to match template size for comparison
        resized = cv2.resize(gray, (tmpl.shape[1], tmpl.shape[0]))
        result = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score

SUIT_X_OFF, SUIT_Y_OFF = 5, 42
SUIT_W, SUIT_H = 35, 22

for col_idx, x, y_top, height in tableau_boxes:
    print(f"--- Column {col_idx} ---")
    num_rows = round((height - 135) / 50) + 1
    for row in range(num_rows):
        y = y_top + row * STEP
        is_last = (row == num_rows - 1)

        if is_last:
            rank_patch = img[y:y+90, x:x+95]
            color = classify_suit_color(img[y+20:y+40, x+55:x+90])
        else:
            rank_patch = img[y:y+RANK_H, x:x+RANK_W]
            suit_patch = img[y+SUIT_Y_OFF:y+SUIT_Y_OFF+SUIT_H, x+SUIT_X_OFF:x+SUIT_X_OFF+SUIT_W]
            color = classify_suit_color(suit_patch)

        name, score = match_rank(rank_patch)
        print(f"  row {row} (y={y}) {'[LAST]' if is_last else ''}: rank='{name}' (score={score:.2f}) suit={color}")
