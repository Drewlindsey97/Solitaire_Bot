import cv2
import glob
import os
import numpy as np

STEP = 50
RANK_W, RANK_H = 45, 45
SUIT_X_OFF, SUIT_Y_OFF = 5, 42
SUIT_W, SUIT_H = 35, 22
PAD = 15
HIDDEN_CARD_H = 30
TOP_RESIDUAL_TOLERANCE = 5

# fixed x-start positions for each tableau column (UI layout doesn't move)
TABLEAU_X = [10, 111, 212, 313, 414, 512, 611]
TABLEAU_Y_TOP = 507
COL_WIDTH = 95

FREE_CELL_X = [10, 110, 210, 310]
FOUNDATION_X = [472]
SLOT_Y = 303
SLOT_W, SLOT_H = 95, 90

def load_templates(folder):
    t = {}
    for path in glob.glob(f"{folder}/*.png"):
        name = os.path.splitext(os.path.basename(path))[0]
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        t[name] = binary
    return t

TEMPLATES = load_templates("templates")
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
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    padded = cv2.copyMakeBorder(binary, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)

    best_name, best_score = "?", -1
    for name, tmpl in template_set.items():
        if tmpl.shape[0] > padded.shape[0] or tmpl.shape[1] > padded.shape[1]:
            continue
        result = cv2.matchTemplate(padded, tmpl, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score

def detect_column_height(img, x):
    """Detect how tall the card stack in this column currently is, using
    white-region contour detection (same approach as find_cards.py).

    The white mask only picks up face-up (revealed) cards; face-down cards at
    the top of a column render as a uniform ~30px sliver per card that the
    mask doesn't see, so the revealed region's top edge sits that much lower
    than the column origin. Returns (height, hidden_count, reliable):
    - height: pixel bottom of the revealed region (unchanged meaning from
      before hidden-card support)
    - hidden_count: how many face-down cards sit above the revealed region,
      inferred from that top offset
    - reliable: False when the top offset doesn't cleanly fit a whole number
      of hidden cards (~30px each) - a sign of a mid-animation render rather
      than a real, stable hidden-card count
    """
    col_slice = img[TABLEAU_Y_TOP:, x:x+COL_WIDTH]
    hsv = cv2.cvtColor(col_slice, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, True

    max_y, min_y = 0, None
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cw > 30 and ch > 20:
            max_y = max(max_y, cy + ch)
            min_y = cy if min_y is None else min(min_y, cy)

    if min_y is None:
        return 0, 0, True

    hidden_count = round(min_y / HIDDEN_CARD_H)
    residual = abs(min_y - hidden_count * HIDDEN_CARD_H)
    reliable = residual <= TOP_RESIDUAL_TOLERANCE
    return max_y, hidden_count, reliable  # bottom pixel of the revealed region

def read_board(frame_path):
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(frame_path)

    board = {}
    for col_idx, x in enumerate(TABLEAU_X):
        height, hidden_count, reliable = detect_column_height(img, x)
        col_cards = []

        if height < 100:  # empty or noise, treat as empty column
            board[f"col{col_idx}"] = col_cards
            continue

        revealed_span = height - hidden_count * HIDDEN_CARD_H
        num_rows = round((revealed_span - 135) / 50) + 1
        num_rows = max(1, num_rows)

        # face-down cards have no readable rank; represent them as unknown
        # rather than feeding their pixels into the rank matcher
        col_cards.extend({"rank": "?", "color": "?", "score": 0.0} for _ in range(hidden_count))

        y_start = TABLEAU_Y_TOP + hidden_count * HIDDEN_CARD_H
        for row in range(num_rows):
            y = y_start + row * STEP
            is_last = (row == num_rows - 1)

            if is_last:
                rank_patch = img[y:y+90, x:x+95]
                name, score = match_rank(rank_patch, TEMPLATES_LAST)
                color = classify_suit_color(img[y+20:y+40, x+55:x+90])
            else:
                rank_patch = img[y:y+RANK_H, x:x+RANK_W]
                name, score = match_rank(rank_patch, TEMPLATES)
                suit_patch = img[y+SUIT_Y_OFF:y+SUIT_Y_OFF+SUIT_H, x+SUIT_X_OFF:x+SUIT_X_OFF+SUIT_W]
                color = classify_suit_color(suit_patch)

            # a top offset that doesn't cleanly fit a whole number of hidden
            # cards means this frame is mid-animation, not a stable state -
            # every row crop here is misaligned regardless of match_rank's score
            if not reliable:
                score = 0.0

            col_cards.append({"rank": name, "color": color, "score": round(float(score), 2)})

        board[f"col{col_idx}"] = col_cards

    def read_slot(x):
        patch = img[SLOT_Y:SLOT_Y+SLOT_H, x:x+SLOT_W]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if gray.std() < 15:
            return None
        name, score = match_rank(patch, TEMPLATES_LAST)
        color = classify_suit_color(img[SLOT_Y+20:SLOT_Y+40, x+55:x+90])
        return {"rank": name, "color": color, "score": round(float(score), 2)}

    board["free_cells"] = [read_slot(x) for x in FREE_CELL_X]
    board["foundation"] = [read_slot(x) for x in FOUNDATION_X]

    return board
