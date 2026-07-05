import cv2
import glob
import os

STEP = 50
RANK_W, RANK_H = 45, 45
SUIT_X_OFF, SUIT_Y_OFF = 5, 42
SUIT_W, SUIT_H = 35, 22
PAD = 15

TABLEAU_BOXES = [
    (0, 10, 507, 471),
    (1, 111, 507, 423),
    (2, 212, 507, 519),
    (3, 313, 507, 135),
    (4, 414, 507, 327),
    (5, 512, 507, 0),
    (6, 611, 507, 0),
]

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

def read_board(frame_path):
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(frame_path)

    board = {}
    for col_idx, x, y_top, height in TABLEAU_BOXES:
        col_cards = []
        if height == 0:
            board[f"col{col_idx}"] = col_cards
            continue

        num_rows = round((height - 135) / 50) + 1
        for row in range(num_rows):
            y = y_top + row * STEP
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
