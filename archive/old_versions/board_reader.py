import cv2
import numpy as np
import pytesseract

img = cv2.imread("Gameplay/frame_0108.png")
h, w = img.shape[:2]

columns = [
    (0, 100),
    (100, 200),
    (200, 300),
    (300, 410),
    (410, 510),
    (510, 610),
    (610, 720),
]

Y_START = 500
STEP = 50
MAX_ROWS = 12

# tighter sub-crops within each card label area
RANK_BOX = (8, 5, 30, 25)   # x_off, y_off, width, height
SUIT_BOX = (8, 30, 25, 20)

def classify_suit_color(patch):
    avg = patch.mean(axis=(0,1))  # BGR
    b, g, r = avg
    if r > g + 20 and r > b:
        return "RED"
    elif r < 100 and g < 100 and b < 100:
        return "BLACK"
    return "?"

for col_idx, (x1, x2) in enumerate(columns):
    print(f"--- Column {col_idx} ---")
    for row in range(MAX_ROWS):
        y1 = Y_START + row * STEP
        if y1 + 50 > h:
            break

        rx, ry, rw, rh = RANK_BOX
        rank_patch = img[y1+ry:y1+ry+rh, x1+rx:x1+rx+rw]

        sx, sy, sw, sh = SUIT_BOX
        suit_patch = img[y1+sy:y1+sy+sh, x1+sx:x1+sx+sw]

        if rank_patch.size == 0:
            break

        avg_val = rank_patch.mean()
        if avg_val > 100 and rank_patch.std() < 15:
            # likely flat green background, stop this column
            break

        cv2.imwrite(f"debug_corners/col{col_idx}_row{row}_rank.png", rank_patch)
        cv2.imwrite(f"debug_corners/col{col_idx}_row{row}_suit.png", suit_patch)

        gray = cv2.cvtColor(rank_patch, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(
            thresh, config="--psm 10 -c tessedit_char_whitelist=A0123456789JQK"
        ).strip()

        color = classify_suit_color(suit_patch)
        print(f"  row {row} (y={y1}): rank='{text}' suit_color={color}")
