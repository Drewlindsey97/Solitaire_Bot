import cv2
import pytesseract

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
SUIT_W, SUIT_H = 45, 30

def classify_suit_color(patch):
    if patch.size == 0:
        return "?"
    avg = patch.mean(axis=(0,1))
    b, g, r = avg
    if r > g + 20 and r > b:
        return "RED"
    elif r < 110 and g < 110 and b < 110:
        return "BLACK"
    return "?"

for col_idx, x, y_top, height in tableau_boxes:
    print(f"--- Column {col_idx} ---")
    y = y_top
    row = 0
    while y < y_top + height - 20:
        rank_patch = img[y:y+RANK_H, x:x+RANK_W]
        suit_patch = img[y+RANK_H:y+RANK_H+SUIT_H, x:x+SUIT_W]

        gray = cv2.cvtColor(rank_patch, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(
            thresh, config="--psm 10 -c tessedit_char_whitelist=A0123456789JQK"
        ).strip()

        color = classify_suit_color(suit_patch)

        cv2.imwrite(f"debug_corners/c{col_idx}_r{row}_rank.png", rank_patch)

        print(f"  row {row} (y={y}): rank='{text}' suit={color}")
        y += STEP
        row += 1
