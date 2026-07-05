import cv2
import numpy as np
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
SUIT_X_OFF, SUIT_Y_OFF = 5, 42
SUIT_W, SUIT_H = 35, 22

def classify_suit_color(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mask = gray < 200
    if mask.sum() < 5:
        return "?"
    b = patch[:, :, 0][mask].mean()
    g = patch[:, :, 1][mask].mean()
    r = patch[:, :, 2][mask].mean()
    if r > g + 15 and r > b:
        return "RED"
    return "BLACK"

def ocr_rank(patch):
    # distance from white, per-pixel -- works equally well for red or black ink
    diff = 255 - patch.astype(np.int16)
    dist = np.sqrt((diff**2).sum(axis=2))
    dist = (dist / dist.max() * 255).astype(np.uint8) if dist.max() > 0 else dist.astype(np.uint8)

    dist = cv2.resize(dist, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(dist, 60, 255, cv2.THRESH_BINARY)

    return pytesseract.image_to_string(
        thresh, config="--psm 10 -c tessedit_char_whitelist=A0123456789JQK"
    ).strip()

for col_idx, x, y_top, height in tableau_boxes:
    print(f"--- Column {col_idx} ---")
    num_rows = round((height - 135) / 50) + 1
    for row in range(num_rows):
        y = y_top + row * STEP

        rank_patch = img[y:y+RANK_H, x:x+RANK_W]
        suit_patch = img[y+SUIT_Y_OFF:y+SUIT_Y_OFF+SUIT_H, x+SUIT_X_OFF:x+SUIT_X_OFF+SUIT_W]

        text = ocr_rank(rank_patch)
        color = classify_suit_color(suit_patch)

        print(f"  row {row} (y={y}): rank='{text}' suit={color}")
