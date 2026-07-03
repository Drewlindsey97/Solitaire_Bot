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
SUIT_X_OFF, SUIT_Y_OFF = 5, 42
SUIT_W, SUIT_H = 35, 22

def classify_suit_color(patch):
    if patch.size == 0:
        return "?"
    avg = patch.mean(axis=(0,1))
    b, g, r = avg
    if r > g + 15 and r > b:
        return "RED"
    elif r < 130 and g < 130 and b < 130:
        return "BLACK"
    return "?"

def ocr_rank(patch, psm=10):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    return pytesseract.image_to_string(
        thresh, config=f"--psm {psm} -c tessedit_char_whitelist=A0123456789JQK"
    ).strip()

for col_idx, x, y_top, height in tableau_boxes:
    print(f"--- Column {col_idx} ---")
    num_rows = round(height / STEP)  # last card is the "extra tall" one
    for row in range(num_rows):
        y = y_top + row * STEP
        is_last = (row == num_rows - 1)

        if is_last:
            # bottom card: bigger crop for the enlarged centered digit
            rank_patch = img[y:y+90, x:x+95]
            text = ocr_rank(rank_patch, psm=7)
            color = classify_suit_color(img[y+20:y+40, x+55:x+90])
        else:
            rank_patch = img[y:y+RANK_H, x:x+RANK_W]
            text = ocr_rank(rank_patch, psm=10)
            suit_patch = img[y+SUIT_Y_OFF:y+SUIT_Y_OFF+SUIT_H, x+SUIT_X_OFF:x+SUIT_X_OFF+SUIT_W]
            color = classify_suit_color(suit_patch)

        cv2.imwrite(f"debug_corners/c{col_idx}_r{row}_rank.png", rank_patch)
        print(f"  row {row} (y={y}) {'[LAST]' if is_last else ''}: rank='{text}' suit={color}")
