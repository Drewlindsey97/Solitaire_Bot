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

VALID_RANKS = {"A","2","3","4","5","6","7","8","9","10","J","Q","K"}

def classify_suit_color(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mask = gray < 200
    if mask.sum() < 5:
        return "?"
    b, g, r = patch[mask].mean(axis=0)
    if r > g + 15 and r > b:
        return "RED"
    return "BLACK"

def try_ocr(thresh_img, psm):
    text = pytesseract.image_to_string(
        thresh_img, config=f"--psm {psm} -c tessedit_char_whitelist=A0123456789JQK"
    ).strip()
    return text

def ocr_rank(patch, psm=10):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)

    # attempt 1: fixed threshold
    _, t1 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    r1 = try_ocr(t1, psm)
    if r1 in VALID_RANKS:
        return r1

    # attempt 2: Otsu
    _, t2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    r2 = try_ocr(t2, psm)
    if r2 in VALID_RANKS:
        return r2

    # attempt 3: adaptive threshold
    t3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 25, 10)
    r3 = try_ocr(t3, psm)
    if r3 in VALID_RANKS:
        return r3

    # nothing matched cleanly, return best guess
    return r1 or r2 or r3

for col_idx, x, y_top, height in tableau_boxes:
    print(f"--- Column {col_idx} ---")
    num_rows = round((height - 135) / 50) + 1
    for row in range(num_rows):
        y = y_top + row * STEP
        is_last = (row == num_rows - 1)

        if is_last:
            rank_patch = img[y:y+90, x:x+95]
            text = ocr_rank(rank_patch, psm=7)
            color = classify_suit_color(img[y+20:y+40, x+55:x+90])
        else:
            rank_patch = img[y:y+RANK_H, x:x+RANK_W]
            text = ocr_rank(rank_patch, psm=10)
            suit_patch = img[y+SUIT_Y_OFF:y+SUIT_Y_OFF+SUIT_H, x+SUIT_X_OFF:x+SUIT_X_OFF+SUIT_W]
            color = classify_suit_color(suit_patch)

        print(f"  row {row} (y={y}) {'[LAST]' if is_last else ''}: rank='{text}' suit={color}")
