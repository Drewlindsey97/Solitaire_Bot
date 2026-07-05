import cv2
import glob
import os
import pytesseract

paths = sorted(glob.glob("cards_out/*.png"))
print("Cards found:", len(paths))

os.makedirs("debug_corners", exist_ok=True)

for idx, p in enumerate(paths):
    img = cv2.imread(p)
    if img is None:
        continue

    h, w = img.shape[:2]
    corner = img[0:int(h*0.35), 0:int(w*0.45)]
    corner = cv2.resize(corner, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # save debug images for first 5 cards only
    if idx < 5:
        cv2.imwrite(f"debug_corners/orig_{idx}.png", img)
        cv2.imwrite(f"debug_corners/corner_{idx}.png", corner)
        cv2.imwrite(f"debug_corners/thresh_{idx}.png", thresh)

    text = pytesseract.image_to_string(
        thresh,
        config="--psm 10 -c tessedit_char_whitelist=A2345678910JQK"
    ).strip()

    print(f"{p}: '{text}'")
