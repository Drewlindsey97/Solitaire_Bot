import cv2
import glob
import numpy as np
import os
import shutil

# ---------------- LOAD FRAMES ----------------
paths = sorted(glob.glob("Gameplay/*.png"))
print("Frames found:", len(paths))

# ---------------- UNIQUE STATES ----------------
def board_hash(img):
    h, w = img.shape[:2]
    crop = img[int(h*0.15):int(h*0.95), 0:w]
    small = cv2.resize(crop, (64, 64))
    return hash(small.tobytes())

unique_states = []
seen = set()
for p in paths:
    img = cv2.imread(p)
    if img is None:
        continue
    h = board_hash(img)
    if h in seen:
        continue
    seen.add(h)
    unique_states.append(img)
print("Unique game states:", len(unique_states))

# ---------------- COLUMN SAMPLES ----------------
img = unique_states[0]
h, w = img.shape[:2]
col_width = w // 7

for i in range(7):
    x1 = i * col_width
    x2 = (i + 1) * col_width
    col = img[:, x1:x2]
    cv2.imwrite(f"col_{i}.png", col)
print("Saved 7 column samples")

# ---------------- CARD EXTRACTION ----------------
shutil.rmtree("cards_out", ignore_errors=True)
os.makedirs("cards_out", exist_ok=True)

CARD_H = 140
STRIDE = 15
card_count = 0

for i in range(7):
    x1 = i * col_width
    x2 = (i + 1) * col_width
    col = img[:, x1:x2]
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
    y = 0
    while y < h - CARD_H:
        patch = gray[y:y+CARD_H, :]
        edges = cv2.Canny(patch, 50, 150)
        score = np.sum(edges)
        if score > 15000:
            card = col[y:y+CARD_H, :]
            if card.shape[0] < 120 or card.shape[1] < 40:
                y += int(CARD_H * 0.75)
                continue
            cv2.imwrite(f"cards_out/card_{card_count}.png", card)
            card_count += 1
            y += int(CARD_H * 0.75)
        else:
            y += STRIDE

print("Cards extracted:", card_count)
