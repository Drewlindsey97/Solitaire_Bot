import cv2
import os

img = cv2.imread("Gameplay/frame_0108.png")
os.makedirs("templates", exist_ok=True)

STEP = 50
RANK_W, RANK_H = 45, 45

# (col_x, y_top, [ranks in order down the column], last_card_box)
columns = [
    (10,  507, ["K","Q","J","10","9","8","7","6"]),
    (111, 507, ["K","Q","J","10","9","8","7"]),
    (212, 507, ["K","Q","J","10","9","8","7","6","5"]),
    (313, 507, ["K"]),
    (414, 507, ["Q","J","10","9","8"]),
]

saved = set()

for x, y_top, ranks in columns:
    for row, rank in enumerate(ranks):
        is_last = (row == len(ranks) - 1)
        y = y_top + row * STEP

        if is_last:
            patch = img[y:y+90, x:x+95]
        else:
            patch = img[y:y+RANK_H, x:x+RANK_W]

        # only save the first clean example of each rank (prefer non-last-card versions)
        if rank not in saved or not is_last:
            cv2.imwrite(f"templates/{rank}.png", patch)
            saved.add(rank)

print("Saved templates for:", sorted(saved))
