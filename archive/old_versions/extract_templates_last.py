import cv2
import os

img = cv2.imread("Gameplay/frame_0108.png")
os.makedirs("templates_last", exist_ok=True)

# (x, y, rank) for each bottom/last card
last_cards = [
    (10,  857, "6"),
    (111, 807, "7"),
    (212, 907, "5"),
    (313, 507, "K"),
    (414, 707, "8"),
]

for x, y, rank in last_cards:
    patch = img[y:y+90, x:x+95]
    cv2.imwrite(f"templates_last/{rank}.png", patch)

print("Saved last-card templates:", [r for _,_,r in last_cards])
