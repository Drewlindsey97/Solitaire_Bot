import sys
import os
import cv2

# Set path to local project workspace
project_path = "/Users/mastercontrol/.gemini/antigravity/scratch/Solitaire_Bot"
sys.path.insert(0, project_path)
os.chdir(project_path)

from board_reader_lib import TEMPLATES_LAST, match_rank, classify_suit_color, SLOT_Y, SLOT_W, SLOT_H

SLOTS_X = [10, 110, 210, 310, 472]

for frame in ["Gameplay/frame_0001.png", "Gameplay/frame_0100.png"]:
    img = cv2.imread(frame)
    if img is None:
        print(f"[Error] Frame {frame} not found.")
        continue
    print(f"\n=== {frame} ===")
    for x in SLOTS_X:
        patch = img[SLOT_Y:SLOT_Y+SLOT_H, x:x+SLOT_W]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        name, score = match_rank(patch, TEMPLATES_LAST)
        color = classify_suit_color(img[SLOT_Y+20:SLOT_Y+40, x+55:x+90])
        
        # Calculate scores for all templates to inspect ranking
        scores = {}
        for tname in TEMPLATES_LAST:
            s_patch = patch
            import numpy as np
            g = cv2.cvtColor(s_patch, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            padded = cv2.copyMakeBorder(binary, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=0)
            t = TEMPLATES_LAST[tname]
            if t.shape[0] > padded.shape[0] or t.shape[1] > padded.shape[1]:
                continue
            r = cv2.matchTemplate(padded, t, cv2.TM_CCOEFF_NORMED)
            scores[tname] = round(float(r.max()), 3)
            
        top3 = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        print(f"x={x:3d}: std={gray.std():6.1f} best={name}({score:.2f}) color={color} top3={top3}")
