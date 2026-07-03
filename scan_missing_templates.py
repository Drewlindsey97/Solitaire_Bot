import cv2
import glob
import os
from board_reader_lib import read_board, TABLEAU_BOXES, FREE_CELL_X, FOUNDATION_X, SLOT_Y, SLOT_W, SLOT_H, STEP

os.makedirs("template_candidates", exist_ok=True)

frames = sorted(glob.glob("Gameplay/*.png"))
print(f"Scanning {len(frames)} frames for low-confidence 'last card' style reads...\n")

LOW_CONF_THRESHOLD = 0.85
saved = 0

for frame_path in frames:
    try:
        board = read_board(frame_path)
    except Exception as e:
        continue

    frame_name = os.path.splitext(os.path.basename(frame_path))[0]
    img = cv2.imread(frame_path)

    # check free cells
    for i, x in enumerate(FREE_CELL_X):
        slot = board["free_cells"][i]
        if slot and slot["score"] < LOW_CONF_THRESHOLD:
            patch = img[SLOT_Y:SLOT_Y+SLOT_H, x:x+SLOT_W]
            out_name = f"template_candidates/{frame_name}_free{i}_guess-{slot['rank']}_score{slot['score']:.2f}.png"
            cv2.imwrite(out_name, patch)
            saved += 1

    # check foundation
    for i, x in enumerate(FOUNDATION_X):
        slot = board["foundation"][i]
        if slot and slot["score"] < LOW_CONF_THRESHOLD:
            patch = img[SLOT_Y:SLOT_Y+SLOT_H, x:x+SLOT_W]
            out_name = f"template_candidates/{frame_name}_found{i}_guess-{slot['rank']}_score{slot['score']:.2f}.png"
            cv2.imwrite(out_name, patch)
            saved += 1

    # check last-card (bottom) tableau cards
    for col_idx, x, y_top, height in TABLEAU_BOXES:
        if height == 0:
            continue
        col_key = f"col{col_idx}"
        cards = board.get(col_key, [])
        if not cards:
            continue
        last_card = cards[-1]
        if last_card["score"] < LOW_CONF_THRESHOLD:
            num_rows = round((height - 135) / 50) + 1
            y = y_top + (num_rows - 1) * STEP
            patch = img[y:y+90, x:x+95]
            out_name = f"template_candidates/{frame_name}_col{col_idx}last_guess-{last_card['rank']}_score{last_card['score']:.2f}.png"
            cv2.imwrite(out_name, patch)
            saved += 1

print(f"Saved {saved} low-confidence candidate images to template_candidates/")
