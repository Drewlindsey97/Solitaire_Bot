import cv2
import glob
import os
from board_reader_lib import read_board, TABLEAU_X, TABLEAU_Y_TOP, STEP, HIDDEN_CARD_H, detect_column_height

os.makedirs("missing_rank_candidates", exist_ok=True)
frames = sorted(glob.glob("Gameplay/*.png"))

saved = 0
for frame_path in frames:
    img = cv2.imread(frame_path)
    board = read_board(frame_path)
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]

    for col_idx, x in enumerate(TABLEAU_X):
        cards = board.get(f"col{col_idx}", [])
        if not cards:
            continue
        last = cards[-1]
        if last["score"] < 0.6:  # very low = likely an untemplated rank
            height, hidden_count, _ = detect_column_height(img, x)
            revealed_span = height - hidden_count * HIDDEN_CARD_H
            num_rows = max(1, round((revealed_span - 135) / 50) + 1)
            y = TABLEAU_Y_TOP + hidden_count * HIDDEN_CARD_H + (num_rows - 1) * STEP
            patch = img[y:y+90, x:x+95]
            out_name = f"missing_rank_candidates/{frame_name}_col{col_idx}_guess-{last['rank']}_score{last['score']:.2f}.png"
            cv2.imwrite(out_name, patch)
            saved += 1

print(f"Saved {saved} candidate images")
