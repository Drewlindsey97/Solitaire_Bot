import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from board_reader_lib import (  # noqa: E402
    BoardLayout,
    TEMPLATES_LAST,
    classify_suit_color,
    match_rank,
)


def inspect_frame(frame: Path) -> None:
    img = cv2.imread(str(frame))
    if img is None:
        print(f"[Error] Frame {frame} not found.")
        return

    layout = BoardLayout()
    slots_x = list(layout.free_cell_x) + list(layout.foundation_x)
    print(f"\n=== {frame} ===")
    for x in slots_x:
        patch = img[layout.slot_y:layout.slot_y + layout.slot_height, x:x + layout.slot_width]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        name, score = match_rank(patch, TEMPLATES_LAST)
        color = classify_suit_color(img[layout.slot_y + 20:layout.slot_y + 40, x + 55:x + 90])

        scores = {}
        for tname, template in TEMPLATES_LAST.items():
            g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            padded = cv2.copyMakeBorder(binary, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=0)
            if template.shape[0] > padded.shape[0] or template.shape[1] > padded.shape[1]:
                continue
            result = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
            scores[tname] = round(float(result.max()), 3)

        top3 = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        print(f"x={x:3d}: std={gray.std():6.1f} best={name}({score:.2f}) color={color} top3={top3}")


def main() -> None:
    frames = [Path(arg) for arg in sys.argv[1:]]
    if not frames:
        frames = [ROOT / "Gameplay/frame_0001.png", ROOT / "Gameplay/frame_0100.png"]
    for frame in frames:
        inspect_frame(frame)


if __name__ == "__main__":
    main()
