import glob
import time
from board_reader_lib import read_board

CONFIDENCE_THRESHOLD = 0.85

def board_confidence(board):
    scores = []
    for key, val in board.items():
        if key.startswith("col"):
            # face-down cards are honestly "unknown", not a bad read - don't
            # let them drag down the confidence of an otherwise-good frame
            scores.extend(c["score"] for c in val if c["rank"] != "?")
        elif key in ("free_cells", "foundation"):
            scores.extend(c["score"] for c in val if c is not None)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def read_board_gated(frame_path, threshold=CONFIDENCE_THRESHOLD):
    """
    Returns (board, confidence, accepted_bool).
    accepted_bool is False if confidence is too low to trust this read.
    """
    board = read_board(frame_path)
    conf = board_confidence(board)
    accepted = conf >= threshold
    return board, conf, accepted

if __name__ == "__main__":
    frames = sorted(glob.glob("Gameplay/*.png"))
    print(f"Scanning {len(frames)} frames with confidence gate ({CONFIDENCE_THRESHOLD})...\n")

    accepted_count = 0
    rejected_count = 0

    for frame_path in frames:
        board, conf, accepted = read_board_gated(frame_path)
        status = "ACCEPTED" if accepted else "rejected"
        if accepted:
            accepted_count += 1
        else:
            rejected_count += 1
        print(f"{frame_path}: confidence={conf:.2f} [{status}]")

    print(f"\nSummary: {accepted_count} accepted, {rejected_count} rejected out of {len(frames)} frames")
