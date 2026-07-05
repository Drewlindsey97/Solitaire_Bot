import json
from board_reader_lib import read_board

for frame in ["Gameplay/frame_0001.png", "Gameplay/frame_0050.png", "Gameplay/frame_0108.png"]:
    print(f"=== {frame} ===")
    board = read_board(frame)
    for k in ["col0","col1","col2","col3","col4","col5","col6"]:
        cards = board[k]
        ranks = [c["rank"] for c in cards]
        scores = [c["score"] for c in cards]
        avg_score = sum(scores)/len(scores) if scores else 0
        print(f"  {k}: {ranks} (avg score {avg_score:.2f})")
    print()
