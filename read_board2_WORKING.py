import json
from board_reader_lib import read_board

board = read_board("Gameplay/frame_0108.png")
print(json.dumps(board, indent=2))

with open("board_state.json", "w") as f:
    json.dump(board, f, indent=2)
print("\nSaved to board_state.json")
