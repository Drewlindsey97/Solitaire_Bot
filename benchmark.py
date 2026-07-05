import time
from board_reader_lib import read_board

start = time.time()
board = read_board("Gameplay/frame_0108.png")
elapsed = time.time() - start

print(f"Board read in {elapsed*1000:.1f} ms")
