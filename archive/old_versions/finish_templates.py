import cv2

img = cv2.imread("Gameplay/frame_0108.png")

# Free cell row cards visible in frame_0108: 7,5,6,3 (free cells) and 4 (foundation)
# From your screenshot: free cells at y~300, x positions ~10,110,210,310, foundation ~470
free_cell_cards = [
    (10,  303, "7freecell"),   # already have 7, skip
    (472, 303, "4"),           # this is our missing 4!
]

for x, y, rank in free_cell_cards:
    if rank == "4":
        patch = img[y:y+90, x:x+95]
        cv2.imwrite("templates_last/4.png", patch)
        # also make a small-label version by using same crop resized down for regular slot use
        small = img[y:y+45, x:x+45]
        cv2.imwrite("templates/4.png", small)
        print("Saved template for 4")
