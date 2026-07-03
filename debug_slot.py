import cv2

img = cv2.imread("Gameplay/frame_0108.png")
patch = img[303:303+90, 10:10+95]
cv2.imwrite("debug_slot0.png", patch)
print("Saved debug_slot0.png")
