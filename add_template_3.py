import cv2

img = cv2.imread("Gameplay/frame_0108.png")
patch = img[303:303+90, 310:310+95]
cv2.imwrite("templates_last/3.png", patch)
print("Saved templates_last/3.png")
