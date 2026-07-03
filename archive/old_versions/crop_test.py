import cv2

img = cv2.imread("Gameplay/frame_0108.png")

# test box - wider and taller than before
x, y = 10, 507   # column 0 top-left
BOX_W, BOX_H = 45, 45

crop = img[y:y+BOX_H, x:x+BOX_W]
crop_big = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
cv2.imwrite("debug_corners/test_crop.png", crop_big)
print("Saved debug_corners/test_crop.png")
