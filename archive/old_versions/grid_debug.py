import cv2

img = cv2.imread("Gameplay/frame_0108.png")
h, w = img.shape[:2]

# draw vertical lines every 50px, horizontal every 50px
for x in range(0, w, 50):
    cv2.line(img, (x, 0), (x, h), (0, 0, 255), 1)
    cv2.putText(img, str(x), (x+2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)

for y in range(0, h, 50):
    cv2.line(img, (0, y), (w, y), (255, 0, 0), 1)
    cv2.putText(img, str(y), (2, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,0,0), 1)

cv2.imwrite("grid_overlay.png", img)
print("Saved grid_overlay.png")
