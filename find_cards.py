import cv2
import numpy as np

img = cv2.imread("Gameplay/frame_0108.png")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_white = np.array([0, 0, 180])
upper_white = np.array([180, 60, 255])
mask = cv2.inRange(hsv, lower_white, upper_white)

kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w > 30 and h > 20:
        boxes.append((x, y, w, h))

boxes.sort(key=lambda b: (b[0]//50, b[1]))

print(f"Found {len(boxes)} white regions")

debug = img.copy()
for i, (x, y, w, h) in enumerate(boxes):
    cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(debug, str(i), (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

cv2.imwrite("card_boxes_debug.png", debug)

for i, b in enumerate(boxes):
    print(i, b)
