import cv2
import numpy as np

# test image
img = cv2.imread("debug_slot0.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, test_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite("debug_test_binary.png", test_binary)

# template
tmpl_gray = cv2.imread("templates_last/7.png", cv2.IMREAD_GRAYSCALE)
_, tmpl_binary = cv2.threshold(tmpl_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite("debug_template_binary.png", tmpl_binary)

print("Saved debug_test_binary.png and debug_template_binary.png")
print("Test shape:", test_binary.shape, "Template shape:", tmpl_binary.shape)
