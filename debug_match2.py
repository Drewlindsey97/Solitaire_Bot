import cv2
import glob
import os

def load_templates(folder):
    t = {}
    for path in glob.glob(f"{folder}/*.png"):
        name = os.path.splitext(os.path.basename(path))[0]
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        t[name] = binary
    return t

templates_last = load_templates("templates_last")

img = cv2.imread("debug_slot0.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, test_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

print("Test image shape:", test_binary.shape)
print()

for name, tmpl in templates_last.items():
    resized = cv2.resize(test_binary, (tmpl.shape[1], tmpl.shape[0]))
    result = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
    score = result.max()
    print(f"score vs '{name}': {score:.3f}")
