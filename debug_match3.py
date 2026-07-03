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

# pad the test image so template can slide and find best alignment
pad = 15
padded = cv2.copyMakeBorder(test_binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

print("Test padded shape:", padded.shape)
print()

for name, tmpl in templates_last.items():
    if tmpl.shape[0] > padded.shape[0] or tmpl.shape[1] > padded.shape[1]:
        continue
    result = cv2.matchTemplate(padded, tmpl, cv2.TM_CCOEFF_NORMED)
    score = result.max()
    print(f"score vs '{name}': {score:.3f}")
