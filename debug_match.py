import cv2
import glob
import os

def load_templates(folder):
    t = {}
    for path in glob.glob(f"{folder}/*.png"):
        name = os.path.splitext(os.path.basename(path))[0]
        t[name] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return t

templates_last = load_templates("templates_last")

img = cv2.imread("debug_slot0.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Test image shape:", gray.shape)
print()

for name, tmpl in templates_last.items():
    print(f"Template '{name}' shape: {tmpl.shape}")
    resized = cv2.resize(gray, (tmpl.shape[1], tmpl.shape[0]))
    result = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
    score = result.max()
    print(f"  -> score vs '{name}': {score:.3f}")
