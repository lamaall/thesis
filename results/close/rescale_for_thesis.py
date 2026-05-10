import os
import re
import numpy as np
from skimage.io import imread, imsave
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
PADDING = 25  # pixels of extra margin around content
TARGET_SIZE = 384 

# match filenames like: anything123.png (ending with digits before .png)
pattern = re.compile(r".*(\d+\.png|_binarized\.png)$")

# ----------------------------
# LOAD + BINARIZE (black = foreground)
# ----------------------------
def to_binary(img):
    img = img.astype(np.float32)
    if img.max() > 1:
        img /= 255.0
    return (img < 0.5).astype(np.uint8)  # black pixels = 1

# ----------------------------
# FIND BOUNDING BOX OF BLACK PIXELS
# ----------------------------
def get_bbox(img):
    ys, xs = np.where(img == 1)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()

# ----------------------------
# PAD AND CROP
# ----------------------------
def crop_with_padding(img, pad):
    bbox = get_bbox(img)
    if bbox is None:
        return img

    y1, y2, x1, x2 = bbox

    h, w = img.shape

    y1 = max(0, y1 - pad)
    y2 = min(h - 1, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(w - 1, x2 + pad)

    return img[y1:y2+1, x1:x2+1]

# ----------------------------
# PROCESS SINGLE IMAGE
# ----------------------------
def process(path, out_path):
    img = imread(path, as_gray=True)
    img = to_binary(img)

    # 1. crop using black pixels
    img = crop_with_padding(img, PADDING)

    # 2. invert back to black-on-white
    img = (1 - img).astype(np.uint8)

    h, w = img.shape

    # 3. SCALE BY HEIGHT ONLY (fixed reference)
    target_h = TARGET_SIZE
    scale = target_h / h

    new_h = target_h
    new_w = max(1, int(w * scale))

    resized = np.array(
        Image.fromarray((img * 255).astype(np.uint8)).resize((new_w, new_h))
    )

    resized = (resized > 128).astype(np.uint8)

    # 4. save directly
    imsave(out_path, (resized * 255).astype(np.uint8))
# ----------------------------
# MAIN LOOP
# ----------------------------
root = "./"

for r, _, files in os.walk(root):
    for f in files:

        if pattern.match(f):
            inp = os.path.join(r, f)
            out = os.path.join(r, f.replace(".png", "_cropped_for_thesis.png"))

            process(inp, out)
            print("saved:", out)