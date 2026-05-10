import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.morphology import skeletonize, dilation, disk

# ----------------------------
# CONFIG
# ----------------------------
OUT_SIZE = 350
DILATION_RADIUS = 3  # <- as requested

# ----------------------------
# BINARY CONVERSION
# ----------------------------
def to_foreground(img):
    img = img.astype(np.float32)

    if img.max() > 1:
        img /= 255.0

    return (img < 0.5).astype(np.uint8)

# ----------------------------
# GET BOUNDING BOX
# ----------------------------
def get_bbox(img):
    ys, xs = np.where(img == 1)

    if len(ys) == 0:
        return None

    return ys.min(), ys.max(), xs.min(), xs.max()

# ----------------------------
# CROP TO CONTENT
# ----------------------------
def crop(img):
    bbox = get_bbox(img)
    if bbox is None:
        return img

    y1, y2, x1, x2 = bbox
    return img[y1:y2+1, x1:x2+1]

# ----------------------------
# RESIZE WITH ASPECT RATIO
# ----------------------------
def resize_keep_aspect(img, target_size):
    h, w = img.shape

    scale = target_size / max(h, w)

    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    resized = resize(
        img,
        (new_h, new_w),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    )

    return (resized > 0.5).astype(np.uint8)

# ----------------------------
# PAD TO SQUARE CANVAS
# ----------------------------
def pad_to_square(img, size):
    h, w = img.shape

    canvas = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img

    return canvas

# ----------------------------
# SKELETON + DILATION (POST-NORMALIZATION)
# ----------------------------
def skeletonize_and_dilate(img, base_size):
    skel = skeletonize(img.astype(bool))

    # scale-aware dilation
    radius = max(1, int(base_size * 0.01))  # ~0.6% of image size

    selem = disk(radius)
    return dilation(skel, selem).astype(np.uint8)

# ----------------------------
# FULL PIPELINE
# ----------------------------
def process(path, out_path):
    img = imread(path, as_gray=True)
    img = to_foreground(img)

    # 1. geometric normalization first
    img = crop(img)
    img = resize_keep_aspect(img, OUT_SIZE)
    img = pad_to_square(img, OUT_SIZE)

    # 2. THEN structural smoothing (on aligned images)
    img = skeletonize_and_dilate(img, OUT_SIZE)

    # convert to clean black/white image
    out = np.ones_like(img) * 255
    out[img == 1] = 0

    imsave(out_path, out)

# ----------------------------
# MAIN LOOP
# ----------------------------
root = "./"

for r, _, files in os.walk(root):
    for f in files:

        if f.endswith(".png") and not f.endswith("_cropped.png"):

            inp = os.path.join(r, f)
            out = os.path.join(r, f.replace(".png", "_cropped.png"))

            process(inp, out)
            print("saved:", out)