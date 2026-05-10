import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rotate
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

# ----------------------------
# LOG FILE SETUP
# ----------------------------
log_path = "results_log.txt"

with open(log_path, "w") as f:
    f.write("category, filename, alignment_score, chamfer, fp, fn\n")

# ----------------------------
# LOAD BINARY
# ----------------------------
def load_bw(path):
    img = imread(path, as_gray=True)

    if img.max() > 1:
        img = img / 255.0

    return (img < 0.5).astype(np.uint8)

# ----------------------------
# PAD IMAGE
# ----------------------------
def pad(img, pad):
    h, w = img.shape
    out = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
    out[pad:pad + h, pad:pad + w] = img
    return out

# ----------------------------
# FAST SHIFT
# ----------------------------
def shift(img, dy, dx):
    out = np.zeros_like(img)

    y1 = max(0, dy)
    y2 = min(img.shape[0], img.shape[0] + dy)

    x1 = max(0, dx)
    x2 = min(img.shape[1], img.shape[1] + dx)

    out[y1:y2, x1:x2] = img[y1 - dy:y2 - dy, x1 - dx:x2 - dx]

    return out

# ----------------------------
# ROTATE IMAGE
# ----------------------------
def rot(img, angle):
    return (rotate(img, angle, resize=False, preserve_range=True) > 0.5).astype(bool)

# ----------------------------
# BEST ALIGNMENT
# ----------------------------
def best_align(img1, img2, search=10, angle_range=10):
    best_score = -1
    best_img = img2

    a = img1.astype(bool)

    for angle in range(-angle_range, angle_range + 1):
        rotated = rot(img2, angle)

        for dy in range(-search, search + 1):
            for dx in range(-search, search + 1):

                shifted = shift(rotated, dy, dx)
                b = shifted

                inter = np.sum(a & b)
                union = np.sum(a) + np.sum(b) - inter

                score = inter / (union + 1e-8)

                if score > best_score:
                    best_score = score
                    best_img = shifted

    return best_img.astype(np.uint8), best_score

# ----------------------------
# SKELETON + CHAMFER
# ----------------------------
def chamfer_distance(a, b):
    a = skeletonize(a.astype(bool))
    b = skeletonize(b.astype(bool))

    dt_a = distance_transform_edt(~a)
    dt_b = distance_transform_edt(~b)

    # mean distance from A→B and B→A normalized by points
    d1 = dt_b[a].mean() if np.any(a) else 0
    d2 = dt_a[b].mean() if np.any(b) else 0

    raw = (d1 + d2) / 2

    diag = np.sqrt(a.shape[0]**2 + a.shape[1]**2)

    return raw / (diag + 1e-8)

# ----------------------------
# STATS
# ----------------------------
def compute_stats(img1, img2):
    a = img1.astype(bool)
    b = img2.astype(bool)

    chamfer = chamfer_distance(a, b)

    fp = np.sum(~a & b)
    fn = np.sum(a & ~b)

    return chamfer, fp, fn

# ----------------------------
# OVERLAY
# ----------------------------
def overlay(img1, img2):
    h, w = img1.shape
    out = np.zeros((h, w, 3), dtype=np.float32)

    out[..., 0] = img1
    out[..., 1] = img2

    both = (img1 == 1) & (img2 == 1)
    out[both] = [1, 1, 0]

    return out

# ----------------------------
# FIND PAIR
# ----------------------------
def find_pair(root, base):
    matches = []

    for r, _, files in os.walk(root):
        for f in files:
            if f.startswith(base) and "_cropped" in f:
                matches.append(os.path.join(r, f))

    return matches[:2]

# ----------------------------
# MAIN
# ----------------------------
root = "./"
seen = set()

for r, _, files in os.walk(root):
    for f in files:

        if f.endswith(".ps"):

            base = f.replace(".ps", "")

            if base in seen:
                continue
            seen.add(base)

            pair = find_pair(root, base)

            if len(pair) < 2:
                continue

            img1 = load_bw(pair[0])
            img2 = load_bw(pair[1])

            if img1.shape != img2.shape:
                print("skip (size mismatch):", base)
                continue

            # ----------------------------
            # CATEGORY = folder name
            # ----------------------------
            category = os.path.basename(os.path.dirname(pair[0]))

            # ----------------------------
            # PAD
            # ----------------------------
            pad_size = 20
            img1 = pad(img1, pad_size)
            img2 = pad(img2, pad_size)

            # ----------------------------
            # ALIGN
            # ----------------------------
            img2_aligned, best_score = best_align(
                img1,
                img2,
                search=10,
                angle_range=10
            )

            # ----------------------------
            # METRICS
            # ----------------------------
            chamfer, fp, fn = compute_stats(img1, img2_aligned)

            # ----------------------------
            # OVERLAY SAVE
            # ----------------------------
            diff = overlay(img1, img2_aligned)

            save_path = pair[0].replace("_cropped.png", "_overlay.png")
            imsave(save_path, (diff * 255).astype(np.uint8))

            # ----------------------------
            # PRINT
            # ----------------------------
            print(f"\n{base} ({category})")
            print(f"Alignment score: {best_score:.4f}")
            print(f"Chamfer distance: {chamfer:.4f}")
            print(f"FP: {fp} | FN: {fn}")
            print("saved:", save_path)

            # ----------------------------
            # LOG WITH CATEGORY
            # ----------------------------
            with open(log_path, "a") as f:
                f.write(f"{category}, {base}, {best_score:.6f}, {chamfer:.6f}, {fp}, {fn}\n")