import os
import numpy as np

from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_ubyte
from skimage.morphology import (
    disk,
    reconstruction,
    remove_small_objects,
    remove_small_holes,
)
from skimage.filters import gaussian, threshold_sauvola
from skimage.transform import resize


# =========================
# UTIL
# =========================

def downscale(image, max_min_side=1000):
    """Resize so smallest side is at most max_min_side."""
    h, w = image.shape
    min_side = min(h, w)

    if min_side <= max_min_side:
        return image

    scale = max_min_side / min_side
    new_h = int(h * scale)
    new_w = int(w * scale)

    return resize(image, (new_h, new_w), preserve_range=True, anti_aliasing=True)


# =========================
# BINARIZATION
# =========================

def binarize_whiteboard(
    image_path,
    dbg_dir=None,
    out_dir=None,
    name="debug",
    debug=False,
    *,
    smooth_sigma=0.3,
    window_frac=0.15,
    sauvola_k=0.13,
    seed_band=0.0,
    relaxed_offset=0.04,
    min_size_frac=0.00025,
    hole_size_frac=0.00025,
):
    def save_dbg(img, suffix):
        if dbg_dir is None:
            return
        path = os.path.join(dbg_dir, f"{name}_{suffix}.png")
        imsave(path, img_as_ubyte(img))

    # --- Load ---
    image = img_as_float(imread(image_path, as_gray=True))
    if debug:
        save_dbg(image, "1_input")

    h, w = image.shape
    total_pixels = h * w

    min_size = int(min_size_frac * total_pixels)
    hole_size = int(hole_size_frac * total_pixels)

    # --- Smooth ---
    image_smooth = gaussian(image, sigma=smooth_sigma) if smooth_sigma > 0 else image
    if debug:
        save_dbg(image_smooth, "2_smooth")

    # --- Sauvola threshold (FULL RESOLUTION) ---
    window_size = int(min(h, w) * window_frac)
    if window_size % 2 == 0:
        window_size += 1

    thresh = threshold_sauvola(image_smooth, window_size=window_size, k=sauvola_k)

    # --- Seed mask ---
    base = image_smooth < (thresh + seed_band)

    weak = image_smooth < (thresh + relaxed_offset)

    mask = reconstruction(
        base.astype(np.uint8),
        weak.astype(np.uint8),
        method="dilation",
    ) > 0

    if debug:
        save_dbg(mask.astype(float), "3_raw_mask")

    # --- Cleanup ---
    mask = remove_small_objects(mask, min_size=min_size)
    mask = remove_small_holes(mask, area_threshold=hole_size)

    # --- DOWNscale LAST ---
    mask = downscale(mask.astype(float), max_min_side=1000)

    if debug:
        save_dbg(mask.astype(float), "4_final")

    
    if out_dir is not None:
        path = os.path.join(out_dir, f"{name}_binarized.png")
        imsave(path, img_as_ubyte(1 - mask.astype(float)))

    return mask