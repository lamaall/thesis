import numpy as np
from skimage.morphology import skeletonize
import cv2
import os

def skeletonize_binary_image(binary_clean, name, dbg_dir, debug=False):
    skeleton = skeletonize(binary_clean)
    if debug:
        cv2.imwrite(os.path.join(dbg_dir, f"{name}_14_skeleton.png"), 255 - skeleton.astype(np.uint8) * 255)
    return skeleton.astype(np.uint8)