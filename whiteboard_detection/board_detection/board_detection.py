import cv2
import numpy as np
import itertools
import os

def undistort_fisheye(img):
    h, w = img.shape[:2]

    # Approximate fisheye camera parameters (tune if needed)
    K = np.array([
        [w/2, 0, w/2],
        [0, w/2, h/2],
        [0, 0, 1]
    ], dtype=np.float32)

    D = np.array([-0.35, 0.15, 0, 0], dtype=np.float32)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w,h), cv2.CV_16SC2
    )

    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

def order_points(pts):
    pts = np.array(pts, dtype="float32")

    # --- 1. Find top edge from all combinations ---
    best_pair = None
    best_y = float("inf")

    for p1, p2 in itertools.combinations(pts, 2):
        avg_y = (p1[1] + p2[1]) / 2.0
        if avg_y < best_y:
            best_y = avg_y
            best_pair = (p1, p2)

    pA, pB = best_pair

    # --- 2. Determine TL and TR from that edge ---
    if pA[0] < pB[0]:
        tl, tr = pA, pB
    else:
        tl, tr = pB, pA

    # --- 3. Get remaining two points ---
    remaining = []
    for p in pts:
        if not (np.allclose(p, tl) or np.allclose(p, tr)):
            remaining.append(p)

    pC, pD = remaining

    # --- 4. Determine BL and BR ---
    # Compare by y (bottom points have larger y)
    if pC[1] > pD[1]:
        bl, br_candidate = pC, pD
    else:
        bl, br_candidate = pD, pC

    # Now fix left/right
    if bl[0] > br_candidate[0]:
        bl, br = br_candidate, bl
    else:
        br = br_candidate

    return np.array([tl, tr, br, bl], dtype="float32")

def warp_board_from_4points(img, pts):
    pts = np.array(pts, dtype="float32")

    if pts.shape[0] != 4:
        raise ValueError("Exactly 4 points are required")

    hull = cv2.convexHull(pts)
    hull = hull.reshape(-1, 2)

    if len(hull) != 4:
        print("found", len(hull), "dots — skipping")
        return None

    rect = order_points(hull)
    tl, tr, br, bl = rect

    # Compute size
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)

    maxW = int(max(wA, wB))
    maxH = int(max(hA, hB))

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH))

    return warped


def detect_whiteboard(path, dbg_dir, name, debug=False):
    img = cv2.imread(path)
    if img is None:
        print("Failed to load:", path)
        return

    if debug:
        cv2.imwrite(os.path.join(dbg_dir, f"{name}_1_original.png"), img)

    # 1 Fisheye correction
    undist = undistort_fisheye(img)

    if debug:
        cv2.imwrite(os.path.join(dbg_dir, f"{name}_2_undistorted.png"), undist)

    # 2 Green mask
    hsv = cv2.cvtColor(undist, cv2.COLOR_BGR2HSV)
    lower_green = np.array([45, 30, 70])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    if debug:
        cv2.imwrite(os.path.join(dbg_dir, f"{name}_3_greenmask.png"), mask)

    # 3 Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    centers = []
    debug_tmp = undist.copy()
    
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        centers.append((cx, cy))

        cv2.drawContours(debug_tmp, [c], -1, (0, 255, 0), 2)
        cv2.circle(debug_tmp, (cx, cy), 6, (255, 0, 0), -1)

    centers = centers[:4]

    if debug:
        cv2.imwrite(os.path.join(dbg_dir, f"{name}_4_detected_dots.png"), debug_tmp)

    if len(centers) != 4:
        print(path, "found", len(centers), "dots — skipping")
        return

    pts = np.array(centers, dtype="float32")
    
    # 4 Warp
    warped = warp_board_from_4points(undist, pts)

    if warped is None:
        return

    # 5 Remove green pixels from warped image
    warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    green_mask_warped = cv2.inRange(
        warped_hsv,
        np.array([45, 30, 70]),
        np.array([85, 255, 255])
    )

    kernel = np.ones((9, 9), np.uint8)
    expanded_mask = cv2.dilate(green_mask_warped, kernel, iterations=1)

    # Inpaint using expanded mask
    warped_no_green = cv2.inpaint(
        warped,
        expanded_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )

    cv2.imwrite(os.path.join(dbg_dir, f"{name}_5_warped.png"), warped_no_green)

    print("SUCCESS:", path)
    return os.path.join(dbg_dir, f"{name}_5_warped.png")