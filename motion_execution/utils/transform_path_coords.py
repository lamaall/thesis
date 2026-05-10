import math

def traversal_to_xyz_rescaled(path, strokes,
                              x_range=(0.3, 0.475),
                              y_range=(-0.2, 0.075),
                              z_solid=0.07,
                              z_dotted=0.11,
                              step=0.005):   # <-- new parameter

    norm_path = [(float(p[0]), float(p[1])) for p in path]

    xs = [p[1] for p in norm_path]
    ys = [p[0] for p in norm_path]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x
    height = max_y - min_y

    tx_min, tx_max = x_range
    ty_min, ty_max = y_range

    target_w = tx_max - tx_min
    target_h = ty_max - ty_min

    scale = min(target_w / width if width != 0 else 1,
                target_h / height if height != 0 else 1)

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    tcx = (tx_min + tx_max) / 2
    tcy = (ty_min + ty_max) / 2

    stroke_edges = set()
    for s in strokes:
        pts = s["path"]
        for i in range(len(pts) - 1):
            a = (float(pts[i][0]), float(pts[i][1]))
            b = (float(pts[i+1][0]), float(pts[i+1][1]))
            stroke_edges.add(tuple(sorted([a, b])))

    def transform(p):
        x = p[1] - cx
        y = p[0] - cy

        x *= scale
        y *= scale

        x = -x
        y = -y

        x += tcx
        y += tcy

        return x, y

    def interpolate_points(p1, p2):
        """Split segment into smaller steps of max length `step`."""
        x1, y1 = transform(p1)
        x2, y2 = transform(p2)

        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)

        if dist == 0:
            return [(x1, y1)]

        n_steps = max(1, int(math.ceil(dist / step)))

        points = []
        for i in range(1, n_steps + 1):
            t = i / n_steps
            xi = x1 + t * dx
            yi = y1 + t * dy
            points.append((xi, yi))

        return points

    xyz = []

    # start
    x0, y0 = transform(norm_path[0])
    xyz.append((x0, y0, z_dotted))
    xyz.append((x0, y0, z_solid))

    for i in range(1, len(norm_path)):
        prev = norm_path[i - 1]
        curr = norm_path[i]

        edge = tuple(sorted([prev, curr]))
        #segment_points = interpolate_points(prev, curr)

        if edge not in stroke_edges:
            # pen up → NO interpolation (just jump)
            x_last, y_last, _ = xyz[-1]
            x, y = transform(curr)

            xyz.append((x_last, y_last, z_dotted))
            xyz.append((x, y, z_dotted))

        else:
            # pen down → interpolate
            segment_points = interpolate_points(prev, curr)

            for (x, y) in segment_points:
                xyz.append((x, y, z_solid))
    
    x_last, y_last, _ = xyz[-1]
    xyz.append((x_last, y_last, z_dotted))

    return xyz