import networkx as nx

def skeleton_to_graph(skeleton):
    """
    Converts a skeletonized binary image into an undirected graph.
    Each white pixel (1) becomes a node, and edges connect 8-connected neighbors.

    Parameters:
        skeleton (np.ndarray): Binary skeleton image (0s and 1s).

    Returns:
        networkx.Graph: Graph representing the skeleton structure.
    """
    G = nx.Graph()
    h, w = skeleton.shape  # Image dimensions

    for y in range(h):
        for x in range(w):
            if skeleton[y, x]:
                G.add_node((y, x))

                # 4-connected neighbors
                for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx_ < w and skeleton[ny, nx_]:
                        G.add_edge((y, x), (ny, nx_))

                # Diagonal neighbors (conditionally)
                for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx_ < w and skeleton[ny, nx_]:
                        # Check if both side neighbors are absent
                        side1 = (y, nx_)  # horizontal neighbor
                        side2 = (ny, x)   # vertical neighbor
                        if (not skeleton[side1]) and (not skeleton[side2]):
                            G.add_edge((y, x), (ny, nx_))

    return G



def get_skeleton_branches_and_cycles(G):
    """
    Extract all skeletal branches and cycles from a skeleton graph.

    Returns:
        branches: list of open branches (list of (y,x) coords)
        cycles: list of cycles (list of (y,x) coords)
    """
    branches = []
    cycles = []
    visited_edges = set()
    visited_nodes = set()

    # Identify nodes with degree != 2 (endpoints and junctions)
    end_nodes = sorted(
        [n for n in G.nodes if G.degree[n] != 2],
        key=lambda x: G.degree[x],
        reverse=True
    )

    # Traverse from each end node
    for node in end_nodes:
        for neighbor in G.neighbors(node):
            edge = tuple(sorted([node, neighbor]))
            if edge in visited_edges:
                continue

            path = [node, neighbor]
            visited_edges.add(edge)

            current = neighbor
            prev = node

            while G.degree[current] == 2:
                neighbors = list(G.neighbors(current))
                next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]

                edge = tuple(sorted([current, next_node]))
                if edge in visited_edges:
                    break  # Prevents infinite loops in malformed graphs

                path.append(next_node)
                visited_edges.add(edge)
                prev, current = current, next_node

            #if len(path) > 10 or (G.degree[path[0]] != 1 and G.degree[path[-1]] != 1):
            branches.append(path)
            visited_nodes.update(path)

    # Remaining unvisited degree-2 nodes might form cycles
    unvisited = set(n for n in G.nodes if G.degree[n] == 2 and n not in visited_nodes)

    while unvisited:
        start = unvisited.pop()
        cycle = [start]
        current = start
        prev = None

        while True:
            neighbors = [n for n in G.neighbors(current) if n != prev]
            if not neighbors:
                break  # Not a cycle
            next_node = neighbors[0]
            if next_node == start:
                cycle.append(start)
                break  # Closed loop
            cycle.append(next_node)
            unvisited.discard(next_node)
            prev, current = current, next_node

        if len(cycle) > 2 and cycle[0] == cycle[-1]:
            cycles.append(cycle)

    return branches, cycles


import numpy as np
import random


def loss(distances):
    return np.mean(distances)/len(distances)

def is_line_segment(branch, min_ratio=0.5, close_threshold=1.5, max_allowed_distance=3.0):
    """
    Determine if a list of connected pixels forms a line segment.

    Args:
        branch: List of (x, y) pixel coordinates.
        min_ratio: Minimum ratio of pixels that must lie close to the line.
        close_threshold: Distance threshold to count a pixel as "close" to the line.
        max_allowed_distance: Max distance any pixel can be from the line.

    Returns:
        True if branch resembles a line segment.
    """
    if len(branch) < 2:
        return False

    points = np.array(branch)
    centroid = points.mean(axis=0)

    # Center the points
    centered = points - centroid

    # PCA: first principal component
    _, _, vh = np.linalg.svd(centered)
    direction = vh[0]

    # Project onto the line
    projections = centered @ direction
    closest_points = np.outer(projections, direction)
    distances = np.linalg.norm(centered - closest_points, axis=1)

    # Check conditions
    num_close = np.sum(distances <= close_threshold)
    ratio = num_close / len(branch)
    all_within_max = np.all(distances <= max_allowed_distance)

    return (ratio >= min_ratio and all_within_max, loss(distances))


def find_largest_line_segment(branch, i, j, **kwargs):
    """
    Expand around branch[i:j] to find largest line segment containing it.

    Returns:
        (i0, j0): Indices such that branch[i0:j0] is the largest valid line segment.
    """
    n = len(branch)

    # Binary search left
    lo, hi = 0, i
    best_i = i
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_line_segment(branch[mid:j], **kwargs)[0]:
            best_i = mid
            hi = mid - 1  # try to expand more left
        else:
            lo = mid + 1  # can't go that far left

    # Binary search right
    lo, hi = j, n
    best_j = j
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_line_segment(branch[best_i:mid], **kwargs)[0]:
            best_j = mid
            lo = mid + 1  # try to expand more right
        else:
            hi = mid - 1  # can't go that far right

    return best_i, best_j




def extract_line_segments(branch, num_attempts=10, min_length=7, **kwargs):
    """
    Recursively find all line segments within a branch that are at least `min_length` pixels long.

    Args:
        branch: List of (y, x) pixel coordinates.
        num_attempts: Number of random (i, j) tries per recursion level.
        min_length: Minimum number of pixels in a valid segment.
        **kwargs: Passed to is_line_segment.

    Returns:
        List of (i0, j0) index pairs where branch[i0:j0] is a valid line segment.
    """
    if len(branch) < min_length:
        return []

    segments = []

    best_i, best_j, best_lo = 0, 0, None

    for _ in range(num_attempts):
        i = random.randint(0, len(branch) - min_length)
        j = random.randint(i + min_length, len(branch))

        is_line, _ = is_line_segment(branch[i:j], **kwargs)

        if is_line:
            i0, j0 = find_largest_line_segment(branch, i, j, **kwargs)
            _, lo = is_line_segment(branch[i0:j0], **kwargs)

            if best_lo is None or lo < best_lo:
                best_i, best_j = i0, j0
                best_lo = lo

    if best_j - best_i >= min_length:
        segments.append((best_i, best_j))

        # Recurse
        left = extract_line_segments(branch[:best_i], num_attempts, min_length, **kwargs)
        right = extract_line_segments(branch[best_j:], num_attempts, min_length, **kwargs)
        right = [(i_ + best_j, j_ + best_j) for (i_, j_) in right]


        return left + [(best_i, best_j)] + right

    return []


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def fit_line_endpoints(segment_pixels):
    """
    Fit a line to the segment and return start/end points on that line.
    
    Returns:
        (x1, y1), (x2, y2) — endpoints of the line covering the segment
    """
    points = np.array(segment_pixels)
    mean = points.mean(axis=0)
    centered = points - mean
    u, s, vh = np.linalg.svd(centered)
    direction = vh[0]

    # Project points onto the direction vector
    projections = centered @ direction
    min_proj = projections.min()
    max_proj = projections.max()

    start = mean + min_proj * direction
    end = mean + max_proj * direction

    return (start[1], start[0]), (end[1], end[0])


def save_segments_as_svg(all_segments, image_shape, output_path="output.svg", stroke_width=1.5):
    """
    Save all fitted segments as straight lines into an SVG,
    and also return a NetworkX graph where each line is an edge.
    """
    h, w = image_shape
    svg_lines = []
    G = nx.Graph()  # graph to store line connections

    def add_line(p1, p2):
        """Add SVG line and graph edge."""
        x1, y1 = p1
        x2, y2 = p2
        svg_lines.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="black" stroke-width="{stroke_width}"/>'
        )
        G.add_edge((round(x1, 2), round(y1, 2)), (round(x2, 2), round(y2, 2)))

    for item in all_segments:
        branch = item["branch"]
        fitted_segments = []

        # --- Fit each segment ---
        for (i0, j0) in item["segments"]:
            segment_pixels = branch[i0:j0]
            if len(segment_pixels) < 2:
                continue

            (x1, y1), (x2, y2) = fit_line_endpoints(segment_pixels)

            # Optional: orient using branch order to prevent flipping
            first_px = segment_pixels[0][::-1]
            last_px  = segment_pixels[-1][::-1]
            if ((x2 - first_px[0])**2 + (y2 - first_px[1])**2) < ((x1 - first_px[0])**2 + (y1 - first_px[1])**2):
                (x1, y1), (x2, y2) = (x2, y2), (x1, y1)

            fitted_segments.append(((x1, y1), (x2, y2)))

        if not fitted_segments:
            continue

        # --- Add all the same SVG lines as before ---
        # 1. Each fitted segment
        for (p1, p2) in fitted_segments:
            add_line(p1, p2)

        # 2. Branch start → first fitted start
        branch_start = branch[0][::-1]
        first_start = fitted_segments[0][0]
        add_line(branch_start, first_start)

        # 3. Bridges between consecutive fitted segments
        for (p1, p2), (q1, q2) in zip(fitted_segments[:-1], fitted_segments[1:]):
            add_line(p2, q1)

        # 4. Last fitted end → branch end
        branch_end = branch[-1][::-1]
        last_end = fitted_segments[-1][1]
        add_line(last_end, branch_end)

    # --- Save SVG ---
    svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
{chr(10).join(svg_lines)}
</svg>
"""
    with open(output_path, "w") as f:
        f.write(svg_content)

    print(f"Saved SVG to: {output_path}")

    # --- Return the graph ---
    return G




def vectorize_skeleton(skeleton, name, dbg_dir):
    G = skeleton_to_graph(skeleton)
    branches, cycles = get_skeleton_branches_and_cycles(G)

    all_segments = []

    for branch in branches + cycles:
        segments = extract_line_segments(branch,
                                          num_attempts=20,
                                          min_length=2,
                                          close_threshold=1.5,
                                          max_allowed_distance=3.0,
                                          min_ratio=0.5)
        all_segments.append({
            "branch": branch,
            "segments": segments
        })

    lineG = save_segments_as_svg(all_segments, skeleton.shape, output_path=dbg_dir+f"/{name}_15_vectorized.svg")
    return lineG