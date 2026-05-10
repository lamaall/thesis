import numpy as np
import networkx as nx


# ----------------------------
# Skeleton → Graph
# ----------------------------

def skeleton_to_graph(skeleton):
    G = nx.Graph()
    h, w = skeleton.shape

    for y in range(h):
        for x in range(w):
            if skeleton[y, x]:
                G.add_node((y, x))

                # 4-connected
                for dy, dx in [(0,1),(1,0),(0,-1),(-1,0)]:
                    ny, nx_ = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx_ < w and skeleton[ny, nx_]:
                        G.add_edge((y,x),(ny,nx_))

                # diagonals (conditional)
                for dy, dx in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    ny, nx_ = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx_ < w and skeleton[ny, nx_]:
                        side1 = (y, nx_)
                        side2 = (ny, x)
                        if (not skeleton[side1]) and (not skeleton[side2]):
                            G.add_edge((y,x),(ny,nx_))

    return G


# ----------------------------
# Branch / Cycle extraction
# ----------------------------

def get_skeleton_branches_and_cycles(G):
    branches = []
    cycles = []
    visited_edges = set()
    visited_nodes = set()

    end_nodes = sorted(
        [n for n in G.nodes if G.degree[n] != 2],
        key=lambda x: G.degree[x],
        reverse=True
    )

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
                nbs = list(G.neighbors(current))
                nxt = nbs[0] if nbs[0] != prev else nbs[1]

                edge = tuple(sorted([current, nxt]))
                if edge in visited_edges:
                    break

                path.append(nxt)
                visited_edges.add(edge)
                prev, current = current, nxt

            branches.append(path)
            visited_nodes.update(path)

    # cycles
    unvisited = set(n for n in G.nodes if G.degree[n] == 2 and n not in visited_nodes)

    while unvisited:
        start = unvisited.pop()
        cycle = [start]
        current = start
        prev = None

        while True:
            nbs = [n for n in G.neighbors(current) if n != prev]
            if not nbs:
                break
            nxt = nbs[0]
            if nxt == start:
                cycle.append(start)
                break
            cycle.append(nxt)
            unvisited.discard(nxt)
            prev, current = current, nxt

        if len(cycle) > 2 and cycle[0] == cycle[-1]:
            cycles.append(cycle)

    return branches, cycles


# ----------------------------
# RDP Simplification
# ----------------------------

def point_line_distance(p, a, b):
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)

    if np.allclose(a, b):
        return np.linalg.norm(p - a)

    return np.abs(np.cross(b - a, a - p)) / np.linalg.norm(b - a)


def rdp(points, epsilon):
    if len(points) < 3:
        return points

    start = points[0]
    end = points[-1]

    max_dist = -1
    idx = -1

    for i in range(1, len(points)-1):
        d = point_line_distance(points[i], start, end)
        if d > max_dist:
            max_dist = d
            idx = i

    if max_dist > epsilon:
        left = rdp(points[:idx+1], epsilon)
        right = rdp(points[idx:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]


def rdp_cycle(branch, epsilon):
    if branch[0] == branch[-1]:
        branch = branch[:-1]

    simplified = rdp(branch, epsilon)

    if simplified[0] != simplified[-1]:
        simplified.append(simplified[0])

    return simplified


# ----------------------------
# Convert to line graph
# ----------------------------

def simplified_to_graph(simplified_paths):
    G = nx.Graph()

    for path in simplified_paths:
        for i in range(len(path)-1):
            p1 = path[i]
            p2 = path[i+1]

            # flip to (x, y) for consistency with your SVG logic
            p1 = (p1[1], p1[0])
            p2 = (p2[1], p2[0])

            G.add_edge(p1, p2)

    return G


# ----------------------------
# SVG export (optional but preserved)
# ----------------------------

def save_svg_from_graph(G, shape, path):
    h, w = shape

    lines = []
    for (x1, y1), (x2, y2) in G.edges:
        lines.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="black" stroke-width="1.5"/>'
        )

    svg = f"""<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
{chr(10).join(lines)}
</svg>
"""
    with open(path, "w") as f:
        f.write(svg)


# ----------------------------
# MAIN VECTORIZATION (RDP)
# ----------------------------

def vectorize_skeleton_rdp(skeleton, name, dbg_dir, debug=False, epsilon=1.5):
    G = skeleton_to_graph(skeleton)

    branches, cycles = get_skeleton_branches_and_cycles(G)

    simplified_paths = []

    # simplify branches
    for b in branches:
        simplified_paths.append(rdp(b, epsilon))

    # simplify cycles
    for c in cycles:
        simplified_paths.append(rdp_cycle(c, epsilon))

    # build final graph
    lineG = simplified_to_graph(simplified_paths)

    # optional debug SVG
    if debug:
        save_svg_from_graph(
            lineG,
            skeleton.shape,
            dbg_dir + f"/{name}_rdp.svg"
        )

    return lineG