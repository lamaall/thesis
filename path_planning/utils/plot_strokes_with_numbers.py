import matplotlib.pyplot as plt
import numpy as np
import os


def get_pos(node, G):
    if isinstance(node, tuple) and len(node) == 2:
        return node
    return G.nodes[node].get("pos", None)


# ----------------------------
# Stroke geometry helpers
# ----------------------------

def stroke_edges(stroke):
    """Return set of undirected edges in the stroke."""
    path = stroke["path"]
    edges = set()

    for i in range(len(path) - 1):
        a = tuple(path[i])
        b = tuple(path[i + 1])
        edges.add((a, b))
        edges.add((b, a))  # allow reverse direction

    return edges


def stroke_label_position(stroke, G=None):
    """
    Return label position:
    - if 2 points → geometric midpoint
    - else → middle vertex of the path
    """
    path = stroke["path"]
    n = len(path)

    if n == 0:
        return None

    # ----------------------------
    # CASE 1: exactly 2 points → centroid
    # ----------------------------
    if n == 2:
        def getp(node):
            return get_pos(node, G) if G is not None else node

        p1 = getp(path[0])
        p2 = getp(path[1])

        if p1 is None or p2 is None:
            return None

        x = (p1[0] + p2[0]) / 2
        y = (p1[1] + p2[1]) / 2

        return [x, -y]

    # ----------------------------
    # CASE 2: longer stroke → middle node
    # ----------------------------
    mid_idx = n // 2
    node = path[mid_idx]

    p = get_pos(node, G) if G is not None else node
    if p is None:
        return None

    return [p[0], -p[1]]

# ----------------------------
# CORRECT ORDER DETECTION
# ----------------------------

def stroke_order_from_path(strokes, P):
    """
    Assign order based on first actual traversal of a stroke edge.
    """

    # Precompute stroke edges
    stroke_edge_map = {
        s["id"]: stroke_edges(s)
        for s in strokes
    }

    order = {}
    current_id = 0

    # Walk along path edges
    for i in range(len(P) - 1):
        a = tuple(P[i])
        b = tuple(P[i + 1])

        for sid, edges in stroke_edge_map.items():
            if sid in order:
                continue

            if (a, b) in edges:
                order[sid] = current_id
                current_id += 1
                break  # only one stroke per segment

    return order


# ----------------------------
# MAIN PLOT
# ----------------------------

def plot_strokes_with_numbers(strokes, P, G=None,
                              title="Strokes Debug View",
                              dbg_folder=None, name=None):

    save_path = None
    if dbg_folder is not None and name is not None:
        save_path = os.path.join(dbg_folder, f"{name}_constructed_strokes.png")

    plt.figure(figsize=(8, 8))
    colors = plt.cm.get_cmap("tab20", len(strokes))

    # ----------------------------
    # DRAW STROKES
    # ----------------------------
    for i, stroke in enumerate(strokes):
        path = stroke["path"]

        coords = []
        for n in path:
            p = get_pos(n, G) if G is not None else n
            if p is not None:
                coords.append([p[0], -p[1]])

        coords = np.array(coords)

        if len(coords) < 2:
            continue

        plt.plot(coords[:, 0], coords[:, 1],
                 color=colors(i), linewidth=2)

    # ----------------------------
    # LABEL STROKES (FIXED)
    # ----------------------------
    order_map = stroke_order_from_path(strokes, P)

    for stroke in strokes:
        sid = stroke["id"]

        if sid not in order_map:
            continue

        c = stroke_label_position(stroke, G)
        if c is None:
            continue

        plt.text(
            c[0], c[1],
            str(order_map[sid]),
            fontsize=14,
            color="black",
            ha="center",
            va="center",
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )

    #plt.title(title)
    plt.axis("equal")
    plt.axis("off")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()