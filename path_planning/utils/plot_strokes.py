import matplotlib.pyplot as plt
import numpy as np
import os


def get_pos(node, G):
    """Safely get node position."""
    if isinstance(node, tuple) and len(node) == 2:
        return node
    return G.nodes[node].get("pos", None)


def plot_strokes(strokes, G=None, title="Strokes Debug View", dbg_folder=None, name=None):
    """
    Visualize each stroke in a different color and optionally save to disk.
    """

    save_path = os.path.join(dbg_folder, f"{name}_constructed_strokes.png")

    plt.figure(figsize=(8, 8))

    colors = plt.cm.get_cmap("tab20", len(strokes))

    for i, stroke in enumerate(strokes):
        path = stroke["path"]

        coords = []
        for n in path:
            if G is not None:
                p = get_pos(n, G)
            else:
                p = n  # assume already (x, y)

            if p is not None:
                coords.append([p[0], -p[1]])  # invert y for plotting

        coords = np.array(coords)

        if len(coords) < 2:
            continue

        plt.plot(
            coords[:, 0],
            coords[:, 1],
            color=colors(i),
            linewidth=2
        )

        plt.scatter(coords[0, 0], coords[0, 1], color=colors(i), s=20)
        plt.scatter(coords[-1, 0], coords[-1, 1], color=colors(i), s=20)

    plt.title(title)
    plt.axis("equal")
    plt.axis("off")

    # ----------------------------
    # SAVE DEBUG IMAGE
    # ----------------------------
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()