import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(G, filename=None, show=True, figsize=(6, 6), flip_x=False, flip_y=True):
    """
    Draws a graph and optionally saves it as an image.
    Can flip X and/or Y axes.

    Parameters:
        G         : networkx graph
        filename  : if provided, saves image (e.g. "graph.png")
        show      : whether to display the plot
        figsize   : size of the figure
        flip_x    : mirror across Y-axis
        flip_y    : mirror across X-axis
    """

    def get_pos(G):
        pos = {}
        for n, data in G.nodes(data=True):
            if "pos" in data:
                x, y = data["pos"]
            elif isinstance(n, tuple):
                x, y = n
            else:
                x, y = (0, 0)

            if flip_x:
                x = -x
            if flip_y:
                y = -y

            pos[n] = (x, y)
        return pos

    pos = get_pos(G)

    plt.figure(figsize=figsize)

    nx.draw(
        G,
        pos,
        node_size=20,
        node_color="black",
        edge_color="gray",
        width=1,
        with_labels=False
    )

    plt.axis("equal")
    plt.axis("off")

    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    plt.close()