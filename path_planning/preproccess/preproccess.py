import networkx as nx
from math import dist as euclidean

def preprocess_graph(G, ndigits=3):

    G = nx.MultiGraph(G)  # ensure multigraph ONCE

    def norm(p):
        return (round(float(p[0]), ndigits),
                round(float(p[1]), ndigits))

    H = nx.MultiGraph()

    # --- nodes ---
    for n, data in G.nodes(data=True):
        nn = norm(n) if isinstance(n, tuple) else n

        pos = data.get("pos", nn)
        if isinstance(pos, tuple):
            pos = norm(pos)

        H.add_node(nn, pos=pos)

    # --- edges (KEEP KEYS!) ---
    for u, v, k, data in G.edges(keys=True, data=True):

        if isinstance(u, tuple):
            u = norm(u)
        if isinstance(v, tuple):
            v = norm(v)

        if u == v:
            continue

        # preserve edge identity in MultiGraph
        H.add_edge(u, v, key=k, **data)

    return H


def remove_degree_2_nodes(G):

    # DO NOT convert again if already MultiGraph
    if not isinstance(G, nx.MultiGraph):
        G = nx.MultiGraph(G)

    def get_pos(node):
        return G.nodes[node].get("pos", node)

    # init edges safely
    for u, v, k, data in G.edges(keys=True, data=True):
        if "path" not in data:
            data["path"] = [u, v]
            data["weight"] = euclidean(get_pos(u), get_pos(v))

    deg2_nodes = [n for n in list(G.nodes) if G.degree(n) == 2]

    for n in deg2_nodes:
        if n not in G:
            continue

        neighbors = list(G.neighbors(n))
        if len(neighbors) != 2:
            continue

        u, v = neighbors

        # SAFER: pick actual edge dicts safely
        e1 = next(iter(G.get_edge_data(u, n).values()))
        e2 = next(iter(G.get_edge_data(n, v).values()))

        p1 = e1["path"]
        p2 = e2["path"]

        if p1[-1] != n:
            p1 = p1[::-1]
        if p2[0] != n:
            p2 = p2[::-1]

        path = p1[:-1] + p2
        weight = e1["weight"] + e2["weight"]

        G.add_edge(u, v, weight=weight, path=path)
        G.remove_node(n)

    return G
