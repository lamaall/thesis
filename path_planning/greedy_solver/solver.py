import networkx as nx
from math import dist as euclidean

def build_fully_connected_graph(G):
    """
    Take a (possibly MultiGraph) G with required edges (nodes as 2D points)
    and return a fully connected MultiGraph with Euclidean weights.
    Marks which edges are required.
    """
    nodes = list(G.nodes())
    FC = nx.MultiGraph()

    def get_pos(node):
        """Get coordinates for Euclidean distance computation."""
        if isinstance(node, tuple) and len(node) == 2:
            return node
        return G.nodes[node].get("pos")

    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            FC.add_edge(u, v, weight=euclidean(get_pos(u), get_pos(v)), required=False)

    for u, v, data in G.edges(data=True):
        w = data.get("weight", euclidean(get_pos(u), get_pos(v)))
        FC.add_edge(u, v, weight=w, required=True)

    return FC

def open_rural_postman_fully_connected(G):
    """
    Greedy Open Rural Postman solver that:
      - Tries all possible start nodes
      - Always picks the closest required edge from the current node
      - Works with MultiGraphs (keeps distinct parallel edges)
    """
    FC = build_fully_connected_graph(G)

    # Include edge keys to preserve multiple required edges
    required_edges = [
        (u, v, k) for u, v, k, d in FC.edges(keys=True, data=True) if d['required']
    ]

    if not required_edges:
        return [], 0.0

    best_path = None
    best_cost = float('inf')

    def edge_length(u, v):
        return euclidean(u, v)

    for start_node in FC.nodes():
        # Keep unvisited edges as a *list* (not a set), since parallel edges are distinct
        unvisited = required_edges.copy()
        current = start_node
        path = [current]
        total_cost = 0.0

        while unvisited:
            best_edge = None
            best_dist = float('inf')
            best_entry = None

            for u, v, k in unvisited:
                d_u = euclidean(current, u)
                d_v = euclidean(current, v)
                edge_len = edge_length(u, v)

                for d, entry in [(d_u, u), (d_v, v)]:
                    if (
                        d < best_dist
                        or (d == best_dist and edge_len < edge_length(*best_edge[:2]))
                        or (d == best_dist and edge_len == edge_length(*best_edge[:2]) and (u, v) < best_edge[:2])
                    ):
                        best_dist = d
                        best_edge = (u, v, k)
                        best_entry = entry

            # Move to the chosen edge
            if best_entry != path[-1]:
                total_cost += best_dist
                current = best_entry
                path.append(current)

            # Traverse the edge
            u, v, _ = best_edge
            target = v if current == u else u
            total_cost += euclidean(current, target)
            path.append(target)
            current = target

            # Remove this exact edge (not just its endpoints)
            unvisited.remove(best_edge)

        if total_cost < best_cost:
            best_cost = total_cost
            best_path = path

    return best_path, best_cost