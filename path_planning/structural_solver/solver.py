import os
import math
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------
# Helper functions
# -----------------------------
def face_perimeter(face, pos):
    """Sum of 2D lengths of edges in a face"""
    return sum(math.hypot(pos[face[i]][0] - pos[face[(i + 1) % len(face)]][0],
                          pos[face[i]][1] - pos[face[(i + 1) % len(face)]][1])
               for i in range(len(face)))

def chain_distance(chain, pos):
    """Sum of 2D lengths of edges in a bridge chain"""
    return sum(math.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1]) for u, v in chain)

def face_edges(face):
    return [(face[i], face[(i + 1) % len(face)]) for i in range(len(face))]

def find_faces_degree2(G):
    """
    Strictly find cycles where each node in the cycle has exactly 2 edges from the cycle.
    """
    faces = []
    visited_edges = set()

    for u in G.nodes():
        for v in G.neighbors(u):
            edge = tuple(sorted((u, v)))
            if edge in visited_edges:
                continue

            # Start DFS to find cycles
            stack = [(u, v, [u, v])]
            while stack:
                prev, curr, path = stack.pop()
                for n in G.neighbors(curr):
                    if n == prev:
                        continue
                    if n == path[0] and len(path) >= 3:
                        # candidate cycle found
                        candidate = path[:]
                        # check degree-2 constraint in the cycle
                        counts = {node: 0 for node in candidate}
                        for i in range(len(candidate)):
                            a, b = candidate[i], candidate[(i + 1) % len(candidate)]
                            counts[a] += 1
                            counts[b] += 1
                        if all(c == 2 for c in counts.values()):
                            # mark edges as visited
                            for i in range(len(candidate)):
                                e = tuple(sorted((candidate[i], candidate[(i + 1) % len(candidate)])))
                                visited_edges.add(e)
                            faces.append(candidate)
                        continue
                    elif n not in path:
                        stack.append((curr, n, path + [n]))
    return faces

def extract_bridge_chains(G, used_edges):
    """
    Extract connected chains of edges not in any face (bridges)
    """
    remaining_edges = [e for e in G.edges() if tuple(sorted(e)) not in used_edges]
    B = nx.Graph()
    B.add_edges_from(remaining_edges)
    chains = []
    for comp in nx.connected_components(B):
        sub = B.subgraph(comp)
        endpoints = [n for n in sub.nodes() if sub.degree[n] == 1]
        if endpoints:
            start = endpoints[0]
        else:
            start = list(sub.nodes())[0]
        path = []
        visited_edges = set()
        curr = start
        while True:
            neighbors = [n for n in sub.neighbors(curr)
                         if (curr, n) not in visited_edges and (n, curr) not in visited_edges]
            if not neighbors:
                break
            nxt = neighbors[0]
            path.append((curr, nxt))
            visited_edges.add((curr, nxt))
            curr = nxt
        if path:
            chains.append(path)
    return chains

# -----------------------------
# Main function
# -----------------------------
def build_human_like_path(G, pos, dbg_folder):
    os.makedirs(dbg_folder, exist_ok=True)
    pos = {n: (x, -y) for n, (x, y) in pos.items()}  # flip y for plotting

    full_path = []
    visited_nodes = set()
    step_index = 0

    G_work = G.copy()

    while G_work.number_of_edges() > 0:

        # Find faces
        faces = find_faces_degree2(G_work)

        # Mark edges used by faces
        used_edges = set()
        for face in faces:
            for e in face_edges(face):
                used_edges.add(tuple(sorted(e)))

        # Find bridges / chains
        bridge_chains = extract_bridge_chains(G_work, used_edges)

        # Build candidate list
        candidates = []

        # Faces: perimeter metric
        for face in faces:
            score = sum(1 for n in face if n in visited_nodes)
            metric = face_perimeter(face, pos)
            candidates.append({"type": "face", "obj": face, "score": score, "metric": metric})

        # Bridge chains: 2D distance metric
        for chain in bridge_chains:
            nodes = set()
            for u, v in chain:
                nodes.add(u)
                nodes.add(v)
            score = sum(1 for n in nodes if n in visited_nodes)
            metric = chain_distance(chain, pos)
            candidates.append({"type": "bridge_chain", "obj": chain, "score": score, "metric": metric})

        if not candidates:
            break

        # Distance-aware selection
        if visited_nodes:
            last_node = max(visited_nodes, key=lambda n: pos[n][0] + pos[n][1])
            for c in candidates:
                if c["type"] == "face":
                    x = sum(pos[n][0] for n in c["obj"]) / len(c["obj"])
                    y = sum(pos[n][1] for n in c["obj"]) / len(c["obj"])
                    c["distance_from_last"] = math.hypot(x - pos[last_node][0], y - pos[last_node][1])
                else:
                    endpoints = [c["obj"][0][0], c["obj"][-1][1]]
                    c["distance_from_last"] = min(
                        math.hypot(pos[n][0]-pos[last_node][0], pos[n][1]-pos[last_node][1])
                        for n in endpoints
                    )
        else:
            for c in candidates:
                c["distance_from_last"] = 0

        # Sort candidates: prefer connection, face, metric (perimeter/length), penalize distance
        alpha = 1.0
        beta = 0.5
        candidates.sort(key=lambda x: (x["score"], x["type"] == "face", alpha*x["metric"] - beta*x["distance_from_last"]), reverse=True)
        best = candidates[0]

        # Rotate / select edges
        if best["type"] == "face":
            face = best["obj"]
            if visited_nodes:
                # rotate so first node is closest to visited
                for i, n in enumerate(face):
                    if n in visited_nodes:
                        face = face[i:] + face[:i]
                        break
            edges = face_edges(face)
        else:
            edges = best["obj"]

        # Append edges to path
        for u, v in edges:
            full_path.append((u, v))
            visited_nodes.add(u)
            visited_nodes.add(v)
            if G_work.has_edge(u, v):
                G_work.remove_edge(u, v)
            elif G_work.has_edge(v, u):
                G_work.remove_edge(v, u)


        # Debug visualization (clean stroke-only view)
        plt.figure(figsize=(6, 6))

        # draw remaining graph in grey
        for u, v in G_work.edges():
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            plt.plot(x, y, linewidth=0.5, color='black')

        # draw current selected edges in red
        for u, v in edges:
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            plt.plot(x, y, linewidth=2, color='red')

        # remove all axes / padding / decorations
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(
            os.path.join(dbg_folder, f"step_{step_index}.png"),
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close()

        step_index += 1

    # Save final path
    with open(os.path.join(dbg_folder, "final_path.txt"), "w") as f:
        for u, v in full_path:
            f.write(f"{u} {v}\n")

    print(f"Debug saved in {dbg_folder}")
    return full_path