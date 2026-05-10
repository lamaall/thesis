import numpy as np
import itertools


# ----------------------------
# Helpers
# ----------------------------

def unit(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    return v / n


def tangent_at_node(path, node):
    if path[0] == node:
        return unit(np.array(path[0]) - np.array(path[1]))
    elif path[-1] == node:
        return unit(np.array(path[-1]) - np.array(path[-2]))
    return None


def merge_paths(p1, p2, junction):
    if p1[-1] != junction:
        p1 = p1[::-1]
    if p2[0] != junction:
        p2 = p2[::-1]
    return p1[:-1] + p2


def angle_cost(u, v):
    return float(np.dot(u, v))

# ----------------------------
# Main
# ----------------------------

def construct_strokes(G):

    id_counter = itertools.count()

    strokes = [
        {
            "id": next(id_counter),
            "path": data["path"],
            "weight": data["weight"],
        }
        for _, _, data in G.edges(data=True)
    ]

    # ----------------------------
    # build node map
    # ----------------------------
    def build_node_map(strokes):
        node_map = {}
        for s in strokes:
            a, b = s["path"][0], s["path"][-1]
            node_map.setdefault(a, []).append(s)
            node_map.setdefault(b, []).append(s)
        return node_map

    # ----------------------------
    # main contraction loop
    # ----------------------------
    while True:

        node_map = build_node_map(strokes)
        junctions = [n for n, lst in node_map.items() if len(lst) > 1]

        if not junctions:
            break

        merged_any = False

        for j in junctions:

            incident = node_map.get(j, [])

            # remove duplicates by ID (critical fix)
            unique = {}
            for s in incident:
                unique[s["id"]] = s
            incident = list(unique.values())

            if len(incident) < 2:
                continue

            best_pair = None
            best_score = float("inf")

            # find best compatible pair
            for i in range(len(incident)):
                for k in range(i + 1, len(incident)):

                    s1 = incident[i]
                    s2 = incident[k]

                    if s1["id"] == s2["id"]:
                        continue

                    t1 = tangent_at_node(s1["path"], j)
                    t2 = tangent_at_node(s2["path"], j)

                    if t1 is None or t2 is None:
                        continue

                    score = angle_cost(t1, t2)

                    if score < best_score:
                        best_score = score
                        best_pair = (s1, s2)

            if best_pair is None:
                continue

            s1, s2 = best_pair

            new_path = merge_paths(s1["path"], s2["path"], j)
            new_weight = s1["weight"] + s2["weight"]

            new_stroke = {
                "id": next(id_counter),
                "path": new_path,
                "weight": new_weight,
            }

            # remove old strokes safely
            remove_ids = {s1["id"], s2["id"]}
            strokes = [s for s in strokes if s["id"] not in remove_ids]
            strokes.append(new_stroke)

            merged_any = True
            break  # restart because topology changed

        if not merged_any:
            break

    return strokes