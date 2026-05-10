import numpy as np


def endpoints(stroke):
    return stroke["path"][0], stroke["path"][-1]


def dist(p, q):
    return np.linalg.norm(np.array(p) - np.array(q))


def stroke_length(stroke):
    return stroke["weight"]


def stroke_points(stroke):
    return set(tuple(p) for p in stroke["path"])


def closest_endpoint(p, stroke):
    a, b = endpoints(stroke)
    return (a, 0) if dist(p, a) <= dist(p, b) else (b, 1)

def choose_start_point(p, stroke):
    a, b = endpoints(stroke)

    path = stroke["path"]
    if path.count(a) > 1:
        return a
    if path.count(b) > 1:
        return b

    return a if dist(p, a) <= dist(p, b) else b

def choose_endpoint(p, stroke, visited_points):
    a, b = endpoints(stroke)
    path = stroke["path"]

    a_in = a in visited_points
    b_in = b in visited_points

    # 1. current point lies on stroke → closer endpoint
    if p in stroke_points(stroke):
        return a if dist(p, a) <= dist(p, b) else b

    # 2. one endpoint is connected to already drawn part
    if a_in and not b_in:
        return a
    if b_in and not a_in:
        return b

    # 3. NEW: endpoint repeats in stroke path (degree hint)
    if path.count(a) > 1 and path.count(b) <= 1:
        return a
    if path.count(b) > 1 and path.count(a) <= 1:
        return b

    if path.count(a) > 1 and path.count(b) > 1:
        return a if dist(p, a) <= dist(p, b) else b

    # 4. fallback → closer endpoint
    return a if dist(p, a) <= dist(p, b) else b


def traverse_from(stroke, start_point):
    a, b = endpoints(stroke)
    return stroke["path"] if start_point == a else stroke["path"][::-1]


# ----------------------------
# Main (FIXED)
# ----------------------------

def plan_stroke_path(strokes):

    strokes = list(strokes)

    visited_ids = set()
    visited_points = set()
    P = []

    # ----------------------------
    # initial stroke
    # ----------------------------
    T_star = max(strokes, key=stroke_length)

    #P.extend(T_star["path"])
    #visited_ids.add(T_star["id"])
    #visited_points |= stroke_points(T_star)

    path = T_star["path"]
    a, b = endpoints(T_star)

    if path.count(a) > 1:
        start = a
    elif path.count(b) > 1:
        start = b
    else:
        start = a

    traversal = traverse_from(T_star, start)

    P.extend(traversal)
    visited_ids.add(T_star["id"])
    visited_points |= stroke_points(T_star)

    p = traversal[-1]

    #p = T_star["path"][-1]

    # ----------------------------
    # main loop
    # ----------------------------
    while len(visited_ids) < len(strokes):

        unvisited = [s for s in strokes if s["id"] not in visited_ids]

        # ----------------------------
        # NEW: connectivity = ANY overlap
        # ----------------------------
        connected = []
        for s in unvisited:
            if len(stroke_points(s) & visited_points) > 0:
                connected.append(s)

        # ----------------------------
        # selection
        # ----------------------------
        if connected:
            T_star = min(
                connected,
                key=lambda s: min(dist(p, endpoints(s)[0]),
                                  dist(p, endpoints(s)[1]))
            )
        else:
            T_star = max(unvisited, key=stroke_length)

        # ----------------------------
        # direction
        # ----------------------------
        #q, _ = closest_endpoint(p, T_star)

        q = choose_endpoint(p, T_star, visited_points)

        traversal = traverse_from(T_star, q)

        # ----------------------------
        # append safely
        # ----------------------------
        if P and traversal[0] == P[-1]:
            P.extend(traversal[1:])
        else:
            P.extend(traversal)

        # ----------------------------
        # update global state
        # ----------------------------
        visited_ids.add(T_star["id"])
        visited_points |= stroke_points(T_star)
        p = traversal[-1]

    return P