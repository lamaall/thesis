import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_traversal(path, strokes):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # ----------------------------
    # normalize path
    # ----------------------------
    norm_path = [(float(p[0]), float(p[1])) for p in path]

    xs = [p[0] for p in norm_path]
    ys = [p[1] for p in norm_path]

    # ----------------------------
    # build stroke edge set
    # ----------------------------
    stroke_edges = set()

    for s in strokes:
        pts = s["path"]
        for i in range(len(pts) - 1):
            a = (float(pts[i][0]), float(pts[i][1]))
            b = (float(pts[i+1][0]), float(pts[i+1][1]))
            stroke_edges.add(tuple(sorted([a, b])))

    # ----------------------------
    # plot setup
    # ----------------------------
    point, = ax.plot([], [], 'ro', markersize=8)

    ax.set_xlim(min(xs) - 5, max(xs) + 5)
    ax.set_ylim(min(ys) - 5, max(ys) + 5)
    ax.invert_yaxis()

    solid_lines = []
    dotted_lines = []

    # ----------------------------
    # init
    # ----------------------------
    def init():
        point.set_data([], [])
        return point,

    # ----------------------------
    # update
    # ----------------------------
    def update(frame):
        if frame == 0:
            return point,

        prev = norm_path[frame - 1]
        curr = norm_path[frame]

        edge = tuple(sorted([prev, curr]))

        # decide if this follows a real stroke
        if edge in stroke_edges:
            line, = ax.plot(
                [prev[0], curr[0]],
                [prev[1], curr[1]],
                'b-',
                lw=2
            )
            solid_lines.append(line)
        else:
            line, = ax.plot(
                [prev[0], curr[0]],
                [prev[1], curr[1]],
                color='gray',
                linestyle='dotted',
                lw=2
            )
            dotted_lines.append(line)

        point.set_data([curr[0]], [curr[1]])

        return [point] + solid_lines + dotted_lines

    # ----------------------------
    # animation
    # ----------------------------
    ani = FuncAnimation(
        fig,
        update,
        frames=len(norm_path),
        init_func=init,
        blit=True,
        interval=400,
        repeat=False
    )

    plt.show()