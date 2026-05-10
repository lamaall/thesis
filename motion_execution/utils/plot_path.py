import matplotlib.pyplot as plt

def plot_xyz_path(xyz, show_arrows=False):
    xs = [p[0] for p in xyz]
    ys = [p[1] for p in xyz]
    zs = [p[2] for p in xyz]

    plt.figure(figsize=(6, 6))

    # draw segments with color depending on z
    for i in range(1, len(xyz)):
        x1, y1, z1 = xyz[i - 1]
        x2, y2, z2 = xyz[i]

        if z2 < 0.2:
            color = "blue"   # solid (drawing)
        else:
            color = "red"    # dotted (pen up)

        plt.plot([x1, x2], [y1, y2], color=color, linewidth=2)

        if show_arrows:
            plt.arrow(x1, y1, x2 - x1, y2 - y1,
                      head_width=0.002, length_includes_head=True,
                      color=color, alpha=0.6)

    # start/end markers
    plt.scatter(xs[0], ys[0], color="green", s=60, label="start")
    plt.scatter(xs[-1], ys[-1], color="black", s=60, label="end")

    plt.title("XYZ Path Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()