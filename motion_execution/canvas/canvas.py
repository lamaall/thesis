import tkinter as tk
from screeninfo import get_monitors
import time
import os

# -----------------------
# GLOBAL STATE
# -----------------------
prev_x = None
prev_y = None
drawing = False

x = []
y = []
strokes = []

canvas = None
root = None

LINE_WIDTH = 35

# -----------------------
# DEBUG
# -----------------------
last_time = None

def log_event(event, name):
    global last_time

    now = time.time()
    dt = 0 if last_time is None else now - last_time
    last_time = now

    print(f"{name:5s} | x={event.x:4d} y={event.y:4d} | dt={dt:.3f}")

    # highlight gaps
    if dt > 0.08:
        print("⚠️ GAP DETECTED")


# -----------------------
# MOUSE / TOUCH EVENTS
# -----------------------
def on_mouse_down(event):
    global prev_x, prev_y, drawing

    log_event(event, "DOWN")

    drawing = True
    prev_x, prev_y = event.x, event.y


def on_mouse_move(event):
    global prev_x, prev_y, x, y, canvas, drawing

    log_event(event, "MOVE")

    if not drawing:
        return

    cx, cy = event.x, event.y
    x.append(cx)
    y.append(cy)

    if prev_x is not None and prev_y is not None:
        canvas.create_line(
            prev_x, prev_y, cx, cy,
            fill="black",
            width=LINE_WIDTH,
            capstyle=tk.ROUND,
            smooth=True
        )

    prev_x, prev_y = cx, cy


def on_mouse_release(event):
    global x, y, prev_x, prev_y, strokes, drawing

    log_event(event, "UP")

    drawing = False

    if x and y:
        strokes.append([x.copy(), y.copy()])
        print("Stroke:", len(strokes))

    x.clear()
    y.clear()
    prev_x, prev_y = None, None


# -----------------------
# PICK NON-PRIMARY DISPLAY
# -----------------------
def get_non_primary_monitor():
    monitors = get_monitors()

    print("\nDetected monitors:")
    for i, m in enumerate(monitors):
        print(f"{i}: {m}")

    primary = None
    secondary = None

    for m in monitors:
        if m.x == 0 and m.y == 0:
            primary = m
        else:
            secondary = m
            break

    if secondary is None:
        secondary = primary

    print("\nUsing monitor:", secondary)
    return secondary


# -----------------------
# OPEN CANVAS
# -----------------------
def open_canvas():
    global canvas, root

    monitor = get_non_primary_monitor()

    root = tk.Tk()
    root.title("Robot Canvas")

    root.geometry(f"{monitor.width}x{monitor.height}+{monitor.x}+{monitor.y}")
    root.overrideredirect(True)

    canvas = tk.Canvas(root, bg="white", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    canvas.bind('<ButtonPress-1>', on_mouse_down)
    canvas.bind('<B1-Motion>', on_mouse_move)
    canvas.bind('<ButtonRelease-1>', on_mouse_release)

    # exit key
    root.bind("<Escape>", lambda e: root.destroy())

    root.lift()
    root.focus_force()

    return root, canvas

def save_canvas(canvas, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)

    ps_path = os.path.join(out_dir, f"{name}.ps")

    # make sure everything is drawn
    canvas.update_idletasks()
    canvas.update()

    canvas.postscript(file=ps_path, colormode="color")

    print("Saved canvas as:", ps_path)


if __name__ == "__main__":
    open_canvas()