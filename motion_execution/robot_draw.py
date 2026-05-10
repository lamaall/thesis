#from motion_execution.canvas.canvas import open_canvas, save_canvas
#from motion_execution.real_robot.movement import robot_drawing
from motion_execution.utils.transform_path_coords import traversal_to_xyz_rescaled
from motion_execution.virtual_robot.enviroment import robot_sim

import threading

def robot_draw(planned_path, strokes, out_dir, name, real_robot=True):

    path = traversal_to_xyz_rescaled(planned_path, strokes)

    if not real_robot:
        robot_sim(path)

"""
    if real_robot:
        root, canvas = open_canvas()

        def run_robot():
            robot_drawing(path)

            save_canvas(canvas, out_dir, name)

            root.after(0, root.destroy)

        robot_thread = threading.Thread(target=run_robot)
        robot_thread.start()

        root.mainloop()

        robot_thread.join()
"""