# import stage
import os
from whiteboard_detection.whiteboard_detection import whiteboard_detection
from graph_construction.graph_construction import graph_construction
from path_planning.plan_path import plan_path
from motion_execution.robot_draw import robot_draw
#from measure.measure import StageProfiler

# input stage
# Load screen shot

INPUT_FOLDER = "dataset"
DEBUG_FOLDER = "debug"
OUTPUT_FOLDER = "results"

DEBUG = True

os.makedirs(DEBUG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


for distance in os.listdir(INPUT_FOLDER):  # close / middle / far

    #if distance != "close":
    #    continue

    distance_path = os.path.join(INPUT_FOLDER, distance)

    if not os.path.isdir(distance_path):
        continue

    for subdir in os.listdir(distance_path):  # folderA, folderB...
        subdir_path = os.path.join(distance_path, subdir)

        if not os.path.isdir(subdir_path):
            continue

        # Create mirrored structure
        out_dir = os.path.join(OUTPUT_FOLDER, distance, subdir)
        dbg_dir = os.path.join(DEBUG_FOLDER, distance, subdir)

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(dbg_dir, exist_ok=True)

        if subdir.lower() in ("car", "cup", "face", "flowe", "hat", "house", "pig", "stickman", "sun", "sword"):
            print("Skipping car folder:", subdir_path)
            continue

        for file in os.listdir(subdir_path):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(subdir_path, file)

            name = os.path.splitext(os.path.basename(path))[0]

            # whiteboard detection stage
            # Detect and rectify the whiteboard

            #prof = StageProfiler(out_dir=out_dir, name=name)

            #prof.start("whiteboard_detection")
            warped_path = whiteboard_detection(path, dbg_dir, name, DEBUG)
            #prof.end("whiteboard_detection")

            if warped_path is None:
                continue

            # graph construction stage
            # Construct a graph from the screen shot

            #prof.start("graph_construction")
            lineG = graph_construction(warped_path, dbg_dir, out_dir, name, DEBUG)
            #prof.end("graph_construction")
            
            if lineG is None:
                continue
            # path planning stage
            # Plan a path from the graph

            #prof.start("path_planning")
            path, strokes = plan_path(lineG, dbg_dir, name, DEBUG)
            #prof.end("path_planning")

            # motion execution stage
            # Execute the planned path

            #prof.start("robot_execution")
            robot_draw(path, strokes, out_dir, name, real_robot=False)
            #prof.end("robot_execution")
            
            #prof.save()
            exit(0)